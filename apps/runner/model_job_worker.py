from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from apps.runner.verify import ModelBenchmark, RealModelVerifier
from april_common.errors import RuntimeUnavailableError
from april_common.hardware_profile import safe_hardware_profile
from april_common.settings import load_settings
from april_common.thermal_state import summarize_thermal_samples
from services.april_runtime.colibri_backend import ColibriBackend
from services.april_runtime.model_registry import ModelRegistry
from services.evaluation.model_quality import ColibriEvaluationClient, evaluate_model_quality
from services.jobs.model_jobs import validate_registered_model
from services.tool_worker.client import ToolWorkerProcessManager, ToolWorkerUnavailable


def _verification(home: Path, model_id: str) -> dict[str, Any]:  # pragma: no cover
    settings = load_settings(root=home, legacy_credential_migration=True)
    model = validate_registered_model(settings, model_id)
    if model.artifact_kind == "colibri_model_directory":
        return asyncio.run(_colibri_verification(settings, model_id))
    verifier = RealModelVerifier(
        home=home,
        model_path=model.path,
        inherit_process_group=True,
    )
    checks = verifier.run()
    return {
        "schema_version": 1,
        "report_type": "model_import_verification",
        "model_id": model_id,
        "passed": all(check.ok for check in checks),
        "checks": [
            {"name": check.name, "ok": check.ok, "status": check.status} for check in checks
        ],
        "measurements": {
            "load_time_seconds": verifier.load_time_seconds,
            "first_token_latency_seconds": verifier.first_token_latency_seconds,
            "generation_time_seconds": verifier.generation_time_seconds,
            "output_tokens": verifier.output_tokens,
            "tokens_per_second": verifier.tokens_per_second,
            "runtime_rss_bytes": verifier.runtime_rss_bytes,
        },
    }


def _benchmark(home: Path, model_id: str) -> dict[str, Any]:  # pragma: no cover
    settings = load_settings(root=home, legacy_credential_migration=True)
    model = validate_registered_model(settings, model_id)
    if model.artifact_kind == "colibri_model_directory":
        return asyncio.run(_colibri_benchmark(settings, model_id))
    benchmark = ModelBenchmark(
        home=home,
        model_path=model.path,
        prompt="Return a compact valid JSON object.",
        runs=settings.benchmark.runs,
        max_output_tokens=64,
        keep_loaded=False,
        inherit_process_group=True,
    )

    def quality(session: ModelBenchmark) -> dict[str, Any]:
        return asyncio.run(
            _quality_evaluation(
                settings,
                session=session,
                model_id="april-brain",
            )
        )

    runs, quality_result = benchmark.run_with_evaluation(quality)
    successful = [run for run in runs if run.ok]
    thermal_evidence = summarize_thermal_samples(
        benchmark.thermal_samples,
        performance_degradation_suggested=None,
    )
    quality_result = quality_result or {}
    load_times = [run.load_time_seconds for run in successful]
    warm_load_times = [
        run.warm_load_time_seconds for run in successful if run.warm_load_time_seconds is not None
    ]
    unload_times = [
        run.unload_time_seconds for run in successful if run.unload_time_seconds is not None
    ]
    return {
        "schema_version": 2,
        "report_type": "model_benchmark",
        "model_id": model_id,
        "passed": len(successful) == len(runs) and bool(runs),
        "runs": [
            {
                "run_index": run.run_index,
                "ok": run.ok,
                "load_time_seconds": run.load_time_seconds,
                "warm_load_time_seconds": run.warm_load_time_seconds,
                "first_token_latency_seconds": run.first_token_latency_seconds,
                "generation_time_seconds": run.generation_time_seconds,
                "output_tokens": run.output_tokens,
                "tokens_per_second": run.tokens_per_second,
                "unload_success": run.unload_success,
                "unload_time_seconds": run.unload_time_seconds,
                "process_rss_bytes": run.process_rss_bytes,
                "peak_process_rss_bytes": run.peak_process_rss_bytes,
                "prompt_token_count": run.prompt_token_count,
                "prompt_eval_duration_seconds": run.prompt_eval_duration_seconds,
            }
            for run in runs
        ],
        "fixture_set": quality_result.get("fixture_set"),
        "quality": {
            key: quality_result.get(key) for key in ("routing", "strict_json", "coding", "context")
        },
        "routing_accuracy": quality_result.get("routing_accuracy"),
        "strict_json_first_pass_reliability": quality_result.get(
            "strict_json_first_pass_reliability"
        ),
        "structured_json_reliability": quality_result.get("structured_json_reliability"),
        "coding_fixture_pass_rate": quality_result.get("coding_fixture_pass_rate"),
        "context_handling_reliability": quality_result.get("context_handling_reliability"),
        "lifecycle": {
            "cold_load_time_seconds": load_times[0] if load_times else None,
            "warm_load_time_seconds": (
                sum(warm_load_times) / len(warm_load_times) if warm_load_times else None
            ),
            "unload_time_seconds": (
                sum(unload_times) / len(unload_times) if unload_times else None
            ),
            "model_switch_time_seconds": (
                (sum(load_times) + sum(unload_times)) / len(load_times) if load_times else None
            ),
            "load_failures": sum(not run.ok for run in runs),
            "unload_failures": sum(run.ok and not run.unload_success for run in runs),
            "prompt_processing_duration_source": "first_token_latency_proxy",
        },
        "measurements_unavailable": [
            *(["thermal_throttling"] if thermal_evidence.direct_measurement_unavailable else []),
            *(
                ["coding_fixture_pass_rate"]
                if quality_result.get("coding_fixture_pass_rate") is None
                else []
            ),
        ],
        "thermal_evidence": thermal_evidence.model_dump(mode="json"),
        "simulated": settings.runtime.backend == "fake",
        "hardware_profile": safe_hardware_profile(),
    }


async def _quality_evaluation(
    settings: Any,
    *,
    session: ModelBenchmark,
    model_id: str,
) -> dict[str, Any]:  # pragma: no cover
    coding_root = session.temp / "coding-fixtures"
    coding_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    manager = ToolWorkerProcessManager(
        april_home=settings.home,
        allowed_roots=(coding_root,),
        runtime_directory=session.temp / "tool-worker",
        environment=settings.environment,
        development_unsandboxed_override=settings.workers.development_unsandboxed_override,
    )
    client = None
    try:
        client = await manager.start()
    except ToolWorkerUnavailable:
        if settings.environment == "production":
            client = None
    try:
        return await evaluate_model_quality(
            settings,
            runtime_url=session.runtime_url,
            runtime_token=session.runtime_token,
            model_id=model_id,
            coding_root=coding_root,
            tool_worker=client,
            fixture_home=Path(__file__).resolve().parents[2],
        )
    finally:
        await manager.stop()


async def _colibri_backend(settings: Any, model_id: str) -> ColibriBackend:
    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml", root=settings.home
    )
    model = registry.get(model_id)
    tokenizer_path = model.colibri_tokenizer_path
    if tokenizer_path is not None and not tokenizer_path.is_absolute():
        tokenizer_path = model.resolved_path(registry.root) / tokenizer_path
    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name or model.name,
        tokenizer_path=tokenizer_path,
    )
    await backend.load(model.model_copy(update={"path": model.resolved_path(registry.root)}))
    return backend


async def _colibri_verification(settings: Any, model_id: str) -> dict[str, Any]:
    started = time.monotonic()
    backend = await _colibri_backend(settings, model_id)
    try:
        result = await backend.generate(
            "Return a compact valid JSON object.",
            temperature=0.0,
            max_output_tokens=64,
        )
        return {
            "schema_version": 2,
            "report_type": "model_import_verification",
            "model_id": model_id,
            "passed": bool(result.text),
            "checks": [
                {"name": "loopback_health_identity", "ok": True, "status": "passed"},
                {"name": "exact_local_tokenizer", "ok": True, "status": "passed"},
                {
                    "name": "generation",
                    "ok": bool(result.text),
                    "status": "passed" if result.text else "failed",
                },
            ],
            "measurements": {
                "generation_time_seconds": time.monotonic() - started,
                "output_tokens": result.output_tokens,
                "tokens_per_second": (
                    result.output_tokens / max(time.monotonic() - started, 0.000001)
                ),
                "seed_control": "unsupported",
            },
            "artifact_kind": "colibri_model_directory",
            "simulated": False,
        }
    finally:
        await backend.unload()


async def _colibri_benchmark(settings: Any, model_id: str) -> dict[str, Any]:
    backend = await _colibri_backend(settings, model_id)
    runs: list[dict[str, Any]] = []
    try:
        for run_index in range(settings.benchmark.runs):
            started = time.monotonic()
            prompt = "Return a compact valid JSON object."
            prompt_tokens = len(await backend.tokenize(prompt))
            result = await backend.generate(
                prompt,
                temperature=0.0,
                max_output_tokens=64,
            )
            elapsed = time.monotonic() - started
            runs.append(
                {
                    "run_index": run_index,
                    "ok": bool(result.text),
                    "generation_time_seconds": elapsed,
                    "first_token_latency_seconds": elapsed,
                    "output_tokens": result.output_tokens,
                    "tokens_per_second": result.output_tokens / max(elapsed, 0.000001),
                    "prompt_token_count": prompt_tokens,
                    "seed_control": "unsupported",
                    "process_rss_bytes": None,
                }
            )
    finally:
        await backend.unload()
    quality = await _colibri_quality(settings, model_id)
    return {
        "schema_version": 2,
        "report_type": "model_benchmark",
        "model_id": model_id,
        "passed": bool(runs) and all(run["ok"] for run in runs),
        "runs": runs,
        "artifact_kind": "colibri_model_directory",
        "fixture_set": quality.get("fixture_set"),
        "quality": {
            key: quality.get(key) for key in ("routing", "strict_json", "coding", "context")
        },
        "routing_accuracy": quality.get("routing_accuracy"),
        "strict_json_first_pass_reliability": quality.get("strict_json_first_pass_reliability"),
        "structured_json_reliability": quality.get("structured_json_reliability"),
        "coding_fixture_pass_rate": quality.get("coding_fixture_pass_rate"),
        "context_handling_reliability": quality.get("context_handling_reliability"),
        "measurements_unavailable": ["rss", "coding_fixture_pass_rate"],
        "seed_control": "unsupported",
        "simulated": False,
    }


async def _colibri_quality(settings: Any, model_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="april-colibri-eval-") as temporary:
        root = Path(temporary)
        coding_root = root / "coding-fixtures"
        coding_root.mkdir(mode=0o700)
        manager = ToolWorkerProcessManager(
            april_home=settings.home,
            allowed_roots=(coding_root,),
            runtime_directory=root / "tool-worker",
            environment=settings.environment,
            development_unsandboxed_override=settings.workers.development_unsandboxed_override,
        )
        tool_worker = None
        try:
            tool_worker = await manager.start()
            backend = await _colibri_backend(settings, model_id)
            try:
                return await evaluate_model_quality(
                    settings,
                    runtime_url="",
                    runtime_token=None,
                    model_id=model_id,
                    coding_root=coding_root,
                    tool_worker=tool_worker,
                    client=ColibriEvaluationClient(backend=backend, model_id=model_id),
                    fixture_home=Path(__file__).resolve().parents[2],
                )
            finally:
                await backend.unload()
        except ToolWorkerUnavailable:
            return {
                "fixture_set": None,
                "coding_fixture_pass_rate": None,
                "unavailable_reason": "tool_worker_unavailable",
            }
        finally:
            await manager.stop()


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", required=True, choices=("verify", "benchmark"))
    args = parser.parse_args()
    try:
        payload = (
            _verification(args.home, args.model_id)
            if args.mode == "verify"
            else _benchmark(args.home, args.model_id)
        )
    except RuntimeUnavailableError as exc:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "passed": False,
                    "unavailable_reason": "configured_but_unavailable",
                    "error_code": exc.code,
                    "artifact_kind": "colibri_model_directory",
                },
                sort_keys=True,
            )
        )
    except Exception:
        print(json.dumps({"error_code": f"model_{args.mode}_worker_failed"}, sort_keys=True))
        raise SystemExit(1) from None
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
