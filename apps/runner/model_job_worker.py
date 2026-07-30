from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from apps.runner.verify import ModelBenchmark, RealModelVerifier
from april_common.hardware_profile import safe_hardware_profile
from april_common.settings import load_settings
from april_common.thermal_state import summarize_thermal_samples
from services.evaluation.model_quality import evaluate_model_quality
from services.jobs.model_jobs import validate_registered_model
from services.tool_worker.client import ToolWorkerProcessManager, ToolWorkerUnavailable


def _verification(home: Path, model_id: str) -> dict[str, Any]:  # pragma: no cover
    settings = load_settings(root=home, legacy_credential_migration=True)
    model = validate_registered_model(settings, model_id)
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
        )
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
    except Exception:
        print(json.dumps({"error_code": f"model_{args.mode}_worker_failed"}, sort_keys=True))
        raise SystemExit(1) from None
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
