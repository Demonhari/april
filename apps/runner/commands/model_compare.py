from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import typer

from apps.cli.render import console
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.hardware_profile import safe_hardware_profile
from april_common.settings import BenchmarkSettings, load_settings
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import ModelRegistry
from services.evaluation.model_quality import fixture_set_metadata
from services.jobs.model_jobs import run_model_utility_job
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore
from services.memory.database import Database
from services.memory.migrations import run_migrations

_FIXTURES = (
    "routing_intent_cases_v1",
    "structured_json_cases_v1",
    "coding_cases_v1",
    "long_context_tokenizer_measured_v1",
    "load_unload_cycles_v1",
)
_FIXTURE_SET_ID = hashlib.sha256(
    json.dumps(_FIXTURES, separators=(",", ":")).encode("utf-8")
).hexdigest()
_QUALITY_METRICS = (
    "routing_accuracy",
    "structured_json_reliability",
    "coding_fixture_pass_rate",
    "context_handling_reliability",
    "load_unload_reliability",
)


def register_model_compare(model_app: typer.Typer) -> None:
    @model_app.command("compare-setups")
    def compare_setups(
        shared_model_id: str = typer.Option(..., "--shared-model-id"),
        cooldown_seconds: float = typer.Option(
            0.0,
            "--cooldown-seconds",
            min=0.0,
            max=600.0,
        ),
        wait: bool = typer.Option(False, "--wait"),
        json_output: bool = typer.Option(False, "--json"),
        wait_timeout: float = typer.Option(14_400.0, "--wait-timeout", min=1.0, max=86_400.0),
    ) -> None:
        job = asyncio.run(_submit_comparison(shared_model_id, cooldown_seconds))
        if wait:
            job = asyncio.run(_wait_for_comparison(job["id"], wait_timeout))
        if json_output:
            console.print_json(data=job)
            return
        console.print(f"Durable model setup comparison job {job['id']} is {job['status']}.")
        console.print(f"  run april jobs show {job['id']}")
        console.print(f"  run april jobs cancel {job['id']}")
        console.print(f"  run april jobs retry {job['id']}")


async def _compare(
    shared_model_id: str,
    *,
    cooldown_seconds: float = 0.0,
    settings: Any | None = None,
    cancellation_event: asyncio.Event | None = None,
    progress: Callable[[int, str, dict[str, Any]], Awaitable[None]] | None = None,
    resume: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    cancellation = cancellation_event or asyncio.Event()
    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml",
        root=settings.home,
    )
    shared = registry.get(shared_model_id)
    specialists = [
        model
        for model in registry.list()
        if model.id != shared.id and model.role not in {"embedding", "router"}
    ]
    if not specialists:
        raise ValueError("Current specialist configuration has no comparable registered models.")
    completed = resume.get("completed_results") if isinstance(resume, Mapping) else None
    restored = completed if isinstance(completed, Mapping) else {}
    specialist_results: list[dict[str, Any]] = []
    total = len(specialists) + 1
    for completed_count, model in enumerate(specialists, start=1):
        if cancellation.is_set():
            raise asyncio.CancelledError
        previous = restored.get(model.id)
        if isinstance(previous, Mapping):
            specialist_results.append(dict(previous))
        else:
            specialist_results.append(
                await run_model_utility_job(
                    settings,
                    model_id=model.id,
                    mode="benchmark",
                    cancellation_event=cancellation,
                    timeout_seconds=3600.0,
                )
            )
        restored = {
            **restored,
            model.id: _redacted_benchmark(specialist_results[-1]),
        }
        if progress is not None:
            await progress(
                int(90 * completed_count / total),
                "model_setup_comparison_specialist_completed",
                {
                    "checkpoint_type": "model_setup_comparison",
                    "completed_results": dict(restored),
                    "completed_model_ids": sorted(restored),
                },
            )
        if cooldown_seconds:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(cancellation.wait(), timeout=cooldown_seconds)
            if cancellation.is_set():
                raise asyncio.CancelledError
    previous_shared = restored.get(shared.id)
    shared_result = (
        dict(previous_shared)
        if isinstance(previous_shared, Mapping)
        else await run_model_utility_job(
            settings,
            model_id=shared.id,
            mode="benchmark",
            cancellation_event=cancellation,
            timeout_seconds=3600.0,
        )
    )
    restored = {**restored, shared.id: _redacted_benchmark(shared_result)}
    if progress is not None:
        await progress(
            95,
            "model_setup_comparison_shared_completed",
            {
                "checkpoint_type": "model_setup_comparison",
                "completed_results": dict(restored),
                "completed_model_ids": sorted(restored),
            },
        )
    specialist_summary = summarize_setup(specialist_results)
    shared_summary = summarize_setup([shared_result])
    installed_fixture_set = fixture_set_metadata(settings.home)
    simulated = any(
        bool(result.get("simulated")) for result in [*specialist_results, shared_result]
    )
    recommendation, gate_failures = recommend_setup(
        specialist_summary,
        shared_summary,
        settings.benchmark,
        simulated=simulated,
    )
    unavailable = sorted(
        {
            *specialist_summary["unavailable_measurements"],
            *shared_summary["unavailable_measurements"],
        }
    )
    recommendation_status = {
        "shared_model_recommended_for_manual_review": "recommended",
        "retain_specialist_configuration": "manual_review_required",
        "candidate_failed_no_regression_gates": "comparison_failed",
        "insufficient_evidence": "insufficient_evidence",
    }[recommendation]
    return {
        "schema_version": 2,
        "report_type": "model_setup_comparison",
        "generated_at": utc_now_iso(),
        "config_fingerprint": config_fingerprint_digest(settings.home),
        "fixture_set": {
            "id": installed_fixture_set["sha256"],
            "version": installed_fixture_set["version"],
            "sha256": installed_fixture_set["sha256"],
            "installed": installed_fixture_set["installed"],
            "fixtures": list(_FIXTURES),
            "identical_for_both_setups": True,
            "long_context_token_count_source": "model_tokenizer_only",
            "character_estimation_used": False,
        },
        "current_specialist_configuration": {
            "fixture_set_id": installed_fixture_set["sha256"],
            "models": [_redacted_benchmark(result) for result in specialist_results],
            "summary": specialist_summary,
        },
        "shared_model": {
            "fixture_set_id": installed_fixture_set["sha256"],
            "model": _redacted_benchmark(shared_result),
            "summary": shared_summary,
        },
        "benchmark_rounds": settings.benchmark.runs,
        "cooldown_seconds": cooldown_seconds,
        "thresholds": _thresholds(settings.benchmark),
        "gate_failures": gate_failures,
        "recommendation": recommendation,
        "recommendation_status": recommendation_status,
        "production_recommendation": (
            recommendation == "shared_model_recommended_for_manual_review" and not simulated
        ),
        "automatic_selection_performed": False,
        "automatic_activation_performed": False,
        "thermal_measurement": {
            "available": False,
            "temperature_celsius": None,
            "hardware_throttle_state": None,
            "reason": "safe unprivileged thermal measurement was not exposed",
            "thermal_throttling_suspected": _thermal_suspected(
                shared_summary,
                settings.benchmark.maximum_decline_fraction,
            ),
            "inference_only": True,
        },
        "simulated": simulated,
        "hardware_profile": safe_hardware_profile(),
        "production_eligible": (
            recommendation_status == "recommended"
            and not simulated
            and not [item for item in unavailable if item != "thermal_throttling"]
        ),
        "unavailable_measurements": unavailable,
    }


async def _submit_comparison(shared_model_id: str, cooldown_seconds: float) -> dict[str, Any]:
    settings = load_settings()
    database = Database(settings.database_path)
    await database.connect()
    try:
        await run_migrations(database)
        store = JobStore(
            database,
            default_job_registry(
                finetune_enabled=settings.finetune.enabled,
                evolution_enabled=settings.evolution.enabled,
            ),
        )
        job = await store.submit(
            job_type="model_setup_comparison",
            payload={
                "shared_model_id": shared_model_id,
                "cooldown_seconds": cooldown_seconds,
            },
            owner="local-user",
        )
        return job.model_dump(mode="json")
    finally:
        await database.close()


async def _wait_for_comparison(job_id: str, timeout_seconds: float) -> dict[str, Any]:
    settings = load_settings()
    database = Database(settings.database_path)
    await database.connect()
    try:
        await run_migrations(database)
        store = JobStore(
            database,
            default_job_registry(
                finetune_enabled=settings.finetune.enabled,
                evolution_enabled=settings.evolution.enabled,
            ),
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = await store.require(job_id)
            if job.status.value in {"cancelled", "succeeded", "failed", "interrupted"}:
                return job.model_dump(mode="json")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting; the durable comparison job is still running."
                )
            await asyncio.sleep(0.25)
    finally:
        await database.close()


def summarize_setup(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runs = [
        dict(run)
        for result in results
        for run in result.get("runs", [])
        if isinstance(run, Mapping)
    ]
    successful = [run for run in runs if run.get("ok") is True]
    sustained = calculate_sustained_performance(successful)
    metrics: dict[str, float | int | None] = {
        "model_load_time_seconds": _average(successful, "load_time_seconds"),
        "first_token_latency_seconds": _average(
            successful,
            "first_token_latency_seconds",
        ),
        "output_tokens_per_second": _average(successful, "tokens_per_second"),
        "prompt_processing_tokens_per_second": _measured_prompt_rate(successful),
        "process_rss_bytes": _average(successful, "process_rss_bytes"),
        "peak_process_rss_bytes": _maximum(successful, "peak_process_rss_bytes"),
        "specialist_switching_overhead_seconds": _average_results(
            results,
            "specialist_switching_overhead_seconds",
        ),
        "strict_json_first_pass_reliability": _quality_metric(
            results,
            "strict_json_first_pass_reliability",
        ),
        **{metric: _quality_metric(results, metric) for metric in _QUALITY_METRICS},
    }
    unavailable = [name for name, value in metrics.items() if value is None]
    if sustained["sustained_performance_degradation_fraction"] is None:
        unavailable.append("sustained_performance_degradation_fraction")
    declared_unavailable = {
        str(name) for result in results for name in result.get("measurements_unavailable", [])
    }
    return {
        "passed": bool(results) and all(result.get("passed") is True for result in results),
        "measurements": metrics,
        "repeated_runs": {
            "latency_seconds": [run.get("first_token_latency_seconds") for run in successful],
            "throughput_tokens_per_second": [run.get("tokens_per_second") for run in successful],
        },
        "sustained_performance": sustained,
        "unavailable_measurements": sorted(set(unavailable) | declared_unavailable),
        "evidence": _evidence_summary(results),
    }


def calculate_sustained_performance(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        return {
            "first_round_tokens_per_second": None,
            "final_round_tokens_per_second": None,
            "first_round_latency_seconds": None,
            "final_round_latency_seconds": None,
            "throughput_degradation_fraction": None,
            "latency_degradation_fraction": None,
            "sustained_performance_degradation_fraction": None,
        }
    first = runs[0]
    final = runs[-1]
    first_tps = _number(first.get("tokens_per_second"))
    final_tps = _number(final.get("tokens_per_second"))
    first_latency = _number(first.get("first_token_latency_seconds"))
    final_latency = _number(final.get("first_token_latency_seconds"))
    throughput_degradation = (
        max(0.0, (first_tps - final_tps) / first_tps)
        if first_tps is not None and first_tps > 0 and final_tps is not None
        else None
    )
    latency_degradation = (
        max(0.0, (final_latency - first_latency) / first_latency)
        if first_latency is not None and first_latency > 0 and final_latency is not None
        else None
    )
    available = [
        value for value in (throughput_degradation, latency_degradation) if value is not None
    ]
    return {
        "first_round_tokens_per_second": first_tps,
        "final_round_tokens_per_second": final_tps,
        "first_round_latency_seconds": first_latency,
        "final_round_latency_seconds": final_latency,
        "throughput_degradation_fraction": throughput_degradation,
        "latency_degradation_fraction": latency_degradation,
        "sustained_performance_degradation_fraction": max(available) if available else None,
    }


def recommend_setup(
    specialist: Mapping[str, Any],
    shared: Mapping[str, Any],
    thresholds: BenchmarkSettings,
    *,
    simulated: bool,
) -> tuple[str, list[str]]:
    if simulated:
        return "insufficient_evidence", ["simulated_results_are_not_production_evidence"]
    for label, summary in (("specialist", specialist), ("shared", shared)):
        evidence = summary.get("evidence")
        if isinstance(evidence, Mapping):
            if evidence.get("fixtures_installed") is not True:
                return "insufficient_evidence", [f"{label}_fixtures_missing"]
            if evidence.get("identical_fixture_hash") is not True:
                return "insufficient_evidence", [f"{label}_fixture_hash_inconsistent"]
    if shared.get("passed") is not True:
        return "candidate_failed_no_regression_gates", ["shared_model_benchmark_failed"]
    specialist_metrics = _metrics(specialist)
    shared_metrics = _metrics(shared)
    required = {
        "output_tokens_per_second",
        *_QUALITY_METRICS,
    }
    conditional_required = {
        "strict_json_first_pass_reliability",
        "first_token_latency_seconds",
        "prompt_processing_tokens_per_second",
        "peak_process_rss_bytes",
        "model_load_time_seconds",
        "specialist_switching_overhead_seconds",
    }
    required.update(
        metric
        for metric in conditional_required
        if metric in specialist_metrics or metric in shared_metrics
    )
    specialist_sustained = _sustained_fraction(specialist)
    shared_sustained = _sustained_fraction(shared)
    missing = sorted(
        metric
        for metric in required
        if specialist_metrics.get(metric) is None or shared_metrics.get(metric) is None
    )
    if specialist_sustained is None or shared_sustained is None:
        missing.append("sustained_performance_degradation_fraction")
    if missing:
        return "insufficient_evidence", [f"missing:{metric}" for metric in sorted(set(missing))]
    assert specialist_sustained is not None
    assert shared_sustained is not None
    absolute_failures: list[str] = []
    absolute_thresholds = {
        "routing_accuracy": thresholds.minimum_routing_accuracy,
        "structured_json_reliability": thresholds.minimum_structured_json_reliability,
        "coding_fixture_pass_rate": thresholds.minimum_coding_pass_rate,
    }
    if "strict_json_first_pass_reliability" in required:
        absolute_thresholds["strict_json_first_pass_reliability"] = (
            thresholds.minimum_structured_json_reliability
        )
    for metric, minimum in absolute_thresholds.items():
        value = float(shared_metrics[metric])
        if value < minimum:
            absolute_failures.append(f"{metric}_below_configured_minimum")
    if shared_sustained > thresholds.maximum_decline_fraction:
        absolute_failures.append("sustained_degradation_above_configured_maximum")
    if absolute_failures:
        return "candidate_failed_no_regression_gates", absolute_failures
    relative_failures: list[str] = []
    lower_is_better = {
        "first_token_latency_seconds",
        "peak_process_rss_bytes",
        "model_load_time_seconds",
        "specialist_switching_overhead_seconds",
    }
    for metric in required:
        specialist_value = float(specialist_metrics[metric])
        shared_value = float(shared_metrics[metric])
        regressed = (
            shared_value > specialist_value * (1.0 + thresholds.maximum_decline_fraction)
            if metric in lower_is_better
            else shared_value < specialist_value * (1.0 - thresholds.maximum_decline_fraction)
        )
        if regressed:
            relative_failures.append(f"{metric}_regressed")
    if relative_failures:
        return "retain_specialist_configuration", relative_failures
    return "shared_model_recommended_for_manual_review", []


def _redacted_benchmark(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": result["model_id"],
        "role": result["role"],
        "model_basename": result["model_basename"],
        "model_sha256": result["model_sha256"],
        "model_size": result["model_size"],
        "passed": bool(result.get("passed")),
        "simulated": bool(result.get("simulated")),
        "fixture_set": result.get("fixture_set"),
        "routing_accuracy": result.get("routing_accuracy"),
        "strict_json_first_pass_reliability": result.get("strict_json_first_pass_reliability"),
        "structured_json_reliability": result.get("structured_json_reliability"),
        "coding_fixture_pass_rate": result.get("coding_fixture_pass_rate"),
        "context_handling_reliability": result.get("context_handling_reliability"),
        "lifecycle": result.get("lifecycle"),
        "runs": [
            {
                "run_index": run.get("run_index"),
                "ok": run.get("ok"),
                "load_time_seconds": run.get("load_time_seconds"),
                "warm_load_time_seconds": run.get("warm_load_time_seconds"),
                "first_token_latency_seconds": run.get("first_token_latency_seconds"),
                "generation_time_seconds": run.get("generation_time_seconds"),
                "tokens_per_second": run.get("tokens_per_second"),
                "prompt_token_count": run.get("prompt_token_count"),
                "prompt_eval_duration_seconds": run.get("prompt_eval_duration_seconds"),
                "prompt_processing_tokens_per_second": _run_prompt_rate(run),
                "process_rss_bytes": run.get("process_rss_bytes"),
                "peak_process_rss_bytes": run.get("peak_process_rss_bytes"),
                "unload_success": run.get("unload_success"),
                "unload_time_seconds": run.get("unload_time_seconds"),
            }
            for run in result.get("runs", [])
            if isinstance(run, Mapping)
        ],
    }


def _measured_prompt_rate(runs: Sequence[Mapping[str, Any]]) -> float | None:
    values = [_run_prompt_rate(run) for run in runs]
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _run_prompt_rate(run: Mapping[str, Any]) -> float | None:
    tokens = _number(run.get("prompt_token_count"))
    duration = _number(run.get("prompt_eval_duration_seconds"))
    if tokens is None or duration is None or duration <= 0:
        return None
    return tokens / duration


def _quality_metric(results: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for result in results:
        direct = _number(result.get(key))
        measurements = result.get("measurements")
        nested = _number(measurements.get(key)) if isinstance(measurements, Mapping) else None
        value = direct if direct is not None else nested
        if value is not None:
            values.append(value)
    if not values and key == "load_unload_reliability":
        runs = [
            run for result in results for run in result.get("runs", []) if isinstance(run, Mapping)
        ]
        if runs:
            return sum(
                run.get("ok") is True and run.get("unload_success") is True for run in runs
            ) / len(runs)
    return sum(values) / len(values) if values else None


def _average(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _maximum(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _average_results(results: Sequence[Mapping[str, Any]], key: str) -> float | None:
    direct = _average(results, key)
    if direct is not None:
        return direct
    values: list[float] = []
    lifecycle_key = {
        "specialist_switching_overhead_seconds": "model_switch_time_seconds",
    }.get(key)
    if lifecycle_key is None:
        return None
    for result in results:
        lifecycle = result.get("lifecycle")
        if isinstance(lifecycle, Mapping):
            value = _number(lifecycle.get(lifecycle_key))
            if value is not None:
                values.append(value)
    return sum(values) / len(values) if values else None


def _evidence_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixture_hashes = {
        str(fixture.get("sha256"))
        for result in results
        if isinstance((fixture := result.get("fixture_set")), Mapping) and fixture.get("sha256")
    }
    installed = bool(results) and all(
        isinstance(result.get("fixture_set"), Mapping)
        and result["fixture_set"].get("installed") is True
        for result in results
    )
    return {
        "fixtures_installed": installed,
        "identical_fixture_hash": len(fixture_hashes) == 1,
        "fixture_hash": next(iter(fixture_hashes)) if len(fixture_hashes) == 1 else None,
        "model_count": len(results),
        "all_real": bool(results) and not any(bool(result.get("simulated")) for result in results),
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = summary.get("measurements")
    return value if isinstance(value, Mapping) else {}


def _sustained_fraction(summary: Mapping[str, Any]) -> float | None:
    sustained = summary.get("sustained_performance")
    if not isinstance(sustained, Mapping):
        return None
    return _number(sustained.get("sustained_performance_degradation_fraction"))


def _thermal_suspected(summary: Mapping[str, Any], threshold: float) -> bool | None:
    degradation = _sustained_fraction(summary)
    return degradation > threshold if degradation is not None else None


def _thresholds(settings: BenchmarkSettings) -> dict[str, float]:
    return {
        "maximum_decline_fraction": settings.maximum_decline_fraction,
        "minimum_structured_json_reliability": (settings.minimum_structured_json_reliability),
        "minimum_routing_accuracy": settings.minimum_routing_accuracy,
        "minimum_coding_pass_rate": settings.minimum_coding_pass_rate,
    }
