from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import typer

from apps.cli.render import console
from april_common.settings import BenchmarkSettings, load_settings
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import ModelRegistry
from services.jobs.model_jobs import run_model_utility_job

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
        output: Path = typer.Option(
            Path("data/verification/model-setup-comparison.json"),
            "--output",
        ),
        cooldown_seconds: float = typer.Option(
            0.0,
            "--cooldown-seconds",
            min=0.0,
            max=600.0,
        ),
    ) -> None:
        report = asyncio.run(_compare(shared_model_id, cooldown_seconds=cooldown_seconds))
        target = output.expanduser()
        settings = load_settings()
        if not target.is_absolute():
            target = settings.home / target
        _atomic_json(target, report)
        console.print_json(data=report)
        if report["recommendation"] in {
            "insufficient_evidence",
            "candidate_failed_no_regression_gates",
        }:
            raise typer.Exit(1)


async def _compare(
    shared_model_id: str,
    *,
    cooldown_seconds: float = 0.0,
) -> dict[str, Any]:
    settings = load_settings()
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
    cancellation = asyncio.Event()
    specialist_results: list[dict[str, Any]] = []
    for model in specialists:
        specialist_results.append(
            await run_model_utility_job(
                settings,
                model_id=model.id,
                mode="benchmark",
                cancellation_event=cancellation,
                timeout_seconds=3600.0,
            )
        )
        if cooldown_seconds:
            await asyncio.sleep(cooldown_seconds)
    shared_result = await run_model_utility_job(
        settings,
        model_id=shared.id,
        mode="benchmark",
        cancellation_event=cancellation,
        timeout_seconds=3600.0,
    )
    specialist_summary = summarize_setup(specialist_results)
    shared_summary = summarize_setup([shared_result])
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
    return {
        "schema_version": 2,
        "report_type": "model_setup_comparison",
        "generated_at": utc_now_iso(),
        "fixture_set": {
            "id": _FIXTURE_SET_ID,
            "fixtures": list(_FIXTURES),
            "identical_for_both_setups": True,
            "long_context_token_count_source": "model_tokenizer_only",
            "character_estimation_used": False,
        },
        "current_specialist_configuration": {
            "fixture_set_id": _FIXTURE_SET_ID,
            "models": [_redacted_benchmark(result) for result in specialist_results],
            "summary": specialist_summary,
        },
        "shared_model": {
            "fixture_set_id": _FIXTURE_SET_ID,
            "model": _redacted_benchmark(shared_result),
            "summary": shared_summary,
        },
        "benchmark_rounds": settings.benchmark.runs,
        "cooldown_seconds": cooldown_seconds,
        "thresholds": _thresholds(settings.benchmark),
        "gate_failures": gate_failures,
        "recommendation": recommendation,
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
        "unavailable_measurements": unavailable,
    }


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
    if shared.get("passed") is not True:
        return "candidate_failed_no_regression_gates", ["shared_model_benchmark_failed"]
    specialist_metrics = _metrics(specialist)
    shared_metrics = _metrics(shared)
    required = {
        "output_tokens_per_second",
        *_QUALITY_METRICS,
    }
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
    for metric, minimum in absolute_thresholds.items():
        value = float(shared_metrics[metric])
        if value < minimum:
            absolute_failures.append(f"{metric}_below_configured_minimum")
    if shared_sustained > thresholds.maximum_decline_fraction:
        absolute_failures.append("sustained_degradation_above_configured_maximum")
    if absolute_failures:
        return "candidate_failed_no_regression_gates", absolute_failures
    relative_failures: list[str] = []
    for metric in required:
        specialist_value = float(specialist_metrics[metric])
        shared_value = float(shared_metrics[metric])
        if shared_value < specialist_value * (1.0 - thresholds.maximum_decline_fraction):
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
        "runs": [
            {
                "run_index": run.get("run_index"),
                "ok": run.get("ok"),
                "load_time_seconds": run.get("load_time_seconds"),
                "first_token_latency_seconds": run.get("first_token_latency_seconds"),
                "generation_time_seconds": run.get("generation_time_seconds"),
                "tokens_per_second": run.get("tokens_per_second"),
                "prompt_token_count": run.get("prompt_token_count"),
                "prompt_eval_duration_seconds": run.get("prompt_eval_duration_seconds"),
                "prompt_processing_tokens_per_second": _run_prompt_rate(run),
                "process_rss_bytes": run.get("process_rss_bytes"),
                "peak_process_rss_bytes": run.get("peak_process_rss_bytes"),
                "unload_success": run.get("unload_success"),
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
    return _average(results, key)


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
