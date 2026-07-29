from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import typer

from apps.cli.render import console
from april_common.settings import load_settings
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import ModelRegistry
from services.jobs.model_jobs import run_model_utility_job


def register_model_compare(model_app: typer.Typer) -> None:
    @model_app.command("compare-setups")
    def compare_setups(
        shared_model_id: str = typer.Option(..., "--shared-model-id"),
        output: Path = typer.Option(
            Path("data/verification/model-setup-comparison.json"),
            "--output",
        ),
    ) -> None:
        report = asyncio.run(_compare(shared_model_id))
        target = output.expanduser()
        settings = load_settings()
        if not target.is_absolute():
            target = settings.home / target
        _atomic_json(target, report)
        console.print_json(data=report)
        if report["recommendation"] == "insufficient_evidence":
            raise typer.Exit(1)


async def _compare(shared_model_id: str) -> dict[str, Any]:
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
    specialist_results = [
        await run_model_utility_job(
            settings,
            model_id=model.id,
            mode="benchmark",
            cancellation_event=cancellation,
            timeout_seconds=3600.0,
        )
        for model in specialists
    ]
    shared_result = await run_model_utility_job(
        settings,
        model_id=shared.id,
        mode="benchmark",
        cancellation_event=cancellation,
        timeout_seconds=3600.0,
    )
    specialist_tps = _average_tps(specialist_results)
    shared_tps = _average_tps([shared_result])
    if specialist_tps is None or shared_tps is None:
        recommendation = "insufficient_evidence"
    elif shared_tps >= specialist_tps * (1.0 - settings.benchmark.maximum_decline_fraction):
        recommendation = "shared_model_meets_configured_performance_threshold"
    else:
        recommendation = "retain_current_specialist_configuration"
    unavailable = sorted(
        {
            item
            for result in [*specialist_results, shared_result]
            for item in result.get("measurements_unavailable", [])
        }
    )
    return {
        "schema_version": 1,
        "report_type": "model_setup_comparison",
        "generated_at": utc_now_iso(),
        "current_specialist_configuration": [
            _redacted_benchmark(result) for result in specialist_results
        ],
        "shared_model": _redacted_benchmark(shared_result),
        "thresholds": {
            "maximum_decline_fraction": settings.benchmark.maximum_decline_fraction,
            "minimum_structured_json_reliability": (
                settings.benchmark.minimum_structured_json_reliability
            ),
            "minimum_routing_accuracy": settings.benchmark.minimum_routing_accuracy,
            "minimum_coding_pass_rate": settings.benchmark.minimum_coding_pass_rate,
        },
        "recommendation": recommendation,
        "automatic_selection_performed": False,
        "thermal_measurement": {
            "available": False,
            "reason": "safe unprivileged thermal measurement was not exposed",
        },
        "unavailable_measurements": unavailable,
    }


def _redacted_benchmark(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": result["model_id"],
        "role": result["role"],
        "model_basename": result["model_basename"],
        "model_sha256": result["model_sha256"],
        "model_size": result["model_size"],
        "passed": bool(result.get("passed")),
        "runs": [
            {
                "run_index": run.get("run_index"),
                "ok": run.get("ok"),
                "load_time_seconds": run.get("load_time_seconds"),
                "first_token_latency_seconds": run.get("first_token_latency_seconds"),
                "generation_time_seconds": run.get("generation_time_seconds"),
                "tokens_per_second": run.get("tokens_per_second"),
                "unload_success": run.get("unload_success"),
            }
            for run in result.get("runs", [])
        ],
    }


def _average_tps(results: list[dict[str, Any]]) -> float | None:
    values = [
        float(run["tokens_per_second"])
        for result in results
        for run in result.get("runs", [])
        if run.get("ok") and isinstance(run.get("tokens_per_second"), (int, float))
    ]
    return sum(values) / len(values) if values else None


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
