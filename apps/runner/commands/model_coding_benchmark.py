from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer

from apps.cli.render import console
from apps.runner.coding_benchmark_reports import (
    build_coding_benchmark_report,
    compare_coding_benchmark_reports,
    load_coding_report,
    write_coding_comparison_report,
    write_coding_report,
)
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry
from services.evaluation.coding_benchmark import coding_benchmark_timeout_profile
from services.evaluation.model_quality import fixture_set_metadata
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore
from services.memory.database import Database
from services.memory.migrations import run_migrations


def register_coding_benchmark_commands(model_app: typer.Typer) -> None:
    @model_app.command("benchmark-coding")
    def benchmark_coding(
        model_id: str = typer.Argument(..., help="One explicitly registered local model."),
        suite: str = typer.Option("full", "--suite", help="smoke or full"),
        timeout_profile: str = typer.Option(
            "full-local", "--timeout-profile", help="smoke or full-local"
        ),
        case_timeout_multiplier: float = typer.Option(
            1.0, "--case-timeout-multiplier", min=0.5, max=4.0
        ),
        report: Path | None = typer.Option(None, "--report"),
        wait: bool = typer.Option(True, "--wait/--no-wait"),
        wait_timeout: float = typer.Option(21_600.0, "--wait-timeout", min=1.0, max=86_400.0),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Benchmark one registered model; compare saved reports offline later."""
        if suite not in {"smoke", "full"}:
            raise typer.BadParameter("suite must be smoke or full")
        profile = coding_benchmark_timeout_profile(timeout_profile)
        settings = load_settings()
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml", root=settings.home
        )
        definition = registry.get(model_id)
        if definition.role not in {"coding", "brain", "reasoning"}:
            raise typer.BadParameter("model must be an explicitly registered coding-capable model")
        target = report or settings.home / "data" / "verification" / (
            f"coding-benchmark-{definition.id}.json"
        )
        console.print(
            f"Coding benchmark report destination: {target} "
            f"(model={definition.id}, suite={suite}, timeout_profile={profile.name})"
        )
        job = asyncio.run(
            _submit_coding_benchmark(
                definition.id,
                suite=suite,
                timeout_profile=profile.name,
                case_timeout_multiplier=case_timeout_multiplier,
                report_path=str(target),
            )
        )
        if not wait:
            console.print(f"Coding benchmark job {job['id']} is {job['status']}.")
            console.print(f"  run april jobs show {job['id']}")
            console.print(f"  run april jobs cancel {job['id']}")
            return
        completed = asyncio.run(_wait_for_job(str(job["id"]), wait_timeout))
        succeeded = completed.get("status") == "succeeded" and isinstance(
            completed.get("result"), Mapping
        )
        result = dict(completed["result"]) if succeeded else {}
        if not succeeded:
            result.update(
                {
                    "benchmark_status": completed.get("status", "failed"),
                    "failure_reason": completed.get("error_code", "benchmark_failed"),
                }
            )
        if succeeded and result.get("report_written") is True and target.is_file():
            payload = load_coding_report(target)
            if json_output:
                console.print_json(data=payload)
            else:
                console.print(f"Wrote coding benchmark report to {target}")
                console.print(f"Comparison eligible: {payload['comparison_eligible']}")
            return
        fixture_metadata = fixture_set_metadata(settings.home)
        result_profile = result.get("timeout_profile")
        if not isinstance(result_profile, Mapping):
            result_profile = {
                "name": profile.name,
                "case_timeout_seconds": profile.case_timeout_seconds * case_timeout_multiplier,
                "job_timeout_seconds": profile.job_timeout_seconds,
                "case_timeout_multiplier": case_timeout_multiplier,
            }
        payload = build_coding_benchmark_report(
            result,
            model_id=definition.id,
            role=str(definition.role),
            backend=definition.backend,
            artifact_kind=definition.artifact_kind,
            configuration={
                "context_size": definition.context_size,
                "max_output_tokens": definition.max_output_tokens,
                "threads": definition.threads,
                "threads_batch": definition.threads_batch,
                "chat_format": definition.chat_format,
            },
            suite=suite,
            timeout_profile=result_profile,
            fixture_set=fixture_metadata,
            simulated=bool(result.get("simulated")),
            hardware_profile=result.get("hardware_profile"),
        )
        write_coding_report(payload, target)
        if json_output:
            console.print_json(data=payload)
        else:
            console.print(f"Wrote coding benchmark report to {target}")
            console.print(f"Comparison eligible: {payload['comparison_eligible']}")
        if not succeeded:
            console.print(
                "[red]Coding benchmark did not complete: "
                f"{completed.get('error_code', 'failed')}[/red]"
            )
            raise typer.Exit(1)

    @model_app.command("compare-coding-results")
    def compare_coding_results(
        report_a: Path = typer.Argument(..., help="Saved single-candidate report."),
        report_b: Path = typer.Argument(..., help="Saved single-candidate report."),
        report: Path | None = typer.Option(None, "--report"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Compare two completed coding reports without model inference."""
        try:
            first = load_coding_report(report_a)
            second = load_coding_report(report_b)
            comparison = compare_coding_benchmark_reports(first, second)
            target = report or Path("data/verification/coding-model-comparison.json")
            write_coding_comparison_report(comparison, target)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        if json_output:
            console.print_json(data=comparison)
        else:
            console.print(f"Wrote offline coding comparison to {target}")
            console.print(f"Recommendation: {comparison['recommendation']}")


async def _submit_coding_benchmark(
    model_id: str,
    *,
    suite: str,
    timeout_profile: str,
    case_timeout_multiplier: float,
    report_path: str | None = None,
) -> dict[str, Any]:
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
            job_type="model_coding_benchmark",
            payload={
                "model_id": model_id,
                "suite": suite,
                "timeout_profile": timeout_profile,
                "case_timeout_multiplier": case_timeout_multiplier,
                "report_path": report_path,
            },
            owner="local-user",
        )
        return job.model_dump(mode="json")
    finally:
        await database.close()


async def _wait_for_job(job_id: str, timeout_seconds: float) -> dict[str, Any]:
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
                raise TimeoutError("Timed out waiting for the coding benchmark job.")
            await asyncio.sleep(0.25)
    finally:
        await database.close()
