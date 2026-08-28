from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, TypeVar

import typer

from apps.cli.render import console
from apps.runner.commands.registry import rollout_app
from april_common.audit import audit_logger_for_settings
from april_common.effective_config import load_agents_file
from april_common.settings import AprilSettings, load_settings
from services.april_runtime.client import RuntimeClient
from services.evolution.rollouts import (
    PromotionReadiness,
    RealPromptShadowEvaluator,
    RolloutError,
    RolloutRecord,
    RolloutService,
)
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore

_T = TypeVar("_T")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _with_service(
    action: Any,
) -> Any:
    settings = load_settings()
    database = Database(settings.database_path)
    await database.connect()
    try:
        await run_migrations(database)
        service = RolloutService(
            settings,
            database,
            audit=audit_logger_for_settings(settings),
            runtime_client=RuntimeClient(
                settings.runtime.url,
                timeout=settings.runtime.request_timeout_seconds,
                token=settings.runtime.token,
            ),
        )
        return await action(settings, database, service)
    finally:
        await database.close()


def _emit(record: RolloutRecord, *, json_output: bool) -> None:
    payload = record.to_safe_payload()
    if json_output:
        console.print_json(data=payload)
        return
    reason = f" reason={record.reason_code}" if record.reason_code else ""
    console.print(f"{record.id} state={record.state}{reason}")


def _emit_many(records: list[RolloutRecord], *, json_output: bool) -> None:
    if json_output:
        console.print_json(data={"rollouts": [item.to_safe_payload() for item in records]})
        return
    if not records:
        console.print("No rollout records.")
        return
    for record in records:
        _emit(record, json_output=False)


def _command(coro: Any, *, json_output: bool = False, many: bool = False) -> None:
    try:
        result = _run(coro)
        if many:
            _emit_many(result, json_output=json_output)
        elif isinstance(result, RolloutRecord):
            _emit(result, json_output=json_output)
        elif json_output:
            console.print_json(data=result)
        else:
            console.print(str(result))
    except (RolloutError, ValueError, OSError) as exc:
        payload = {"ok": False, "reason": str(exc)}
        if json_output:
            console.print_json(data=payload)
        else:
            console.print(f"[red]Blocked: {exc}[/red]")
        raise typer.Exit(1) from exc


@rollout_app.command("create")
def create(
    candidate_type: str = typer.Option(..., "--type"),
    target_id: str = typer.Option(..., "--target-id"),
    candidate_id: str = typer.Option(..., "--candidate-id"),
    candidate_path: Path = typer.Option(..., "--candidate-path"),
    baseline_id: str | None = typer.Option(None, "--baseline-id"),
    baseline_sha256: str | None = typer.Option(None, "--baseline-sha256"),
    baseline_path: Path | None = typer.Option(None, "--baseline-path"),
    minimum_samples: int | None = typer.Option(None, "--minimum-samples", min=1),
    canary_fraction: float | None = typer.Option(None, "--canary-fraction", min=0.001, max=0.25),
    canary_max_turns: int | None = typer.Option(None, "--canary-max-turns", min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        _settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> RolloutRecord:
        if candidate_type not in {"prompt_overlay", "lora_adapter"}:
            raise ValueError("type_must_be_prompt_overlay_or_lora_adapter")
        return await service.create(
            candidate_type=candidate_type,  # type: ignore[arg-type]
            target_id=target_id,
            candidate_id=candidate_id,
            candidate_artifact_path=candidate_path,
            baseline_id=baseline_id,
            baseline_sha256=baseline_sha256,
            baseline_artifact_path=baseline_path,
            minimum_samples=minimum_samples,
            canary_fraction=canary_fraction,
            canary_max_eligible_turns=canary_max_turns,
        )

    _command(_with_service(action), json_output=json_output)


@rollout_app.command("list")
def list_rollouts(
    state: str | None = typer.Option(None, "--state"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        _settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> list[RolloutRecord]:
        return await service.list(state=state)

    _command(_with_service(action), json_output=json_output, many=True)


@rollout_app.command("show")
@rollout_app.command("status")
def show(
    rollout_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        _settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> RolloutRecord:
        return await service.require(rollout_id)

    _command(_with_service(action), json_output=json_output)


@rollout_app.command("shadow-start")
def shadow_start(
    rollout_id: str,
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Run in this process instead of the durable Job Worker.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> Any:
        record = await service.require(rollout_id)
        if not foreground:
            store = JobStore(
                _database,
                default_job_registry(
                    finetune_enabled=settings.finetune.enabled,
                    evolution_enabled=settings.evolution.enabled,
                ),
            )
            queued, job = await service.queue_shadow(rollout_id, store=store)
            return {
                **queued.to_safe_payload(),
                "job": job.model_dump(mode="json"),
                "next_command": f"run april jobs status {job.id}",
            }
        agents = load_agents_file(settings.home).agents
        target = agents.get(record.target_id)
        judge = agents.get("reading_agent")
        evaluator = RealPromptShadowEvaluator(
            settings,
            RuntimeClient(
                settings.runtime.url,
                timeout=settings.runtime.request_timeout_seconds,
                token=settings.runtime.token,
            ),
            model_id=target.model_id if target is not None else None,
            judge_model_id=judge.model_id if judge is not None else None,
        )
        return await service.start_shadow(rollout_id, evaluator=evaluator)

    _command(_with_service(action), json_output=json_output)


@rollout_app.command("approval-request")
def approval_request(
    rollout_id: str,
    stage: str = typer.Option(..., "--stage", help="canary or activation"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        settings: AprilSettings,
        database: Database,
        service: RolloutService,
    ) -> dict[str, Any]:
        if stage not in {"canary", "activation"}:
            raise ValueError("stage_must_be_canary_or_activation")
        approvals = ApprovalStore(
            database,
            audit_logger_for_settings(settings),
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        approval_id = await service.request_approval(
            rollout_id,
            stage=stage,  # type: ignore[arg-type]
            approvals=approvals,
            request_id=str(uuid.uuid4()),
        )
        return {
            "ok": False,
            "state": "pending_approval",
            "approval_id": approval_id,
            "next_command": (
                f"run april approve {approval_id}; then retry the {stage} transition "
                f"with --approval-id {approval_id}"
            ),
        }

    _command(_with_service(action), json_output=bool(json_output))
    # Creating a pending approval is not a completed transition.
    raise typer.Exit(1)


@rollout_app.command("canary-start")
def canary_start(
    rollout_id: str,
    approval_id: str = typer.Option(..., "--approval-id"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        _settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> RolloutRecord:
        return await service.start_canary(rollout_id, approval_id=approval_id)

    _command(_with_service(action), json_output=json_output)


@rollout_app.command("promote")
def promote(
    rollout_id: str,
    approval_id: str = typer.Option(..., "--approval-id"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        settings: AprilSettings,
        database: Database,
        service: RolloutService,
    ) -> RolloutRecord:
        database_healthy = (await database.fetchone("SELECT 1")) is not None
        runtime_healthy = False
        try:
            health = await RuntimeClient(
                settings.runtime.url,
                timeout=min(5.0, settings.runtime.request_timeout_seconds),
                token=settings.runtime.token,
            ).health(timeout=5.0)
            runtime_healthy = str(health.get("status")) in {"ok", "degraded"}
        except Exception:
            runtime_healthy = False
        return await service.promote(
            rollout_id,
            approval_id=approval_id,
            readiness=PromotionReadiness(
                runtime_healthy=runtime_healthy,
                database_healthy=database_healthy,
            ),
        )

    _command(_with_service(action), json_output=json_output)


@rollout_app.command("cancel")
def cancel(
    rollout_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        _settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> RolloutRecord:
        return await service.cancel(rollout_id)

    _command(_with_service(action), json_output=json_output)


@rollout_app.command("rollback")
def rollback(
    rollout_id: str,
    reason: str = typer.Option("operator_rollback", "--reason"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def action(
        _settings: AprilSettings,
        _database: Database,
        service: RolloutService,
    ) -> RolloutRecord:
        return await service.rollback(rollout_id, reason_code=reason)

    _command(_with_service(action), json_output=json_output)
