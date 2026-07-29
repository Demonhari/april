from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import typer

from apps.cli.render import console
from april_common.audit import audit_logger_for_settings
from april_common.settings import AprilSettings, load_settings
from services.finetune.dataset import FinetunePlan, create_finetune_plan, load_finetune_plan
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest

finetune_app = typer.Typer(
    help="Plan and submit reviewed, local-only fine-tuning.",
    invoke_without_command=True,
)


@finetune_app.callback()
def finetune_guided(
    ctx: typer.Context,
    dataset: Path | None = typer.Option(None, "--dataset"),
    base_model_id: str = typer.Option("april-brain", "--base-model-id"),
    plan_id: str | None = typer.Option(None, "--plan-id"),
    approval_id: str | None = typer.Option(None, "--approval-id"),
) -> None:
    """Create a plan or launch an exact approved plan."""
    if ctx.invoked_subcommand is not None:
        return
    if dataset is not None:
        _print_plan(_run_async(_create_plan_and_approval(dataset, base_model_id)))
        return
    if plan_id is not None and approval_id is not None:
        job = _run_async(_launch(plan_id, approval_id))
        console.print_json(data=job)
        return
    console.print(
        "Use `run april finetune plan --dataset DATASET --base-model-id MODEL`, "
        "review the manifest and approval, then run "
        "`run april finetune --plan-id PLAN --approval-id APPROVAL`."
    )


@finetune_app.command("doctor")
def doctor() -> None:
    settings = load_settings()
    trainer = _executable_status(settings.finetune.trainer_executable)
    evaluator = _executable_status(settings.finetune.evaluator_executable)
    console.print_json(
        data={
            "enabled": settings.finetune.enabled,
            "minimum_samples": settings.finetune.minimum_samples,
            "trainer": trainer,
            "evaluator": evaluator,
            "network_installation": False,
            "activation_after_training": False,
        }
    )
    if not settings.finetune.enabled or not trainer["available"] or not evaluator["available"]:
        raise typer.Exit(1)


@finetune_app.command("plan")
def plan(
    dataset: Path = typer.Option(..., "--dataset", exists=True, dir_okay=False),
    base_model_id: str = typer.Option("april-brain", "--base-model-id"),
) -> None:
    _print_plan(_run_async(_create_plan_and_approval(dataset, base_model_id)))


@finetune_app.command("status")
def status(job_id: str) -> None:
    console.print_json(data=_run_async(_job_status(job_id)))


@finetune_app.command("cancel")
def cancel(job_id: str) -> None:
    console.print_json(data=_run_async(_cancel(job_id)))


async def _create_plan_and_approval(
    dataset: Path,
    base_model_id: str,
) -> tuple[FinetunePlan, str]:
    settings = load_settings()
    plan = create_finetune_plan(settings, source=dataset, base_model_id=base_model_id)
    database = Database(settings.database_path)
    await database.connect()
    try:
        await run_migrations(database)
        approvals = ApprovalStore(
            database,
            audit_logger_for_settings(settings),
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        response = await approvals.create(
            ApprovalRequest(
                tool="finetune",
                args=_approval_args(plan),
                agent="local-operator",
                permission_level=4,
                risk_level="system_action",
                affected_paths=[plan.adapter_candidate_basename],
                expected_side_effects=[
                    "launch configured local trainer",
                    "write inactive adapter candidate and evaluation evidence",
                ],
                metadata={"plan_id": plan.plan_id, "adapter_activation": False},
            ),
            actor="local-user",
            request_id=str(uuid.uuid4()),
        )
        return plan, response.approval_id
    finally:
        await database.close()


async def _launch(plan_id: str, approval_id: str) -> dict[str, Any]:
    settings = load_settings()
    if not settings.finetune.enabled:
        raise ValueError("Fine-tuning is disabled in reviewed configuration.")
    plan = load_finetune_plan(settings, plan_id)
    database = Database(settings.database_path)
    await database.connect()
    try:
        await run_migrations(database)
        audit = audit_logger_for_settings(settings)
        approvals = ApprovalStore(
            database,
            audit,
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        )
        record = await approvals.get(approval_id)
        args = _approval_args(plan)
        if record.tool != "finetune" or record.args != args:
            raise ValueError("Fine-tune approval does not match the immutable plan.")
        request_id = str(uuid.uuid4())
        await approvals.approve_exact(
            approval_id=approval_id,
            tool="finetune",
            args=args,
            actor="local-user",
            request_id=request_id,
        )
        store = JobStore(
            database,
            default_job_registry(
                finetune_enabled=True,
                evolution_enabled=settings.evolution.enabled,
            ),
        )
        job = await store.submit(
            job_type="finetune",
            payload={"plan_id": plan.plan_id},
            owner="local-user",
            approved=True,
        )
        await approvals.consume(
            approval_id=approval_id,
            result={"ok": True, "job_id": job.id, "status": job.status.value},
            actor="local-user",
            request_id=request_id,
        )
        return {
            "job_id": job.id,
            "status": job.status.value,
            "plan_id": plan.plan_id,
            "adapter_active": False,
            "next_commands": [
                f"run april finetune status {job.id}",
                "run april verify --all-configured-models --require-real-model "
                f"--candidate-adapter-model-id {plan.base_model_id} "
                f"--candidate-adapter-path data/evolution/adapters/candidates/"
                f"{plan.adapter_candidate_basename}",
                f"run april evolve adapter activate {plan.base_model_id} "
                f"data/evolution/adapters/candidates/{plan.adapter_candidate_basename} "
                f"--evidence data/evolution/adapters/evidence/{plan.plan_id}.json",
            ],
        }
    finally:
        await database.close()


async def _job_status(job_id: str) -> dict[str, Any]:
    settings, database, store = await _job_store()
    del settings
    try:
        job = await store.require(job_id)
        if job.job_type != "finetune":
            raise ValueError("Job is not a fine-tune job.")
        return job.model_dump(mode="json")
    finally:
        await database.close()


async def _cancel(job_id: str) -> dict[str, Any]:
    settings, database, store = await _job_store()
    del settings
    try:
        job = await store.require(job_id)
        if job.job_type != "finetune":
            raise ValueError("Job is not a fine-tune job.")
        cancelled, already_terminal = await store.request_cancel(job_id)
        return {
            "job_id": cancelled.id,
            "status": cancelled.status.value,
            "already_terminal": already_terminal,
        }
    finally:
        await database.close()


async def _job_store() -> tuple[AprilSettings, Database, JobStore]:
    settings = load_settings()
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    store = JobStore(
        database,
        default_job_registry(
            finetune_enabled=settings.finetune.enabled,
            evolution_enabled=settings.evolution.enabled,
        ),
    )
    return settings, database, store


def _approval_args(plan: FinetunePlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "dataset_sha256": plan.dataset_sha256,
        "configuration_sha256": plan.configuration_sha256,
        "base_model_sha256": plan.base_model_sha256,
        "trainer_sha256": plan.trainer_sha256,
        "evaluator_sha256": plan.evaluator_sha256,
        "adapter_candidate_basename": plan.adapter_candidate_basename,
    }


def _executable_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"configured": False, "available": False, "basename": None}
    resolved = path.expanduser().resolve(strict=False)
    return {
        "configured": True,
        "available": resolved.is_file() and os.access(resolved, os.X_OK),
        "basename": resolved.name,
    }


def _print_plan(value: tuple[FinetunePlan, str]) -> None:
    plan, approval_id = value
    console.print_json(
        data={
            **plan.to_dict(),
            "approval_id": approval_id,
            "launch_command": (
                f"run april finetune --plan-id {plan.plan_id} --approval-id {approval_id}"
            ),
            "adapter_active": False,
        }
    )


def _run_async(coroutine: Any) -> Any:
    return asyncio.run(coroutine)
