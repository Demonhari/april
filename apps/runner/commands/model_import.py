from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import typer

from apps.cli.render import console
from april_common.audit import audit_logger_for_settings
from april_common.settings import AprilSettings, load_settings
from services.jobs.model_import import ModelImportService
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest


def register_model_import_commands(model_app: typer.Typer) -> None:
    @model_app.command("import-enqueue")
    def enqueue(
        role: str = typer.Option(..., "--role"),
        model_id: str = typer.Option(..., "--id"),
        name: str = typer.Option(..., "--name"),
        path: Path = typer.Option(..., "--path"),
        approval_id: str | None = typer.Option(None, "--approval-id"),
    ) -> None:
        """Create an exact import approval, then enqueue the approved local copy."""
        payload = _run(_approval_payload(path, model_id=model_id, role=role, name=name))
        if approval_id is None:
            created = _run(_create_approval(payload))
            console.print_json(
                data={
                    "status": "approval_required",
                    "approval_id": created,
                    "model_id": model_id,
                    "logical_role": role,
                    "basename": path.name,
                    "enqueue_command": (
                        "run april model import-enqueue "
                        f"--role {role} --id {model_id} --name {name!r} "
                        f"--path {str(path)!r} --approval-id {created}"
                    ),
                    "automatic_selection_performed": False,
                    "automatic_load_performed": False,
                }
            )
            return
        result = _run(_enqueue(payload, approval_id))
        console.print_json(data=result)

    @model_app.command("import-show")
    def show(job_id: str) -> None:
        console.print_json(data=_run(_job(job_id)))

    @model_app.command("import-cancel")
    def cancel(job_id: str) -> None:
        console.print_json(data=_run(_cancel(job_id)))

    @model_app.command("import-retry")
    def retry(job_id: str) -> None:
        console.print_json(data=_run(_retry(job_id)))


async def _approval_payload(
    source: Path,
    *,
    model_id: str,
    role: str,
    name: str,
) -> dict[str, Any]:
    settings = load_settings()
    service = ModelImportService(settings)
    return await asyncio.to_thread(
        service.prepare_payload,
        source_path=str(source),
        model_id=model_id,
        role=role,
        name=name,
    )


async def _create_approval(payload: dict[str, Any]) -> str:
    _settings, database, store, approvals = await _stores()
    del store
    try:
        source = Path(str(payload["source_path"]))
        response = await approvals.create(
            ApprovalRequest(
                tool="model_import",
                args=payload,
                agent="local-operator",
                permission_level=4,
                risk_level="system_action",
                affected_paths=[
                    source.name,
                    "configs/models.yaml",
                ],
                expected_side_effects=[
                    "copy approved local GGUF bytes into APRIL models storage",
                    "register an inactive low-priority model candidate",
                ],
                metadata={
                    "model_id": payload["model_id"],
                    "automatic_selection": False,
                    "automatic_load": False,
                },
            ),
            actor="local-user",
            request_id=str(uuid.uuid4()),
        )
        return response.approval_id
    finally:
        await database.close()


async def _enqueue(payload: dict[str, Any], approval_id: str) -> dict[str, Any]:
    _settings, database, store, approvals = await _stores()
    try:
        record = await approvals.get(approval_id)
        if record.tool != "model_import" or record.args != payload:
            raise ValueError("Model-import approval does not match the exact local artifact.")
        request_id = str(uuid.uuid4())
        await approvals.approve_exact(
            approval_id=approval_id,
            tool="model_import",
            args=payload,
            actor="local-user",
            request_id=request_id,
        )
        job = await store.submit(
            job_type="model_import",
            payload=payload,
            owner="local-user",
            approved=True,
        )
        await approvals.consume(
            approval_id=approval_id,
            result={"ok": True, "job_id": job.id, "status": job.status.value},
            actor="local-user",
            request_id=request_id,
        )
        model_id = str(payload["model_id"])
        return {
            "job_id": job.id,
            "status": job.status.value,
            "model_id": model_id,
            "logical_role": payload["role"],
            "basename": Path(str(payload["source_path"])).name,
            "automatic_selection_performed": False,
            "automatic_load_performed": False,
            "next_commands": [
                f"run april model import-show {job.id}",
                "run april jobs submit model_import_verification "
                f'--payload \'{{"model_id":"{model_id}"}}\' --wait',
                "run april jobs submit model_benchmark "
                f'--payload \'{{"model_id":"{model_id}"}}\' --wait',
            ],
        }
    finally:
        await database.close()


async def _job(job_id: str) -> dict[str, Any]:
    _settings, database, store, _approvals = await _stores()
    try:
        job = await store.require(job_id)
        _require_import_job(job.job_type)
        return job.model_dump(mode="json")
    finally:
        await database.close()


async def _cancel(job_id: str) -> dict[str, Any]:
    _settings, database, store, _approvals = await _stores()
    try:
        existing = await store.require(job_id)
        _require_import_job(existing.job_type)
        job, already_terminal = await store.request_cancel(job_id)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "already_terminal": already_terminal,
        }
    finally:
        await database.close()


async def _retry(job_id: str) -> dict[str, Any]:
    _settings, database, store, _approvals = await _stores()
    try:
        existing = await store.require(job_id)
        _require_import_job(existing.job_type)
        job, already_queued = await store.retry(job_id)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "already_queued": already_queued,
        }
    finally:
        await database.close()


async def _stores() -> tuple[AprilSettings, Database, JobStore, ApprovalStore]:
    settings = load_settings()
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    audit = audit_logger_for_settings(settings)
    return (
        settings,
        database,
        JobStore(
            database,
            default_job_registry(
                finetune_enabled=settings.finetune.enabled,
                evolution_enabled=settings.evolution.enabled,
            ),
        ),
        ApprovalStore(
            database,
            audit,
            expiry_seconds=settings.permissions.approval_expiry_seconds,
        ),
    )


def _require_import_job(job_type: str) -> None:
    if job_type != "model_import":
        raise ValueError("Job is not a model-import job.")


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)
