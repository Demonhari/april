from __future__ import annotations

import asyncio
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

import typer

from apps.cli.render import console
from april_common.audit import audit_logger_for_settings
from april_common.errors import PermissionDeniedError
from april_common.settings import AprilSettings, load_settings
from services.jobs.model_import import ModelImportError, ModelImportService
from services.jobs.registry import default_job_registry
from services.jobs.store import JobStore, JobTransitionError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest


def register_model_import_commands(model_app: typer.Typer) -> None:
    def options(
        role: str = typer.Option(..., "--role"),
        model_id: str = typer.Option(..., "--id"),
        name: str = typer.Option(..., "--name"),
        path: Path = typer.Option(..., "--path"),
        expected_sha256: str = typer.Option(
            ...,
            "--sha256",
            help="Expected SHA-256 calculated independently for the local GGUF.",
        ),
        verify: bool = typer.Option(
            False,
            "--verify",
            help="Bind a requested follow-up verification to the approval (never auto-loads).",
        ),
        approval_id: str | None = typer.Option(None, "--approval-id"),
        wait: bool = typer.Option(False, "--wait"),
        json_output: bool = typer.Option(False, "--json"),
        wait_timeout: float = typer.Option(14_400.0, "--wait-timeout", min=1.0, max=86_400.0),
    ) -> None:
        _import_command(
            role=role,
            model_id=model_id,
            name=name,
            path=path,
            expected_sha256=expected_sha256,
            verify=verify,
            approval_id=approval_id,
            wait=wait,
            json_output=json_output,
            wait_timeout=wait_timeout,
        )

    model_app.command("import")(options)

    @model_app.command("import-enqueue", hidden=True)
    def enqueue_alias(
        role: str = typer.Option(..., "--role"),
        model_id: str = typer.Option(..., "--id"),
        name: str = typer.Option(..., "--name"),
        path: Path = typer.Option(..., "--path"),
        expected_sha256: str = typer.Option(..., "--sha256"),
        verify: bool = typer.Option(False, "--verify"),
        approval_id: str | None = typer.Option(None, "--approval-id"),
        wait: bool = typer.Option(False, "--wait"),
        json_output: bool = typer.Option(False, "--json"),
        wait_timeout: float = typer.Option(14_400.0, "--wait-timeout", min=1.0, max=86_400.0),
    ) -> None:
        """Deprecated compatibility alias for ``model import``."""
        console.print(
            "[yellow]Deprecated: use `run april model import`; "
            "this alias uses the same durable implementation.[/yellow]"
        )
        _import_command(
            role=role,
            model_id=model_id,
            name=name,
            path=path,
            expected_sha256=expected_sha256,
            verify=verify,
            approval_id=approval_id,
            wait=wait,
            json_output=json_output,
            wait_timeout=wait_timeout,
        )


def _import_command(
    *,
    role: str,
    model_id: str,
    name: str,
    path: Path,
    expected_sha256: str,
    verify: bool,
    approval_id: str | None,
    wait: bool,
    json_output: bool,
    wait_timeout: float,
) -> None:
    """Create an exact approval, then atomically accept one durable import job."""
    try:
        payload = _run(
            _approval_payload(
                path,
                model_id=model_id,
                role=role,
                name=name,
                expected_sha256=expected_sha256,
                requested_verification=verify,
            )
        )
        if approval_id is None:
            created = _run(_create_approval(payload))
            command = shlex.join(
                [
                    "run",
                    "april",
                    "model",
                    "import",
                    "--role",
                    role,
                    "--id",
                    model_id,
                    "--name",
                    name,
                    "--path",
                    str(path),
                    "--sha256",
                    expected_sha256,
                    "--approval-id",
                    created,
                    *(["--verify"] if verify else []),
                    *(["--wait"] if wait else []),
                    *(["--json"] if json_output else []),
                ]
            )
            result = {
                "status": "approval_required",
                "approval_id": created,
                "model_id": model_id,
                "logical_role": role,
                "basename": path.name,
                "destination": payload["destination"],
                "format": payload["format"],
                "requested_verification": verify,
                "submit_command": command,
                "automatic_selection_performed": False,
                "automatic_load_performed": False,
            }
            if json_output:
                console.print_json(data=result)
            else:
                console.print(f"Approval required: {created}")
                console.print(
                    f"To approve this exact action and submit its durable job, run:\n  {command}",
                    markup=False,
                )
            return
        result = _run(_enqueue(payload, approval_id))
        if wait:
            result["job"] = _run(_wait_for_job(str(result["job_id"]), wait_timeout))
        if json_output:
            console.print_json(data=result)
        else:
            console.print(
                f"Accepted durable model import job {result['job_id']} ({result['status']})."
            )
            for command in result["next_commands"]:
                console.print(f"  {command}", markup=False)
    except (
        ModelImportError,
        JobTransitionError,
        PermissionDeniedError,
        ValueError,
        TimeoutError,
    ) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


async def _approval_payload(
    source: Path,
    *,
    model_id: str,
    role: str,
    name: str,
    expected_sha256: str,
    requested_verification: bool,
) -> dict[str, Any]:
    settings = load_settings()
    service = ModelImportService(settings)
    return await asyncio.to_thread(
        service.prepare_payload,
        source_path=str(source),
        model_id=model_id,
        role=role,
        name=name,
        expected_sha256=expected_sha256,
        requested_verification=requested_verification,
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
        if record.status == "pending":
            try:
                await approvals.approve_exact(
                    approval_id=approval_id,
                    tool="model_import",
                    args=payload,
                    actor="local-user",
                    request_id=str(uuid.uuid4()),
                )
            except PermissionDeniedError:
                record = await approvals.get(approval_id)
                if record.status not in {"approved", "consumed"}:
                    raise
        job, created = await store.submit_with_exact_approval(
            job_type="model_import",
            payload=payload,
            owner="local-user",
            approval_id=approval_id,
            approval_tool="model_import",
            approval_args=payload,
        )
        if created:
            approvals.audit.write(
                {
                    "actor": "local-user",
                    "request_id": str(uuid.uuid4()),
                    "event_type": "approval_consumed",
                    "tool": "model_import",
                    "approval_id": approval_id,
                    "outcome": "consumed",
                    "job_id": job.id,
                }
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
                f"run april jobs show {job.id}",
                f"run april jobs cancel {job.id}",
                f"run april model verify {model_id} --wait",
                f"run april model benchmark {model_id} --wait",
            ],
        }
    finally:
        await database.close()


async def _wait_for_job(job_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        job = await _job(job_id)
        if job["status"] in {"cancelled", "succeeded", "failed", "interrupted"}:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting; the durable import job is still running.")
        await asyncio.sleep(0.25)


async def _job(job_id: str) -> dict[str, Any]:
    _settings, database, store, _approvals = await _stores()
    try:
        job = await store.require(job_id)
        return job.model_dump(mode="json")
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


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)
