from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from services.api.dependencies import ApiContainer
from services.jobs.schemas import (
    DEFAULT_JOB_LIST_LIMIT,
    MAX_JOB_LIST_LIMIT,
    JobSubmission,
)
from services.jobs.store import JobNotFoundError, JobTransitionError


async def _scoped_job(active: ApiContainer, job_id: str) -> Any:
    if active.job_store is None:
        raise HTTPException(status_code=503, detail="job_store_unavailable")
    try:
        job = await active.job_store.require(job_id)
    except (JobNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="job_not_found") from exc
    if job.owner != "local-user":
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.project_id is not None and await active.memory.get_project(job.project_id) is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return job


def register_job_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
) -> None:
    @app.post("/jobs")
    async def submit_job(
        request: JobSubmission,
        http_request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if active.job_store is None:
            raise HTTPException(status_code=503, detail="job_store_unavailable")
        if request.project_id is not None:
            project = await active.memory.get_project(request.project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="project_not_found")
        if request.conversation_id is not None:
            conversation = await active.memory.get_conversation(request.conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="conversation_not_found")
            if (
                request.project_id is not None
                and conversation.project_id is not None
                and conversation.project_id != request.project_id
            ):
                raise HTTPException(status_code=409, detail="conversation_project_mismatch")
        try:
            definition = active.job_store.registry.require(request.job_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        approved = False
        approval_id = request.approval_id
        request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())
        if definition.approval_required:
            if approval_id is None:
                raise HTTPException(status_code=409, detail="approval_required")
            record = await active.approvals.get(approval_id)
            expected_payload: dict[str, Any]
            if request.job_type == "configured_test" and record.tool == "test_runner":
                expected_payload = {
                    "argv": list(record.args.get("argv", ["pytest"])),
                    "cwd": str(record.args.get("repo_path", "")),
                }
            elif request.job_type == "finetune" and record.tool == "finetune":
                expected_payload = {"plan_id": str(record.args.get("plan_id", ""))}
            elif request.job_type == "model_import" and record.tool == "model_import":
                expected_payload = {
                    "source_path": str(record.args.get("source_path", "")),
                    "model_id": str(record.args.get("model_id", "")),
                    "role": str(record.args.get("role", "")),
                    "name": str(record.args.get("name", "")),
                    "expected_sha256": record.args.get("expected_sha256"),
                }
            else:
                raise HTTPException(status_code=409, detail="approval_action_mismatch")
            validated_payload = definition.validate_payload(request.payload)
            if validated_payload != expected_payload:
                raise HTTPException(status_code=409, detail="approval_action_mismatch")
            await active.approvals.approve_exact(
                approval_id=approval_id,
                tool=record.tool,
                args=record.args,
                actor="local-user",
                request_id=request_id,
            )
            approved = True
        try:
            job = await active.job_store.submit(
                job_type=request.job_type,
                payload=request.payload,
                owner="local-user",
                conversation_id=request.conversation_id,
                project_id=request.project_id,
                approved=approved,
            )
        except (ValueError, JobTransitionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if approval_id is not None and approved:
            await active.approvals.consume(
                approval_id=approval_id,
                result={"ok": True, "job_id": job.id, "status": job.status.value},
                actor="local-user",
                request_id=request_id,
            )
        return job.model_dump(mode="json")

    @app.get("/jobs")
    async def list_jobs(
        project_id: str | None = None,
        limit: int = DEFAULT_JOB_LIST_LIMIT,
        offset: int = 0,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if active.job_store is None:
            raise HTTPException(status_code=503, detail="job_store_unavailable")
        if not 1 <= limit <= MAX_JOB_LIST_LIMIT or not 0 <= offset <= 10_000:
            raise HTTPException(status_code=400, detail="pagination_out_of_bounds")
        jobs = await active.job_store.list(
            owner="local-user",
            project_id=project_id,
            limit=limit,
            offset=offset,
        )
        return {
            "jobs": [job.model_dump(mode="json") for job in jobs],
            "limit": limit,
            "offset": offset,
        }

    @app.get("/jobs/{job_id}")
    async def show_job(
        job_id: str,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        job = await _scoped_job(active, job_id)
        return job.model_dump(mode="json")

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(
        job_id: str,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        await _scoped_job(active, job_id)
        assert active.job_store is not None
        job, already_terminal = await active.job_store.request_cancel(job_id)
        return {
            "job": job.model_dump(mode="json"),
            "already_terminal": already_terminal,
        }

    @app.post("/jobs/{job_id}/retry")
    async def retry_job(
        job_id: str,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        await _scoped_job(active, job_id)
        assert active.job_store is not None
        try:
            job, already_queued = await active.job_store.retry(job_id)
        except JobTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "job": job.model_dump(mode="json"),
            "already_queued": already_queued,
        }
