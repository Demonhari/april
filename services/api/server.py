from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import platform
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from april_common.config_fingerprint import config_fingerprint_digest
from april_common.errors import (
    AprilError,
    PermissionDeniedError,
    RequestTooLargeError,
    error_payload,
)
from april_common.path_security import PathPolicy, normalize_existing_path
from april_common.process_environment import PROCESS_ENVIRONMENT_POLICY_VERSION
from april_common.process_runner import ResourceLimitProfile, resource_limit_report
from april_common.report_freshness import freshness_from_payload
from april_common.service_health import ServiceHealthResult, probe_service_health
from april_common.settings import (
    INSECURE_API_TOKENS,
    INSECURE_RUNTIME_TOKENS,
    AprilSettings,
    get_settings,
)
from april_common.time import utc_now
from april_common.token_setup import legacy_plaintext_credentials_detected
from services.api.auth import require_bearer_token
from services.api.dependencies import ApiContainer, build_container
from services.api.schemas import (
    AdapterActivateRequest,
    AdapterRollbackRequest,
    AgentRunRequest,
    ChatRequest,
    ChatResponse,
    DatasetExportRequest,
    DocumentCreateRequest,
    EvalPromoteRequest,
    EvalRejectRequest,
    EvolutionRollbackRequest,
    FeedbackRequest,
    MemoryCreateRequest,
    OverlayApprovalRequest,
    PlaybookResumeRequest,
    PlaybookRunRequest,
    ProjectCreateRequest,
    ReminderCreateRequest,
    SessionAttachRequest,
    ToolApprovalAction,
    ToolRequestEnvelope,
    WakeMuteRequest,
    WakeRequest,
)
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import LoadModelRequest
from services.evolution.adapters import AdapterLifecycleManager
from services.evolution.approval import PromptOverlayApprovalService
from services.evolution.dataset_export import export_finetune_dataset
from services.evolution.dreamer import latest_report
from services.evolution.eval_review import (
    EvalReviewError,
    get_pending_case,
    list_pending_cases,
    promote_pending_case,
    reject_pending_case,
)
from services.evolution.feedback_eval import count_pending_eval_cases, stage_feedback_eval_case
from services.evolution.inspect import (
    count_pending_write_capable_overlay_candidates,
    evolution_history,
    evolution_kill_switch_active,
    evolution_status,
    overlay_diff,
    set_evolution_kill_switch,
)
from services.evolution.playbook_miner import mine_playbook_candidates
from services.evolution.versions import PromptOverlayManager
from services.evolution.write_guard import EvolutionWriteGuard
from services.jobs.schemas import (
    DEFAULT_JOB_LIST_LIMIT,
    MAX_JOB_LIST_LIMIT,
    JobSubmission,
)
from services.jobs.store import JobNotFoundError, JobTransitionError
from services.memory.maintenance import check_database
from services.memory.migrations import SCHEMA_VERSION
from services.memory.writer import MemoryWriter
from services.pool.agent_pool import AgentPool
from services.scheduler import compose_briefing, compute_repo_activity
from services.tool_worker.limits import UnsafeToolWorkerSocket, validate_live_socket
from services.voice.health import (
    microphone_access,
    query_audio_devices,
    voice_health,
    voice_readiness_summary,
)
from services.wake.feedback import WakeFeedback, classify_wake_feedback
from services.wake.schemas import WakeEvent
from services.wake.sentinel import MuteSwitch
from services.wake.status import read_wake_status
from services.wake.wake_bus import WakeBus
from skills.playbooks import (
    PlaybookAdoptionService,
    PlaybookDefinition,
    PlaybookLoader,
    PlaybookRunner,
)

_DESKTOP_WEB_DIR = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "web"

_ACTIVITY_MAX_LIMIT = 200

# Strict allowlist for the Activity/Logs feed. Only these keys are ever exposed,
# so audit fields that may carry prompt content, file contents, tool arguments,
# metadata, reminder/notification text, tokens, or secrets are dropped even if
# new event types add them later. This is deny-by-default, not redact-by-key.
_ACTIVITY_ALLOWED_KEYS = frozenset(
    {
        "timestamp",
        "event_type",
        "event",
        "actor",
        "request_id",
        "audit_correlation_id",
        "approval_id",
        "reference_id",
        "reminder_id",
        "memory_id",
        "memory_type",
        "agent",
        "tool",
        "permission_level",
        "risk",
        "risk_level",
        "outcome",
        "status",
        "project_id",
        "content_length",
        "reason_length",
        "kind",
        "sink",
        "date",
        # wake_mute_changed / eval review events: safe booleans and digests.
        "muted",
        "case_id",
    }
)


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


_PATH_TEXT_RE = re.compile(r"~?(?:/[\w.\-]+){2,}/?")
_VERIFICATION_REPORT_TYPES = {
    "multi_model",
    "target_mac",
    "voice_live",
    "voice_conversation_live",
    "workflow",
    "soak",
}
_VERIFICATION_REPORT_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$")


def _read_activity_events(audit_path: Path, limit: int) -> list[dict[str, Any]]:
    if not audit_path.exists():
        return []
    try:
        lines = audit_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        safe_source = dict(record)
        payload = record.get("payload")
        if isinstance(payload, dict):
            safe_source.update(payload)
        projected = {
            key: value for key, value in safe_source.items() if key in _ACTIVITY_ALLOWED_KEYS
        }
        if projected:
            events.append(projected)
        if len(events) >= limit:
            break
    return events


def create_app(container: ApiContainer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.container is None:
            try:
                app.state.container = await build_container()
            except Exception:
                # Liveness must remain available when dependency assembly fails.
                # Authenticated endpoints retry assembly and surface the real error.
                app.state.container_error = True
        active: ApiContainer | None = app.state.container
        if active is not None and active.scheduler is not None:
            # start() is a no-op unless scheduler.enabled, so this is safe in tests.
            await active.scheduler.start()
        wake_bus: WakeBus | None = None
        if active is not None and active.settings.wake.enabled:
            # Local wake bus: owner-only Unix socket for hotkey/desktop wakes.
            async def bus_handler(event: WakeEvent) -> dict[str, Any]:
                return await _handle_wake_event(active, event, request_id=str(uuid.uuid4()))

            wake_bus = WakeBus(active.settings.wake_socket_path, bus_handler)
            await wake_bus.start()
        app.state.wake_bus = wake_bus
        yield
        if app.state.wake_bus is not None:
            await app.state.wake_bus.stop()
            app.state.wake_bus = None
        if app.state.container is not None:
            await app.state.container.aclose()

    app = FastAPI(title="APRIL Core API", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.state.container_error = False
    app.state.wake_bus = None
    initial_settings = container.settings if container is not None else get_settings()
    if initial_settings.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1", "http://localhost"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    @app.exception_handler(AprilError)
    async def april_error_handler(request: Request, exc: AprilError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return JSONResponse(error_payload(exc, request_id), status_code=exc.status_code)

    @app.middleware("http")
    async def enforce_request_size(request: Request, call_next: Any) -> object:
        active_settings = (
            app.state.container.settings if app.state.container is not None else initial_settings
        )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > active_settings.api.max_request_bytes:
                request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
                error = RequestTooLargeError(
                    "Request body exceeds configured maximum size.",
                    {"max_request_bytes": active_settings.api.max_request_bytes},
                )
                return JSONResponse(error_payload(error, request_id), status_code=413)
        return await call_next(request)

    async def get_container() -> ApiContainer:
        if app.state.container is None:
            app.state.container = await build_container()
        return app.state.container

    async def authorized(
        authorization: str | None = Header(default=None),
        active: ApiContainer = Depends(get_container),
    ) -> ApiContainer:
        await require_bearer_token(active.settings, authorization)
        return active

    @app.get("/health")
    async def health() -> object:
        return {"status": "ok", "service": "april-core-api"}

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
            if request.job_type != "configured_test" or record.tool != "test_runner":
                raise HTTPException(status_code=409, detail="approval_action_mismatch")
            expected_payload = {
                "argv": list(record.args.get("argv", ["pytest"])),
                "cwd": str(record.args.get("repo_path", "")),
            }
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

    @app.get("/diagnostics")
    async def diagnostics(active: ApiContainer = Depends(authorized)) -> object:
        diagnostic_status = "ok"
        memory_index = await active.memory_repository.health()
        if memory_index.repair_required:
            diagnostic_status = "degraded"
        try:
            runtime = await active.runtime_client.health(timeout=1.0)
            if str(runtime.get("status", "ok")) not in {"ok", "degraded"}:
                diagnostic_status = "degraded"
        except AprilError as exc:
            runtime = {"status": "unavailable", "error": exc.message}
            diagnostic_status = "degraded"
        return {
            "status": diagnostic_status,
            "database": {"ok": active.database.path.exists(), "path": str(active.database.path)},
            "vector_index": active.vector_memory.health(),
            "memory_index": asdict(memory_index),
            "voice": voice_health(active.settings).model_dump(),
            "wake": {
                "enabled": active.settings.wake.enabled,
                "muted": active.settings.mute_flag_path.exists(),
                "state": read_wake_status(active.settings)["state"],
            },
            "scheduler": {
                "enabled": active.settings.scheduler.enabled,
                "running": active.scheduler.running if active.scheduler else False,
                "briefing_enabled": active.settings.scheduler.briefing_enabled,
                "fired_reminders": (
                    active.scheduler.fired_reminder_count if active.scheduler else 0
                ),
            },
            "runtime_url": active.settings.runtime.url,
            "runtime": runtime,
        }

    @app.get("/diagnostics/activity")
    async def diagnostics_activity(
        limit: int = 50,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        capped = max(1, min(limit, _ACTIVITY_MAX_LIMIT))
        events = _read_activity_events(active.settings.audit_path, capped)
        return {"events": events, "count": len(events)}

    @app.get("/readiness")
    async def readiness(active: ApiContainer = Depends(authorized)) -> object:
        return await _readiness_payload(active)

    @app.get("/verification/report/latest")
    async def verification_report_latest(
        type: str = "any",
        active: ApiContainer = Depends(authorized),
    ) -> object:
        # ?type=any (default) | real_model (multi_model+target_mac)
        # | voice_live | voice_conversation_live | workflow. Unknown values fall back to
        # "any", so existing callers (and the ignored-?path= probe) keep their
        # behaviour.
        return _latest_verification_report(active.settings, report_type=type)

    @app.get("/verification/reports")
    async def verification_reports(
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return _verification_report_history(active.settings)

    @app.get("/verification/reports/{report_basename}")
    async def verification_report_by_basename(
        report_basename: str,
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return _verification_report_detail(active.settings, report_basename)

    @app.get("/reports")
    async def reports_index(
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return _browser_reports(active.settings)

    @app.get("/reports/latest")
    async def reports_latest_any(
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return _browser_latest(active.settings)

    @app.get("/reports/latest/{report_type}")
    async def reports_latest_typed(
        report_type: str,
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        if report_type not in _BROWSER_REPORT_TYPES:
            raise HTTPException(status_code=404, detail="unknown report type")
        return _browser_latest(active.settings, report_type=report_type)

    @app.post("/chat")
    async def chat(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> ChatResponse:
        request_id = x_request_id or str(uuid.uuid4())
        async with active.require_session_manager().interaction(request.conversation_id):
            result = await active.orchestrator.chat(
                request.message,
                conversation_id=request.conversation_id,
                request_id=request_id,
                project_id=request.project_id,
                repo_path=request.repo_path,
                mode=request.mode,
            )
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        request_id = x_request_id or str(uuid.uuid4())
        interaction = active.require_session_manager().interaction(request.conversation_id)
        # Enter before returning the response: request acceptance itself is
        # activity, even if streaming later fails or the client disconnects.
        await interaction.__aenter__()

        async def events() -> AsyncIterator[str]:
            try:
                async for event_name, payload in active.orchestrator.stream_chat(
                    request.message,
                    conversation_id=request.conversation_id,
                    request_id=request_id,
                    project_id=request.project_id,
                    repo_path=request.repo_path,
                    mode=request.mode,
                ):
                    yield _sse_event(event_name, request_id, payload)
            finally:
                await interaction.__aexit__(None, None, None)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/voice/input")
    async def voice_input(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
    ) -> ChatResponse:
        request_id = str(uuid.uuid4())
        async with active.require_session_manager().interaction(request.conversation_id):
            result = await active.orchestrator.chat(
                request.message,
                conversation_id=request.conversation_id,
                request_id=request_id,
                project_id=request.project_id,
                repo_path=request.repo_path,
                mode=request.mode,
            )
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/wake")
    async def wake(
        request: WakeRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        event = WakeEvent(
            source=request.source,
            score=request.score,
            text=request.text,
            reason=request.reason,
            captured_at=request.captured_at,
            session_hint=request.session_hint,
        )
        return await _handle_wake_event(active, event, request_id=request_id)

    @app.get("/wake/mute")
    async def wake_mute_status(active: ApiContainer = Depends(authorized)) -> object:
        # Redacted state only; the flag/status paths themselves are never exposed.
        return {
            "muted": MuteSwitch(active.settings.mute_flag_path).is_muted(),
            "state": read_wake_status(active.settings)["state"],
        }

    @app.post("/wake/mute")
    async def wake_mute_set(
        request: WakeMuteRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        switch = MuteSwitch(active.settings.mute_flag_path)
        if request.muted:
            switch.mute()
        else:
            switch.unmute()
        active.approvals.audit.write(
            {
                "event_type": "wake_mute_changed",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "muted": request.muted,
            }
        )
        return {
            "muted": switch.is_muted(),
            "state": read_wake_status(active.settings)["state"],
            "audited": True,
        }

    @app.get("/sessions")
    async def sessions(
        limit: int = 50,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        records = await active.memory.list_sessions(limit=limit)
        return {"sessions": [record.model_dump() for record in records]}

    @app.post("/sessions")
    async def session_attach(
        request: SessionAttachRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        event = WakeEvent(source=request.source, reason="session_attach")
        return await _handle_wake_event(active, event, request_id=request_id)

    @app.post("/sessions/{session_id}/close")
    async def session_close(
        session_id: str,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        session_manager = active.require_session_manager()
        record = await active.memory.get_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session not found")
        closed = await session_manager.close(session_id)
        active.approvals.audit.write(
            {
                "event_type": "session_closed",
                "request_id": request_id,
                "actor": "local-user",
                "reference_id": session_id,
                "outcome": "closed" if closed else "already_closed",
            }
        )
        return {"session_id": session_id, "closed": closed}

    @app.post("/agents/run")
    async def agents_run(
        request: AgentRunRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> ChatResponse:
        request_id = x_request_id or str(uuid.uuid4())
        if not request.options.structured:
            raise PermissionDeniedError(
                "Direct agent runs only support structured execution.",
                {"agent": request.agent},
            )
        async with active.require_session_manager().interaction(request.conversation_id):
            result = await active.orchestrator.run_agent(
                agent_id=request.agent,
                message=request.message,
                conversation_id=request.conversation_id,
                request_id=request_id,
                project_id=request.project_id,
                repo_path=request.repo_path,
            )
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/tools/request")
    async def tool_request(
        request: ToolRequestEnvelope,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        context = await active.tool_executor.context(
            request_id=request_id,
            actor="local-user",
            agent_id=request.agent,
            project_id=str(request.args["project_id"]) if request.args.get("project_id") else None,
            source="api",
        )
        outcome = await active.tool_executor.request_or_execute(
            tool=request.tool,
            args=request.args,
            context=context,
        )
        if outcome.approval is not None:
            return {"status": "pending_approval", "approval": outcome.approval}
        return {"status": outcome.status, "result": outcome.result}

    @app.post("/tools/approve")
    async def approve(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        approval = await active.approvals.get(request.approval_id)
        playbook_run_id = approval.metadata.get("playbook_run_id")
        if isinstance(playbook_run_id, str):
            if request.tool is not None:
                raise PermissionDeniedError(
                    "Playbook approvals resume only their persisted exact action."
                )
            result = await PlaybookRunner(active.tool_executor, memory=active.memory).resume(
                playbook_run_id,
                approval_id=request.approval_id,
                actor="local-user",
            )
            return {"playbook_run": asdict(result)}
        return await active.orchestrator.approve_tool(
            approval_id=request.approval_id,
            actor="local-user",
            request_id=request_id,
            tool=request.tool,
            args=request.args if request.tool is not None else None,
        )

    @app.post("/tools/deny")
    async def deny(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        approval = await active.approvals.get(request.approval_id)
        result = await active.orchestrator.deny_tool(
            approval_id=request.approval_id,
            actor="local-user",
            request_id=x_request_id or str(uuid.uuid4()),
        )
        playbook_run_id = approval.metadata.get("playbook_run_id")
        if isinstance(playbook_run_id, str):
            playbook = await PlaybookRunner(active.tool_executor, memory=active.memory).mark_denied(
                playbook_run_id, approval_id=request.approval_id
            )
            return {"approval": result, "playbook_run": asdict(playbook)}
        return result

    @app.get("/approvals")
    async def approvals(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "approvals": [record.model_dump() for record in await active.approvals.list_pending()]
        }

    @app.post("/memory")
    async def memory_create(
        request: MemoryCreateRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> object:
        if (
            request.project_id is not None
            and await active.memory.get_project(request.project_id) is None
        ):
            raise PermissionDeniedError(
                "Unknown project for project-scoped memory.",
                {"project_id": request.project_id},
            )
        if request.source_conversation_id is not None:
            conversation = await active.memory.get_conversation(request.source_conversation_id)
            if conversation is None:
                raise PermissionDeniedError(
                    "Unknown source conversation for memory write.",
                    {"conversation_id": request.source_conversation_id},
                )
            if conversation.project_id != request.project_id:
                raise PermissionDeniedError(
                    "Memory source conversation project scope does not match.",
                    {
                        "conversation_project_id": conversation.project_id,
                        "memory_project_id": request.project_id,
                    },
                )

        writer = MemoryWriter(active.memory_repository)
        record = await writer.write(
            request.content,
            reason=request.reason,
            memory_type=request.memory_type,
            requested_by_user=True,
            project_id=request.project_id,
        )
        await active.memory_repository.set_provenance(
            record.id,
            source_conversation_id=request.source_conversation_id,
        )
        index_health = await active.memory_repository.health()
        active.approvals.audit.write(
            {
                "event_type": "memory_written",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "memory_id": record.id,
                "memory_type": record.kind,
                "project_id": record.project_id,
                "source_conversation_id": request.source_conversation_id,
                "content_length": len(record.content),
                "reason_length": len(record.reason),
            }
        )
        return {
            "memory": record.model_dump(),
            "stored": f"Stored {record.kind} memory.",
            "index_repair_required": index_health.repair_required,
        }

    @app.get("/memory/search")
    async def memory_search(
        q: str,
        project_id: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if project_id is not None and await active.memory.get_project(project_id) is None:
            raise PermissionDeniedError(
                "Unknown project for memory search.", {"project_id": project_id}
            )
        results = await active.memory.search_memories(q, project_id=project_id)
        return {"results": [result.model_dump() for result in results]}

    @app.delete("/memory/{memory_id}")
    async def memory_delete(memory_id: str, active: ApiContainer = Depends(authorized)) -> object:
        deleted = await active.memory_repository.delete_memory(memory_id)
        index_health = await active.memory_repository.health()
        return {
            "deleted": deleted,
            "index_repair_required": index_health.repair_required,
        }

    @app.get("/memory/inspect")
    async def memory_inspect(
        state: str = "machine",
        limit: int = 100,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            records = await active.memory.list_memories_by_state(state, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "state": state,
            "memories": [record.model_dump() for record in records],
        }

    @app.get("/memory/export")
    async def memory_export(
        project_id: str | None = None, active: ApiContainer = Depends(authorized)
    ) -> object:
        if project_id is not None and await active.memory.get_project(project_id) is None:
            raise PermissionDeniedError(
                "Unknown project for memory export.", {"project_id": project_id}
            )
        return {"export": await active.memory.export_memories(project_id=project_id)}

    @app.post("/memory/reindex")
    async def memory_reindex(active: ApiContainer = Depends(authorized)) -> object:
        reindexed = await active.memory_repository.rebuild()
        configured_provider = active.settings.memory.embedding_provider
        active_provider = active.vector_memory.embedding.name
        vector_health = active.vector_memory.health()
        # A degraded index is one where the configured provider fell back to
        # hashed-token, or the persisted index does not match the active
        # provider. Callers must never see a silent mix.
        fallback_active = configured_provider == "runtime-local" and (
            active_provider == "hashed-token"
        )
        return {
            "reindexed": reindexed,
            "provider": active_provider,
            "configured_provider": configured_provider,
            "dimensions": active.vector_memory.embedding.dimensions,
            "index_compatible": bool(vector_health.get("compatible", True)),
            "fallback_active": fallback_active,
            "degraded": fallback_active or not bool(vector_health.get("compatible", True)),
        }

    @app.post("/memory/repair-index")
    async def memory_repair_index(
        apply: bool = False,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return active.vector_memory.repair_index(apply=apply)

    @app.post("/feedback")
    async def feedback(
        request: FeedbackRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        conversation_id = request.conversation_id
        session_id: str | None = None
        if conversation_id is None:
            # Bind to the active session's conversation when one exists.
            session = await active.memory.latest_open_session()
            if session is not None:
                session_id = session.id
                conversation_id = session.conversation_id
        agent_run_id = request.agent_run_id
        if agent_run_id is None:
            agent_run_id = await active.memory.latest_agent_run_id(conversation_id=conversation_id)
        record = await active.memory.record_feedback_event(
            rating=request.rating,
            reason=request.reason,
            session_id=session_id,
            conversation_id=conversation_id,
            agent_run_id=agent_run_id,
        )
        if record.rating == "bad" and record.agent_run_id is not None:
            await active.orchestrator.routing_reliability.mark_negative_feedback(
                agent_run_id=record.agent_run_id
            )
        active.approvals.audit.write(
            {
                "event_type": "feedback_recorded",
                "request_id": request_id,
                "actor": "local-user",
                "rating": record.rating,
                "reason_length": len(record.reason or ""),
                "agent_run_bound": record.agent_run_id is not None,
            }
        )
        if record.rating == "bad":
            # Stage a reviewable pending eval case; never let staging failures
            # break feedback recording itself.
            with contextlib.suppress(Exception):
                await stage_feedback_eval_case(
                    active.settings,
                    active.memory,
                    record,
                    kind="explicit_feedback",
                    audit=active.approvals.audit,
                )
        return {"feedback": record.model_dump()}

    @app.get("/reminders")
    async def reminders(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "reminders": [
                reminder.model_dump() for reminder in await active.memory.list_reminders()
            ]
        }

    @app.post("/reminders")
    async def reminder_create(
        request: ReminderCreateRequest, active: ApiContainer = Depends(authorized)
    ) -> object:
        reminder = await active.memory.create_reminder(request.content, due_at=request.due_at)
        return {"reminder": reminder.model_dump()}

    @app.delete("/reminders/{reminder_id}")
    async def reminder_delete(
        reminder_id: str, active: ApiContainer = Depends(authorized)
    ) -> object:
        return {"deleted": await active.memory.delete_reminder(reminder_id)}

    @app.get("/tasks")
    async def tasks(active: ApiContainer = Depends(authorized)) -> object:
        return {"tasks": [task.model_dump() for task in await active.memory.list_tasks()]}

    @app.get("/playbooks")
    async def playbooks(active: ApiContainer = Depends(authorized)) -> object:
        loader = PlaybookLoader(active.settings.playbooks_path)
        return {"playbooks": [playbook.model_dump() for playbook in loader.list()]}

    @app.post("/playbooks/adopt")
    async def playbook_adopt(
        request: PlaybookDefinition,
        approval_id: str | None = None,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        # Level 3+ playbooks require an exact-action adoption approval: the
        # first call returns pending_approval; re-posting the identical
        # definition with ?approval_id=... completes adoption.
        adoption = PlaybookAdoptionService(
            loader=PlaybookLoader(active.settings.playbooks_path),
            tool_registry=active.tool_registry,
            approvals=active.approvals,
            memory=active.memory,
            approval_required_at=active.permission_engine.approval_required_at,
        )
        return await adoption.adopt(
            request,
            actor="local-user",
            request_id=x_request_id or str(uuid.uuid4()),
            approval_id=approval_id,
        )

    @app.post("/playbooks/mine")
    async def playbook_mine(
        support_threshold: int = 3,
        lookback_days: int = 14,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        report = await mine_playbook_candidates(
            active.memory,
            active.settings,
            guard=EvolutionWriteGuard(active.settings, audit=active.approvals.audit),
            support_threshold=max(2, support_threshold),
            lookback_days=max(1, lookback_days),
        )
        return {"mine": report.to_payload()}

    @app.post("/playbooks/{playbook_id}/run")
    async def playbook_run(
        playbook_id: str,
        request: PlaybookRunRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        loader = PlaybookLoader(active.settings.playbooks_path)
        playbook = loader.get(playbook_id)
        if playbook is None:
            raise HTTPException(status_code=404, detail="playbook not found")
        async with active.require_session_manager().interaction(request.conversation_id):
            result = await PlaybookRunner(active.tool_executor, memory=active.memory).run(
                playbook,
                conversation_id=request.conversation_id,
                project_id=request.project_id,
            )
        return {"run": asdict(result)}

    @app.get("/playbooks/{playbook_id}/runs")
    async def playbook_runs(
        playbook_id: str,
        limit: int = 50,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        runs = await active.memory.list_playbook_runs(playbook_id=playbook_id, limit=limit)
        return {"runs": runs}

    @app.post("/playbooks/runs/{run_id}/resume")
    async def playbook_resume(
        run_id: str,
        request: PlaybookResumeRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        row = await active.memory.get_playbook_run(run_id)
        conversation_id = (
            str(row["conversation_id"])
            if row is not None and row["conversation_id"] is not None
            else None
        )
        async with active.require_session_manager().interaction(conversation_id):
            result = await PlaybookRunner(active.tool_executor, memory=active.memory).resume(
                run_id, approval_id=request.approval_id
            )
        return {"run": asdict(result)}

    @app.get("/evolution/versions")
    async def evolution_versions(
        agent: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = PromptOverlayManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        return {"versions": await manager.versions(agent=agent)}

    @app.post("/evolution/rollback")
    async def evolution_rollback(
        request: EvolutionRollbackRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = PromptOverlayManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        result = await manager.rollback(agent=request.agent, version=request.version)
        payload = asdict(result)
        if payload.get("path") is not None:
            payload["path"] = str(payload["path"])
        return {"rollback": payload}

    @app.get("/evolution/adapters")
    async def evolution_adapters(
        model_id: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = AdapterLifecycleManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        return {"adapters": await manager.list(model_id=model_id)}

    @app.post("/evolution/adapters/activate")
    async def evolution_adapters_activate(
        request: AdapterActivateRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = AdapterLifecycleManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        result = await manager.activate(
            model_id=request.model_id,
            adapter_path=Path(request.adapter_path),
            evidence_path=(Path(request.evidence_path) if request.evidence_path else None),
            verification_report_path=(
                Path(request.verification_report_path) if request.verification_report_path else None
            ),
        )
        return {"activation": result.to_payload()}

    @app.post("/evolution/adapters/rollback")
    async def evolution_adapters_rollback(
        request: AdapterRollbackRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        manager = AdapterLifecycleManager(
            active.settings,
            active.database,
            audit=active.approvals.audit,
        )
        result = await manager.rollback(model_id=request.model_id, version=request.version)
        return {"rollback": result.to_payload()}

    @app.get("/evolution/report/latest")
    async def evolution_report_latest(active: ApiContainer = Depends(authorized)) -> object:
        return {"report": latest_report(active.settings)}

    @app.get("/evolution/status")
    async def evolution_status_endpoint(active: ApiContainer = Depends(authorized)) -> object:
        status = await evolution_status(active.settings, active.database)
        status["scheduler_running"] = active.scheduler.running if active.scheduler else False
        return {"status": status}

    @app.get("/evolution/history")
    async def evolution_history_endpoint(
        limit: int = 20,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return {"runs": await evolution_history(active.database, limit=limit)}

    @app.get("/evolution/diff")
    async def evolution_diff_endpoint(
        agent: str,
        from_version: int | None = None,
        to_version: int | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return await overlay_diff(
            active.settings,
            active.database,
            agent=agent,
            from_version=from_version,
            to_version=to_version,
        )

    @app.post("/evolution/off")
    async def evolution_off(
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        result = set_evolution_kill_switch(active.settings, disabled=True)
        active.approvals.audit.write(
            {
                "event_type": "evolution_kill_switch_set",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "outcome": "disabled",
            }
        )
        return result

    @app.post("/evolution/on")
    async def evolution_on(
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        result = set_evolution_kill_switch(active.settings, disabled=False)
        active.approvals.audit.write(
            {
                "event_type": "evolution_kill_switch_set",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "outcome": "cleared",
            }
        )
        return result

    @app.post("/evolution/dataset/export")
    async def evolution_dataset_export(
        request: DatasetExportRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        result = await export_finetune_dataset(
            active.memory,
            active.settings,
            dataset_name=request.name,
        )
        active.approvals.audit.write(
            {
                "event_type": "evolution_dataset_exported",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "outcome": "written",
            }
        )
        return {"export": result.to_payload()}

    @app.get("/evolution/overlays/pending")
    async def evolution_overlays_pending(
        active: ApiContainer = Depends(authorized),
    ) -> object:
        service = PromptOverlayApprovalService(
            active.settings,
            active.database,
            audit=active.approvals.audit,
            runtime_client=active.runtime_client,
        )
        return {"pending": [item.to_payload() for item in await service.list_pending()]}

    @app.post("/evolution/overlays/approve")
    async def evolution_overlays_approve(
        request: OverlayApprovalRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        service = PromptOverlayApprovalService(
            active.settings,
            active.database,
            audit=active.approvals.audit,
            runtime_client=active.runtime_client,
        )
        result = await service.approve(agent=request.agent, content_hash=request.content_hash)
        payload = asdict(result)
        if payload.get("path") is not None:
            payload["path"] = str(payload["path"])
        return {"approval": payload}

    @app.get("/evolution/evals/pending")
    async def evolution_evals_pending(active: ApiContainer = Depends(authorized)) -> object:
        return {"pending": list_pending_cases(active.settings)}

    @app.get("/evolution/evals/pending/{case_id}")
    async def evolution_evals_pending_case(
        case_id: str,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            case = get_pending_case(active.settings, case_id)
        except EvalReviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if case is None:
            raise HTTPException(status_code=404, detail="pending eval case not found")
        return {"case": case}

    @app.post("/evolution/evals/promote")
    async def evolution_evals_promote(
        request: EvalPromoteRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            result = promote_pending_case(
                active.settings,
                request.case_id,
                expected_behavior=request.expected_behavior,
                audit=active.approvals.audit,
            )
        except EvalReviewError as exc:
            status_code = 404 if "unknown" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return {"promoted": result}

    @app.post("/evolution/evals/reject")
    async def evolution_evals_reject(
        request: EvalRejectRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            result = reject_pending_case(
                active.settings,
                request.case_id,
                reason=request.reason,
                audit=active.approvals.audit,
            )
        except EvalReviewError as exc:
            status_code = 404 if "unknown" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return {"rejected": result}

    @app.get("/scheduler/briefing/preview")
    async def scheduler_briefing_preview(
        active: ApiContainer = Depends(authorized),
    ) -> object:
        now = utc_now()
        until = now + timedelta(hours=24)
        repo_activity = None
        if active.settings.scheduler.repo_monitor_enabled:
            # Preview must not advance the baseline (persist=False, idempotent).
            repo_activity = await compute_repo_activity(active.memory, persist=False)
        notification = await compose_briefing(
            active.memory,
            now_iso=now.isoformat().replace("+00:00", "Z"),
            until_iso=until.isoformat().replace("+00:00", "Z"),
            repo_activity=repo_activity,
            evolution_report=latest_report(active.settings),
        )
        return notification.model_dump()

    @app.delete("/conversations/{conversation_id}")
    async def conversation_delete(
        conversation_id: str, active: ApiContainer = Depends(authorized)
    ) -> object:
        return {"deleted": await active.memory.delete_conversation(conversation_id)}

    @app.get("/projects")
    async def projects(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "projects": [project.model_dump() for project in await active.memory.list_projects()]
        }

    @app.post("/projects")
    async def project_add(
        request: ProjectCreateRequest, active: ApiContainer = Depends(authorized)
    ) -> object:
        normalized = _normalize_project_path(request.path, active.settings)
        project = await active.memory.add_project(str(normalized), name=request.name)
        return project

    @app.post("/projects/{project_id}/index")
    async def project_index(project_id: str, active: ApiContainer = Depends(authorized)) -> object:
        project = await active.memory.get_project(project_id)
        if project is None:
            raise PermissionDeniedError("Project not found.")
        request_id = str(uuid.uuid4())
        context = await active.tool_executor.context(
            request_id=request_id,
            actor="local-user",
            agent_id="coding_agent",
            project_id=project_id,
            source="api",
        )
        outcome = await active.tool_executor.request_or_execute(
            tool="repo_indexer",
            args={"repo_path": project.path, "project_id": project_id},
            context=context,
        )
        return {"result": outcome.result}

    @app.post("/documents")
    async def document_add(
        request: DocumentCreateRequest, active: ApiContainer = Depends(authorized)
    ) -> object:
        request_id = str(uuid.uuid4())
        context = await active.tool_executor.context(
            request_id=request_id,
            actor="local-user",
            agent_id="reading_agent",
            source="api",
        )
        outcome = await active.tool_executor.request_or_execute(
            tool="document_indexer",
            args={"folder_path": request.path},
            context=context,
        )
        return {"result": outcome.result}

    @app.get("/documents")
    async def documents(active: ApiContainer = Depends(authorized)) -> object:
        return {"documents": active.vector_memory.sources(source_type="document")}

    @app.get("/documents/search")
    async def documents_search(q: str, active: ApiContainer = Depends(authorized)) -> object:
        chunks = active.memory_retriever.document_chunks(q)
        return {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "citations": [
                {
                    "path": chunk.metadata.get("path"),
                    "start_line": chunk.metadata.get("start_line"),
                    "end_line": chunk.metadata.get("end_line"),
                }
                for chunk in chunks
                if chunk.metadata.get("path")
            ],
        }

    @app.get("/pool/agents")
    async def pool_agents(active: ApiContainer = Depends(authorized)) -> object:
        pool = AgentPool(
            active.memory,
            known_agents=[agent.name for agent in active.agent_registry.list()],
        )
        return {"agents": [card.to_payload() for card in await pool.scorecards()]}

    @app.get("/runtime/models")
    async def runtime_models(active: ApiContainer = Depends(authorized)) -> object:
        return await active.runtime_client.models()

    @app.post("/runtime/models/load")
    async def runtime_model_load(
        request: LoadModelRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return await active.runtime_client.load(request.model_id, request_id=request.request_id)

    @app.post("/runtime/models/unload")
    async def runtime_model_unload(
        request: LoadModelRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return await active.runtime_client.unload(request.model_id, request_id=request.request_id)

    # Serve the local Desktop SPA from the Core API (same-origin, loopback only).
    # The static assets ship no secrets; all data still flows through the
    # authenticated endpoints above. Mounted last so it never shadows API routes.
    if _DESKTOP_WEB_DIR.is_dir():
        app.mount(
            "/desktop",
            StaticFiles(directory=str(_DESKTOP_WEB_DIR), html=True),
            name="desktop",
        )

    return app


async def _handle_wake_event(
    active: ApiContainer, event: WakeEvent, *, request_id: str
) -> dict[str, Any]:
    """Converge one wake event: resolve the session, then route optional text.

    ``event.text`` (typed or STT-confirmed command) is executed as a normal chat
    turn inside the session's conversation — through the standard orchestrator,
    Brain routing, and permission engine. Wake events never bypass anything.
    """
    session_manager = active.require_session_manager()
    resolution = await session_manager.handle_wake(event)
    active.approvals.audit.write(
        {
            "event_type": "wake_event",
            "request_id": request_id,
            "actor": "local-user",
            "source": event.source,
            "session_id": resolution.session_id,
            "joined_existing": resolution.joined_existing,
            "score": event.score,
            "reason": event.reason,
            "transcript_present": bool(event.text),
        }
    )
    payload: dict[str, Any] = {
        "request_id": request_id,
        **resolution.model_dump(),
    }
    if event.text:
        feedback = classify_wake_feedback(event.text)
        if feedback is not None:
            payload["result"] = await _record_wake_feedback(
                active,
                feedback,
                session_id=resolution.session_id,
                conversation_id=resolution.conversation_id,
                request_id=request_id,
            )
            await session_manager.touch(resolution.session_id)
            return payload
        async with session_manager.interaction(resolution.conversation_id):
            result = await active.orchestrator.chat(
                event.text,
                conversation_id=resolution.conversation_id,
                request_id=request_id,
            )
        payload["result"] = result.model_dump()
    return payload


async def _record_wake_feedback(
    active: ApiContainer,
    feedback: WakeFeedback,
    *,
    session_id: str,
    conversation_id: str | None,
    request_id: str,
) -> dict[str, Any]:
    agent_run_id = await active.memory.latest_agent_run_id(conversation_id=conversation_id)
    if agent_run_id is None:
        active.approvals.audit.write(
            {
                "event_type": "wake_feedback_noop",
                "request_id": request_id,
                "actor": "local-user",
                "kind": "wake_feedback",
                "rating": feedback.rating,
                "reason_length": len(feedback.phrase),
                "agent_run_bound": False,
            }
        )
        return {
            "final_message": "I do not have a recent answer to attach that feedback to.",
            "feedback_recorded": False,
        }
    record = await active.memory.record_feedback_event(
        rating=feedback.rating,
        reason=f"wake_feedback: {feedback.phrase}",
        session_id=session_id,
        conversation_id=conversation_id,
        agent_run_id=agent_run_id,
    )
    active.approvals.audit.write(
        {
            "event_type": "feedback_recorded",
            "request_id": request_id,
            "actor": "local-user",
            "kind": "wake_feedback",
            "rating": record.rating,
            "reason_length": len(record.reason or ""),
            "agent_run_bound": True,
        }
    )
    if record.rating == "bad":
        with contextlib.suppress(Exception):
            await stage_feedback_eval_case(
                active.settings,
                active.memory,
                record,
                kind="wake_feedback",
                audit=active.approvals.audit,
            )
    return {"final_message": "Feedback recorded.", "feedback_recorded": True}


def _redact_health_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"path", "model_path", "binary_path"}:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_health_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_health_payload(item) for item in value]
    return value


async def _readiness_payload(active: ApiContainer) -> dict[str, Any]:
    runtime_status = "unavailable"
    runtime_backend = "unknown"
    runtime_simulated: bool | None = None
    runtime_health: dict[str, Any]
    runtime_probe = await asyncio.to_thread(
        probe_service_health,
        active.settings.runtime.url.rstrip("/") + "/runtime/health",
        bearer_token=active.settings.runtime.token,
        timeout=1.0,
    )
    if runtime_probe.ok:
        try:
            raw_runtime = await active.runtime_client.health(timeout=1.0)
            runtime_health = _safe_runtime_health(raw_runtime)
            runtime_status = str(raw_runtime.get("status", "unknown"))
            runtime_backend = str(raw_runtime.get("backend", "unknown"))
            simulated = raw_runtime.get("simulated")
            runtime_simulated = simulated if isinstance(simulated, bool) else None
        except AprilError as exc:
            runtime_health = {"status": "unavailable", "error": exc.message}
            runtime_probe = ServiceHealthResult(
                ok=False,
                status_code=None,
                reason="invalid_response",
                message="Runtime health response could not be read.",
            )
    else:
        runtime_health = {
            "status": "unavailable",
            "probe_reason": runtime_probe.reason,
            "http_status": runtime_probe.status_code,
            "error": runtime_probe.message,
        }

    if runtime_probe.ok:
        try:
            raw_models = await active.runtime_client.models()
            models = [
                _safe_model_entry(model, runtime_backend) for model in raw_models.get("models", [])
            ]
        except AprilError:
            models = []
    else:
        models = []
    if not models and isinstance(runtime_health.get("models"), list):
        models = [
            _safe_model_entry(model, runtime_backend)
            for model in runtime_health.get("models", [])
            if isinstance(model, dict)
        ]

    vector_health = active.vector_memory.health()
    memory_index_health = await active.memory_repository.health()
    vector = _redact_health_payload(vector_health)
    # The embedding provider is a first-class readiness axis: hashed-token is the
    # safe/offline default, runtime-local is the recommended hardened path. The
    # active provider, whether it fell back, and whether the on-disk index matches
    # are surfaced (booleans/enums only — never a path) so a silent mix is visible.
    configured_embedding_provider = active.settings.memory.embedding_provider
    active_embedding_provider = str(vector_health.get("embedding", "hashed-token"))
    embedding_index_compatible = bool(vector_health.get("compatible", True))
    embedding_model_status = _embedding_model_status(active.settings)
    fell_back_to_hashed_token = (
        configured_embedding_provider == "runtime-local"
        and active_embedding_provider == "hashed-token"
    )
    embedding_warnings: list[str] = []
    if (
        active.settings.environment == "production"
        and configured_embedding_provider != "runtime-local"
    ):
        embedding_warnings.append(
            "runtime-local embeddings are not configured in production-like mode"
        )
    if fell_back_to_hashed_token:
        embedding_warnings.append("runtime-local embeddings fell back to hashed-token")
    if not embedding_model_status["embedding_model_registered"]:
        embedding_warnings.append("no embedding-role model is registered")
    embeddings = {
        "configured_provider": configured_embedding_provider,
        "active_provider": active_embedding_provider,
        "runtime_local_requested": configured_embedding_provider == "runtime-local",
        "fell_back_to_hashed_token": fell_back_to_hashed_token,
        "hashed_token_active": active_embedding_provider == "hashed-token",
        "hashed_token_fallback_active": fell_back_to_hashed_token,
        "embedding_model_id": active.settings.memory.embedding_model_id,
        "dimensions": vector_health.get("dimensions"),
        "index_compatible": embedding_index_compatible,
        "persisted_provider": vector_health.get("persisted_provider"),
        "reindex_required": not embedding_index_compatible,
        "reindex_command": "run april memory reindex",
        "warnings": embedding_warnings,
        "active_generation": vector_health.get("effective_generation"),
        "last_successful_reindex_at": vector_health.get("last_successful_reindex_at"),
        "vector_index_status": vector_health.get("status"),
        "repair_command": vector_health.get("repair_command"),
    }
    embeddings.update(embedding_model_status)
    # query_audio_devices() only *enumerates* devices; it never opens the
    # microphone or starts a stream. Readiness stays inert by construction.
    devices = query_audio_devices()
    voice_readiness = voice_readiness_summary(active.settings, devices)
    # Lift the offline milestone to a live rung only when a redacted live report
    # proves it. wake_live_verified outranks live_verified.
    live_flags = _latest_live_voice_flags(active.settings)
    voice_milestone = str(voice_readiness.get("voice_milestone", "not_configured"))
    if active.settings.voice.enabled:
        if live_flags["voice_conversation_live_verified"]:
            voice_milestone = "conversation_live_verified"
        elif live_flags["wake_word_live_verified"]:
            voice_milestone = "wake_live_verified"
        elif live_flags["voice_live_verified"]:
            voice_milestone = "live_verified"
    voice_artifacts = [
        _voice_artifact(
            active.settings,
            "wake confirmation whisper binary",
            active.settings.voice.effective_confirmation_whisper_binary_path,
        ),
        _voice_artifact(
            active.settings,
            "wake confirmation whisper model",
            active.settings.voice.effective_confirmation_whisper_model_path,
        ),
        _voice_artifact(
            active.settings,
            "transcription whisper binary",
            active.settings.voice.effective_transcription_whisper_binary_path,
        ),
        _voice_artifact(
            active.settings,
            "transcription whisper model",
            active.settings.voice.effective_transcription_whisper_model_path,
        ),
        _voice_artifact(active.settings, "piper binary", active.settings.voice.piper_binary_path),
        _voice_artifact(active.settings, "piper model", active.settings.voice.piper_model_path),
    ]
    wake_word_model_paths = _wake_word_model_artifacts(active.settings)
    voice_artifacts.extend(wake_word_model_paths)
    api_localhost = active.settings.api.host in {"127.0.0.1", "localhost"}
    runtime_localhost = active.settings.runtime.url.startswith(
        ("http://127.0.0.1", "http://localhost")
    )
    try:
        database_available = (await active.database.fetchone("SELECT 1")) is not None
    except Exception:
        database_available = False
    database_integrity = await asyncio.to_thread(
        check_database,
        active.settings.database_path,
        home=active.settings.home,
    )
    try:
        audit_size = (
            active.settings.audit_path.stat().st_size if active.settings.audit_path.exists() else 0
        )
        audit_chain_status = (
            active.approvals.audit.verify().status
            if audit_size <= 4 * 1024 * 1024
            else "explicit_verification_required"
        )
    except (OSError, RuntimeError, AprilError):
        audit_chain_status = "unavailable"
    legacy_plaintext = legacy_plaintext_credentials_detected(active.settings.home)
    credential_store_selected: str = active.settings.security.credential_store
    if credential_store_selected == "auto":
        credential_store_selected = (
            "macos-keychain"
            if active.settings.environment == "production" and platform.system() == "Darwin"
            else "legacy-development-default"
        )
    model_registry = _model_registry_readiness(active.settings)
    summary_readiness = _conversation_summary_readiness(
        active,
        runtime_available=runtime_probe.ok,
    )
    adapter_state = await AdapterLifecycleManager(
        active.settings,
        active.database,
        audit=active.approvals.audit,
    ).state_health()
    job_counts = (
        await active.job_store.counts()
        if active.job_store is not None
        else {"queued": 0, "running": 0, "interrupted": 0, "expired_leases": 0}
    )
    tool_worker_live = bool(
        active.tool_worker_manager is not None
        and active.tool_worker_manager.process is not None
        and active.tool_worker_manager.process.returncode is None
    )
    tool_worker_socket_mode: str | None = None
    tool_worker_protocol_ready = False
    tool_worker_self_check = False
    if active.tool_worker_manager is not None:
        try:
            tool_worker_socket_mode = validate_live_socket(
                active.tool_worker_manager.socket_path,
                runtime_directory=active.tool_worker_manager.runtime_directory,
            )
            tool_worker_protocol_ready = active.tool_worker_client is not None
            if active.tool_worker_client is not None and active.settings.allowed_roots:
                response = await active.tool_worker_client.self_check(
                    project_root=active.settings.allowed_roots[0]
                )
                tool_worker_self_check = response.ok
        except (OSError, UnsafeToolWorkerSocket, Exception):
            tool_worker_protocol_ready = False
            tool_worker_self_check = False
    job_worker_live = bool(
        active.job_worker_manager is not None
        and active.job_worker_manager.process is not None
        and active.job_worker_manager.process.returncode is None
    )
    job_worker_ready = bool(
        active.job_worker_manager is not None and active.job_worker_manager.status_path.exists()
    )
    scheduler_required = active.settings.scheduler.enabled
    scheduler_available = active.scheduler is not None and active.scheduler.running
    failure_reasons = _readiness_failure_reasons(
        runtime_probe=runtime_probe,
        runtime_status=runtime_status,
        database_available=database_available,
        model_registry=model_registry,
        scheduler_required=scheduler_required,
        scheduler_available=scheduler_available,
        vector_health=vector_health,
    )
    if not bool(adapter_state["consistent"]):
        failure_reasons.append(
            {
                "code": "adapter_state_inconsistent",
                "message": "Adapter lifecycle state requires reconciliation.",
            }
        )
    if active.settings.workers.tool_worker_enabled and (
        not tool_worker_protocol_ready or not tool_worker_self_check
    ):
        failure_reasons.append(
            {
                "code": "tool_worker_unavailable",
                "message": "Tool Worker is not ready; risky tools fail closed.",
            }
        )
    if active.settings.workers.job_worker_enabled and not job_worker_ready:
        failure_reasons.append(
            {
                "code": "job_worker_unavailable",
                "message": "Job Worker is not ready; durable jobs remain queued.",
            }
        )
    ready = not failure_reasons
    overlay_approval_service = PromptOverlayApprovalService(
        active.settings,
        active.database,
        audit=active.approvals.audit,
        runtime_client=active.runtime_client,
    )
    pending_write_capable_overlays = await overlay_approval_service.list_pending()
    pending_real_runtime_blockers = _pending_real_runtime_overlay_blockers(active.settings)
    real_runtime_required = active.settings.environment == "production"
    return {
        "ready": ready,
        "status": "ok" if ready else "degraded",
        "failure_reasons": failure_reasons,
        "core": {
            "api_health": "ok",
            "runtime_health": runtime_status,
            "runtime_backend": runtime_backend,
            "runtime_simulated": runtime_simulated,
            "runtime": runtime_health,
            "database": {
                "status": "ok" if database_available else "unavailable",
                "configured": True,
                "available": database_available,
            },
            "vector_index": vector,
            "scheduler": {
                "enabled": active.settings.scheduler.enabled,
                "running": active.scheduler.running if active.scheduler else False,
                "briefing_enabled": active.settings.scheduler.briefing_enabled,
            },
            "required_services": {
                "runtime": runtime_probe.ok,
                "database": database_available,
                "scheduler": not scheduler_required or scheduler_available,
            },
        },
        "memory_index": asdict(memory_index_health),
        "models": {
            "llama_cpp_python_available": importlib.util.find_spec("llama_cpp") is not None,
            "registered": models,
            "lora_adapters": _lora_adapter_readiness(active.settings),
            "adapter_lifecycle": adapter_state,
            "registry": model_registry,
        },
        "embeddings": embeddings,
        "conversation_summarization": summary_readiness,
        "jobs": {
            "schema_version": SCHEMA_VERSION,
            "worker_enabled": active.settings.workers.job_worker_enabled,
            "queued": job_counts.get("queued", 0),
            "running": job_counts.get("running", 0),
            "interrupted": job_counts.get("interrupted", 0),
            "expired_leases": job_counts.get("expired_leases", 0),
            "worker_liveness": job_worker_live,
            "worker_readiness": job_worker_ready,
        },
        "tool_worker": {
            "enabled": active.settings.workers.tool_worker_enabled,
            "process_liveness": tool_worker_live,
            "socket_available": tool_worker_socket_mode is not None,
            "socket_mode": tool_worker_socket_mode,
            "protocol_readiness": tool_worker_protocol_ready,
            "self_check": tool_worker_self_check,
        },
        "process_policy": {
            "environment_policy_version": PROCESS_ENVIRONMENT_POLICY_VERSION,
            "unsupported_resource_limits": list(
                resource_limit_report(ResourceLimitProfile.COMMAND).unsupported
            ),
        },
        "evolution": {
            "enabled": active.settings.evolution.enabled,
            "kill_switch_active": evolution_kill_switch_active(active.settings),
            "scheduler_enabled": active.settings.scheduler.enabled,
            "scheduler_running": active.scheduler.running if active.scheduler else False,
            "overlay_eval_mode": (
                "deterministic_fixture_plus_real_runtime"
                if real_runtime_required
                else "deterministic_fixture"
            ),
            "deterministic_fixture_eval_kind": "deterministic_fixture",
            "real_runtime_eval_required": real_runtime_required,
            "pending_real_runtime_overlay_blocker_count": len(pending_real_runtime_blockers),
            "pending_real_runtime_overlay_blockers": pending_real_runtime_blockers,
            "pending_write_capable_overlay_approval_count": len(pending_write_capable_overlays),
            "pending_write_capable_overlay_candidate_count": (
                count_pending_write_capable_overlay_candidates(active.settings)
            ),
            "pending_eval_case_count": count_pending_eval_cases(active.settings),
        },
        # Redacted local config digest + per-type report freshness, so the Desktop
        # operator console and `doctor --daily-driver` can flag stale reports.
        "config_fingerprint": config_fingerprint_digest(active.settings.home),
        "reports": _reports_freshness(active.settings),
        "verification_guidance": {
            "commands": [
                "run april verify --all-configured-models --require-real-model "
                "--report data/verification/mac-readiness.json",
                "run april verify --workflow --real-model "
                "--report data/verification/workflow-real.json",
                "run april verify /absolute/path/to/model.gguf --target-mac "
                "--require-real-model --report data/verification/single-model.json",
                "run april voice verify-wake-live --report data/verification/wake-live.json",
            ],
            "warnings": [
                "Fake verification is not real model verification.",
                "Desktop never loads models or starts voice automatically.",
                "Reports are redacted and show model basenames only.",
                "Generated verification reports and app stubs are ignored by Git.",
            ],
        },
        "voice": {
            "enabled": active.settings.voice.enabled,
            "sounddevice_available": bool(devices.get("sounddevice_installed")),
            "microphone_access": microphone_access(devices)["status"],
            "input_device_count": len(devices.get("input_devices", [])),
            "output_device_count": len(devices.get("output_devices", [])),
            "macos_microphone_permission_guidance": (
                "macOS: System Settings > Privacy & Security > Microphone. "
                "Allow the terminal app used to run APRIL."
            ),
            "artifacts": voice_artifacts,
            "wake_word_model_paths": wake_word_model_paths,
            "wake_live_report_status": (
                "verified" if live_flags["wake_word_live_verified"] else "not_verified"
            ),
            "wake_live_report_missing": not live_flags["wake_word_live_verified"],
            "push_to_talk_available_without_wake_word": True,
            "openwakeword_available": voice_readiness["openwakeword_available"],
            "push_to_talk_ready": voice_readiness["push_to_talk_ready"],
            "wake_word_ready": voice_readiness["wake_word_ready"],
            "full_voice_loop_ready": voice_readiness["full_voice_loop_ready"],
            "sentinel_live_verified": live_flags["wake_word_live_verified"],
            "conversation_endpointing_configured": True,
            "endpoint_silence_ms": active.settings.voice.endpoint_silence_ms,
            "minimum_utterance_ms": active.settings.voice.minimum_utterance_ms,
            "barge_in_trigger": active.settings.voice.barge_in_trigger,
            "barge_in_action": active.settings.voice.barge_in_action,
            "acoustic_echo_cancellation_available": False,
            "complete_live_conversation_verified": live_flags["voice_conversation_live_verified"],
            "complete_live_conversation_command": (
                "run april voice verify-conversation-live "
                "--report data/verification/voice-conversation-live.json"
            ),
            "speaker_gate": {
                "mode": active.settings.wake.speaker_gate,
                "supported": False,
                "detail": (
                    (
                        "speaker_gate=soft is configured, but no production local "
                        "speaker verifier model ships with APRIL; Sentinel audits one "
                        "warning and behaves as off."
                        if active.settings.wake.speaker_gate == "soft"
                        else "speaker_gate is off; enrollment does not enable it by itself."
                    )
                    + " It is a convenience filter, never a security boundary."
                ),
            },
            # Single redacted enum capturing the highest voice milestone reached:
            # disabled / not_configured / push_to_talk_ready / wake_word_ready /
            # full_voice_loop_ready / live_verified / wake_live_verified.
            "voice_milestone": voice_milestone,
        },
        "security": {
            "allowed_filesystem_roots": [
                {
                    "basename": root.name or str(root),
                    "exists": root.exists(),
                    "within_april_home": _is_relative_to(root, active.settings.home),
                }
                for root in active.settings.allowed_roots
            ],
            "api_token": {"status": "configured" if active.settings.api.token else "missing"},
            "runtime_token": {
                "status": "configured" if active.settings.runtime.token else "missing"
            },
            "credential_store": credential_store_selected,
            "legacy_plaintext_credential_detected": legacy_plaintext,
            "audit_chain_status": audit_chain_status,
            "database_integrity": {
                "quick_check": database_integrity.quick_check,
                "foreign_key_consistent": database_integrity.foreign_key_consistent,
                "wal_state": database_integrity.journal_mode,
                "last_successful_backup": database_integrity.last_successful_backup,
                "checked_at": database_integrity.checked_at,
            },
            "api_localhost_binding": api_localhost,
            "runtime_localhost_binding": runtime_localhost,
            "cors_enabled": active.settings.api.cors_enabled,
            "development_token_warning": _development_token_warning(active.settings),
        },
        "daemon": _daemon_readiness(active.settings),
        "next_actions": [
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
            "run april voice verify-live --report data/verification/voice-live.json",
            "run april voice verify-wake-live --report data/verification/wake-live.json",
            "run april setup app-stub",
        ],
    }


def _model_registry_readiness(settings: AprilSettings) -> dict[str, Any]:
    router_model_id = settings.brain.router_model_id or settings.brain.model_id
    router_aliased = settings.brain.router_model_id is None
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml",
            root=settings.home,
        )
    except AprilError:
        return {
            "valid": False,
            "required_model_available": False,
            "required_model_ids": [],
            "unavailable_required_model_ids": [],
            "router_model_id": router_model_id,
            "router_aliased_to_brain": router_aliased,
            "dedicated_router_available": False,
            "router_failure_reason": "model_registry_invalid",
        }
    required_models = [model for model in registry.list() if model.role == "brain"]
    unavailable = [
        model.id
        for model in required_models
        if settings.runtime.backend != "fake"
        and model.backend != "fake"
        and not model.resolved_path(registry.root).is_file()
    ]
    router_failure_reason: str | None = None
    dedicated_router_available = False
    if router_aliased:
        router_valid = registry.exists(settings.brain.model_id)
        if not router_valid:
            router_failure_reason = "aliased_brain_model_not_registered"
    elif not registry.exists(router_model_id):
        router_valid = False
        router_failure_reason = "dedicated_router_not_registered"
    else:
        router_model = registry.get(router_model_id)
        router_valid = router_model.role == "router"
        dedicated_router_available = router_valid and (
            settings.runtime.backend == "fake"
            or router_model.backend == "fake"
            or router_model.resolved_path(registry.root).is_file()
        )
        if not router_valid:
            router_failure_reason = "dedicated_router_role_mismatch"
        elif not dedicated_router_available:
            router_failure_reason = "dedicated_router_artifact_unavailable"
    return {
        "valid": True,
        "required_model_available": (bool(required_models) and not unavailable and router_valid),
        "required_model_ids": [model.id for model in required_models],
        "unavailable_required_model_ids": unavailable,
        "router_model_id": router_model_id,
        "router_aliased_to_brain": router_aliased,
        "dedicated_router_available": dedicated_router_available,
        "router_failure_reason": router_failure_reason,
    }


def _conversation_summary_readiness(
    active: ApiContainer, *, runtime_available: bool
) -> dict[str, Any]:
    enabled = active.settings.conversation_context.summary_enabled
    reading_agent = active.agent_registry.get("reading_agent")
    model_id = reading_agent.model_id if reading_agent is not None else None
    model_entry_exists = False
    model_artifact_available = False
    if model_id:
        try:
            registry = ModelRegistry.from_file(
                active.settings.home / "configs" / "models.yaml",
                root=active.settings.home,
            )
            model_entry_exists = registry.exists(model_id)
            if model_entry_exists:
                model = registry.get(model_id)
                model_artifact_available = (
                    active.settings.runtime.backend == "fake"
                    or model.backend == "fake"
                    or model.resolved_path(registry.root).is_file()
                )
        except AprilError:
            pass
    available = bool(
        enabled
        and model_id
        and model_entry_exists
        and model_artifact_available
        and runtime_available
    )
    return {
        "enabled": enabled,
        "reading_agent_configured": reading_agent is not None,
        "model_entry_exists": model_entry_exists,
        "available": available,
        "degrades_safely": True,
        "status": ("disabled" if not enabled else ("available" if available else "degraded")),
    }


def _readiness_failure_reasons(
    *,
    runtime_probe: ServiceHealthResult,
    runtime_status: str,
    database_available: bool,
    model_registry: dict[str, Any],
    scheduler_required: bool,
    scheduler_available: bool,
    vector_health: dict[str, Any],
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    if not runtime_probe.ok:
        if runtime_probe.reason == "authentication_rejected":
            message = "Runtime authentication was rejected."
        elif runtime_probe.reason == "endpoint_not_found":
            message = "The configured Runtime health endpoint is missing or incorrect."
        else:
            message = "Runtime is not reachable."
        reasons.append({"code": f"runtime_{runtime_probe.reason}", "message": message})
    elif runtime_status not in {"ok", "degraded"}:
        reasons.append(
            {
                "code": "runtime_unhealthy",
                "message": "Runtime reported an unhealthy operational status.",
            }
        )
    if not database_available:
        reasons.append({"code": "database_unavailable", "message": "Database is unavailable."})
    if not bool(model_registry["valid"]):
        reasons.append({"code": "model_registry_invalid", "message": "Model registry is invalid."})
    elif not bool(model_registry["required_model_available"]):
        reasons.append(
            {
                "code": "required_model_unavailable",
                "message": "A required model is unavailable.",
            }
        )
    if scheduler_required and not scheduler_available:
        reasons.append(
            {
                "code": "required_scheduler_unavailable",
                "message": "The required scheduler service is unavailable.",
            }
        )
    vector_status = str(vector_health.get("status", "not_ready"))
    if vector_status == "not_ready":
        reasons.append(
            {
                "code": "vector_index_unavailable",
                "message": "Vector retrieval has no valid index generation.",
            }
        )
    elif bool(vector_health.get("fallback_active")):
        reasons.append(
            {
                "code": "vector_index_recovery_active",
                "message": "Vector retrieval is using a read-only recovery generation.",
            }
        )
    elif vector_status == "degraded":
        reasons.append(
            {
                "code": "vector_index_degraded",
                "message": "Vector retrieval requires index repair or reindexing.",
            }
        )
    return reasons


def _embedding_model_status(settings: AprilSettings) -> dict[str, Any]:
    model_id = settings.memory.embedding_model_id
    status: dict[str, Any] = {
        "embedding_model_registered": False,
        "embedding_model_path_exists": False,
        "embedding_model_missing_reason": None,
    }
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml",
            root=settings.home,
        )
    except Exception:
        status["embedding_model_missing_reason"] = "model registry is unavailable"
        return status
    candidates = [model for model in registry.list() if model.role == "embedding"]
    model = None
    if model_id:
        with contextlib.suppress(Exception):
            candidate = registry.get(model_id)
            if candidate.role == "embedding":
                model = candidate
    elif candidates:
        model = candidates[0]
    if model is None:
        if settings.memory.embedding_provider == "runtime-local":
            status["embedding_model_missing_reason"] = (
                "runtime-local requested without a registered role=embedding model"
                if not model_id
                else "embedding model id is not registered with role=embedding"
            )
        else:
            status["embedding_model_missing_reason"] = "no role=embedding model is registered"
        return status
    path = model.resolved_path(registry.root)
    status["embedding_model_registered"] = True
    status["embedding_model_path_exists"] = path.exists()
    if not path.exists():
        status["embedding_model_missing_reason"] = f"missing model file: {path.name}"
    return status


def _wake_word_model_artifacts(settings: AprilSettings) -> list[dict[str, Any]]:
    paths = settings.voice.effective_wake_word_model_paths
    if not paths:
        return [_voice_artifact(settings, "wake-word model", None)]
    artifacts: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        name = "wake-word model" if index == 0 else f"wake-word model {index + 1}"
        artifacts.append(_voice_artifact(settings, name, path))
    return artifacts


def _lora_adapter_readiness(settings: AprilSettings) -> list[dict[str, Any]]:
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml",
            root=settings.home,
        )
    except Exception:
        return []
    adapters: list[dict[str, Any]] = []
    for model in registry.list():
        adapter = model.resolved_adapter_path(registry.root)
        if adapter is None:
            continue
        exists = adapter.exists()
        adapters.append(
            {
                "model_id": model.id,
                "configured": True,
                "missing": not exists,
                "basename": adapter.name,
                "status": "present_unverified" if exists else "missing_blocker",
                "detail": (
                    "adapter present; real-model verification still required"
                    if exists
                    else "configured adapter file is missing; model load fails closed"
                ),
            }
        )
    return adapters


def _pending_real_runtime_overlay_blockers(settings: AprilSettings) -> list[dict[str, str]]:
    report = latest_report(settings)
    if report is None:
        return []
    phases = report.get("phases")
    examine = phases.get("examine") if isinstance(phases, dict) else None
    pending = examine.get("pending_real_runtime") if isinstance(examine, dict) else None
    if not isinstance(pending, list):
        return []
    blockers: list[dict[str, str]] = []
    for item in pending:
        if not isinstance(item, dict):
            continue
        blockers.append(
            {
                "agent": str(item.get("agent") or "unknown"),
                "status": str(item.get("status") or "unknown"),
                "reason": _redact_path_text(
                    str(item.get("reason") or "real-runtime evaluation did not pass")
                )[:240],
            }
        )
    return blockers


def _daemon_readiness(settings: AprilSettings) -> dict[str, Any]:
    try:
        from apps.daemon.apriald import read_daemon_status

        payload = read_daemon_status(settings)
    except Exception:
        payload = {"status": "unknown", "details_available": False}
    return {
        "status": payload.get("status", "unknown"),
        "details_available": bool(payload.get("details_available", False)),
        "children": payload.get("children", []),
        "governor": payload.get("governor", {}),
    }


def _safe_runtime_health(payload: dict[str, Any]) -> dict[str, Any]:
    safe = _redact_health_payload(payload)
    if isinstance(safe, dict) and isinstance(safe.get("models"), list):
        backend = str(safe.get("backend", "unknown"))
        safe["models"] = [
            _safe_model_entry(model, backend) for model in safe["models"] if isinstance(model, dict)
        ]
    return safe if isinstance(safe, dict) else {"status": "unknown"}


def _safe_model_entry(model: dict[str, Any], runtime_backend: str) -> dict[str, Any]:
    path = model.get("path")
    backend = str(model.get("backend") or runtime_backend or "unknown")
    return {
        "id": str(model.get("id", "unknown")),
        "name": str(model.get("name", "unknown")),
        "role": str(model.get("role", "unknown")),
        "backend": backend,
        "state": str(model.get("state", "unknown")),
        "keep_loaded": bool(model.get("keep_loaded", False)),
        "missing_path": bool(model.get("missing_path", False)),
        "simulated": backend == "fake" or runtime_backend == "fake",
        "path_basename": _basename(path),
        "context_size": model.get("context_size"),
        "load_error": (
            _redact_path_text(str(model.get("load_error"))) if model.get("load_error") else None
        ),
    }


def _voice_artifact(settings: AprilSettings, name: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"name": name, "configured": False, "missing": True, "basename": None}
    resolved = settings.resolve_path(path)
    return {
        "name": name,
        "configured": True,
        "missing": not resolved.exists(),
        "basename": resolved.name,
    }


def _development_token_warning(settings: AprilSettings) -> str | None:
    if not settings.api.token or settings.api.token in INSECURE_API_TOKENS:
        return "API token uses an insecure development/placeholder default or is empty."
    if not settings.runtime.token or settings.runtime.token in INSECURE_RUNTIME_TOKENS:
        return "Runtime token uses an insecure development/placeholder default or is missing."
    return None


def _verification_root(settings: AprilSettings) -> Path:
    return (settings.home / "data" / "verification").resolve()


def _verification_report_files(settings: AprilSettings) -> list[Path]:
    root = _verification_root(settings)
    if not root.exists() or not root.is_dir():
        return []
    candidates: list[Path] = []
    for path in root.glob("*.json"):
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if (
            path.is_file()
            and not path.is_symlink()
            and _VERIFICATION_REPORT_BASENAME_RE.match(path.name)
            and _is_relative_to(resolved, root)
        ):
            candidates.append(path)
    return candidates


def _read_safe_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# A "real model" verification is either the single-model target-Mac report or the
# multi-model report; both genuinely exercise real GGUF models. The voice-live
# report is a separate axis and must never be selected as the real-model latest.
_REAL_MODEL_REPORT_TYPES = {"multi_model", "target_mac"}


def _classified_report_type(payload: dict[str, Any]) -> str:
    report_type = str(payload.get("report_type") or _infer_report_type(payload))
    return report_type if report_type in _VERIFICATION_REPORT_TYPES else "unknown"


def _report_matches_filter(report_type: str, filter_type: str) -> bool:
    if filter_type == "any":
        return True
    if filter_type == "real_model":
        return report_type in _REAL_MODEL_REPORT_TYPES
    return report_type == filter_type


def _latest_verification_report(
    settings: AprilSettings, *, report_type: str = "any"
) -> dict[str, Any]:
    # The latest report is selected *within the requested class* by the safe report
    # timestamp first, falling back to mtime only when the report timestamp is
    # absent/invalid. A newer voice-live report can never overwrite the latest
    # real-model report (or vice versa).
    filter_type = (
        report_type
        if report_type in {"any", "real_model", "voice_live", "voice_conversation_live", "workflow"}
        else "any"
    )
    candidates = _verification_report_files(settings)
    matching: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        payload = _read_safe_report(path)
        if payload is None:
            continue
        if _report_matches_filter(_classified_report_type(payload), filter_type):
            matching.append((path, payload))
    if not matching:
        if filter_type == "any" and candidates:
            # Files exist but none could be read as JSON objects.
            return {
                "status": "unreadable",
                "message": "latest verification report could not be read",
                "report": None,
            }
        return {
            "status": "not_verified",
            "message": "not verified yet",
            "report": None,
        }
    latest_path, latest_payload = max(matching, key=lambda item: _report_order_key(*item))
    return {
        "status": "ok",
        "message": "latest verification report",
        "report": _safe_report_payload(latest_payload, latest_path),
    }


def _reports_freshness(settings: AprilSettings) -> dict[str, Any]:
    """Per-type freshness for the latest report of each kind (redacted).

    Returns only basenames, statuses, ages, and stale booleans/reasons — never a
    token, prompt, transcript, patch, or absolute path. Staleness combines a
    per-type age TTL with the redacted config fingerprint embedded in each report.
    """
    current_fingerprint = config_fingerprint_digest(settings.home)
    latest: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is None:
            continue
        report_type = _browser_report_type(payload)
        if report_type == "unknown":
            continue
        key = _report_order_key(path, payload)
        if report_type not in latest or key > latest[report_type][0]:
            latest[report_type] = (key, path, payload)
    out: dict[str, Any] = {}
    for report_type, (_key, path, payload) in latest.items():
        fresh = freshness_from_payload(
            payload,
            report_type=report_type,
            current_fingerprint=current_fingerprint,
            basename=path.name,
        )
        status = payload.get("final_status") or payload.get("summary")
        out[report_type] = {
            "basename": path.name,
            "report_type": report_type,
            "status": str(status) if status is not None else None,
            "generated_at": fresh.generated_at,
            "age_seconds": fresh.age_seconds,
            "age_human": fresh.age_human,
            "stale": fresh.stale,
            "stale_reason": fresh.stale_reason,
            "config_fingerprint_matches": fresh.config_fingerprint_matches,
        }
    return out


def _latest_live_voice_flags(settings: AprilSettings) -> dict[str, bool]:
    """Read the latest live voice / wake-word verification flags from disk.

    Returns only two booleans (never a transcript, device, or path). Used to lift
    the offline voice milestone to its ``live_verified`` / ``wake_live_verified``
    rungs. Reading a report never opens the microphone.
    """
    voice_verified = False
    wake_verified = False
    conversation_verified = False
    voice_best: float | None = None
    wake_best: float | None = None
    conversation_best: float | None = None
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is None:
            continue
        declared = str(payload.get("report_type") or "")
        if declared == "voice_live":
            key = _report_order_key(path, payload)
            if voice_best is None or key > voice_best:
                voice_best = key
                voice_verified = bool(payload.get("voice_live_verified", False))
        elif declared == "wake_word_live":
            key = _report_order_key(path, payload)
            if wake_best is None or key > wake_best:
                wake_best = key
                wake_verified = bool(payload.get("wake_word_live_verified", False))
        elif declared == "voice_conversation_live":
            key = _report_order_key(path, payload)
            if conversation_best is None or key > conversation_best:
                conversation_best = key
                conversation_verified = bool(
                    payload.get("evidence_mode") == "real_hardware"
                    and payload.get("voice_conversation_live_verified", False)
                )
    return {
        "voice_live_verified": voice_verified,
        "wake_word_live_verified": wake_verified,
        "voice_conversation_live_verified": conversation_verified,
    }


def _verification_report_history(settings: AprilSettings) -> dict[str, Any]:
    matching: list[tuple[Path, dict[str, Any]]] = []
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is None:
            continue
        matching.append((path, payload))
    matching.sort(key=lambda item: _report_order_key(*item), reverse=True)
    reports: list[dict[str, Any]] = []
    for path, payload in matching:
        reports.append(_safe_report_payload(payload, path))
    if not reports:
        return {
            "status": "not_verified",
            "message": "not verified yet",
            "reports": [],
            "count": 0,
        }
    return {
        "status": "ok",
        "message": "verification report history",
        "reports": reports,
        "count": len(reports),
    }


def _verification_report_detail(settings: AprilSettings, report_basename: str) -> dict[str, Any]:
    path = _safe_report_path(settings, report_basename)
    payload = _read_safe_report(path)
    if payload is None:
        raise HTTPException(status_code=404, detail="verification report not found")
    return {
        "status": "ok",
        "message": "verification report",
        "report": _safe_report_payload(payload, path),
    }


# The read-only ``/reports`` browser surface. Its allowlist projection covers the
# acceptance/mac-activation/wake-word axes the older ``/verification`` projection
# does not, while still emitting only basenames, statuses, levels, booleans, and
# redacted next-action commands — never tokens, transcripts, or absolute paths.
_BROWSER_REPORT_TYPES = {
    "acceptance",
    "go_live",
    "mac_activation",
    "voice_live",
    "wake_word_live",
    "voice_conversation_live",
    "multi_model",
    "workflow",
    "fake_soak",
}
_BROWSER_TYPE_ALIASES = {
    "acceptance": "acceptance",
    "go_live": "go_live",
    "mac_activation": "mac_activation",
    "voice_live": "voice_live",
    "wake_word_live": "wake_word_live",
    "voice_conversation_live": "voice_conversation_live",
    "multi_model": "multi_model",
    "workflow": "workflow",
    "soak": "fake_soak",
    "fake_soak": "fake_soak",
}


def _browser_report_type(payload: dict[str, Any]) -> str:
    declared = str(payload.get("report_type") or "")
    if declared in _BROWSER_TYPE_ALIASES:
        return _BROWSER_TYPE_ALIASES[declared]
    if "verification_level" in payload and "models" in payload:
        return "multi_model"
    if "iterations" in payload and "latency_ms" in payload:
        return "fake_soak"
    return "unknown"


def _browser_report_summary(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    report_type = _browser_report_type(payload)
    status = (
        payload.get("final_status")
        if report_type in {"acceptance", "go_live", "mac_activation"}
        else payload.get("summary")
    )
    services = payload.get("services")
    services_summary: dict[str, Any] | None = None
    if isinstance(services, dict) and services.get("requested"):
        services_summary = {
            "mode": str(services.get("mode", "none")),
            "startup_status": str(services.get("startup_status", "unknown")),
            "shutdown_status": str(services.get("shutdown_status", "unknown")),
            "api_reachable": bool(services.get("api_reachable", False)),
            "runtime_reachable": bool(services.get("runtime_reachable", False)),
        }
    level = payload.get("acceptance_level")
    backend = payload.get("runtime_backend")
    summary: dict[str, Any] = {
        "basename": path.name,
        "report_type": report_type,
        "generated_at": str(payload.get("generated_at") or payload.get("timestamp") or ""),
        "status": str(status) if status is not None else None,
        "acceptance_level": str(level) if level else None,
        "runtime_backend": str(backend) if backend else None,
        "services": services_summary,
        "next_actions": _safe_string_list(payload.get("next_actions")),
    }
    if report_type == "go_live":
        # Surface the core-vs-hardened distinction so the browser can show a
        # working real-model core separately from the hardened go-live rung.
        # Every value here is a boolean, a small enum, or a redacted advisory.
        summary["core_real_model_ready"] = bool(payload.get("core_real_model_ready", False))
        summary["real_model_core_status"] = str(payload.get("real_model_core_status") or "not_run")
        summary["hardened_go_live_ready"] = bool(payload.get("hardened_go_live_ready", False))
        summary["hardening_warnings"] = _safe_string_list(payload.get("hardening_warnings"))
        summary["hardening_blockers"] = _safe_string_list(payload.get("hardening_blockers"))
    return summary


def _sorted_browser_items(settings: AprilSettings) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is not None:
            items.append((path, payload))
    items.sort(key=lambda item: _report_order_key(*item), reverse=True)
    return items


def _browser_reports(settings: AprilSettings) -> dict[str, Any]:
    reports = [
        _browser_report_summary(payload, path) for path, payload in _sorted_browser_items(settings)
    ]
    return {
        "status": "ok" if reports else "empty",
        "count": len(reports),
        "reports": reports,
    }


def _browser_latest(settings: AprilSettings, *, report_type: str | None = None) -> dict[str, Any]:
    for path, payload in _sorted_browser_items(settings):
        summary = _browser_report_summary(payload, path)
        if report_type is None:
            if summary["report_type"] in _BROWSER_REPORT_TYPES:
                return {"status": "ok", "report": summary}
        elif summary["report_type"] == report_type:
            return {"status": "ok", "report": summary}
    return {"status": "not_found", "report": None}


def _safe_report_path(settings: AprilSettings, report_basename: str) -> Path:
    if (
        report_basename != Path(report_basename).name
        or "/" in report_basename
        or "\\" in report_basename
        or Path(report_basename).is_absolute()
        or not _VERIFICATION_REPORT_BASENAME_RE.match(report_basename)
    ):
        raise HTTPException(status_code=400, detail="unsafe report basename")
    root = (settings.home / "data" / "verification").resolve()
    path = root / report_basename
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="verification report not found") from exc
    if path.is_symlink() or not path.is_file() or not _is_relative_to(resolved, root):
        raise HTTPException(status_code=400, detail="unsafe report path")
    return path


def _safe_report_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    report_type = str(payload.get("report_type") or _infer_report_type(payload))
    if report_type not in _VERIFICATION_REPORT_TYPES:
        report_type = "unknown"
    summary = str(payload.get("summary", "degraded"))
    safe: dict[str, Any] = {
        "file_basename": path.name,
        "basename": path.name,
        "generated_at": str(payload.get("generated_at") or payload.get("timestamp") or ""),
        "report_type": report_type,
        "summary": summary,
        "real_model_verified": _report_real_model_verified(payload, report_type),
        "verification_level": _safe_verification_level(payload),
        "real_models_exercised": _safe_int(payload.get("real_models_exercised")),
        "real_models_passed": _safe_int(payload.get("real_models_passed")),
        "any_real_model_exercised": bool(payload.get("any_real_model_exercised", False)),
        "any_real_model_passed": bool(payload.get("any_real_model_passed", False)),
        "core_model_set_verified": bool(payload.get("core_model_set_verified", False)),
        "all_available_models_verified": bool(payload.get("all_available_models_verified", False)),
        "all_configured_models_verified": bool(
            payload.get("all_configured_models_verified", False)
        ),
        "skipped": _safe_skipped(payload.get("skipped")),
        "threshold_failures": _safe_string_list(payload.get("threshold_failures")),
    }
    safe["skipped_count"] = len(safe["skipped"])
    safe["threshold_failure_count"] = len(safe["threshold_failures"])
    if isinstance(payload.get("models"), list):
        safe["models"] = [
            {
                "model_id": str(model.get("model_id", model.get("id", "unknown"))),
                "role": str(model.get("role", "unknown")),
                "backend": str(model.get("backend", "unknown")),
                "path_basename": _basename(model.get("path_basename") or model.get("path")),
                "available": bool(model.get("available", False)),
                "skipped_reason": _redact_path_text(str(model.get("skipped_reason")))
                if model.get("skipped_reason")
                else None,
            }
            for model in payload["models"]
            if isinstance(model, dict)
        ]
    if isinstance(payload.get("real_model"), dict):
        real_model = payload["real_model"]
        safe["models"] = [
            {
                "model_id": str(real_model.get("model_id", "unknown")),
                "role": str(real_model.get("role", "unknown")),
                "backend": str(payload.get("runtime_backend", "unknown")),
                "path_basename": _basename(real_model.get("path_basename")),
                "available": bool(real_model.get("attempted", False)),
                "skipped_reason": None,
            }
        ]
    if report_type == "voice_live":
        # Voice-live reports expose only safe booleans/counts: a live-verified flag
        # and per-stage successes. Never a transcript, an audio file path, or a
        # device name — VoiceLiveReport does not store those, and this allowlist
        # projection keeps it that way even if new raw fields are added later.
        safe["voice_live_verified"] = bool(payload.get("voice_live_verified", False))
        safe["recording_success"] = bool(payload.get("recording_success", False))
        safe["stt_success"] = bool(payload.get("stt_success", False))
        safe["tts_success"] = bool(payload.get("tts_success", False))
        safe["playback_user_confirmed"] = bool(payload.get("playback_user_confirmed", False))
    if report_type == "voice_conversation_live":
        safe["voice_conversation_live_verified"] = bool(
            payload.get("voice_conversation_live_verified", False)
        )
        safe["evidence_mode"] = str(payload.get("evidence_mode", "unknown"))
        safe["turn_count"] = _safe_int(payload.get("turn_count"))
        safe["same_conversation"] = bool(payload.get("same_conversation", False))
        safe["barge_in_detected"] = bool(payload.get("barge_in_detected", False))
        safe["two_turns_completed"] = bool(payload.get("two_turns_completed", False))
        safe["follow_up_opened"] = bool(payload.get("follow_up_opened", False))
    if report_type == "workflow":
        safe["real_model_exercised"] = bool(payload.get("real_model_exercised", False))
        safe["checks"] = _safe_workflow_checks(payload.get("checks"))
    if "checks_failed" in payload:
        safe["checks_failed"] = payload.get("checks_failed")
    if "check_failures" in payload:
        safe["check_failures"] = _safe_string_list(payload.get("check_failures"))
    if "failures" in payload:
        safe["failures"] = _safe_string_list(payload.get("failures"))
    return safe


def _safe_verification_level(payload: dict[str, Any]) -> str:
    value = str(payload.get("verification_level", "none"))
    return value if value in {"none", "partial", "core", "all"} else "none"


def _safe_int(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


def _infer_report_type(payload: dict[str, Any]) -> str:
    if "real_model" in payload:
        return "target_mac"
    if "recording_success" in payload or "playback_user_confirmed" in payload:
        return "voice_live"
    if "checks" in payload and str(payload.get("report_type")) == "workflow":
        return "workflow"
    if "iterations" in payload and "latency_ms" in payload:
        return "soak"
    return "unknown"


def _report_real_model_verified(payload: dict[str, Any], report_type: str) -> bool:
    if report_type == "voice_live":
        return False
    if report_type == "workflow":
        return bool(payload.get("real_model_verified", False))
    if report_type in _REAL_MODEL_REPORT_TYPES and isinstance(
        payload.get("real_model_verified"), bool
    ):
        return bool(payload["real_model_verified"])
    if report_type == "target_mac" and isinstance(payload.get("real_model"), dict):
        real_model = payload["real_model"]
        return (
            str(payload.get("runtime_backend")) != "fake"
            and bool(real_model.get("attempted"))
            and bool(real_model.get("load_success"))
            and bool(real_model.get("chat_success"))
            and bool(real_model.get("streaming_success"))
            and bool(real_model.get("unload_success"))
        )
    return False


def _report_order_key(path: Path, payload: dict[str, Any]) -> float:
    parsed = _safe_report_timestamp(payload)
    if parsed is not None:
        return parsed.timestamp()
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _safe_report_timestamp(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("generated_at") or payload.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_workflow_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    checks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "name": str(item.get("name", "unknown")),
                "status": str(item.get("status", "unknown")),
                "ok": bool(item.get("ok", False)),
                "detail": _safe_workflow_detail(str(item.get("detail", ""))),
            }
        )
    return checks


def _safe_workflow_detail(detail: str) -> str:
    lower = detail.lower()
    if "decision_summary" in lower:
        return "decision_summary redacted"
    sensitive_markers = (
        "prompt",
        "transcript",
        "token",
        "authorization",
        "bearer",
        "raw_tool_args",
        "tool args",
    )
    if any(marker in lower for marker in sensitive_markers):
        return "sensitive detail redacted"
    return _redact_path_text(detail)[:240]


def _safe_skipped(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    skipped: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        skipped.append(
            {
                "name": str(item.get("name", "unknown")),
                "reason": _redact_path_text(str(item.get("reason", ""))),
            }
        )
    return skipped


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_redact_path_text(str(item)) for item in value]


def _redact_path_text(text: str) -> str:
    def _basename(match: re.Match[str]) -> str:
        name = Path(match.group(0)).name
        return name or match.group(0)

    return _PATH_TEXT_RE.sub(_basename, text)


def _basename(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text or text == "[REDACTED]":
        return None
    return Path(text).name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


async def _execute_approved_tool(
    active: ApiContainer, request: ToolApprovalAction, *, request_id: str
) -> object:
    outcome = await active.tool_executor.execute_approved(
        approval_id=request.approval_id,
        actor="local-user",
        request_id=request_id,
        tool=request.tool,
        args=request.args if request.tool is not None else None,
    )
    return {"status": outcome.status, "result": outcome.result}


def _normalize_project_path(path: str, settings: AprilSettings) -> Path:
    policy = PathPolicy(
        allowed_roots=tuple(settings.allowed_roots),
        max_read_bytes=settings.paths.max_file_read_bytes,
        max_write_bytes=settings.paths.max_file_write_bytes,
    )
    normalized = normalize_existing_path(path, policy)
    if not normalized.is_dir():
        raise PermissionDeniedError("Project path must be an existing directory.")
    return normalized


def _sse_event(event: str, request_id: str, payload: dict[str, Any]) -> str:
    body = {"request_id": request_id, "event": event, "payload": payload}
    return f"event: {event}\ndata: {json.dumps(body)}\n\n"


app = create_app()


def main() -> None:
    settings: AprilSettings = get_settings()
    uvicorn.run(
        "services.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
