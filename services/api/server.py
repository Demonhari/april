from __future__ import annotations

import importlib.util
import json
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from april_common.errors import (
    AprilError,
    PermissionDeniedError,
    RequestTooLargeError,
    error_payload,
)
from april_common.path_security import PathPolicy, normalize_existing_path
from april_common.settings import (
    INSECURE_API_TOKENS,
    INSECURE_RUNTIME_TOKENS,
    AprilSettings,
    get_settings,
)
from april_common.time import utc_now
from services.api.auth import require_bearer_token
from services.api.dependencies import ApiContainer, build_container
from services.api.schemas import (
    AgentRunRequest,
    ChatRequest,
    ChatResponse,
    DocumentCreateRequest,
    MemoryCreateRequest,
    ProjectCreateRequest,
    ReminderCreateRequest,
    ToolApprovalAction,
    ToolRequestEnvelope,
)
from services.april_runtime.schemas import LoadModelRequest
from services.memory.writer import MemoryWriter
from services.scheduler import compose_briefing, compute_repo_activity
from services.voice.health import microphone_access, query_audio_devices, voice_health

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
    }
)

_PATH_TEXT_RE = re.compile(r"~?(?:/[\w.\-]+){2,}/?")
_VERIFICATION_REPORT_TYPES = {
    "multi_model",
    "target_mac",
    "voice_live",
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
        projected = {key: value for key, value in record.items() if key in _ACTIVITY_ALLOWED_KEYS}
        if projected:
            events.append(projected)
        if len(events) >= limit:
            break
    return events


def create_app(container: ApiContainer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.container is None:
            app.state.container = await build_container()
        scheduler = app.state.container.scheduler
        if scheduler is not None:
            # start() is a no-op unless scheduler.enabled, so this is safe in tests.
            await scheduler.start()
        yield
        if app.state.container is not None:
            await app.state.container.aclose()

    app = FastAPI(title="APRIL Core API", version="0.1.0", lifespan=lifespan)
    app.state.container = container
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
            app.state.container.settings if app.state.container is not None else get_settings()
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
    async def health(active: ApiContainer = Depends(get_container)) -> object:
        status = "ok"
        try:
            runtime = await active.runtime_client.health(timeout=1.0)
            if str(runtime.get("status", "ok")) not in {"ok", "degraded"}:
                status = "degraded"
        except AprilError as exc:
            runtime = {"status": "unavailable", "error": exc.message}
            status = "degraded"
        return _redact_health_payload(
            {
                "status": status,
                "database": {
                    "ok": active.database.path.exists(),
                    "path": str(active.database.path),
                },
                "vector_index": active.vector_memory.health(),
                "voice": voice_health(active.settings).model_dump(),
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
        )

    @app.get("/diagnostics")
    async def diagnostics(active: ApiContainer = Depends(authorized)) -> object:
        diagnostic_status = "ok"
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
            "voice": voice_health(active.settings).model_dump(),
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
        # ?type=any (default) | real_model (multi_model+target_mac) | voice_live.
        # An unknown/extra query value falls back to "any", so existing callers
        # (and the ignored-?path= probe) keep their behaviour.
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

    @app.post("/chat")
    async def chat(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> ChatResponse:
        request_id = x_request_id or str(uuid.uuid4())
        result = await active.orchestrator.chat(
            request.message,
            conversation_id=request.conversation_id,
            request_id=request_id,
            project_id=request.project_id,
            repo_path=request.repo_path,
        )
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        request_id = x_request_id or str(uuid.uuid4())

        async def events() -> AsyncIterator[str]:
            async for event_name, payload in active.orchestrator.stream_chat(
                request.message,
                conversation_id=request.conversation_id,
                request_id=request_id,
                project_id=request.project_id,
                repo_path=request.repo_path,
            ):
                yield _sse_event(event_name, request_id, payload)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/voice/input")
    async def voice_input(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
    ) -> ChatResponse:
        request_id = str(uuid.uuid4())
        result = await active.orchestrator.chat(
            request.message,
            conversation_id=request.conversation_id,
            request_id=request_id,
            project_id=request.project_id,
            repo_path=request.repo_path,
        )
        return ChatResponse(request_id=request_id, result=result)

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
        return await active.orchestrator.deny_tool(
            approval_id=request.approval_id,
            actor="local-user",
            request_id=x_request_id or str(uuid.uuid4()),
        )

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

        writer = MemoryWriter(active.memory)
        record = await writer.write(
            request.content,
            reason=request.reason,
            memory_type=request.memory_type,
            requested_by_user=True,
            project_id=request.project_id,
        )
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
        return {"deleted": await active.memory.delete_memory(memory_id)}

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
        reindexed = active.vector_memory.reindex()
        return {
            "reindexed": reindexed,
            "provider": active.vector_memory.embedding.name,
            "dimensions": active.vector_memory.embedding.dimensions,
        }

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
    try:
        raw_runtime = await active.runtime_client.health(timeout=1.0)
        runtime_health = _safe_runtime_health(raw_runtime)
        runtime_status = str(raw_runtime.get("status", "unknown"))
        runtime_backend = str(raw_runtime.get("backend", "unknown"))
        simulated = raw_runtime.get("simulated")
        runtime_simulated = simulated if isinstance(simulated, bool) else None
    except AprilError as exc:
        runtime_health = {"status": "unavailable", "error": exc.message}

    try:
        raw_models = await active.runtime_client.models()
        models = [
            _safe_model_entry(model, runtime_backend) for model in raw_models.get("models", [])
        ]
    except AprilError:
        models = []
    if not models and isinstance(runtime_health.get("models"), list):
        models = [
            _safe_model_entry(model, runtime_backend)
            for model in runtime_health.get("models", [])
            if isinstance(model, dict)
        ]

    vector = _redact_health_payload(active.vector_memory.health())
    devices = query_audio_devices()
    voice_artifacts = [
        _voice_artifact(
            active.settings, "whisper binary", active.settings.voice.whisper_binary_path
        ),
        _voice_artifact(active.settings, "whisper model", active.settings.voice.whisper_model_path),
        _voice_artifact(active.settings, "piper binary", active.settings.voice.piper_binary_path),
        _voice_artifact(active.settings, "piper model", active.settings.voice.piper_model_path),
        _voice_artifact(
            active.settings, "wake-word model", active.settings.voice.wake_word_model_path
        ),
    ]
    api_localhost = active.settings.api.host in {"127.0.0.1", "localhost"}
    runtime_localhost = active.settings.runtime.url.startswith(
        ("http://127.0.0.1", "http://localhost")
    )
    degraded = (
        str(runtime_status) not in {"ok", "degraded"}
        or not active.database.path.exists()
        or runtime_simulated is True
    )
    return {
        "status": "degraded" if degraded else "ok",
        "core": {
            "api_health": "ok",
            "runtime_health": runtime_status,
            "runtime_backend": runtime_backend,
            "runtime_simulated": runtime_simulated,
            "database": {
                "status": "ok" if active.database.path.exists() else "missing",
                "configured": True,
            },
            "vector_index": vector,
            "scheduler": {
                "enabled": active.settings.scheduler.enabled,
                "running": active.scheduler.running if active.scheduler else False,
                "briefing_enabled": active.settings.scheduler.briefing_enabled,
            },
        },
        "models": {
            "llama_cpp_python_available": importlib.util.find_spec("llama_cpp") is not None,
            "registered": models,
        },
        "verification_guidance": {
            "commands": [
                "run april verify --all-configured-models --require-real-model "
                "--report data/verification/mac-readiness.json",
                "run april verify --workflow --real-model "
                "--report data/verification/workflow-real.json",
                "run april verify /absolute/path/to/model.gguf --target-mac "
                "--require-real-model --report data/verification/single-model.json",
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
            "push_to_talk_available_without_wake_word": True,
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
            "api_localhost_binding": api_localhost,
            "runtime_localhost_binding": runtime_localhost,
            "cors_enabled": active.settings.api.cors_enabled,
            "development_token_warning": _development_token_warning(active.settings),
        },
        "next_actions": [
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
            "run april voice verify-live --report data/verification/voice-live.json",
            "run april setup app-stub",
        ],
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
    filter_type = report_type if report_type in {"any", "real_model", "voice_live"} else "any"
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


def _verification_report_history(settings: AprilSettings) -> dict[str, Any]:
    candidates = sorted(
        _verification_report_files(settings),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    reports: list[dict[str, Any]] = []
    for path in candidates:
        payload = _read_safe_report(path)
        if payload is None:
            continue
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
    if "decision_summary" in detail.lower():
        return "decision_summary redacted"
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
