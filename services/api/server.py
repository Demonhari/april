from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from april_common.errors import (
    AprilError,
    PermissionDeniedError,
    RequestTooLargeError,
    error_payload,
)
from april_common.path_security import PathPolicy, normalize_existing_path
from april_common.settings import AprilSettings, get_settings
from services.api.auth import require_bearer_token
from services.api.dependencies import ApiContainer, build_container
from services.api.schemas import (
    ChatRequest,
    ChatResponse,
    ProjectCreateRequest,
    ToolApprovalAction,
    ToolRequestEnvelope,
)
from services.april_runtime.schemas import LoadModelRequest
from services.permissions.artifacts import (
    apply_approved_patch,
    build_git_commit_metadata,
    build_patch_approval_metadata,
    verify_approval_artifact,
)
from services.permissions.schemas import ApprovalRequest
from services.voice.health import voice_health
from skills.schemas import ToolResult


def create_app(container: ApiContainer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if app.state.container is not None:
            await app.state.container.database.close()

    app = FastAPI(title="APRIL Core API", version="0.1.0", lifespan=lifespan)
    app.state.container = container

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
        return {
            "status": "ok",
            "database": {"ok": active.database.path.exists(), "path": str(active.database.path)},
            "vector_index": active.vector_memory.health(),
            "voice": voice_health(active.settings).model_dump(),
            "runtime_url": active.settings.runtime.url,
        }

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
            request_id=request_id,
            project_id=request.project_id,
            repo_path=request.repo_path,
        )
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/agents/run")
    async def agents_run(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
    ) -> ChatResponse:
        request_id = str(uuid.uuid4())
        result = await active.orchestrator.chat(
            request.message,
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
        decision = active.permission_engine.evaluate(
            tool=request.tool,
            args=request.args,
            agent=request.agent,
        )
        if decision.confirmation_required:
            side_effects = ["Execute requested tool once."]
            approval = await active.approvals.create(
                ApprovalRequest(
                    tool=request.tool,
                    args=request.args,
                    agent=request.agent,
                    permission_level=decision.permission_level,
                    risk_level=decision.risk_level,
                    affected_paths=decision.affected_paths,
                    expected_side_effects=side_effects,
                    metadata=await _approval_metadata(request.tool, request.args, side_effects),
                ),
                actor="local-user",
                request_id=request_id,
            )
            return {"status": "pending_approval", "approval": approval}
        result = await active.tool_registry.execute(request.tool, request.args)
        await active.memory.record_tool_call(
            tool=request.tool,
            args=request.args,
            status="ok" if result.ok else "failed",
            permission_level=result.permission_level,
            risk_level=result.risk_level,
            result=result.model_dump(),
        )
        return {"status": "executed", "result": result}

    @app.post("/tools/approve")
    async def approve(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        return await _execute_approved_tool(active, request, request_id=request_id)

    @app.post("/tools/deny")
    async def deny(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        await active.approvals.deny(
            approval_id=request.approval_id,
            actor="local-user",
            request_id=x_request_id or str(uuid.uuid4()),
        )
        return {"status": "denied", "approval_id": request.approval_id}

    @app.get("/approvals")
    async def approvals(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "approvals": [record.model_dump() for record in await active.approvals.list_pending()]
        }

    @app.get("/memory/search")
    async def memory_search(q: str, active: ApiContainer = Depends(authorized)) -> object:
        results = await active.memory.search_memories(q)
        return {"results": [result.model_dump() for result in results]}

    @app.delete("/memory/{memory_id}")
    async def memory_delete(memory_id: str, active: ApiContainer = Depends(authorized)) -> object:
        return {"deleted": await active.memory.delete_memory(memory_id)}

    @app.get("/memory/export")
    async def memory_export(active: ApiContainer = Depends(authorized)) -> object:
        return {"export": await active.memory.export_memories()}

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
        result = await active.tool_registry.execute(
            "repo_indexer",
            {"repo_path": project.path, "project_id": project_id},
        )
        return {"result": result}

    @app.get("/runtime/models")
    async def runtime_models(active: ApiContainer = Depends(authorized)) -> object:
        return await active.runtime_client.models()

    @app.post("/runtime/models/load")
    async def runtime_model_load(
        request: LoadModelRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        async with httpx.AsyncClient(
            timeout=active.settings.runtime.request_timeout_seconds
        ) as client:
            response = await client.post(
                f"{active.settings.runtime.url}/runtime/models/load",
                json=request.model_dump(),
            )
        return response.json()

    @app.post("/runtime/models/unload")
    async def runtime_model_unload(
        request: LoadModelRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        async with httpx.AsyncClient(
            timeout=active.settings.runtime.request_timeout_seconds
        ) as client:
            response = await client.post(
                f"{active.settings.runtime.url}/runtime/models/unload",
                json=request.model_dump(),
            )
        return response.json()

    return app


async def _execute_approved_tool(
    active: ApiContainer, request: ToolApprovalAction, *, request_id: str
) -> object:
    if request.tool is None:
        record = await active.approvals.get(request.approval_id)
        tool = record.tool
        args = record.args
    else:
        tool = request.tool
        args = request.args
    record = await active.approvals.approve_exact(
        approval_id=request.approval_id,
        tool=tool,
        args=args,
        actor="local-user",
        request_id=request_id,
    )
    permission = active.permission_engine.evaluate(
        tool=record.tool,
        args=record.args,
        agent=record.agent,
        model_permission_level=record.permission_level,
        model_risk_level=record.risk_level,
    )
    if permission.permission_level >= 3:
        active.approvals.audit.write(
            {
                "actor": "local-user",
                "request_id": request_id,
                "event_type": "approved_tool_execution_started",
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": permission.permission_level,
                "risk": permission.risk_level,
                "metadata": record.metadata,
                "approval_id": record.id,
                "outcome": "started",
            }
        )
    precondition_failure = (
        None if record.tool == "patch_applier" else await verify_approval_artifact(record)
    )
    if precondition_failure is not None:
        await active.memory.record_tool_call(
            tool=record.tool,
            args=record.args,
            status="failed",
            permission_level=permission.permission_level,
            risk_level=permission.risk_level,
            result=precondition_failure.model_dump(),
        )
        await active.approvals.consume(
            approval_id=request.approval_id,
            result=precondition_failure.model_dump(),
            actor="local-user",
            request_id=request_id,
        )
        active.approvals.audit.write(
            {
                "actor": "local-user",
                "request_id": request_id,
                "event_type": "approved_tool_rejected",
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": permission.permission_level,
                "risk": permission.risk_level,
                "metadata": record.metadata,
                "approval_id": record.id,
                "outcome": "failed",
                "result": precondition_failure.model_dump(),
            }
        )
        return {"status": "failed", "result": precondition_failure}
    if record.tool == "patch_applier":
        active.approvals.audit.write(
            {
                "actor": "local-user",
                "request_id": request_id,
                "event_type": "approved_patch_verified",
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": permission.permission_level,
                "risk": permission.risk_level,
                "metadata": record.metadata,
                "approval_id": record.id,
                "outcome": "verified",
            }
        )
    try:
        if record.tool == "patch_applier":
            tool_result = await apply_approved_patch(record)
        else:
            tool_result = await active.tool_registry.execute(record.tool, record.args)
    except Exception as exc:
        tool_result = ToolResult(
            ok=False,
            stderr=str(exc),
            risk_level=permission.risk_level,
            permission_level=permission.permission_level,
        )
    await active.memory.record_tool_call(
        tool=record.tool,
        args=record.args,
        status="ok" if tool_result.ok else "failed",
        permission_level=permission.permission_level,
        risk_level=permission.risk_level,
        result=tool_result.model_dump(),
    )
    await active.approvals.consume(
        approval_id=request.approval_id,
        result=tool_result.model_dump(),
        actor="local-user",
        request_id=request_id,
    )
    active.approvals.audit.write(
        {
            "actor": "local-user",
            "request_id": request_id,
            "event_type": "approved_tool_executed",
            "tool": record.tool,
            "arguments": record.args,
            "agent": record.agent,
            "permission_level": permission.permission_level,
            "risk": permission.risk_level,
            "metadata": record.metadata,
            "approval_id": record.id,
            "outcome": "ok" if tool_result.ok else "failed",
        }
    )
    return {"status": "executed" if tool_result.ok else "failed", "result": tool_result}


async def _approval_metadata(
    tool: str, args: dict[str, Any], expected_side_effects: list[str]
) -> dict[str, Any]:
    if tool == "patch_applier":
        return await build_patch_approval_metadata(
            repo_path=str(args["repo_path"]),
            patch_path=str(args["patch_path"]),
            expected_side_effects=expected_side_effects,
            project_id=str(args["project_id"]) if args.get("project_id") is not None else None,
        )
    if tool == "git_commit":
        return await build_git_commit_metadata(
            repo_path=str(args["repo_path"]),
            message=str(args.get("message")) if args.get("message") is not None else None,
            project_id=str(args["project_id"]) if args.get("project_id") is not None else None,
        )
    return {}


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
