from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from april_common.errors import AprilError, PermissionDeniedError, error_payload
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
from services.permissions.schemas import ApprovalRequest
from services.voice.health import voice_health


def create_app(container: ApiContainer | None = None) -> FastAPI:
    app = FastAPI(title="APRIL Core API", version="0.1.0")
    app.state.container = container

    @app.exception_handler(AprilError)
    async def april_error_handler(request: Request, exc: AprilError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return JSONResponse(error_payload(exc, request_id), status_code=exc.status_code)

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

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if app.state.container is not None:
            await app.state.container.database.close()

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
            result = await active.orchestrator.chat(
                request.message,
                conversation_id=request.conversation_id,
                request_id=request_id,
            )
            for token in result.final_message.split():
                data = json.dumps({"request_id": request_id, "text": token + " "})
                yield f"event: token\ndata: {data}\n\n"
            yield f"event: done\ndata: {json.dumps({'request_id': request_id})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/voice/input")
    async def voice_input(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
    ) -> ChatResponse:
        request_id = str(uuid.uuid4())
        result = await active.orchestrator.chat(request.message, request_id=request_id)
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/agents/run")
    async def agents_run(
        request: ChatRequest,
        active: ApiContainer = Depends(authorized),
    ) -> ChatResponse:
        request_id = str(uuid.uuid4())
        result = await active.orchestrator.chat(request.message, request_id=request_id)
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
            approval = await active.approvals.create(
                ApprovalRequest(
                    tool=request.tool,
                    args=request.args,
                    permission_level=decision.permission_level,
                    risk_level=decision.risk_level,
                    affected_paths=decision.affected_paths,
                    expected_side_effects=["Execute requested tool once."],
                ),
                actor="local-user",
                request_id=request_id,
            )
            return {"status": "pending_approval", "approval": approval}
        result = await active.tool_registry.execute(request.tool, request.args)
        return {"status": "executed", "result": result}

    @app.post("/tools/approve")
    async def approve(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        if request.tool is None:
            pending = await active.approvals.list_pending()
            match = next((record for record in pending if record.id == request.approval_id), None)
            if match is None:
                raise PermissionDeniedError("Pending approval was not found.")
            tool = match.tool
            args = match.args
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
        result = await active.tool_registry.execute(record.tool, record.args)
        await active.approvals.consume(
            approval_id=request.approval_id,
            result=result.model_dump(),
            actor="local-user",
            request_id=request_id,
        )
        return {"status": "executed", "result": result}

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
        return await active.memory.add_project(request.path, name=request.name)

    @app.post("/projects/{project_id}/index")
    async def project_index(project_id: str, active: ApiContainer = Depends(authorized)) -> object:
        projects = await active.memory.list_projects()
        project = next((item for item in projects if item.id == project_id), None)
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
        # Direct HTTP proxy keeps model paths hidden behind registered IDs.
        import httpx

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
        import httpx

        async with httpx.AsyncClient(
            timeout=active.settings.runtime.request_timeout_seconds
        ) as client:
            response = await client.post(
                f"{active.settings.runtime.url}/runtime/models/unload",
                json=request.model_dump(),
            )
        return response.json()

    return app


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
