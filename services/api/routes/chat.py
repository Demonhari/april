from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import Depends, FastAPI, Header
from fastapi.responses import StreamingResponse

from services.api.dependencies import ApiContainer
from services.api.schemas import ChatRequest, ChatResponse


def register_chat_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
    *,
    sse_event: Callable[[str, str, dict[str, Any]], str],
) -> None:
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
                    yield sse_event(event_name, request_id, payload)
            finally:
                await interaction.__aexit__(None, None, None)

        return StreamingResponse(events(), media_type="text/event-stream")
