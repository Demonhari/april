from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException

from services.api.dependencies import ApiContainer
from services.api.schemas import (
    ChatRequest,
    ChatResponse,
    SessionAttachRequest,
    WakeMuteRequest,
    WakeRequest,
)
from services.wake.schemas import WakeEvent
from services.wake.sentinel import MuteSwitch
from services.wake.status import read_wake_status


class WakeHandler(Protocol):
    async def __call__(
        self,
        active: ApiContainer,
        event: WakeEvent,
        *,
        request_id: str,
    ) -> object: ...


def register_voice_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
    *,
    wake_handler: WakeHandler,
) -> None:
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
        event = WakeEvent(
            source=request.source,
            score=request.score,
            text=request.text,
            reason=request.reason,
            captured_at=request.captured_at,
            session_hint=request.session_hint,
        )
        return await wake_handler(
            active,
            event,
            request_id=x_request_id or str(uuid.uuid4()),
        )

    @app.get("/wake/mute")
    async def wake_mute_status(active: ApiContainer = Depends(authorized)) -> object:
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
        switch.mute() if request.muted else switch.unmute()
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
        event = WakeEvent(source=request.source, reason="session_attach")
        return await wake_handler(
            active,
            event,
            request_id=x_request_id or str(uuid.uuid4()),
        )

    @app.post("/sessions/{session_id}/close")
    async def session_close(
        session_id: str,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        session_manager = active.require_session_manager()
        if await active.memory.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        closed = await session_manager.close(session_id)
        active.approvals.audit.write(
            {
                "event_type": "session_closed",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "reference_id": session_id,
                "outcome": "closed" if closed else "already_closed",
            }
        )
        return {"session_id": session_id, "closed": closed}
