from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from april_common.audit import AuditLogger
from april_common.time import parse_utc_iso, utc_now
from services.memory.schemas import SessionRecord
from services.memory.sqlite_memory import SqliteMemory
from services.wake.schemas import WakeEvent, WakeResolution


class SessionManager:
    """Converges every wake surface onto one continuity-aware session stream.

    A wake that arrives within ``continuity_minutes`` of the last activity joins
    the existing open session (and therefore its conversation); anything later
    closes over into a new session with a fresh conversation. Every wake is
    persisted to ``wake_events`` (transcript text itself is never stored there).
    """

    def __init__(
        self,
        memory: SqliteMemory,
        *,
        continuity_minutes: float,
        clock: Callable[[], datetime] | None = None,
        on_close: Callable[[str], Awaitable[object]] | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self.memory = memory
        self.continuity_minutes = continuity_minutes
        self.clock = clock or utc_now
        self.on_close = on_close
        self.audit = audit
        self._state_lock = asyncio.Lock()
        self._active_interactions: dict[str, int] = {}

    async def handle_wake(self, event: WakeEvent) -> WakeResolution:
        now = self.clock()
        now_iso = now.isoformat().replace("+00:00", "Z")
        session, joined_existing = await self._resolve_session(event, now, now_iso)
        wake_event = await self.memory.record_wake_event(
            session_id=session.id,
            source=event.source,
            score=event.score,
            accepted=event.accepted,
            reason=event.reason,
            transcript_present=bool(event.text),
            captured_at=event.captured_at,
            session_hint=event.session_hint,
        )
        return WakeResolution(
            session_id=session.id,
            conversation_id=session.conversation_id,
            joined_existing=joined_existing,
            wake_event_id=wake_event.id,
        )

    async def touch(self, session_id: str) -> None:
        """Record activity so follow-up wakes keep joining this session."""
        now_iso = self.clock().isoformat().replace("+00:00", "Z")
        await self.memory.touch_session(session_id, at=now_iso)

    async def touch_conversation(self, conversation_id: str) -> str | None:
        """Touch the open session for a conversation without fabricating one."""
        async with self._state_lock:
            session = await self.memory.open_session_for_conversation(conversation_id)
            if session is None:
                return None
            await self.touch(session.id)
            return session.id

    @asynccontextmanager
    async def interaction(self, conversation_id: str | None) -> AsyncIterator[str | None]:
        """Keep a conversation's session active for one accepted interaction.

        Activity is refreshed on entry and exit (including failures).  The
        in-flight counter is protected by the same lock used by idle closing,
        so a long stream cannot be closed between its initial touch and final
        touch. Unknown conversation ids degrade to no session and create
        nothing.
        """
        session_id: str | None = None
        if conversation_id is not None:
            async with self._state_lock:
                session = await self.memory.open_session_for_conversation(conversation_id)
                if session is not None:
                    session_id = session.id
                    self._active_interactions[session_id] = (
                        self._active_interactions.get(session_id, 0) + 1
                    )
                    await self.touch(session_id)
        try:
            yield session_id
        finally:
            if session_id is not None:
                async with self._state_lock:
                    await self.touch(session_id)
                    remaining = self._active_interactions.get(session_id, 1) - 1
                    if remaining > 0:
                        self._active_interactions[session_id] = remaining
                    else:
                        self._active_interactions.pop(session_id, None)

    async def close(self, session_id: str) -> bool:
        now_iso = self.clock().isoformat().replace("+00:00", "Z")
        closed = await self.memory.close_session(session_id, at=now_iso)
        if closed and self.on_close is not None:
            try:
                await self.on_close(session_id)
            except Exception as exc:
                self._audit_reflection_failure(session_id, exc)
        return closed

    async def close_idle_sessions(self) -> list[str]:
        """Close every open session idle past the continuity window.

        This is the idle-reflection rule: a session that would no longer be
        joined by a new wake (its continuity window elapsed) is closed, which
        triggers the same ``on_close`` reflection path as an explicit close.
        """
        now = self.clock()
        closed: list[str] = []
        async with self._state_lock:
            for session in await self.memory.list_open_sessions():
                if self._active_interactions.get(session.id, 0) > 0:
                    continue
                if self._within_continuity(session, now):
                    continue
                if await self.close(session.id):
                    closed.append(session.id)
        return closed

    async def _resolve_session(
        self, event: WakeEvent, now: datetime, now_iso: str
    ) -> tuple[SessionRecord, bool]:
        # A session hint is advisory: it may only join a session that is still
        # open. Hints naming closed or unknown sessions fall back to the normal
        # continuity flow, so a stale hint can never resurrect old context.
        if event.session_hint:
            hinted = await self.memory.get_session(event.session_hint)
            if hinted is not None and hinted.closed_at is None:
                await self.memory.touch_session(hinted.id, at=now_iso)
                return hinted.model_copy(update={"last_activity_at": now_iso}), True
        latest = await self.memory.latest_open_session()
        if latest is not None and self._within_continuity(latest, now):
            await self.memory.touch_session(latest.id, at=now_iso)
            refreshed = latest.model_copy(update={"last_activity_at": now_iso})
            return refreshed, True
        if latest is not None:
            # The stale session is closed so exactly one session is ever open.
            await self.close(latest.id)
        conversation_id = await self.memory.create_conversation(actor="local-user")
        session = await self.memory.create_session(
            source=event.source,
            conversation_id=conversation_id,
            started_at=now_iso,
        )
        return session, False

    def _within_continuity(self, session: SessionRecord, now: datetime) -> bool:
        try:
            last_activity = parse_utc_iso(session.last_activity_at)
        except ValueError:
            return False
        window = timedelta(minutes=self.continuity_minutes)
        return now - last_activity <= window

    def _audit_reflection_failure(self, session_id: str, exc: Exception) -> None:
        if self.audit is None:
            return
        self.audit.write(
            {
                "event_type": "archive_reflection_failed",
                "actor": "archive_agent",
                "reference_id": session_id,
                "error_type": type(exc).__name__,
                "error_message_length": len(str(exc)),
            }
        )
