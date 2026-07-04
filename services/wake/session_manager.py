from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

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
    ) -> None:
        self.memory = memory
        self.continuity_minutes = continuity_minutes
        self.clock = clock or utc_now
        self.on_close = on_close

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

    async def close(self, session_id: str) -> bool:
        now_iso = self.clock().isoformat().replace("+00:00", "Z")
        closed = await self.memory.close_session(session_id, at=now_iso)
        if closed and self.on_close is not None:
            await self.on_close(session_id)
        return closed

    async def close_idle_sessions(self) -> list[str]:
        """Close every open session idle past the continuity window.

        This is the idle-reflection rule: a session that would no longer be
        joined by a new wake (its continuity window elapsed) is closed, which
        triggers the same ``on_close`` reflection path as an explicit close.
        """
        now = self.clock()
        closed: list[str] = []
        for session in await self.memory.list_open_sessions():
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
