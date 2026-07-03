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

    async def _resolve_session(
        self, event: WakeEvent, now: datetime, now_iso: str
    ) -> tuple[SessionRecord, bool]:
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
