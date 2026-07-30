from __future__ import annotations

# Mechanical extraction keeps the original type vocabulary available.
# ruff: noqa: F401
# mypy: disable-error-code="attr-defined"
import json
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from april_common.errors import PermissionDeniedError
from april_common.text_normalization import normalize_text, word_tokens
from april_common.time import utc_now_iso
from services.brain.planner import TaskPlan, TaskStep
from services.memory.encryption import UNAVAILABLE_CONTENT, SensitiveMemoryEncryption
from services.memory.schemas import (
    Conversation,
    ConversationSummary,
    ConversationSummaryContent,
    FeedbackEventRecord,
    LexicalHit,
    MemoryContradictionRecord,
    MemoryRecord,
    Message,
    Project,
    ReminderRecord,
    SessionRecord,
    SuspendedAgentRun,
    WakeEventRecord,
)
from services.memory.sqlite_base import SqliteRepositoryBase


class SessionFeedbackRepository(SqliteRepositoryBase):
    async def create_session(
        self,
        *,
        source: str,
        conversation_id: str | None,
        started_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        session_id = str(uuid.uuid4())
        now = started_at or utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO sessions(
                id, conversation_id, source, started_at, last_activity_at, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (session_id, conversation_id, source, now, now, json.dumps(metadata or {})),
        )
        return SessionRecord(
            id=session_id,
            conversation_id=conversation_id,
            source=source,
            started_at=now,
            last_activity_at=now,
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        row = await self.database.fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            return None
        return self._session_from_row(row)

    async def latest_open_session(self) -> SessionRecord | None:
        row = await self.database.fetchone(
            """
            SELECT * FROM sessions
            WHERE closed_at IS NULL
            ORDER BY last_activity_at DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        return self._session_from_row(row)

    async def open_session_for_conversation(self, conversation_id: str) -> SessionRecord | None:
        """Return the one open session bound to ``conversation_id``.

        Conversation ids are accepted by every legacy chat surface, so this is
        the authoritative bridge from those requests to session activity.  It
        never creates a session for an unknown conversation.
        """
        row = await self.database.fetchone(
            """
            SELECT * FROM sessions
            WHERE conversation_id = ? AND closed_at IS NULL
            ORDER BY last_activity_at DESC
            LIMIT 1
            """,
            (conversation_id,),
        )
        if row is None:
            return None
        return self._session_from_row(row)

    async def list_open_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        rows = await self.database.fetchall(
            """
            SELECT * FROM sessions
            WHERE closed_at IS NULL
            ORDER BY last_activity_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        )
        return [self._session_from_row(row) for row in rows]

    async def touch_session(self, session_id: str, *, at: str | None = None) -> None:
        await self.database.execute(
            "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
            (at or utc_now_iso(), session_id),
        )

    async def close_session(self, session_id: str, *, at: str | None = None) -> bool:
        cursor = await self.database.execute(
            "UPDATE sessions SET closed_at = ? WHERE id = ? AND closed_at IS NULL",
            (at or utc_now_iso(), session_id),
        )
        return cursor.rowcount > 0

    async def list_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        rows = await self.database.fetchall(
            "SELECT * FROM sessions ORDER BY last_activity_at DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        )
        return [self._session_from_row(row) for row in rows]

    async def record_wake_event(
        self,
        *,
        session_id: str | None,
        source: str,
        score: float | None = None,
        accepted: bool = True,
        reason: str | None = None,
        transcript_present: bool = False,
        captured_at: str | None = None,
        session_hint: str | None = None,
    ) -> WakeEventRecord:
        event_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO wake_events(
                id, session_id, source, score, accepted, reason,
                transcript_present, captured_at, session_hint, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                source,
                score,
                1 if accepted else 0,
                reason,
                1 if transcript_present else 0,
                captured_at,
                session_hint,
                created_at,
            ),
        )
        return WakeEventRecord(
            id=event_id,
            session_id=session_id,
            source=source,
            score=score,
            accepted=accepted,
            reason=reason,
            transcript_present=transcript_present,
            captured_at=captured_at,
            session_hint=session_hint,
            created_at=created_at,
        )

    async def list_wake_events(
        self, *, session_id: str | None = None, limit: int = 100
    ) -> list[WakeEventRecord]:
        capped = max(1, min(limit, 500))
        if session_id is None:
            rows = await self.database.fetchall(
                "SELECT * FROM wake_events ORDER BY created_at DESC LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM wake_events
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, capped),
            )
        return [WakeEventRecord.model_validate(dict(row)) for row in rows]

    async def record_feedback_event(
        self,
        *,
        rating: Literal["good", "bad"],
        reason: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        agent_run_id: str | None = None,
    ) -> FeedbackEventRecord:
        event_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO feedback_events(
                id, session_id, conversation_id, agent_run_id, rating, reason, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, session_id, conversation_id, agent_run_id, rating, reason, created_at),
        )
        return FeedbackEventRecord(
            id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            agent_run_id=agent_run_id,
            rating=rating,
            reason=reason,
            created_at=created_at,
        )

    async def list_feedback_events(
        self, *, conversation_id: str | None = None, limit: int = 100
    ) -> list[FeedbackEventRecord]:
        capped = max(1, min(limit, 500))
        if conversation_id is None:
            rows = await self.database.fetchall(
                "SELECT * FROM feedback_events ORDER BY created_at DESC LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM feedback_events
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, capped),
            )
        return [FeedbackEventRecord.model_validate(dict(row)) for row in rows]

    async def list_tool_call_summaries(
        self, *, conversation_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Sanitized tool-call summaries for reflection: no args, no results."""
        rows = await self.database.fetchall(
            """
            SELECT tool, status, risk_level, permission_level, created_at
            FROM tool_calls
            WHERE conversation_id = ?
            ORDER BY created_at
            LIMIT ?
            """,
            (conversation_id, max(1, min(limit, 200))),
        )
        return [dict(row) for row in rows]

    async def latest_agent_run_id(self, *, conversation_id: str | None = None) -> str | None:
        if conversation_id is None:
            row = await self.database.fetchone(
                "SELECT id FROM agent_runs ORDER BY created_at DESC LIMIT 1"
            )
        else:
            row = await self.database.fetchone(
                """
                SELECT id FROM agent_runs
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
        if row is None:
            return None
        return str(row["id"])

    def _session_from_row(self, row: Any) -> SessionRecord:
        data = dict(row)
        raw_metadata = data.pop("metadata_json", None) or "{}"
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            metadata = {}
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return SessionRecord.model_validate(data)

    async def record_tool_call(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        status: str,
        permission_level: int,
        risk_level: str,
        result: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO tool_calls(
                id, conversation_id, tool, args_json, result_json, status,
                permission_level, risk_level, created_at, completed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call_id,
                conversation_id,
                tool,
                json.dumps(args, sort_keys=True),
                json.dumps(result or {}, sort_keys=True),
                status,
                permission_level,
                risk_level,
                utc_now_iso(),
                utc_now_iso() if result is not None else None,
            ),
        )
        return call_id
