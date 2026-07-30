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


class ConversationRepository(SqliteRepositoryBase):
    async def create_conversation(
        self,
        title: str | None = None,
        *,
        project_id: str | None = None,
        actor: str = "local-user",
    ) -> str:
        conversation_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO conversations(id, title, project_id, actor, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, title, project_id, actor, created_at, created_at),
        )
        return conversation_id

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = await self.database.fetchone(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        if row is None:
            return None
        return Conversation.model_validate(dict(row))

    async def ensure_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        *,
        project_id: str | None = None,
        actor: str = "local-user",
    ) -> str:
        existing = await self.get_conversation(conversation_id)
        if existing is not None:
            if existing.project_id != project_id:
                raise PermissionDeniedError(
                    "Conversation project scope cannot change.",
                    {
                        "conversation_id": conversation_id,
                        "existing_project_id": existing.project_id,
                        "requested_project_id": project_id,
                    },
                )
            return conversation_id
        now = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO conversations(id, title, project_id, actor, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, title, project_id, actor, now, now),
        )
        return conversation_id

    async def add_message(self, conversation_id: str, role: str, content: str) -> str:
        message_id = str(uuid.uuid4())
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, now),
            )
            await conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return message_id

    async def recent_messages(self, conversation_id: str, *, limit: int = 8) -> list[Message]:
        rows = await self.database.fetchall(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        messages = [Message.model_validate(dict(row)) for row in rows]
        return list(reversed(messages))

    async def list_messages_paginated(
        self,
        conversation_id: str,
        *,
        after_created_at: str | None = None,
        after_message_id: str | None = None,
        limit: int = 200,
    ) -> list[Message]:
        """Return a deterministic page ordered by the checkpoint pair."""

        if (after_created_at is None) != (after_message_id is None):
            raise ValueError("both checkpoint fields must be supplied together")
        if limit < 1:
            return []
        if after_created_at is None:
            rows = await self.database.fetchall(
                """
                SELECT *
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT *
                FROM messages
                WHERE conversation_id = ?
                  AND (created_at > ? OR (created_at = ? AND id > ?))
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (
                    conversation_id,
                    after_created_at,
                    after_created_at,
                    after_message_id,
                    limit,
                ),
            )
        return [Message.model_validate(dict(row)) for row in rows]

    async def messages_after_summary_checkpoint(
        self,
        conversation_id: str,
        *,
        limit: int = 200,
    ) -> list[Message]:
        summary = await self.get_conversation_summary(conversation_id)
        return await self.list_messages_paginated(
            conversation_id,
            after_created_at=(summary.through_created_at if summary else None),
            after_message_id=(summary.through_message_id if summary else None),
            limit=limit,
        )

    async def get_conversation_summary(self, conversation_id: str) -> ConversationSummary | None:
        row = await self.database.fetchone(
            "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        )
        return self._conversation_summary_from_row(row) if row is not None else None

    async def upsert_conversation_summary(
        self,
        *,
        conversation_id: str,
        content: ConversationSummaryContent,
        through_message_id: str,
        through_created_at: str,
        summarized_message_count: int,
        source_hash: str,
        model_id: str | None,
        expected_version: int | None,
        expected_through_message_id: str | None = None,
        expected_source_hash: str | None = None,
    ) -> ConversationSummary | None:
        """CAS a generated summary without holding a transaction during generation."""

        summary_json = json.dumps(
            content.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            cursor = await conn.execute(
                "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
            existing = await cursor.fetchone()
            if existing is None:
                if expected_version is not None:
                    return None
                version = 1
                await conn.execute(
                    """
                    INSERT INTO conversation_summaries(
                        conversation_id, summary_json, through_message_id,
                        through_created_at, summarized_message_count, source_hash,
                        model_id, version, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        summary_json,
                        through_message_id,
                        through_created_at,
                        summarized_message_count,
                        source_hash,
                        model_id,
                        version,
                        now,
                        now,
                    ),
                )
            else:
                current = self._conversation_summary_from_row(existing)
                if expected_version != current.version:
                    return None
                if (
                    expected_through_message_id is not None
                    and expected_through_message_id != current.through_message_id
                ):
                    return None
                if expected_source_hash is not None and expected_source_hash != current.source_hash:
                    return None
                new_checkpoint = (through_created_at, through_message_id)
                old_checkpoint = (current.through_created_at, current.through_message_id)
                if new_checkpoint <= old_checkpoint:
                    return None
                version = current.version + 1
                await conn.execute(
                    """
                    UPDATE conversation_summaries
                    SET summary_json = ?, through_message_id = ?,
                        through_created_at = ?, summarized_message_count = ?,
                        source_hash = ?, model_id = ?, version = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    (
                        summary_json,
                        through_message_id,
                        through_created_at,
                        summarized_message_count,
                        source_hash,
                        model_id,
                        version,
                        now,
                        conversation_id,
                    ),
                )
        return await self.get_conversation_summary(conversation_id)

    async def delete_conversation_summary(self, conversation_id: str) -> bool:
        cursor = await self.database.execute(
            "DELETE FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        )
        return cursor.rowcount > 0

    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self.database.transaction() as conn:
            await conn.execute(
                "DELETE FROM suspended_agent_runs WHERE conversation_id = ?",
                (conversation_id,),
            )
            cursor = await conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0

    def _conversation_summary_from_row(self, row: Any) -> ConversationSummary:
        return ConversationSummary(
            conversation_id=str(row["conversation_id"]),
            content=ConversationSummaryContent.model_validate_json(row["summary_json"]),
            through_message_id=str(row["through_message_id"]),
            through_created_at=str(row["through_created_at"]),
            summarized_message_count=int(row["summarized_message_count"]),
            source_hash=str(row["source_hash"]),
            model_id=row["model_id"],
            version=int(row["version"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    async def record_conversation_event(
        self,
        *,
        conversation_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> str:
        event_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO conversation_events(
                id, conversation_id, event_type, payload_json, created_at
            )
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                event_id,
                conversation_id,
                event_type,
                json.dumps(payload, sort_keys=True),
                utc_now_iso(),
            ),
        )
        return event_id
