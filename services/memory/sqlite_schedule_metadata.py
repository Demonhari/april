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


class ScheduleMetadataRepository(SqliteRepositoryBase):
    async def create_reminder(self, content: str, due_at: str | None = None) -> ReminderRecord:
        reminder_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO reminders(id, content, due_at, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (reminder_id, content, due_at, created_at),
        )
        return ReminderRecord(
            id=reminder_id,
            content=content,
            due_at=due_at,
            created_at=created_at,
        )

    async def list_reminders(self) -> list[ReminderRecord]:
        rows = await self.database.fetchall("SELECT * FROM reminders ORDER BY created_at DESC")
        return [ReminderRecord.model_validate(dict(row)) for row in rows]

    async def list_due_reminders(self, now_iso: str) -> list[ReminderRecord]:
        rows = await self.database.fetchall(
            """
            SELECT *
            FROM reminders
            WHERE due_at IS NOT NULL AND fired_at IS NULL AND due_at <= ?
            ORDER BY due_at ASC
            """,
            (now_iso,),
        )
        return [ReminderRecord.model_validate(dict(row)) for row in rows]

    async def mark_reminder_fired(self, reminder_id: str, fired_at: str) -> bool:
        cursor = await self.database.execute(
            "UPDATE reminders SET fired_at = ? WHERE id = ? AND fired_at IS NULL",
            (fired_at, reminder_id),
        )
        return cursor.rowcount > 0

    async def list_upcoming_reminders(self, now_iso: str, until_iso: str) -> list[ReminderRecord]:
        rows = await self.database.fetchall(
            """
            SELECT *
            FROM reminders
            WHERE due_at IS NOT NULL AND fired_at IS NULL AND due_at <= ?
            ORDER BY due_at ASC
            """,
            (until_iso,),
        )
        return [ReminderRecord.model_validate(dict(row)) for row in rows]

    async def delete_reminder(self, reminder_id: str) -> bool:
        cursor = await self.database.execute(
            "DELETE FROM reminders WHERE id = ?",
            (reminder_id,),
        )
        return cursor.rowcount > 0

    async def get_scheduler_state(self, key: str) -> str | None:
        row = await self.database.fetchone(
            "SELECT value FROM scheduler_state WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        return str(row["value"])

    async def set_scheduler_state(self, key: str, value: str) -> None:
        await self.database.execute(
            """
            INSERT INTO scheduler_state(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, utc_now_iso()),
        )

    async def get_repo_snapshot(self, project_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone(
            "SELECT last_head_sha, last_dirty_count, updated_at "
            "FROM repo_snapshots WHERE project_id = ?",
            (project_id,),
        )
        if row is None:
            return None
        return {
            "head_sha": row["last_head_sha"],
            "dirty_count": int(row["last_dirty_count"]),
            "updated_at": str(row["updated_at"]),
        }

    async def upsert_repo_snapshot(
        self,
        project_id: str,
        head_sha: str | None,
        dirty_count: int,
        updated_at: str,
    ) -> None:
        await self.database.execute(
            """
            INSERT INTO repo_snapshots(project_id, last_head_sha, last_dirty_count, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                last_head_sha = excluded.last_head_sha,
                last_dirty_count = excluded.last_dirty_count,
                updated_at = excluded.updated_at
            """,
            (project_id, head_sha, dirty_count, updated_at),
        )
