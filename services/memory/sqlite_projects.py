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


class ProjectRepository(SqliteRepositoryBase):
    async def add_project(self, path: str, name: str | None = None) -> Project:
        existing = await self.get_project_by_path(path)
        if existing is not None:
            return existing
        project_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        project_name = name or path.rstrip("/").split("/")[-1] or path
        await self.database.execute(
            "INSERT INTO projects(id, path, name, created_at) VALUES(?, ?, ?, ?)",
            (project_id, path, project_name, created_at),
        )
        return Project(id=project_id, path=path, name=project_name, created_at=created_at)

    async def get_project(self, project_id: str) -> Project | None:
        row = await self.database.fetchone("SELECT * FROM projects WHERE id = ?", (project_id,))
        if row is None:
            return None
        return Project.model_validate(dict(row))

    async def get_project_by_path(self, path: str) -> Project | None:
        row = await self.database.fetchone("SELECT * FROM projects WHERE path = ?", (path,))
        if row is None:
            return None
        return Project.model_validate(dict(row))

    async def list_projects(self) -> list[Project]:
        rows = await self.database.fetchall("SELECT * FROM projects ORDER BY created_at DESC")
        return [Project.model_validate(dict(row)) for row in rows]
