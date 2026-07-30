"""Backward-compatible facade for the focused SQLite memory repositories."""

from __future__ import annotations

# Re-export the historical module vocabulary for integrations importing it.
# ruff: noqa: F401
from services.memory.database import Database
from services.memory.encryption import SensitiveMemoryEncryption
from services.memory.sqlite_agent_runs import AgentRunRepository
from services.memory.sqlite_conversations import ConversationRepository
from services.memory.sqlite_memories import (
    MemoryRecordRepository,
    _escaped_like_value,
    _fts_query,
)
from services.memory.sqlite_playbooks import PlaybookRepository
from services.memory.sqlite_projects import ProjectRepository
from services.memory.sqlite_schedule_metadata import ScheduleMetadataRepository
from services.memory.sqlite_sessions_feedback import SessionFeedbackRepository


class SqliteMemory(
    ProjectRepository,
    MemoryRecordRepository,
    ConversationRepository,
    ScheduleMetadataRepository,
    PlaybookRepository,
    AgentRunRepository,
    SessionFeedbackRepository,
):
    """Stable facade; all writes still delegate to the shared ``Database``."""

    def __init__(
        self,
        database: Database,
        *,
        sensitive_encryption: SensitiveMemoryEncryption | None = None,
        sensitive_encryption_enabled: bool = False,
    ) -> None:
        self.database = database
        self.sensitive_encryption = sensitive_encryption
        self.sensitive_encryption_enabled = sensitive_encryption_enabled
