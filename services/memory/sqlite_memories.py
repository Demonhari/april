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


def _fts_query(tokens: list[str]) -> str:
    """Build a bounded MATCH expression from tokenizer-produced literals only."""
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens[:16])


def _escaped_like_value(value: str) -> str:
    bounded = value[:512].strip()
    if not bounded:
        return ""
    escaped = bounded.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class MemoryRecordRepository(SqliteRepositoryBase):
    async def create_memory(
        self,
        content: str,
        *,
        kind: str = "fact",
        reason: str,
        project_id: str | None = None,
        confidence: float = 0.7,
        source: str = "user",
        expires_at: str | None = None,
        superseded_by: str | None = None,
        sensitive: bool = False,
    ) -> MemoryRecord:
        memory_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        if sensitive and self.sensitive_encryption is None:
            raise PermissionDeniedError(
                "Sensitive-memory encryption is disabled or its key is unavailable."
            )
        stored_content = (
            self.sensitive_encryption.encrypt(memory_id, content)
            if sensitive and self.sensitive_encryption is not None
            else content
        )
        stored_reason = "Explicitly encrypted local memory." if sensitive else reason
        indexed_content = "" if sensitive else content
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO memories(
                    id, project_id, kind, content, reason, created_at,
                    confidence, source, expires_at, superseded_by, content_encrypted
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    project_id,
                    kind,
                    stored_content,
                    stored_reason,
                    created_at,
                    confidence,
                    source,
                    expires_at,
                    superseded_by,
                    int(sensitive),
                ),
            )
            await conn.execute(
                "INSERT INTO memories_fts(id, content, reason) VALUES(?, ?, ?)",
                (memory_id, indexed_content, stored_reason),
            )
        return MemoryRecord(
            id=memory_id,
            content=content,
            kind=kind,
            project_id=project_id,
            reason=stored_reason,
            created_at=created_at,
            confidence=confidence,
            source=source,
            expires_at=expires_at,
            superseded_by=superseded_by,
            content_encrypted=sensitive,
        )

    async def get_memory(
        self, memory_id: str, *, include_inactive: bool = False
    ) -> MemoryRecord | None:
        if include_inactive:
            row = await self.database.fetchone("SELECT * FROM memories WHERE id = ?", (memory_id,))
        else:
            row = await self.database.fetchone(
                """
                SELECT * FROM memories
                WHERE id = ?
                  AND superseded_by IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (memory_id, utc_now_iso()),
            )
        if row is None:
            return None
        return self._memory_record(row)

    async def find_duplicate_memory(
        self,
        content: str,
        *,
        kind: str,
        project_id: str | None = None,
    ) -> MemoryRecord | None:
        normalized = " ".join(content.casefold().split())
        rows = await self.database.fetchall(
            """
            SELECT * FROM memories
            WHERE kind = ? AND (project_id IS ? OR project_id = ?)
            ORDER BY created_at DESC
            """,
            (kind, project_id, project_id),
        )
        for row in rows:
            record = self._memory_record(row)
            if " ".join(record.content.casefold().split()) == normalized:
                return record
        return None

    async def list_memories(
        self, *, project_id: str | None = None, include_inactive: bool = False
    ) -> list[MemoryRecord]:
        active_clause = (
            ""
            if include_inactive
            else "AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
        )
        params: tuple[object, ...]
        if project_id is None:
            params = (utc_now_iso(),) if not include_inactive else ()
            rows = await self.database.fetchall(
                f"""
                SELECT * FROM memories
                WHERE 1 = 1 {active_clause}
                ORDER BY created_at DESC
                """,
                params,
            )
        else:
            params = (project_id, utc_now_iso()) if not include_inactive else (project_id,)
            rows = await self.database.fetchall(
                f"""
                SELECT * FROM memories
                WHERE project_id = ? {active_clause}
                ORDER BY created_at DESC
                """,
                params,
            )
        return [self._memory_record(row) for row in rows]

    async def search_memories(
        self, query: str, *, project_id: str | None = None
    ) -> list[MemoryRecord]:
        return [
            hit.memory
            for hit in await self.search_memory_lexical_hits(query, project_id=project_id)
        ]

    async def search_memory_lexical_hits(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[LexicalHit]:
        """Return bounded Unicode-safe FTS hits with deterministic rank metadata."""
        capped_limit = max(1, min(limit, 100))
        if query.strip() in {"", "*"}:
            memories = await self.list_memories(project_id=project_id)
            return [
                LexicalHit(
                    memory=memory,
                    lexical_rank=rank,
                    normalized_score=1.0 / rank,
                )
                for rank, memory in enumerate(memories[:capped_limit], start=1)
            ]
        tokens = word_tokens(query, max_tokens=16)
        fts_query = _fts_query(tokens)
        now = utc_now_iso()
        rows: list[Any] = []
        if fts_query:
            if project_id is None:
                rows = await self.database.fetchall(
                    """
                    SELECT m.*, bm25(memories_fts) AS lexical_bm25
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.id
                    WHERE memories_fts MATCH ?
                      AND m.superseded_by IS NULL
                      AND (m.expires_at IS NULL OR m.expires_at > ?)
                    ORDER BY lexical_bm25 ASC, m.id ASC
                    LIMIT ?
                    """,
                    (fts_query, now, capped_limit),
                )
            else:
                rows = await self.database.fetchall(
                    """
                    SELECT m.*, bm25(memories_fts) AS lexical_bm25
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.id
                    WHERE memories_fts MATCH ?
                      AND (m.project_id = ? OR m.project_id IS NULL)
                      AND m.superseded_by IS NULL
                      AND (m.expires_at IS NULL OR m.expires_at > ?)
                    ORDER BY lexical_bm25 ASC, m.id ASC
                    LIMIT ?
                    """,
                    (fts_query, project_id, now, capped_limit),
                )
        if not rows:
            like_value = _escaped_like_value(normalize_text(query))
            if not like_value:
                return []
            if project_id is None:
                rows = await self.database.fetchall(
                    """
                    SELECT m.*
                    FROM memories m
                    WHERE m.superseded_by IS NULL
                      AND (m.expires_at IS NULL OR m.expires_at > ?)
                      AND (
                        lower(m.content) LIKE ? ESCAPE '\\'
                        OR lower(m.reason) LIKE ? ESCAPE '\\'
                      )
                    ORDER BY m.created_at DESC, m.id ASC
                    LIMIT ?
                    """,
                    (now, like_value, like_value, capped_limit),
                )
            else:
                rows = await self.database.fetchall(
                    """
                    SELECT m.*
                    FROM memories m
                    WHERE (m.project_id = ? OR m.project_id IS NULL)
                      AND m.superseded_by IS NULL
                      AND (m.expires_at IS NULL OR m.expires_at > ?)
                      AND (
                        lower(m.content) LIKE ? ESCAPE '\\'
                        OR lower(m.reason) LIKE ? ESCAPE '\\'
                      )
                    ORDER BY m.created_at DESC, m.id ASC
                    LIMIT ?
                    """,
                    (project_id, now, like_value, like_value, capped_limit),
                )

        hits: list[LexicalHit] = []
        for rank, row in enumerate(rows, start=1):
            memory = self._memory_record(row)
            document_tokens = set(word_tokens(f"{memory.content} {memory.reason}", max_tokens=512))
            matched = tuple(token for token in tokens if token in document_tokens)
            hits.append(
                LexicalHit(
                    memory=memory,
                    lexical_rank=rank,
                    normalized_score=1.0 / rank,
                    matched_tokens=tuple(dict.fromkeys(matched)),
                )
            )
        return hits

    async def mark_memories_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            for memory_id in dict.fromkeys(memory_ids):
                await conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1, last_used_at = ?
                    WHERE id = ?
                    """,
                    (now, memory_id),
                )

    async def refresh_memory(
        self, memory_id: str, *, confidence: float | None = None
    ) -> MemoryRecord | None:
        """Auditable duplicate-merge refresh: bump usage and keep max confidence."""
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            if confidence is None:
                await conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1, last_used_at = ?
                    WHERE id = ?
                    """,
                    (now, memory_id),
                )
            else:
                await conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1,
                        last_used_at = ?,
                        confidence = MAX(confidence, ?)
                    WHERE id = ?
                    """,
                    (now, confidence, memory_id),
                )
        return await self.get_memory(memory_id, include_inactive=True)

    async def set_memory_decay(
        self, memory_id: str, *, confidence: float, expires_at: str | None
    ) -> bool:
        """Deterministic decay update: lower confidence, optionally start fading."""
        cursor = await self.database.execute(
            "UPDATE memories SET confidence = ?, expires_at = ? WHERE id = ?",
            (confidence, expires_at, memory_id),
        )
        return cursor.rowcount > 0

    async def list_memories_by_state(self, state: str, *, limit: int = 100) -> list[MemoryRecord]:
        """Inspect memory lifecycle states without hiding or deleting anything.

        States: ``machine`` (machine-written, still active), ``superseded``,
        ``expired`` (expires_at passed), ``fading`` (expires_at set but not yet
        reached), and ``active`` (what retrieval serves).
        """
        now = utc_now_iso()
        capped = max(1, min(limit, 500))
        clauses = {
            "machine": (
                "source != 'user' AND superseded_by IS NULL "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (now, capped),
            ),
            "superseded": ("superseded_by IS NOT NULL", (capped,)),
            "expired": ("expires_at IS NOT NULL AND expires_at <= ?", (now, capped)),
            "fading": (
                "expires_at IS NOT NULL AND expires_at > ? AND superseded_by IS NULL",
                (now, capped),
            ),
            "active": (
                "superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)",
                (now, capped),
            ),
        }
        if state not in clauses:
            raise ValueError("state must be one of machine, superseded, expired, fading, active")
        where, params = clauses[state]
        rows = await self.database.fetchall(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [self._memory_record(row) for row in rows]

    def _memory_record(self, row: Any) -> MemoryRecord:
        payload = dict(row)
        encrypted = bool(payload.get("content_encrypted", False))
        if encrypted:
            if self.sensitive_encryption is None:
                payload["content"] = UNAVAILABLE_CONTENT
            else:
                payload["content"] = self.sensitive_encryption.decrypt(
                    str(payload["id"]),
                    str(payload["content"]),
                )
        payload["content_encrypted"] = encrypted
        return MemoryRecord.model_validate(payload)

    async def supersede_memory(self, memory_id: str, *, superseded_by: str) -> bool:
        """Mark a memory as superseded without deleting the row."""
        cursor = await self.database.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL",
            (superseded_by, memory_id),
        )
        return cursor.rowcount > 0

    async def record_memory_contradiction(
        self, *, memory_id_a: str, memory_id_b: str
    ) -> MemoryContradictionRecord:
        """Flag a contradictory pair for Dreamer adjudication; both rows are kept."""
        pair_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO memory_contradictions(
                id, memory_id_a, memory_id_b, status, created_at
            )
            VALUES(?, ?, ?, 'pending', ?)
            """,
            (pair_id, memory_id_a, memory_id_b, created_at),
        )
        return MemoryContradictionRecord(
            id=pair_id,
            memory_id_a=memory_id_a,
            memory_id_b=memory_id_b,
            status="pending",
            created_at=created_at,
        )

    async def list_memory_contradictions(
        self, *, status: str | None = "pending", limit: int = 100
    ) -> list[MemoryContradictionRecord]:
        capped = max(1, min(limit, 500))
        if status is None:
            rows = await self.database.fetchall(
                "SELECT * FROM memory_contradictions ORDER BY created_at LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM memory_contradictions
                WHERE status = ?
                ORDER BY created_at
                LIMIT ?
                """,
                (status, capped),
            )
        return [MemoryContradictionRecord.model_validate(dict(row)) for row in rows]

    async def resolve_memory_contradiction(self, contradiction_id: str, *, resolution: str) -> bool:
        cursor = await self.database.execute(
            """
            UPDATE memory_contradictions
            SET status = 'resolved', resolution = ?, resolved_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (resolution, utc_now_iso(), contradiction_id),
        )
        return cursor.rowcount > 0

    async def count_machine_memories_since(
        self, since_iso: str, *, source: str | Sequence[str]
    ) -> int:
        sources = (source,) if isinstance(source, str) else tuple(source)
        placeholders = ", ".join("?" for _ in sources)
        row = await self.database.fetchone(
            "SELECT COUNT(*) AS count FROM memories "
            f"WHERE source IN ({placeholders}) AND created_at >= ?",
            (*sources, since_iso),
        )
        return int(row["count"]) if row is not None else 0

    async def delete_memory(self, memory_id: str) -> bool:
        async with self.database.transaction() as conn:
            await conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            cursor = await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    async def export_memories(self, *, project_id: str | None = None) -> str:
        memories = [
            memory.model_dump() for memory in await self.list_memories(project_id=project_id)
        ]
        return json.dumps({"memories": memories}, indent=2)
