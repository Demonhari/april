from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from april_common.audit import AuditLogger
from april_common.time import utc_now_iso
from services.memory.schemas import MemoryRecord, VectorMetadata
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory


@dataclass(frozen=True, slots=True)
class MemoryIndexHealth:
    repair_required: bool
    pending_repairs: int


class MemoryRepository:
    """SQLite-authoritative memory persistence with repairable vector indexing."""

    def __init__(
        self,
        memory: SqliteMemory,
        vector_memory: VectorMemory,
        *,
        audit: AuditLogger | None = None,
    ) -> None:
        self.memory = memory
        self.vector_memory = vector_memory
        self.audit = audit

    async def create_memory(self, content: str, **kwargs: Any) -> MemoryRecord:
        record = await self.memory.create_memory(content, **kwargs)
        await self._index_after_commit(record, operation="upsert")
        return record

    async def refresh_memory(
        self, memory_id: str, *, confidence: float | None = None
    ) -> MemoryRecord | None:
        record = await self.memory.refresh_memory(memory_id, confidence=confidence)
        if record is not None and record.superseded_by is None:
            await self._index_after_commit(record, operation="upsert")
        return record

    async def delete_memory(self, memory_id: str) -> bool:
        deleted = await self.memory.delete_memory(memory_id)
        if deleted:
            await self._delete_after_commit(memory_id, operation="delete")
        return deleted

    async def supersede_memory(self, memory_id: str, *, superseded_by: str) -> bool:
        changed = await self.memory.supersede_memory(memory_id, superseded_by=superseded_by)
        if changed:
            await self._delete_after_commit(memory_id, operation="supersede")
        return changed

    async def set_provenance(
        self,
        memory_id: str,
        *,
        source_session_id: str | None = None,
        source_conversation_id: str | None = None,
        source_message_ids: list[str] | None = None,
    ) -> None:
        await self.memory.database.execute(
            """
            INSERT INTO memory_provenance(
                memory_id, source_session_id, source_conversation_id,
                source_message_ids_json, updated_at
            ) VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                source_session_id = excluded.source_session_id,
                source_conversation_id = excluded.source_conversation_id,
                source_message_ids_json = excluded.source_message_ids_json,
                updated_at = excluded.updated_at
            """,
            (
                memory_id,
                source_session_id,
                source_conversation_id,
                json.dumps(source_message_ids or [], sort_keys=True),
                utc_now_iso(),
            ),
        )

    async def health(self) -> MemoryIndexHealth:
        row = await self.memory.database.fetchone(
            "SELECT COUNT(*) AS count FROM memory_index_repairs"
        )
        count = int(row["count"]) if row is not None else 0
        return MemoryIndexHealth(repair_required=count > 0, pending_repairs=count)

    async def rebuild(self) -> int:
        records = await self.memory.list_memories()
        items = [(record.id, record.content, self._metadata(record)) for record in records]
        try:
            count = self.vector_memory.rebuild_memory_namespace(items)
        except Exception as exc:
            await self._mark_repair("*", "reindex", exc)
            self._audit("memory_index_rebuild_failed", None, exc)
            raise
        await self.memory.database.execute("DELETE FROM memory_index_repairs")
        self._audit("memory_index_rebuilt", None, None, count=count)
        return count

    async def _index_after_commit(self, record: MemoryRecord, *, operation: str) -> None:
        if record.content_encrypted:
            await self._clear_repair(record.id)
            self._audit("sensitive_memory_index_skipped", record.id, None)
            return
        try:
            self.vector_memory.upsert(
                record_id=record.id,
                content=record.content,
                metadata=self._metadata(record),
            )
        except Exception as exc:
            await self._mark_repair(record.id, operation, exc)
            self._audit("memory_index_write_failed", record.id, exc)
            return
        await self._clear_repair(record.id)
        self._audit("memory_index_write_succeeded", record.id, None)

    async def _delete_after_commit(self, memory_id: str, *, operation: str) -> None:
        try:
            self.vector_memory.delete(memory_id)
        except Exception as exc:
            await self._mark_repair(memory_id, operation, exc)
            self._audit("memory_index_delete_failed", memory_id, exc)
            return
        await self._clear_repair(memory_id)
        self._audit("memory_index_delete_succeeded", memory_id, None)

    async def _mark_repair(self, memory_id: str, operation: str, exc: Exception) -> None:
        await self.memory.database.execute(
            """
            INSERT INTO memory_index_repairs(memory_id, operation, error_type, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                operation = excluded.operation,
                error_type = excluded.error_type,
                updated_at = excluded.updated_at
            """,
            (memory_id, operation, type(exc).__name__, utc_now_iso()),
        )

    async def _clear_repair(self, memory_id: str) -> None:
        await self.memory.database.execute(
            "DELETE FROM memory_index_repairs WHERE memory_id = ?", (memory_id,)
        )

    @staticmethod
    def _metadata(record: MemoryRecord) -> VectorMetadata:
        return VectorMetadata(
            source_type="memory",
            source_id=record.id,
            project_id=record.project_id,
            content_hash=hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
            created_at=record.created_at,
        )

    def _audit(
        self,
        event_type: str,
        memory_id: str | None,
        exc: Exception | None,
        *,
        count: int | None = None,
    ) -> None:
        if self.audit is None:
            return
        self.audit.write(
            {
                "event_type": event_type,
                "actor": "memory_repository",
                "memory_id": memory_id,
                "error_type": type(exc).__name__ if exc is not None else None,
                "count": count,
            }
        )
