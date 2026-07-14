from __future__ import annotations

import hashlib

import pytest

from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.repository import MemoryRepository
from services.memory.schemas import VectorMetadata
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory


async def _repository(settings_tmp):
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    vector = VectorMemory(settings_tmp.vector_index_path)
    return database, memory, vector, MemoryRepository(memory, vector)


@pytest.mark.asyncio
async def test_repository_create_delete_and_rebuild_preserves_documents(settings_tmp) -> None:
    database, memory, vector, repository = await _repository(settings_tmp)
    try:
        document = "repository architecture notes"
        vector.upsert(
            record_id="doc-1",
            content=document,
            metadata=VectorMetadata(
                source_type="document",
                source_id="doc-1",
                content_hash=hashlib.sha256(document.encode()).hexdigest(),
                created_at="2026-07-14T00:00:00Z",
            ),
        )
        record = await repository.create_memory(
            "I prefer concise explanations",
            kind="preference",
            reason="explicit",
        )
        assert [item.id for item in vector.search("concise", source_type="memory")] == [record.id]

        # A true SQLite rebuild replaces only the memory namespace.
        vector.delete(record.id)
        assert await repository.rebuild() == 1
        assert vector.search("concise", source_type="memory")[0].id == record.id
        assert vector.search("architecture", source_type="document")[0].id == "doc-1"

        assert await repository.delete_memory(record.id) is True
        assert vector.search("concise", source_type="memory") == []
        assert await memory.get_memory(record.id, include_inactive=True) is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_vector_failure_keeps_sqlite_fact_and_marks_repair(settings_tmp, monkeypatch) -> None:
    database, memory, vector, repository = await _repository(settings_tmp)
    try:
        original = vector.upsert

        def fail_upsert(**_kwargs):
            raise OSError("simulated index write failure")

        monkeypatch.setattr(vector, "upsert", fail_upsert)
        record = await repository.create_memory(
            "The project uses strict local boundaries",
            kind="fact",
            reason="explicit",
        )
        assert await memory.get_memory(record.id) is not None
        health = await repository.health()
        assert health.repair_required is True
        assert health.pending_repairs == 1
        assert (await memory.search_memories("local boundaries"))[0].id == record.id

        monkeypatch.setattr(vector, "upsert", original)
        assert await repository.rebuild() == 1
        assert (await repository.health()).repair_required is False
        assert vector.search("local boundaries", source_type="memory")[0].id == record.id
    finally:
        await database.close()
