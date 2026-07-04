from __future__ import annotations

from datetime import UTC, datetime

import anyio
import pytest
from fastapi.testclient import TestClient

from april_common.time import utc_now
from services.api.server import create_app
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.memory.decay import apply_memory_decay
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from tests.test_core_api import auth, make_container


async def _memory(settings) -> tuple[Database, SqliteMemory]:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


async def _seed(
    memory: SqliteMemory,
    *,
    content: str,
    source: str,
    created_at: str | None = None,
    confidence: float = 0.7,
) -> str:
    record = await memory.create_memory(
        content, reason="test seed", source=source, confidence=confidence
    )
    if created_at is not None:
        await memory.database.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?", (created_at, record.id)
        )
    return record.id


@pytest.mark.asyncio
async def test_decay_targets_stale_machine_memories_only(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        stale_machine = await _seed(
            memory,
            content="machine fact from months ago",
            source="archive",
            created_at="2026-01-01T00:00:00Z",
        )
        fresh_machine = await _seed(memory, content="fresh machine fact", source="archive")
        stale_user = await _seed(
            memory,
            content="user fact from months ago",
            source="user",
            created_at="2026-01-01T00:00:00Z",
        )
        guard = EvolutionWriteGuard(settings_tmp)
        now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)
        report = await apply_memory_decay(memory, guard=guard, now=now)
        assert report.decayed == 1
        assert report.faded == 0

        decayed = await memory.get_memory(stale_machine)
        assert decayed is not None
        assert decayed.confidence == pytest.approx(0.63)
        untouched_machine = await memory.get_memory(fresh_machine)
        assert untouched_machine is not None
        assert untouched_machine.confidence == pytest.approx(0.7)
        untouched_user = await memory.get_memory(stale_user)
        assert untouched_user is not None
        assert untouched_user.confidence == pytest.approx(0.7)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_decay_fades_low_confidence_without_deleting(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        fading_id = await _seed(
            memory,
            content="barely trusted machine fact",
            source="archive",
            created_at="2026-01-01T00:00:00Z",
            confidence=0.31,
        )
        guard = EvolutionWriteGuard(settings_tmp)
        now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)
        report = await apply_memory_decay(memory, guard=guard, now=now)
        assert report.decayed == 1
        assert report.faded == 1

        # The row still exists, has expires_at in the future, and shows up as
        # "fading" — it was not deleted and is still served until expiry.
        record = await memory.get_memory(fading_id, include_inactive=True)
        assert record is not None
        assert record.expires_at is not None
        assert record.expires_at > now.isoformat().replace("+00:00", "Z")
        fading = await memory.list_memories_by_state("fading")
        assert [item.id for item in fading] == [fading_id]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_expired_memories_hidden_from_retrieval_but_inspectable(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        expired_id = (
            await memory.create_memory(
                "expired machine fact",
                reason="seed",
                source="archive",
                expires_at="2026-01-01T00:00:00Z",
            )
        ).id
        active = await memory.list_memories()
        assert expired_id not in [record.id for record in active]
        expired = await memory.list_memories_by_state("expired")
        assert [record.id for record in expired] == [expired_id]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_memory_state_listing_covers_all_states(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        machine_id = await _seed(memory, content="machine fact", source="archive")
        user_id = await _seed(memory, content="user fact", source="user")
        superseded_id = await _seed(memory, content="old fact", source="archive")
        await memory.supersede_memory(superseded_id, superseded_by=machine_id)

        machine = [record.id for record in await memory.list_memories_by_state("machine")]
        assert machine == [machine_id]
        superseded = [
            record.id for record in await memory.list_memories_by_state("superseded")
        ]
        assert superseded == [superseded_id]
        active = [record.id for record in await memory.list_memories_by_state("active")]
        assert set(active) == {machine_id, user_id}

        with pytest.raises(ValueError, match="state must be one of"):
            await memory.list_memories_by_state("bogus")
    finally:
        await database.close()


def test_memory_inspect_api_and_validation(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    async def seed() -> str:
        record = await container.memory.create_memory(
            "machine-written fact", reason="seed", source="archive"
        )
        return record.id

    memory_id = anyio.run(seed)
    response = client.get("/memory/inspect", params={"state": "machine"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "machine"
    assert [record["id"] for record in body["memories"]] == [memory_id]

    bad = client.get("/memory/inspect", params={"state": "bogus"}, headers=headers)
    assert bad.status_code == 400

    unauthorized = client.get("/memory/inspect")
    assert unauthorized.status_code == 403


@pytest.mark.asyncio
async def test_decay_is_bounded_and_repeatable(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        memory_id = await _seed(
            memory,
            content="stale machine fact",
            source="archive",
            created_at="2026-01-01T00:00:00Z",
        )
        guard = EvolutionWriteGuard(settings_tmp)
        await apply_memory_decay(memory, guard=guard, now=utc_now())
        first = await memory.get_memory(memory_id)
        assert first is not None
        await apply_memory_decay(memory, guard=guard, now=utc_now())
        second = await memory.get_memory(memory_id)
        assert second is not None
        # Monotonic, deterministic decay: each pass multiplies by the factor.
        assert second.confidence == pytest.approx(round(first.confidence * 0.9, 4))
    finally:
        await database.close()
