from __future__ import annotations

import pytest

from april_common.audit import AuditLogger
from april_common.errors import PermissionDeniedError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.policy import MemoryPolicy
from services.memory.sqlite_memory import SqliteMemory
from services.memory.writer import MemoryWriter
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest


@pytest.mark.asyncio
async def test_migrations_write_read_delete_and_export(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    record = await memory.create_memory("I prefer local tools", reason="test")
    assert record in await memory.list_memories()
    assert (await memory.search_memories("local"))[0].id == record.id
    assert await memory.delete_memory(record.id)
    assert "memories" in await memory.export_memories()
    await database.close()


def test_memory_policy_decision() -> None:
    policy = MemoryPolicy()
    assert policy.evaluate("my password is secret", requested_by_user=True).allowed is False
    assert policy.evaluate("I prefer concise answers").allowed is False
    assert policy.evaluate("I prefer concise answers", requested_by_user=True).allowed is True


@pytest.mark.asyncio
async def test_writer_rejects_sensitive(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    writer = MemoryWriter(SqliteMemory(database))
    with pytest.raises(PermissionDeniedError):
        await writer.write("api token abc", reason="test", requested_by_user=True)
    await database.close()


@pytest.mark.asyncio
async def test_conversation_deletion(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    conversation_id = await memory.create_conversation()
    await memory.add_message(conversation_id, "user", "hello")
    assert await memory.delete_conversation(conversation_id)
    row = await database.fetchone(
        "SELECT * FROM messages WHERE conversation_id = ?", (conversation_id,)
    )
    assert row is None
    await database.close()


@pytest.mark.asyncio
async def test_approval_transaction_integrity(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    store = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    approval = await store.create(
        ApprovalRequest(
            tool="write_file",
            args={"path": "x", "content": "y"},
            permission_level=3,
            risk_level="code_write",
        ),
        actor="test",
        request_id="r",
    )
    records = await store.list_pending()
    assert records[0].id == approval.approval_id
    await database.close()


@pytest.mark.asyncio
async def test_reminders_are_stored_in_sqlite(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    reminder = await memory.create_reminder("stand up", due_at="2026-06-21T09:00:00Z")
    reminders = await memory.list_reminders()
    assert reminders[0].id == reminder.id
    assert reminders[0].content == "stand up"
    assert await memory.delete_reminder(reminder.id)
    assert await memory.list_reminders() == []
    await database.close()


@pytest.mark.asyncio
async def test_scheduler_state_round_trip(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    assert await memory.get_scheduler_state("last_briefing_date") is None
    await memory.set_scheduler_state("last_briefing_date", "2026-06-21")
    assert await memory.get_scheduler_state("last_briefing_date") == "2026-06-21"
    # Upsert: a second write overwrites rather than duplicating.
    await memory.set_scheduler_state("last_briefing_date", "2026-06-22")
    assert await memory.get_scheduler_state("last_briefing_date") == "2026-06-22"
    await database.close()


@pytest.mark.asyncio
async def test_repo_snapshot_round_trip(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    project = await memory.add_project(str(settings_tmp.home))
    assert await memory.get_repo_snapshot(project.id) is None
    await memory.upsert_repo_snapshot(project.id, "abc123", 2, "2026-06-22T00:00:00Z")
    snapshot = await memory.get_repo_snapshot(project.id)
    assert snapshot == {
        "head_sha": "abc123",
        "dirty_count": 2,
        "updated_at": "2026-06-22T00:00:00Z",
    }
    # Upsert overwrites rather than duplicating.
    await memory.upsert_repo_snapshot(project.id, "def456", 0, "2026-06-23T00:00:00Z")
    snapshot = await memory.get_repo_snapshot(project.id)
    assert snapshot == {
        "head_sha": "def456",
        "dirty_count": 0,
        "updated_at": "2026-06-23T00:00:00Z",
    }
    await database.close()


@pytest.mark.asyncio
async def test_list_upcoming_reminders_window(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    now_iso = "2026-06-21T08:00:00Z"
    until_iso = "2026-06-22T08:00:00Z"
    overdue = await memory.create_reminder("overdue", due_at="2026-06-20T09:00:00Z")
    in_window = await memory.create_reminder("in window", due_at="2026-06-21T18:00:00Z")
    await memory.create_reminder("out of window", due_at="2026-06-23T09:00:00Z")
    upcoming = await memory.list_upcoming_reminders(now_iso, until_iso)
    assert [reminder.id for reminder in upcoming] == [overdue.id, in_window.id]
    await database.close()
