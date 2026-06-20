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
    assert policy.evaluate("I prefer concise answers").allowed is True


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
