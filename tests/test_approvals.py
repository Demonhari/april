from __future__ import annotations

import asyncio
import json

import pytest

from april_common.audit import AuditLogger
from april_common.errors import PermissionDeniedError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest


class FailingAudit:
    def __init__(self) -> None:
        self.path = None

    def write(self, entry: dict[str, object]) -> None:
        raise RuntimeError("audit unavailable")


@pytest.fixture
async def approval_store(settings_tmp):
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    store = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    yield store
    await database.close()


async def _create(store: ApprovalStore):
    return await store.create(
        ApprovalRequest(
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            permission_level=3,
            risk_level="code_write",
        ),
        actor="test",
        request_id="r1",
    )


@pytest.mark.asyncio
async def test_exact_action_approval_succeeds_once(approval_store: ApprovalStore) -> None:
    approval = await _create(approval_store)
    await approval_store.approve_exact(
        approval_id=approval.approval_id,
        tool="write_file",
        args={"path": "a.py", "content": "x"},
        actor="test",
        request_id="r2",
    )
    await approval_store.consume(
        approval_id=approval.approval_id,
        result={"ok": True},
        actor="test",
        request_id="r3",
    )
    with pytest.raises(PermissionDeniedError):
        await approval_store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            actor="test",
            request_id="r4",
        )


@pytest.mark.asyncio
async def test_changed_arguments_denied(approval_store: ApprovalStore) -> None:
    approval = await _create(approval_store)
    with pytest.raises(PermissionDeniedError):
        await approval_store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "b.py", "content": "x"},
            actor="test",
            request_id="r2",
        )


@pytest.mark.asyncio
async def test_denial_prevents_execution(approval_store: ApprovalStore) -> None:
    approval = await _create(approval_store)
    await approval_store.deny(approval_id=approval.approval_id, actor="test", request_id="r2")
    with pytest.raises(PermissionDeniedError):
        await approval_store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            actor="test",
            request_id="r3",
        )


@pytest.mark.asyncio
async def test_expired_approval_denied(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    store = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=-1)
    approval = await _create(store)
    with pytest.raises(PermissionDeniedError):
        await store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            actor="test",
            request_id="r2",
        )
    await database.close()


@pytest.mark.asyncio
async def test_concurrent_approval_transition_succeeds_once(
    approval_store: ApprovalStore,
) -> None:
    approval = await _create(approval_store)

    async def approve(request_id: str) -> object:
        return await approval_store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            actor="test",
            request_id=request_id,
        )

    results = await asyncio.gather(approve("r2"), approve("r3"), return_exceptions=True)
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, PermissionDeniedError) for result in results) == 1
    assert (await approval_store.get(approval.approval_id)).status == "approved"


@pytest.mark.asyncio
async def test_concurrent_consumption_succeeds_once(approval_store: ApprovalStore) -> None:
    approval = await _create(approval_store)
    await approval_store.approve_exact(
        approval_id=approval.approval_id,
        tool="write_file",
        args={"path": "a.py", "content": "x"},
        actor="test",
        request_id="r2",
    )

    async def consume(request_id: str) -> object:
        return await approval_store.consume(
            approval_id=approval.approval_id,
            result={"ok": True},
            actor="test",
            request_id=request_id,
        )

    results = await asyncio.gather(consume("r3"), consume("r4"), return_exceptions=True)
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, PermissionDeniedError) for result in results) == 1
    assert (await approval_store.get(approval.approval_id)).status == "consumed"


@pytest.mark.asyncio
async def test_denial_and_approval_race_has_one_valid_winner(
    approval_store: ApprovalStore,
) -> None:
    approval = await _create(approval_store)

    async def approve() -> object:
        return await approval_store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            actor="test",
            request_id="approve",
        )

    async def deny() -> object:
        return await approval_store.deny(
            approval_id=approval.approval_id,
            actor="test",
            request_id="deny",
        )

    results = await asyncio.gather(approve(), deny(), return_exceptions=True)
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, PermissionDeniedError) for result in results) == 1
    assert (await approval_store.get(approval.approval_id)).status in {"approved", "denied"}


@pytest.mark.asyncio
async def test_failed_denial_rolls_back_approval_and_agent_state(
    approval_store: ApprovalStore,
) -> None:
    approval = await _create(approval_store)
    database = approval_store.database
    async with database.transaction() as connection:
        await connection.execute(
            """
            INSERT INTO conversations(id, actor, created_at)
            VALUES('conversation', 'test', '2026-01-01T00:00:00Z')
            """
        )
        await connection.execute(
            """
            INSERT INTO agent_runs(
                id, conversation_id, agent, status, metadata_json, created_at
            )
            VALUES('run', 'conversation', 'general_agent', 'suspended', '{}',
                   '2026-01-01T00:00:00Z')
            """
        )
        await connection.execute(
            """
            INSERT INTO suspended_agent_runs(
                id, agent_run_id, approval_id, conversation_id, agent, iteration,
                request_id, messages_json, tool_request_json, normalized_args_json,
                context_json, status, created_at
            )
            VALUES('suspended', 'run', ?, 'conversation', 'general_agent', 1,
                   'request', '[]', '{}', '{}', '{}', 'suspended',
                   '2026-01-01T00:00:00Z')
            """,
            (approval.approval_id,),
        )
        await connection.execute(
            """
            CREATE TRIGGER fail_agent_denial
            BEFORE UPDATE OF status ON agent_runs
            WHEN NEW.status = 'denied'
            BEGIN
                SELECT RAISE(ABORT, 'forced denial failure');
            END
            """
        )

    with pytest.raises(Exception, match="forced denial failure"):
        await approval_store.deny(
            approval_id=approval.approval_id,
            actor="test",
            request_id="deny",
        )

    assert (await approval_store.get(approval.approval_id)).status == "pending"
    suspended = await database.fetchone(
        "SELECT status FROM suspended_agent_runs WHERE approval_id = ?",
        (approval.approval_id,),
    )
    run = await database.fetchone("SELECT status FROM agent_runs WHERE id = 'run'")
    assert suspended is not None
    assert suspended["status"] == "suspended"
    assert run is not None
    assert run["status"] == "suspended"
    audit_text = approval_store.audit.path.read_text(encoding="utf-8")
    assert json.loads(audit_text.splitlines()[-1])["event_type"] == "approval_created"


@pytest.mark.asyncio
async def test_failed_audit_during_create_is_terminal_and_persisted(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    store = ApprovalStore(database, FailingAudit(), expiry_seconds=60)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="audit unavailable"):
        await _create(store)

    row = await database.fetchone(
        "SELECT status, result_json FROM approvals ORDER BY created_at DESC LIMIT 1"
    )
    assert row is not None
    assert row["status"] == "audit_failed"
    assert json.loads(row["result_json"])["prior_status"] == "pending"
    await database.close()


@pytest.mark.asyncio
async def test_failed_audit_during_approval_cannot_validate(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    audit = AuditLogger(settings_tmp.audit_path)
    store = ApprovalStore(database, audit, expiry_seconds=60)
    approval = await _create(store)
    store.audit = FailingAudit()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="audit unavailable"):
        await store.approve_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
            actor="test",
            request_id="failed-approval",
        )

    row = await database.fetchone(
        "SELECT status, result_json FROM approvals WHERE id = ?", (approval.approval_id,)
    )
    assert row is not None
    assert row["status"] == "audit_failed"
    assert json.loads(row["result_json"])["prior_status"] == "approved"
    with pytest.raises(PermissionDeniedError):
        await store.validate_exact(
            approval_id=approval.approval_id,
            tool="write_file",
            args={"path": "a.py", "content": "x"},
        )
    await database.close()


@pytest.mark.asyncio
async def test_failed_audit_during_consumption_is_terminal_and_replay_denied(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    audit = AuditLogger(settings_tmp.audit_path)
    store = ApprovalStore(database, audit, expiry_seconds=60)
    approval = await _create(store)
    await store.approve_exact(
        approval_id=approval.approval_id,
        tool="write_file",
        args={"path": "a.py", "content": "x"},
        actor="test",
        request_id="approve",
    )
    store.audit = FailingAudit()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="audit unavailable"):
        await store.consume(
            approval_id=approval.approval_id,
            result={"ok": True},
            actor="test",
            request_id="failed-consumption",
        )

    row = await database.fetchone(
        "SELECT status, result_json FROM approvals WHERE id = ?", (approval.approval_id,)
    )
    assert row is not None
    assert row["status"] == "audit_failed"
    assert json.loads(row["result_json"])["prior_status"] == "consumed"
    with pytest.raises(PermissionDeniedError):
        await store.consume(
            approval_id=approval.approval_id,
            result={"ok": True},
            actor="test",
            request_id="replay",
        )
    await database.close()
