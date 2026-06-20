from __future__ import annotations

import pytest

from april_common.audit import AuditLogger
from april_common.errors import PermissionDeniedError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest


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
