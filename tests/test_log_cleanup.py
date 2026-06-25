from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from april_common.audit import AuditLogger
from april_common.errors import PermissionDeniedError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.schemas import ApprovalRecord
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.cleanup import (
    CleanupLimits,
    apply_approved_log_cleanup,
    build_cleanup_manifest,
    enumerate_candidates,
    load_cleanup_manifest,
    store_cleanup_manifest,
)
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
from skills.cleanup.log_cleanup import apply_log_cleanup, plan_log_cleanup
from skills.registry import default_registry


@pytest.fixture
async def harness(settings_tmp):
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    registry = default_registry()
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    engine = PermissionEngine(registry, approval_required_at=3)
    executor = ToolExecutionService(
        settings=settings_tmp,
        memory=memory,
        tool_registry=registry,
        permission_engine=engine,
        approvals=approvals,
    )
    yield SimpleNamespace(
        executor=executor,
        approvals=approvals,
        database=database,
        settings=settings_tmp,
    )
    await database.close()


async def _ctx(executor: ToolExecutionService):
    return await executor.context(
        request_id="req-1", actor="local-user", agent_id="system_action_agent", source="api"
    )


def _seed_logs(settings, *, count: int = 3, old: bool = True) -> Path:
    root = settings.logs_path
    root.mkdir(parents=True, exist_ok=True)
    old_time = time.time() - 90 * 86400
    for index in range(count):
        path = root / f"old-{index}.log"
        path.write_text(f"log entry {index}\n", encoding="utf-8")
        if old:
            os.utime(path, (old_time, old_time))
    return root


def _record_for(manifest_id: str, *, target: str, root: Path) -> ApprovalRecord:
    return ApprovalRecord(
        id="appr-1",
        tool="apply_log_cleanup",
        args={"manifest_id": manifest_id},
        agent="system_action_agent",
        canonical_hash="unused-in-direct-apply",
        metadata={
            "artifact_type": "log_cleanup",
            "manifest_id": manifest_id,
            "manifest_sha256": manifest_id,
            "target": target,
            "root": str(root),
        },
        permission_level=4,
        risk_level="system_action",
        status="approved",
        expires_at="2999-01-01T00:00:00Z",
        created_at="2026-01-01T00:00:00Z",
    )


# --- enumeration / plan ----------------------------------------------------


def test_plan_enumerates_old_files_only(settings_tmp) -> None:
    root = settings_tmp.logs_path
    root.mkdir(parents=True, exist_ok=True)
    old = root / "old.log"
    old.write_text("old\n", encoding="utf-8")
    os.utime(old, (time.time() - 90 * 86400, time.time() - 90 * 86400))
    (root / "fresh.log").write_text("fresh\n", encoding="utf-8")
    _root, candidates, _bytes = enumerate_candidates(
        target="logs", older_than_days=30, settings=settings_tmp
    )
    names = {item["relpath"] for item in candidates}
    assert "old.log" in names
    assert "fresh.log" not in names


def test_plan_skips_protected_and_symlinks(settings_tmp) -> None:
    root = settings_tmp.logs_path
    root.mkdir(parents=True, exist_ok=True)
    (root / ".gitkeep").write_text("", encoding="utf-8")
    (root / "audit.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "real.log").write_text("data\n", encoding="utf-8")
    outside = settings_tmp.home / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (root / "link.log").symlink_to(outside)
    _root, candidates, _bytes = enumerate_candidates(
        target="logs", older_than_days=0, settings=settings_tmp
    )
    names = {item["relpath"] for item in candidates}
    assert names == {"real.log"}  # protected + symlink excluded


def test_plan_enforces_max_file_limit(settings_tmp) -> None:
    _seed_logs(settings_tmp, count=5)
    with pytest.raises(PermissionDeniedError, match="candidate count"):
        enumerate_candidates(
            target="logs",
            older_than_days=0,
            settings=settings_tmp,
            limits=CleanupLimits(max_candidate_files=2),
        )


def test_plan_enforces_max_byte_limit(settings_tmp) -> None:
    root = settings_tmp.logs_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "big.log").write_text("x" * 5000, encoding="utf-8")
    with pytest.raises(PermissionDeniedError, match="total size"):
        enumerate_candidates(
            target="logs",
            older_than_days=0,
            settings=settings_tmp,
            limits=CleanupLimits(max_total_bytes=100),
        )


def test_plan_rejects_arbitrary_target(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        enumerate_candidates(target="/etc", older_than_days=0, settings=settings_tmp)


def test_manifest_is_content_addressed_and_detects_tampering(settings_tmp) -> None:
    _seed_logs(settings_tmp, count=2)
    result = build_cleanup_manifest(target="logs", older_than_days=0, settings=settings_tmp)
    manifest_id = result["manifest_id"]
    # Reloads and verifies integrity.
    manifest = load_cleanup_manifest(manifest_id)
    assert manifest["candidate_count"] == 2
    # Tamper the stored manifest: digest no longer matches the id -> fail closed.
    path = settings_tmp.resolve_path(Path("data/artifacts/cleanup")) / f"{manifest_id}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PermissionDeniedError, match="tampered"):
        load_cleanup_manifest(manifest_id)


# --- integration through the executor / approval boundary ------------------


async def test_plan_via_executor_deletes_nothing(harness) -> None:
    root = _seed_logs(harness.settings, count=3)
    context = await _ctx(harness.executor)
    outcome = await harness.executor.request_or_execute(
        tool="plan_log_cleanup",
        args={"target": "logs", "older_than_days": 0},
        context=context,
    )
    assert outcome.status == "executed"
    assert outcome.result is not None
    assert outcome.result.ok
    assert outcome.result.data["candidate_count"] == 3
    # Nothing was deleted by planning.
    assert len(list(root.glob("old-*.log"))) == 3


async def test_apply_requires_approval(harness) -> None:
    _seed_logs(harness.settings, count=2)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    manifest_id = plan.result.data["manifest_id"]
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup", args={"manifest_id": manifest_id}, context=context
    )
    assert apply.status == "pending_approval"
    assert apply.approval is not None
    assert len(list(harness.settings.logs_path.glob("old-*.log"))) == 2


async def test_successful_exact_cleanup_and_audit(harness) -> None:
    root = _seed_logs(harness.settings, count=3)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    manifest_id = plan.result.data["manifest_id"]
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup", args={"manifest_id": manifest_id}, context=context
    )
    outcome = await harness.executor.execute_approved(
        approval_id=apply.approval.approval_id,
        actor="local-user",
        request_id="req-apply",
    )
    assert outcome.status == "executed"
    assert outcome.result.data["deleted_count"] == 3
    assert list(root.glob("old-*.log")) == []
    # Audit trail records the approval lifecycle for the cleanup.
    audit_text = harness.settings.audit_path.read_text(encoding="utf-8")
    assert "approval_created" in audit_text
    assert "approved_tool_executed" in audit_text
    assert "apply_log_cleanup" in audit_text


async def test_denied_approval_does_not_delete(harness) -> None:
    root = _seed_logs(harness.settings, count=2)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup",
        args={"manifest_id": plan.result.data["manifest_id"]},
        context=context,
    )
    await harness.approvals.deny(
        approval_id=apply.approval.approval_id, actor="local-user", request_id="deny"
    )
    with pytest.raises(PermissionDeniedError):
        await harness.executor.execute_approved(
            approval_id=apply.approval.approval_id, actor="local-user", request_id="x"
        )
    assert len(list(root.glob("old-*.log"))) == 2


async def test_expired_approval_does_not_delete(harness) -> None:
    root = _seed_logs(harness.settings, count=2)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup",
        args={"manifest_id": plan.result.data["manifest_id"]},
        context=context,
    )
    # Force expiry deterministically.
    async with harness.database.transaction() as conn:
        await conn.execute(
            "UPDATE approvals SET expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00Z", apply.approval.approval_id),
        )
    with pytest.raises(PermissionDeniedError, match="expired"):
        await harness.executor.execute_approved(
            approval_id=apply.approval.approval_id, actor="local-user", request_id="x"
        )
    assert len(list(root.glob("old-*.log"))) == 2


async def test_replay_is_prevented(harness) -> None:
    _seed_logs(harness.settings, count=2)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    manifest_id = plan.result.data["manifest_id"]
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup", args={"manifest_id": manifest_id}, context=context
    )
    await harness.executor.execute_approved(
        approval_id=apply.approval.approval_id, actor="local-user", request_id="a1"
    )
    # Same approval cannot be replayed.
    with pytest.raises(PermissionDeniedError):
        await harness.executor.execute_approved(
            approval_id=apply.approval.approval_id, actor="local-user", request_id="a2"
        )
    # A fresh approval for the same (now consumed) manifest fails closed at verify.
    apply2 = await harness.executor.request_or_execute(
        tool="apply_log_cleanup", args={"manifest_id": manifest_id}, context=context
    )
    outcome = await harness.executor.execute_approved(
        approval_id=apply2.approval.approval_id, actor="local-user", request_id="a3"
    )
    assert outcome.status == "failed"
    assert "already been used" in (outcome.result.stderr or "")


async def test_tampered_manifest_rejected_at_apply(harness) -> None:
    _seed_logs(harness.settings, count=2)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    manifest_id = plan.result.data["manifest_id"]
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup", args={"manifest_id": manifest_id}, context=context
    )
    # Tamper the stored manifest after approval was created.
    manifest_path = (
        harness.settings.resolve_path(Path("data/artifacts/cleanup")) / f"{manifest_id}.json"
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    outcome = await harness.executor.execute_approved(
        approval_id=apply.approval.approval_id, actor="local-user", request_id="x"
    )
    assert outcome.status == "failed"
    assert len(list(harness.settings.logs_path.glob("old-*.log"))) == 2


async def test_file_changed_after_planning_is_not_deleted(harness) -> None:
    root = _seed_logs(harness.settings, count=2)
    context = await _ctx(harness.executor)
    plan = await harness.executor.request_or_execute(
        tool="plan_log_cleanup", args={"target": "logs", "older_than_days": 0}, context=context
    )
    apply = await harness.executor.request_or_execute(
        tool="apply_log_cleanup",
        args={"manifest_id": plan.result.data["manifest_id"]},
        context=context,
    )
    # Mutate one candidate after planning; its identity no longer matches.
    (root / "old-0.log").write_text("MUTATED CONTENT", encoding="utf-8")
    outcome = await harness.executor.execute_approved(
        approval_id=apply.approval.approval_id, actor="local-user", request_id="x"
    )
    assert outcome.status == "executed"
    assert (root / "old-0.log").exists()  # protected: content changed
    assert not (root / "old-1.log").exists()  # unchanged candidate deleted
    skipped = {item["relpath"] for item in outcome.result.data["skipped"]}
    assert "old-0.log" in skipped


# --- direct apply security (hand-crafted manifests) ------------------------


async def test_apply_rejects_path_traversal_candidate(settings_tmp) -> None:
    root = settings_tmp.logs_path
    root.mkdir(parents=True, exist_ok=True)
    outside = settings_tmp.home / "victim.txt"
    outside.write_text("do not delete\n", encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "target": "logs",
        "root": str(root.resolve()),
        "older_than_days": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "candidate_count": 1,
        "total_bytes": 1,
        "candidates": [{"relpath": "../victim.txt", "size": 1, "sha256": "x", "mtime": "t"}],
    }
    stored = store_cleanup_manifest(manifest)
    record = _record_for(stored["manifest_id"], target="logs", root=root.resolve())
    outcome = await apply_approved_log_cleanup(record)
    assert outcome.ok
    assert outside.exists()  # traversal candidate never deleted
    assert outcome.data["deleted_count"] == 0


async def test_apply_rejects_symlink_candidate(settings_tmp) -> None:
    root = settings_tmp.logs_path
    root.mkdir(parents=True, exist_ok=True)
    outside = settings_tmp.home / "target.txt"
    outside.write_text("keep\n", encoding="utf-8")
    (root / "link.log").symlink_to(outside)
    manifest = {
        "manifest_version": 1,
        "target": "logs",
        "root": str(root.resolve()),
        "older_than_days": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "candidate_count": 1,
        "total_bytes": 1,
        "candidates": [{"relpath": "link.log", "size": 1, "sha256": "x", "mtime": "t"}],
    }
    stored = store_cleanup_manifest(manifest)
    record = _record_for(stored["manifest_id"], target="logs", root=root.resolve())
    outcome = await apply_approved_log_cleanup(record)
    assert outcome.ok
    assert (root / "link.log").is_symlink()  # symlink not followed/deleted
    assert outside.exists()
    assert outcome.data["deleted_count"] == 0


async def test_apply_executor_fails_closed_without_approval(settings_tmp) -> None:
    # The apply_log_cleanup executor must never delete when reached directly; it
    # only runs through the approved manifest-bound path.
    result = await apply_log_cleanup({"manifest_id": "x" * 64})
    assert result.ok is False
    assert result.permission_level == 4
    assert "approved cleanup manifest" in (result.stderr or "")


async def test_plan_rejects_non_integer_older_than_days(settings_tmp) -> None:
    result = await plan_log_cleanup({"target": "logs", "older_than_days": "soon"})
    assert result.ok is False
    assert "older_than_days" in (result.stderr or "")


async def test_plan_reports_disallowed_target_as_error(settings_tmp) -> None:
    result = await plan_log_cleanup({"target": "secrets", "older_than_days": 0})
    assert result.ok is False
    assert result.permission_level == 1


async def test_apply_rejects_manifest_root_outside_configured_root(settings_tmp) -> None:
    # A manifest whose root is not the configured target root must fail closed.
    manifest = {
        "manifest_version": 1,
        "target": "logs",
        "root": "/etc",
        "older_than_days": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "candidate_count": 0,
        "total_bytes": 0,
        "candidates": [],
    }
    stored = store_cleanup_manifest(manifest)
    record = _record_for(stored["manifest_id"], target="logs", root=Path("/etc"))
    outcome = await apply_approved_log_cleanup(record)
    assert outcome.ok is False
    assert "outside the configured" in (outcome.stderr or "")
