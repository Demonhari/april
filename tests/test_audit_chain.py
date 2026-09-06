from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apps.runner.main import app
from april_common.audit import AuditLogger, AuditVerification, MemoryAuditAnchor
from april_common.errors import AprilError, PermissionDeniedError
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.permissions.approvals import ApprovalStore
from services.permissions.schemas import ApprovalRequest


def _process_audit_writer(path: str, count: int) -> None:
    logger = AuditLogger(Path(path))
    for index in range(count):
        logger.write({"event_type": "process_event", "index": index})


def test_valid_genesis_multi_event_and_secret_redaction(tmp_path: Path) -> None:
    anchor = MemoryAuditAnchor()
    logger = AuditLogger(tmp_path / "audit.jsonl", anchor=anchor)
    logger.write(
        {
            "event_type": "started",
            "api_token": "secret-value",
            "prompt": "private prompt",
            "safe": "visible",
        }
    )
    logger.write({"event_type": "completed", "count": 2})
    result = logger.verify()
    assert result.status == "valid"
    assert result.record_count == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["sequence"] == 1
    assert records[1]["previous_hash"] == records[0]["record_hash"]
    text = json.dumps(records)
    assert "secret-value" not in text
    assert "private prompt" not in text
    assert records[0]["payload"]["safe"] == "visible"


def test_concurrent_thread_and_process_writers(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: logger.write({"event_type": "thread_event", "index": index}),
                range(40),
            )
        )
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_process_audit_writer, args=(str(path), 10)) for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    result = logger.verify()
    assert result.status == "valid"
    assert result.record_count == 60


def test_mutation_reorder_and_middle_deletion_are_detected(tmp_path: Path) -> None:
    for mutation, expected in (
        ("mutate", "incorrect_event_hash"),
        ("reorder", "reordered_record"),
        ("delete", "sequence_gap"),
    ):
        path = tmp_path / f"{mutation}.jsonl"
        logger = AuditLogger(path)
        for index in range(4):
            logger.write({"event_type": "event", "index": index})
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if mutation == "mutate":
            records[1]["payload"]["index"] = 99
        elif mutation == "reorder":
            records[1], records[2] = records[2], records[1]
        else:
            records.pop(1)
        path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        result = logger.verify()
        assert result.corrupt
        assert expected in {issue.code for issue in result.issues}

    genesis_path = tmp_path / "genesis.jsonl"
    genesis = AuditLogger(genesis_path)
    genesis.write({"event_type": "one"})
    genesis.write({"event_type": "two"})
    genesis_path.write_text(
        genesis_path.read_text(encoding="utf-8").splitlines()[1] + "\n",
        encoding="utf-8",
    )
    assert "missing_genesis" in {issue.code for issue in genesis.verify().issues}


def test_terminal_truncation_malformed_line_and_anchor_lag(tmp_path: Path) -> None:
    anchor = MemoryAuditAnchor()
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path, anchor=anchor)
    logger.write({"event_type": "one"})
    logger.write({"event_type": "two"})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n", encoding="utf-8")
    assert "terminal_truncation" in {issue.code for issue in logger.verify().issues}

    malformed_path = tmp_path / "malformed.jsonl"
    malformed = AuditLogger(malformed_path)
    malformed.write({"event_type": "one"})
    with malformed_path.open("ab") as handle:
        handle.write(b'{"partial":')
    assert "malformed_json" in {issue.code for issue in malformed.verify().issues}
    malformed.write({"event_type": "two"})
    assert malformed.verify().status == "valid"

    lag_anchor = MemoryAuditAnchor()
    lag_logger = AuditLogger(tmp_path / "lag.jsonl", anchor=lag_anchor)
    lag_logger.write({"event_type": "one"})
    terminal = json.loads((tmp_path / "lag.jsonl").read_text(encoding="utf-8").splitlines()[0])
    lag_anchor.value = None
    lag = lag_logger.verify()
    assert lag.status == "anchor_lagged"
    assert lag.anchor_lagged
    assert terminal["sequence"] == 1


def test_audit_verify_cli_exit_codes_and_json(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("APRIL_HOME", str(tmp_path))  # type: ignore[attr-defined]
    monkeypatch.setenv("APRIL_ENV", "test")  # type: ignore[attr-defined]
    logger = AuditLogger(tmp_path / "logs" / "audit.jsonl")
    logger.write({"event_type": "valid"})
    runner = CliRunner()
    valid = runner.invoke(app, ["april", "audit", "verify", "--json"])
    assert valid.exit_code == 0
    assert json.loads(valid.stdout)["status"] == "valid"
    path = tmp_path / "logs" / "audit.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    corrupt = runner.invoke(app, ["april", "audit", "verify", "--json"])
    assert corrupt.exit_code == 1
    assert json.loads(corrupt.stdout)["status"] == "corrupt"


def test_recovery_quarantine_is_byte_exact_and_rechecks_concurrent_changes(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path)
    logger.write({"event_type": "original", "value": "kept"})
    original_bytes = path.read_bytes()
    path.write_bytes(b"{}\n")
    plan = logger.recover(reason="test concurrent recovery", apply=False)
    path.write_bytes(b'{"changed_after_plan":true}\n')
    with pytest.raises(AprilError) as error:
        logger.approve_recovery(plan_id=str(plan.plan_id))
    assert error.value.code == "AUDIT_RECOVERY_PLAN_STALE"
    quarantine_root = tmp_path / "data" / "backups" / "audit-quarantine"
    quarantine_log = next(quarantine_root.glob("recovery-*/audit.jsonl"))
    assert (
        hashlib.sha256(quarantine_log.read_bytes()).hexdigest()
        == hashlib.sha256(b"{}\n").hexdigest()
    )
    assert path.read_bytes() == b'{"changed_after_plan":true}\n'
    assert original_bytes not in path.read_bytes()


def test_recovery_anchor_failure_never_reports_success(tmp_path: Path) -> None:
    class FailingAnchor(MemoryAuditAnchor):
        fail = False

        def set(self, value: str) -> None:
            if self.fail:
                raise AprilError("AUDIT_ANCHOR_FAILED", "injected", 500)
            super().set(value)

    anchor = FailingAnchor()
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path, anchor=anchor)
    logger.write({"event_type": "one"})
    path.write_text("{}\n", encoding="utf-8")
    plan = logger.recover(reason="test anchor failure", apply=False)
    consent = logger.approve_recovery(plan_id=str(plan.plan_id))
    anchor.fail = True
    with pytest.raises(AprilError, match="AUDIT_ANCHOR_FAILED"):
        logger.recover(
            reason="test anchor failure",
            apply=True,
            plan_id=str(plan.plan_id),
            approval_id=str(consent["approval_id"]),
        )
    assert list((tmp_path / "data" / "backups" / "audit-quarantine").glob("recovery-*"))


def test_recovery_plan_consent_claim_and_apply_preserve_evidence(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path)
    logger.write({"event_type": "original", "value": "kept"})
    original_bytes = path.read_bytes()
    path.write_bytes(b"{}\n")

    plan = logger.plan_recovery(reason="owner reviewed test recovery", expiry_seconds=60)
    assert plan.status == "dry_run"
    consent = logger.approve_recovery(plan_id=str(plan.plan_id), plan_digest=plan.plan_digest)
    recovered = logger.recover(
        reason="owner reviewed test recovery",
        apply=True,
        plan_id=str(plan.plan_id),
        approval_id=str(consent["approval_id"]),
    )

    assert recovered.status == "recovered"
    assert logger.verify().valid
    quarantine = Path(str(plan.quarantine_directory))
    backup = tmp_path / "data" / "backups" / "audit-quarantine" / quarantine
    assert (backup / "audit.jsonl").read_bytes() == b"{}\n"
    assert (backup / "manifest.json").stat().st_mode & 0o077 == 0
    assert (
        tmp_path / "data" / "backups" / "audit-recovery-journal.jsonl"
    ).stat().st_mode & 0o077 == 0
    assert original_bytes not in path.read_bytes()
    with pytest.raises(AprilError, match="AUDIT_RECOVERY_REPLAY"):
        logger.recover(
            reason="owner reviewed test recovery",
            apply=True,
            plan_id=str(plan.plan_id),
            approval_id=str(consent["approval_id"]),
        )


def test_recovery_rejects_stale_consent_and_expiry_before_claim(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path)
    logger.write({"event_type": "original"})
    path.write_bytes(b"{}\n")
    stale_plan = logger.plan_recovery(reason="stale test", expiry_seconds=60)
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(AprilError, match="AUDIT_RECOVERY_PLAN_STALE"):
        logger.approve_recovery(plan_id=str(stale_plan.plan_id))

    path.write_bytes(b"{}\n")
    plan = logger.plan_recovery(reason="expiry test", expiry_seconds=60)
    consent = logger.approve_recovery(plan_id=str(plan.plan_id))
    real_now = __import__("april_common.audit", fromlist=["utc_now_iso"]).utc_now_iso
    monkeypatch.setattr(
        "april_common.audit.utc_now_iso",
        lambda: "9999-01-01T00:00:00Z",
    )
    with pytest.raises(AprilError, match="AUDIT_RECOVERY_EXPIRED"):
        logger.recover(
            reason="expiry test",
            apply=True,
            plan_id=str(plan.plan_id),
            approval_id=str(consent["approval_id"]),
        )
    monkeypatch.setattr("april_common.audit.utc_now_iso", real_now)
    assert path.read_bytes() == b"{}\n"


def test_recovery_resumes_after_log_publication_before_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path)
    logger.write({"event_type": "original"})
    path.write_bytes(b"{}\n")
    plan = logger.plan_recovery(reason="interruption test", expiry_seconds=60)
    consent = logger.approve_recovery(plan_id=str(plan.plan_id))
    original_append = logger._append_recovery_event

    def interrupt(event_type: str, payload: dict[str, object]):
        if event_type == "log_published":
            raise RuntimeError("interrupted after publication")
        return original_append(event_type, payload)

    monkeypatch.setattr(logger, "_append_recovery_event", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        logger.recover(
            reason="interruption test",
            apply=True,
            plan_id=str(plan.plan_id),
            approval_id=str(consent["approval_id"]),
        )
    monkeypatch.setattr(logger, "_append_recovery_event", original_append)
    resumed = logger.recover(
        reason="interruption test",
        apply=True,
        plan_id=str(plan.plan_id),
        approval_id=str(consent["approval_id"]),
    )
    assert resumed.status == "recovered"
    assert logger.verify().valid


def test_recovery_journal_tampering_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    logger = AuditLogger(path)
    logger.write({"event_type": "original"})
    path.write_bytes(b"{}\n")
    plan = logger.plan_recovery(reason="journal test")
    journal = tmp_path / "data" / "backups" / "audit-recovery-journal.jsonl"
    record = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    record["payload"]["plan_id"] = "tampered"
    journal.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(AprilError, match="AUDIT_RECOVERY_JOURNAL_CORRUPT"):
        logger.approve_recovery(plan_id=str(plan.plan_id))


def test_unavailable_audit_verification_has_distinct_cli_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APRIL_HOME", str(tmp_path))
    monkeypatch.setenv("APRIL_ENV", "test")

    class UnavailableAudit:
        def verify(self) -> AuditVerification:
            return AuditVerification(
                status="unavailable",
                valid=False,
                corrupt=False,
                anchor_lagged=False,
                record_count=0,
                terminal_sequence=None,
                terminal_hash=None,
            )

    monkeypatch.setattr(
        "apps.runner.audit_commands.audit_logger_for_settings", lambda settings: UnavailableAudit()
    )
    result = CliRunner().invoke(app, ["april", "audit", "verify", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "unavailable"


@pytest.mark.asyncio
async def test_recovery_approval_requires_exact_unexpired_binding(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    audit = AuditLogger(settings_tmp.audit_path)
    store = ApprovalStore(database, audit, expiry_seconds=60)
    args = {"apply": True, "reason": "exact recovery"}
    approval = await store.create(
        ApprovalRequest(
            tool="audit_recovery",
            args=args,
            permission_level=3,
            risk_level="code_write",
        ),
        actor="test",
        request_id="create-recovery",
    )
    await store.approve_exact(
        approval_id=approval.approval_id,
        tool="audit_recovery",
        args=args,
        actor="test",
        request_id="approve-recovery",
    )
    with pytest.raises(PermissionDeniedError):
        await store.consume_exact(
            approval_id=approval.approval_id,
            tool="audit_recovery",
            args={"apply": True, "reason": "different recovery"},
            result={"ok": True},
            actor="test",
            request_id="consume-recovery",
        )
    with pytest.raises(PermissionDeniedError):
        await store.validate_exact(
            approval_id="arbitrary-nonempty-id",
            tool="audit_recovery",
            args=args,
        )
    await database.close()
