from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from april_common.credentials import CredentialKey, CredentialStore, CredentialStoreError
from april_common.errors import AprilError
from april_common.time import utc_now_iso

if TYPE_CHECKING:
    from april_common.settings import AprilSettings

AUDIT_SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
SECRET_KEYWORDS = (
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "api_key",
    "prompt",
    "transcript",
    "audio",
    "environment",
    "cookie",
)
_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_COMPATIBILITY_FIELDS = frozenset(
    {
        "actor",
        "request_id",
        "audit_correlation_id",
        "approval_id",
        "reference_id",
        "reminder_id",
        "memory_id",
        "memory_type",
        "agent",
        "tool",
        "permission_level",
        "risk",
        "risk_level",
        "outcome",
        "status",
        "project_id",
        "content_length",
        "reason_length",
        "kind",
        "sink",
        "date",
        "muted",
        "case_id",
        "error_type",
        "error_message_length",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)\bAPRIL_(?:API|RUNTIME)_TOKEN\s*=\s*\S+"),
)


class AuditAnchor(Protocol):
    def get(self) -> str | None: ...

    def set(self, value: str) -> None: ...


class MemoryAuditAnchor:
    def __init__(self, value: str | None = None) -> None:
        self.value = value

    def get(self) -> str | None:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class CredentialAuditAnchor:
    def __init__(self, store: CredentialStore) -> None:
        self.store = store

    def get(self) -> str | None:
        return self.store.get(CredentialKey.AUDIT_ANCHOR)

    def set(self, value: str) -> None:
        self.store.set(CredentialKey.AUDIT_ANCHOR, value)


class FileAuditAnchor:
    """Development-only durable anchor; production injects a Keychain anchor."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self) -> str | None:
        try:
            mode = self.path.stat().st_mode & 0o777
            if mode & 0o077:
                raise AprilError(
                    "AUDIT_ANCHOR_INSECURE",
                    "Audit anchor permissions are not owner-only.",
                    500,
                )
            return self.path.read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as exc:
            raise AprilError(
                "AUDIT_ANCHOR_INVALID",
                "Protected audit anchor is malformed.",
                422,
            ) from exc
        except OSError as exc:
            raise AprilError(
                "AUDIT_ANCHOR_FAILED",
                "Unable to read the protected audit anchor.",
                500,
            ) from exc

    def set(self, value: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(value + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise AprilError(
                "AUDIT_ANCHOR_FAILED",
                "Unable to update the protected audit anchor.",
                500,
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    line: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class AuditVerification:
    status: str
    valid: bool
    corrupt: bool
    anchor_lagged: bool
    record_count: int
    terminal_sequence: int | None
    terminal_hash: str | None
    issues: tuple[AuditIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditStartupDecision:
    """Safe, redacted classification used before operational startup."""

    accepted: bool
    status: str
    issue_codes: tuple[str, ...]
    issue_lines: tuple[str, ...]
    record_count: int
    reason: str
    next_commands: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "issue_codes": list(self.issue_codes),
            "issue_lines": list(self.issue_lines),
            "record_count": self.record_count,
            "reason": self.reason,
            "next_commands": list(self.next_commands),
        }

    @property
    def operator_message(self) -> str:
        if self.accepted:
            return f"Audit chain accepted ({self.status}); records={self.record_count}."
        lines = [
            f"APRIL cannot start because the local audit chain is {self.status}.",
            "No operational services were started.",
            "Run:",
            "  run april audit verify --json",
        ]
        if self.status == "corrupt":
            lines.extend(
                [
                    "Then review:",
                    '  run april audit recover --reason "owner-reviewed recovery"',
                ]
            )
        else:
            lines.append(
                "Do not create a new chain for an unavailable audit; resolve the access "
                "failure and verify again."
            )
        if self.issue_lines:
            lines.append("Safe diagnosis: " + ", ".join(self.issue_lines) + ".")
        return "\n".join(lines)


class AuditStartupBlocked(RuntimeError):
    """Raised when an operational startup path fails the audit safety gate."""

    def __init__(self, decision: AuditStartupDecision) -> None:
        self.decision = decision
        super().__init__(decision.operator_message)


@dataclass(frozen=True, slots=True)
class AuditRecoveryPlan:
    status: str
    issue_codes: tuple[str, ...]
    record_count: int
    quarantine_directory: str | None = None
    quarantined_log_sha256: str | None = None
    plan_id: str | None = None
    plan_digest: str | None = None
    canonical_target: str | None = None
    original_anchor_sha256: str | None = None
    expires_at: str | None = None
    approval_id: str | None = None
    phase: str | None = None
    log_changed: bool | None = None
    anchor_state: str | None = None
    resume_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Audit payload keys must be strings.")
            if any(secret in key.lower() for secret in SECRET_KEYWORDS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        redacted_value = value
        for pattern in _SECRET_VALUE_PATTERNS:
            redacted_value = pattern.sub("[REDACTED]", redacted_value)
        return redacted_value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Audit payload numbers must be finite.")
        return value
    raise ValueError(f"Unsupported audit payload type: {type(value).__name__}.")


class AuditLogger:
    def __init__(self, path: Path, *, anchor: AuditAnchor | None = None) -> None:
        self.path = path.expanduser().resolve(strict=False)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.anchor = anchor or FileAuditAnchor(self.path.with_name(f"{self.path.name}.anchor"))
        self.recovery_root = self.path.parent.parent / "data" / "backups" / "audit-quarantine"
        self.recovery_journal = (
            self.path.parent.parent / "data" / "backups" / "audit-recovery-journal.jsonl"
        )
        with _THREAD_LOCKS_GUARD:
            self._thread_lock = _THREAD_LOCKS.setdefault(self.lock_path, threading.Lock())

    def write(self, entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            raise AprilError("AUDIT_EVENT_INVALID", "Audit event must be an object.", 422)
        # Older call sites used ``event``. Keep that controlled adapter while
        # persisting only the Phase 4B ``event_type`` field.
        event_type = entry.get("event_type", entry.get("event"))
        if not isinstance(event_type, str) or not event_type.strip():
            raise AprilError(
                "AUDIT_EVENT_INVALID",
                "Audit event_type must be a non-empty string.",
                422,
            )
        try:
            payload = redact(
                {key: value for key, value in entry.items() if key not in {"event_type", "event"}}
            )
            _canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise AprilError(
                "AUDIT_EVENT_INVALID",
                "Audit event payload is malformed or unsupported.",
                422,
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            self._write_under_process_lock(event_type.strip(), payload)

    def verify(self) -> AuditVerification:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _flock(descriptor)
                return verify_audit_chain(self.path, anchor=self.anchor)
            finally:
                _funlock(descriptor)
                os.close(descriptor)

    def _write_under_process_lock(self, event_type: str, payload: Any) -> None:
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _flock(descriptor)
            self._recover_incomplete_final_line()
            verification = verify_audit_chain(self.path, anchor=self.anchor)
            if not verification.valid:
                issue_codes = [issue.code for issue in verification.issues]
                raise AprilError(
                    "AUDIT_CHAIN_CORRUPT"
                    if verification.corrupt
                    else "AUDIT_VERIFICATION_UNAVAILABLE",
                    "Audit verification failed; append refused.",
                    500,
                    {"issues": issue_codes},
                )
            if verification.anchor_lagged and verification.terminal_hash is not None:
                self.anchor.set(
                    _encode_anchor(
                        verification.terminal_sequence or 0,
                        verification.terminal_hash,
                    )
                )
            sequence = (verification.terminal_sequence or 0) + 1
            previous_hash = verification.terminal_hash or GENESIS_HASH
            record: dict[str, Any] = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "sequence": sequence,
                "event_id": str(uuid.uuid4()),
                "timestamp": utc_now_iso(),
                "event_type": event_type,
                "payload": payload,
                "previous_hash": previous_hash,
            }
            record.update(
                {key: value for key, value in payload.items() if key in _COMPATIBILITY_FIELDS}
            )
            record_hash = _record_hash(record)
            record["record_hash"] = record_hash
            encoded = (_canonical_json(record) + "\n").encode("utf-8")
            try:
                append_descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                    0o600,
                )
                try:
                    os.fchmod(append_descriptor, 0o600)
                    _write_all(append_descriptor, encoded)
                    os.fsync(append_descriptor)
                finally:
                    os.close(append_descriptor)
                self.anchor.set(_encode_anchor(sequence, record_hash))
                _fsync_directory(self.path.parent)
            except OSError as exc:
                raise AprilError(
                    "AUDIT_LOG_FAILED",
                    "Unable to durably write the audit log.",
                    500,
                ) from exc
        finally:
            _funlock(descriptor)
            os.close(descriptor)

    def _recover_incomplete_final_line(self) -> None:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise AprilError("AUDIT_LOG_FAILED", "Unable to read the audit log.", 500) from exc
        if not data or data.endswith(b"\n"):
            return
        last_newline = data.rfind(b"\n")
        prefix = data[: last_newline + 1] if last_newline >= 0 else b""
        # Only an unterminated suffix is recoverable. Validate every complete
        # record before truncating so arbitrary earlier corruption is preserved.
        prefix_result = _verify_bytes(prefix, anchor=None)
        if prefix_result.corrupt:
            raise AprilError(
                "AUDIT_CHAIN_CORRUPT",
                "Audit chain is corrupt before its incomplete final line.",
                500,
            )
        file_descriptor = os.open(self.path, os.O_WRONLY)
        try:
            os.ftruncate(file_descriptor, len(prefix))
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)

    def recover(
        self,
        *,
        reason: str,
        apply: bool = False,
        approval_id: str | None = None,
        plan_id: str | None = None,
    ) -> AuditRecoveryPlan:
        """Plan or perform an explicit owner-approved chain recovery."""
        if not reason.strip() or len(reason) > 240:
            raise AprilError(
                "AUDIT_RECOVERY_INVALID",
                "A bounded recovery reason is required.",
                422,
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            plan: dict[str, Any] | None = None
            try:
                _flock(descriptor)
                plan = self._load_recovery_plan(plan_id) if apply else None
                if apply and plan_id is not None and plan is None:
                    raise AprilError(
                        "AUDIT_RECOVERY_PLAN_INVALID",
                        "Recovery plan was not found.",
                        404,
                    )
                verification = verify_audit_chain(self.path, anchor=self.anchor)
                # A claimed operation may be resumed after publication. Its
                # authorization was checked at claim time, so do not re-expire
                # it merely because the newly published chain now verifies.
                if apply and plan is not None:
                    self._check_plan_snapshot_or_publication(plan)
                    if approval_id is None:
                        raise AprilError(
                            "AUDIT_RECOVERY_APPROVAL_REQUIRED",
                            "Owner consent is required for recovery.",
                            403,
                        )
                    self._claim_recovery(plan, approval_id=approval_id)
                    return self._publish_recovery(
                        plan,
                        approval_id=approval_id,
                        reason=str(plan["reason"]),
                        issue_codes=tuple(str(code) for code in plan.get("issue_codes", [])),
                        verification=verification,
                    )
                issue_codes = tuple(sorted({issue.code for issue in verification.issues}))
                if not verification.valid and not verification.corrupt:
                    return AuditRecoveryPlan("unavailable", issue_codes, verification.record_count)
                if not verification.corrupt:
                    return AuditRecoveryPlan("not_required", issue_codes, verification.record_count)
                if not apply:
                    return self._create_recovery_plan(
                        reason=reason.strip(),
                        issue_codes=issue_codes,
                        verification=verification,
                    )

                plan = self._load_recovery_plan(plan_id)
                if plan is None:
                    raise AprilError(
                        "AUDIT_RECOVERY_PLAN_REQUIRED",
                        "Apply requires an immutable recovery plan.",
                        409,
                    )
                self._check_plan_snapshot(plan)
                if approval_id is None:
                    raise AprilError(
                        "AUDIT_RECOVERY_APPROVAL_REQUIRED",
                        "Owner consent is required for recovery.",
                        403,
                    )
                self._claim_recovery(plan, approval_id=approval_id)
                return self._publish_recovery(
                    plan,
                    approval_id=approval_id,
                    reason=reason.strip(),
                    issue_codes=issue_codes,
                    verification=verification,
                )
            except AprilError as exc:
                details: dict[str, Any] = {
                    "phase": "claim" if plan is not None else "preflight",
                    "log_changed": self._recovery_log_changed(plan) if plan else False,
                    "anchor_state": "unchanged",
                    "plan_id": plan.get("plan_id") if plan else plan_id,
                    "approval_id": approval_id,
                }
                details.update(exc.details)
                raise AprilError(exc.code, exc.message, exc.status_code, details) from exc
            finally:
                _funlock(descriptor)
                os.close(descriptor)

    def plan_recovery(self, *, reason: str, expiry_seconds: int = 900) -> AuditRecoveryPlan:
        """Create a durable immutable recovery plan without changing the chain."""
        if not reason.strip() or len(reason) > 240:
            raise AprilError(
                "AUDIT_RECOVERY_INVALID", "A bounded recovery reason is required.", 422
            )
        if expiry_seconds <= 0 or expiry_seconds > 86_400:
            raise AprilError("AUDIT_RECOVERY_INVALID", "Recovery expiry is out of bounds.", 422)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _flock(descriptor)
                verification = verify_audit_chain(self.path, anchor=self.anchor)
                if not verification.corrupt:
                    if not verification.valid:
                        return AuditRecoveryPlan("unavailable", (), verification.record_count)
                    return AuditRecoveryPlan("not_required", (), verification.record_count)
                return self._create_recovery_plan(
                    reason=reason.strip(),
                    issue_codes=tuple(sorted({issue.code for issue in verification.issues})),
                    verification=verification,
                    expiry_seconds=expiry_seconds,
                )
            finally:
                _funlock(descriptor)
                os.close(descriptor)

    def approve_recovery(self, *, plan_id: str, plan_digest: str | None = None) -> dict[str, Any]:
        """Record local-owner consent for exactly one immutable recovery plan."""
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _flock(descriptor)
                plan = self._load_recovery_plan(plan_id)
                if plan is None:
                    raise AprilError(
                        "AUDIT_RECOVERY_PLAN_INVALID", "Recovery plan was not found.", 404
                    )
                if plan_digest is not None and plan_digest != plan["plan_digest"]:
                    raise AprilError(
                        "AUDIT_RECOVERY_PLAN_STALE", "Recovery plan digest changed.", 409
                    )
                self._check_plan_snapshot(plan)
                expires_at = str(plan["expires_at"])
                if expires_at <= utc_now_iso():
                    raise AprilError("AUDIT_RECOVERY_EXPIRED", "Recovery plan has expired.", 409)
                existing = self._recovery_events_for(plan_id)
                if any(event["event_type"] == "consent" for event in existing):
                    raise AprilError(
                        "AUDIT_RECOVERY_REPLAY", "Recovery consent was already recorded.", 409
                    )
                approval_id = f"recovery:{plan_id}:{str(plan['plan_digest'])[:16]}"
                event = self._append_recovery_event(
                    "consent",
                    {
                        "plan_id": plan_id,
                        "plan_digest": plan["plan_digest"],
                        "approval_id": approval_id,
                        "expires_at": expires_at,
                        "canonical_target": plan["canonical_target"],
                        "original_log_sha256": plan["original_log_sha256"],
                        "original_anchor_sha256": plan["original_anchor_sha256"],
                    },
                )
                return {
                    "status": "approved",
                    "approval_id": approval_id,
                    "event_digest": event["record_hash"],
                    **plan,
                }
            finally:
                _funlock(descriptor)
                os.close(descriptor)

    def _create_recovery_plan(
        self,
        *,
        reason: str,
        issue_codes: tuple[str, ...],
        verification: AuditVerification,
        expiry_seconds: int = 900,
    ) -> AuditRecoveryPlan:
        snapshot = self.path.read_bytes() if self.path.exists() else b""
        anchor_snapshot = self.anchor.get()
        self.recovery_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        quarantine = Path(tempfile.mkdtemp(prefix="recovery-", dir=self.recovery_root))
        os.chmod(quarantine, 0o700)
        log_copy = quarantine / self.path.name
        if self.path.exists():
            shutil.copy2(self.path, log_copy)
            os.chmod(log_copy, 0o600)
        if isinstance(self.anchor, FileAuditAnchor) and self.anchor.path.exists():
            anchor_copy = quarantine / self.anchor.path.name
            shutil.copy2(self.anchor.path, anchor_copy)
            os.chmod(anchor_copy, 0o600)
        if (
            self.path.read_bytes() if self.path.exists() else b""
        ) != snapshot or self.anchor.get() != anchor_snapshot:
            raise AprilError(
                "AUDIT_RECOVERY_CONCURRENT_CHANGE",
                "Audit log or protected anchor changed during recovery planning.",
                409,
            )
        log_hash = hashlib.sha256(snapshot).hexdigest()
        anchor_hash = hashlib.sha256((anchor_snapshot or "").encode("utf-8")).hexdigest()
        expires_at = (
            (datetime.now(UTC) + timedelta(seconds=expiry_seconds))
            .isoformat()
            .replace("+00:00", "Z")
        )
        fields = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "canonical_target": str(self.path),
            "original_log_sha256": log_hash,
            "original_anchor_sha256": anchor_hash,
            "reason": reason[:240],
            "issue_codes": list(issue_codes),
            "quarantine_directory": quarantine.name,
            "expires_at": expires_at,
        }
        plan_digest = hashlib.sha256(_canonical_json(fields).encode("utf-8")).hexdigest()
        plan_id = plan_digest[:32]
        manifest = {
            **fields,
            "plan_id": plan_id,
            "plan_digest": plan_digest,
            "original_anchor": anchor_snapshot,
        }
        _write_private_json(quarantine / "manifest.json", manifest)
        self._append_recovery_event("plan_created", manifest)
        return AuditRecoveryPlan(
            "dry_run",
            issue_codes,
            verification.record_count,
            quarantine.name,
            log_hash,
            plan_id,
            plan_digest,
            str(self.path),
            anchor_hash,
            expires_at,
        )

    def _load_recovery_plan(self, plan_id: str | None) -> dict[str, Any] | None:
        if not plan_id or not re.fullmatch(r"[a-f0-9]{32}", plan_id):
            return None
        if not self.recovery_root.exists():
            return None
        for directory in sorted(self.recovery_root.glob("recovery-*/manifest.json")):
            try:
                manifest = json.loads(directory.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            if isinstance(manifest, dict) and manifest.get("plan_id") == plan_id:
                if manifest.get(
                    "quarantine_directory"
                ) != directory.parent.name or not re.fullmatch(
                    r"recovery-[A-Za-z0-9_-]+", directory.parent.name
                ):
                    raise AprilError(
                        "AUDIT_RECOVERY_PLAN_INVALID",
                        "Recovery plan directory binding is invalid.",
                        409,
                    )
                fields = {
                    key: manifest.get(key)
                    for key in (
                        "schema_version",
                        "canonical_target",
                        "original_log_sha256",
                        "original_anchor_sha256",
                        "reason",
                        "issue_codes",
                        "quarantine_directory",
                        "expires_at",
                    )
                }
                digest = hashlib.sha256(_canonical_json(fields).encode("utf-8")).hexdigest()
                if manifest.get("plan_digest") != digest or manifest.get("plan_id") != digest[:32]:
                    raise AprilError(
                        "AUDIT_RECOVERY_PLAN_INVALID", "Recovery plan digest is invalid.", 409
                    )
                return manifest
        return None

    def _check_plan_snapshot(self, plan: dict[str, Any]) -> None:
        current = self.path.read_bytes() if self.path.exists() else b""
        if (
            plan.get("canonical_target") != str(self.path)
            or hashlib.sha256(current).hexdigest() != plan.get("original_log_sha256")
            or self.anchor.get() != plan.get("original_anchor")
        ):
            raise AprilError(
                "AUDIT_RECOVERY_PLAN_STALE", "The planned audit log or anchor changed.", 409
            )

    def _check_plan_snapshot_or_publication(self, plan: dict[str, Any]) -> None:
        """Accept only the original snapshot or this plan's own candidate."""
        try:
            self._check_plan_snapshot(plan)
            return
        except AprilError as exc:
            if exc.code != "AUDIT_RECOVERY_PLAN_STALE":
                raise
        current = self.path.read_bytes() if self.path.exists() else b""
        candidate = self.recovery_root / str(plan["quarantine_directory"]) / "candidate-audit.jsonl"
        candidate_hash = plan.get("candidate_sha256")
        if candidate.exists():
            candidate_bytes = candidate.read_bytes()
            if candidate_hash is None:
                candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
            matches_candidate = (
                current == candidate_bytes
                and hashlib.sha256(candidate_bytes).hexdigest() == candidate_hash
            )
        else:
            matches_candidate = (
                isinstance(candidate_hash, str)
                and hashlib.sha256(current).hexdigest() == candidate_hash
            )
        if not matches_candidate:
            raise AprilError(
                "AUDIT_RECOVERY_CONCURRENT_CHANGE",
                "Audit log changed during recovery publication.",
                409,
            )

    def _recovery_events_for(self, plan_id: str) -> list[dict[str, Any]]:
        return [
            event
            for event in self._read_recovery_journal()
            if isinstance(event.get("payload"), dict) and event["payload"].get("plan_id") == plan_id
        ]

    def _read_recovery_journal(self) -> list[dict[str, Any]]:
        if not self.recovery_journal.exists():
            return []
        try:
            raw = self.recovery_journal.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise AprilError(
                    "AUDIT_RECOVERY_JOURNAL_CORRUPT",
                    "Recovery journal has an incomplete final record.",
                    500,
                )
            lines = raw.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise AprilError(
                "AUDIT_RECOVERY_JOURNAL_CORRUPT", "Recovery journal is unavailable.", 500
            ) from exc
        records: list[dict[str, Any]] = []
        previous_hash = GENESIS_HASH
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AprilError(
                    "AUDIT_RECOVERY_JOURNAL_CORRUPT", "Recovery journal is invalid.", 500
                ) from exc
            if (
                not isinstance(record, dict)
                or record.get("schema_version") != AUDIT_SCHEMA_VERSION
                or not isinstance(record.get("event_id"), str)
                or not isinstance(record.get("timestamp"), str)
                or not isinstance(record.get("event_type"), str)
                or not isinstance(record.get("payload"), dict)
                or record.get("previous_hash") != previous_hash
            ):
                raise AprilError(
                    "AUDIT_RECOVERY_JOURNAL_CORRUPT", "Recovery journal hash chain is invalid.", 500
                )
            record_hash = record.get("record_hash")
            unsigned = {key: value for key, value in record.items() if key != "record_hash"}
            if not isinstance(record_hash, str) or _record_hash(unsigned) != record_hash:
                raise AprilError(
                    "AUDIT_RECOVERY_JOURNAL_CORRUPT",
                    "Recovery journal record hash is invalid.",
                    500,
                )
            records.append(record)
            previous_hash = record_hash
        return records

    def _append_recovery_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.recovery_journal.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.recovery_journal.parent, 0o700)
        previous = GENESIS_HASH
        if self.recovery_journal.exists():
            records = self._read_recovery_journal()
            if records:
                previous = str(records[-1]["record_hash"])
        record: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "payload": redact(payload),
            "previous_hash": previous,
        }
        record["record_hash"] = _record_hash(record)
        descriptor = os.open(self.recovery_journal, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, (_canonical_json(record) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.recovery_journal.parent)
        return record

    def _claim_recovery(self, plan: dict[str, Any], *, approval_id: str) -> None:
        plan_id = str(plan["plan_id"])
        expected = f"recovery:{plan_id}:{str(plan['plan_digest'])[:16]}"
        if approval_id != expected:
            raise AprilError(
                "AUDIT_RECOVERY_APPROVAL_INVALID",
                "Recovery approval is not bound to this plan.",
                403,
            )
        events = self._recovery_events_for(plan_id)
        consent = next((event for event in events if event["event_type"] == "consent"), None)
        if consent is None or consent.get("payload", {}).get("approval_id") != approval_id:
            raise AprilError(
                "AUDIT_RECOVERY_APPROVAL_REQUIRED", "Owner consent is required for recovery.", 403
            )
        if any(event["event_type"] == "completed" for event in events):
            raise AprilError(
                "AUDIT_RECOVERY_REPLAY", "Recovery operation was already completed.", 409
            )
        claimed = next((event for event in events if event["event_type"] == "claimed"), None)
        if claimed is not None:
            if claimed.get("payload", {}).get("approval_id") != approval_id:
                raise AprilError(
                    "AUDIT_RECOVERY_REPLAY", "Recovery plan is claimed by another approval.", 409
                )
            return
        if claimed is None:
            if str(consent["payload"].get("expires_at", "")) <= utc_now_iso():
                raise AprilError(
                    "AUDIT_RECOVERY_EXPIRED", "Recovery consent expired before claim.", 409
                )
            self._append_recovery_event(
                "claimed",
                {
                    "plan_id": plan_id,
                    "approval_id": approval_id,
                    "plan_digest": plan["plan_digest"],
                },
            )

    def _publish_recovery(
        self,
        plan: dict[str, Any],
        *,
        approval_id: str,
        reason: str,
        issue_codes: tuple[str, ...],
        verification: AuditVerification,
    ) -> AuditRecoveryPlan:
        quarantine = self.recovery_root / str(plan["quarantine_directory"])
        candidate = quarantine / "candidate-audit.jsonl"
        candidate_anchor = quarantine / "candidate-anchor.json"
        phase = "staging"
        anchor_state = "not_checked"
        try:
            candidate_bytes, expected_anchor = self._prepare_recovery_candidate(
                plan=plan,
                quarantine=quarantine,
                candidate=candidate,
                candidate_anchor=candidate_anchor,
                approval_id=approval_id,
                reason=reason,
                issue_codes=issue_codes,
                verification=verification,
            )
            current = self.path.read_bytes() if self.path.exists() else b""
            original_bytes = self._original_bytes(plan)
            if current == original_bytes:
                phase = "log_publication"
                staged_path = self.path.with_name(f".{self.path.name}.recovery-{uuid.uuid4().hex}")
                descriptor = os.open(staged_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    _write_all(descriptor, candidate_bytes)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.replace(staged_path, self.path)
                os.chmod(self.path, 0o600)
                _fsync_directory(self.path.parent)
            elif current != candidate_bytes:
                raise AprilError(
                    "AUDIT_RECOVERY_CONCURRENT_CHANGE",
                    "Audit log changed during recovery publication.",
                    409,
                )
            phase = "journal_log_publication"
            events = self._recovery_events_for(str(plan["plan_id"]))
            if not any(event["event_type"] == "log_published" for event in events):
                self._append_recovery_event(
                    "log_published",
                    {
                        "plan_id": plan["plan_id"],
                        "approval_id": approval_id,
                        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                    },
                )
            phase = "anchor_publication"
            anchor_state = "checking"
            current_anchor = self.anchor.get()
            if current_anchor != expected_anchor:
                if current_anchor not in {plan.get("original_anchor"), None}:
                    raise AprilError(
                        "AUDIT_RECOVERY_CONCURRENT_CHANGE",
                        "Protected anchor changed during recovery publication.",
                        409,
                    )
                self.anchor.set(expected_anchor)
                anchor_state = "updated"
            else:
                anchor_state = "already_published"
            phase = "journal_anchor_publication"
            events = self._recovery_events_for(str(plan["plan_id"]))
            if not any(event["event_type"] == "anchor_published" for event in events):
                self._append_recovery_event(
                    "anchor_published",
                    {
                        "plan_id": plan["plan_id"],
                        "approval_id": approval_id,
                        "anchor_sha256": hashlib.sha256(expected_anchor.encode()).hexdigest(),
                    },
                )
            phase = "verification"
            final_result = verify_audit_chain(self.path, anchor=self.anchor)
            if not final_result.valid:
                raise AprilError(
                    "AUDIT_RECOVERY_FAILED",
                    "Recovered audit chain did not verify with its protected anchor.",
                    500,
                )
            phase = "journal_finalization"
            events = self._recovery_events_for(str(plan["plan_id"]))
            if not any(event["event_type"] == "completed" for event in events):
                self._append_recovery_event(
                    "completed",
                    {
                        "plan_id": plan["plan_id"],
                        "approval_id": approval_id,
                        "candidate_sha256": plan["candidate_sha256"],
                    },
                )
            return AuditRecoveryPlan(
                "recovered",
                issue_codes,
                verification.record_count,
                str(plan["quarantine_directory"]),
                plan["original_log_sha256"],
                plan["plan_id"],
                plan["plan_digest"],
                plan["canonical_target"],
                plan["original_anchor_sha256"],
                plan["expires_at"],
                approval_id,
                "completed",
                self._recovery_log_changed(plan),
                "verified",
                self._recovery_resume_command(plan, approval_id),
            )
        except AprilError as exc:
            log_changed = self._recovery_log_changed(plan)
            details = self._recovery_state(
                plan,
                approval_id=approval_id,
                phase=phase,
                log_changed=log_changed,
                anchor_state=anchor_state,
            )
            details.update(exc.details)
            raise AprilError(exc.code, exc.message, exc.status_code, details) from exc
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AprilError(
                "AUDIT_RECOVERY_INCOMPLETE",
                "Recovery publication stopped before the operation was complete.",
                409,
                self._recovery_state(
                    plan,
                    approval_id=approval_id,
                    phase=phase,
                    log_changed=self._recovery_log_changed(plan),
                    anchor_state=anchor_state,
                ),
            ) from exc

    def _prepare_recovery_candidate(
        self,
        *,
        plan: dict[str, Any],
        quarantine: Path,
        candidate: Path,
        candidate_anchor: Path,
        approval_id: str,
        reason: str,
        issue_codes: tuple[str, ...],
        verification: AuditVerification,
    ) -> tuple[bytes, str]:
        """Validate or finish staging; a candidate alone is never sufficient."""
        if not candidate.exists() and "candidate_sha256" not in plan:
            staged_logger = AuditLogger(candidate, anchor=MemoryAuditAnchor())
            staged_logger.write(
                {
                    "event_type": "audit_chain_recovery",
                    "reason": reason[:240],
                    "issue_codes": list(issue_codes),
                    "quarantined_artifact_basename": self.path.name,
                    "quarantined_artifact_sha256": plan["original_log_sha256"],
                    "previous_terminal_sequence": verification.terminal_sequence,
                    "previous_terminal_hash": verification.terminal_hash,
                    "approval_id": approval_id,
                    "recovery_plan_id": plan["plan_id"],
                    "recovery_plan_digest": plan["plan_digest"],
                    "recovery_approval_provenance": {
                        "journal": str(self.recovery_journal),
                        "approval_id": approval_id,
                    },
                }
            )
        if candidate.exists():
            candidate_bytes = candidate.read_bytes()
        else:
            candidate_bytes = self.path.read_bytes() if self.path.exists() else b""
        candidate_sha256, expected_anchor = self._validate_recovery_candidate(
            candidate_bytes, plan=plan, approval_id=approval_id
        )
        if plan.get("candidate_sha256") not in {None, candidate_sha256}:
            raise AprilError(
                "AUDIT_RECOVERY_CONCURRENT_CHANGE",
                "Recovery candidate changed after it was validated.",
                409,
            )
        if candidate_sha256 != hashlib.sha256(candidate_bytes).hexdigest():
            raise AprilError(
                "AUDIT_RECOVERY_INCOMPLETE", "Recovery candidate checksum is invalid.", 409
            )
        if candidate_anchor.exists():
            try:
                anchor_payload = json.loads(candidate_anchor.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AprilError(
                    "AUDIT_RECOVERY_INCOMPLETE",
                    "Recovery candidate anchor metadata is incomplete or invalid.",
                    409,
                ) from exc
            if not isinstance(anchor_payload, dict) or (
                anchor_payload.get("value") != expected_anchor
            ):
                raise AprilError(
                    "AUDIT_RECOVERY_INCOMPLETE",
                    "Recovery candidate anchor metadata does not match its validated candidate.",
                    409,
                )
        else:
            _write_private_json(candidate_anchor, {"value": expected_anchor})
        plan_updates = dict(plan)
        plan_updates["candidate_sha256"] = candidate_sha256
        plan_updates["candidate_anchor"] = expected_anchor
        if (
            plan.get("candidate_sha256") != candidate_sha256
            or plan.get("candidate_anchor") != expected_anchor
        ):
            _write_private_json(quarantine / "manifest.json", plan_updates)
        plan.update(plan_updates)
        return candidate_bytes, expected_anchor

    def _validate_recovery_candidate(
        self, candidate_bytes: bytes, *, plan: dict[str, Any], approval_id: str
    ) -> tuple[str, str]:
        result = _verify_bytes(candidate_bytes, anchor=None)
        if not result.valid or result.record_count != 1 or result.terminal_hash is None:
            raise AprilError(
                "AUDIT_RECOVERY_INCOMPLETE",
                "Recovery candidate is not a complete valid audit chain.",
                409,
            )
        try:
            records = [json.loads(line) for line in candidate_bytes.decode("utf-8").splitlines()]
            payload = records[0]["payload"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise AprilError(
                "AUDIT_RECOVERY_INCOMPLETE",
                "Recovery candidate provenance is invalid.",
                409,
            ) from exc
        if (
            records[0].get("event_type") != "audit_chain_recovery"
            or not isinstance(payload, dict)
            or payload.get("recovery_plan_id") != plan.get("plan_id")
            or payload.get("recovery_plan_digest") != plan.get("plan_digest")
            or payload.get("approval_id") != approval_id
            or payload.get("quarantined_artifact_sha256") != plan.get("original_log_sha256")
        ):
            raise AprilError(
                "AUDIT_RECOVERY_INCOMPLETE",
                "Recovery candidate is not bound to the approved immutable plan.",
                409,
            )
        return hashlib.sha256(candidate_bytes).hexdigest(), _encode_anchor(
            result.terminal_sequence or 0, result.terminal_hash
        )

    def _recovery_log_changed(self, plan: dict[str, Any]) -> bool:
        current = self.path.read_bytes() if self.path.exists() else b""
        return hashlib.sha256(current).hexdigest() != plan.get("original_log_sha256")

    def _recovery_resume_command(self, plan: dict[str, Any], approval_id: str) -> str:
        return (
            "run april audit recover --apply "
            f"--plan-id {plan['plan_id']} --approval-id {approval_id} --json"
        )

    def _recovery_state(
        self,
        plan: dict[str, Any],
        *,
        approval_id: str,
        phase: str,
        log_changed: bool,
        anchor_state: str,
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "log_changed": log_changed,
            "anchor_state": anchor_state,
            "plan_id": plan.get("plan_id"),
            "approval_id": approval_id,
            "resume_command": self._recovery_resume_command(plan, approval_id),
        }

    def _original_bytes(self, plan: dict[str, Any]) -> bytes:
        original = self.recovery_root / str(plan["quarantine_directory"]) / self.path.name
        return original.read_bytes() if original.exists() else b""


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_logger_for_settings(
    settings: AprilSettings,
    *,
    credential_store: CredentialStore | None = None,
) -> AuditLogger:
    """Build the production logger with its protected credential-store anchor."""
    if credential_store is None and (
        settings.environment == "production" or settings.security.credential_store != "auto"
    ):
        from april_common.credentials import select_credential_store

        configured_path = settings.security.credential_file_path
        credential_store = select_credential_store(
            backend=settings.security.credential_store,
            environment=settings.environment,
            repository_root=settings.home,
            file_path=(
                settings.resolve_path(configured_path) if configured_path is not None else None
            ),
        )
    anchor: AuditAnchor | None = (
        CredentialAuditAnchor(credential_store) if credential_store is not None else None
    )
    return AuditLogger(settings.audit_path, anchor=anchor)


def audit_startup_decision(
    settings: AprilSettings,
    *,
    credential_store: CredentialStore | None = None,
    audit: AuditLogger | None = None,
    logger_factory: Callable[..., AuditLogger] | None = None,
) -> AuditStartupDecision:
    """Perform the authoritative full audit check required before startup.

    This intentionally does not use readiness's bounded large-log shortcut and
    never repairs or appends to the chain. ``logger_factory`` is injectable so
    readiness/preflight tests retain their existing synthetic stores.
    """
    try:
        logger = audit
        if logger is None:
            factory = logger_factory or audit_logger_for_settings
            logger = factory(settings, credential_store=credential_store)
        result = logger.verify()
    except (CredentialStoreError, OSError, RuntimeError):
        return AuditStartupDecision(
            accepted=False,
            status="unavailable",
            issue_codes=("verification_unavailable",),
            issue_lines=("verification_unavailable",),
            record_count=0,
            reason="Audit verification could not be completed.",
            next_commands=("run april audit verify --json",),
        )
    except AprilError as exc:
        if exc.code not in {"AUDIT_ANCHOR_FAILED", "AUDIT_VERIFICATION_UNAVAILABLE"}:
            raise
        return AuditStartupDecision(
            accepted=False,
            status="unavailable",
            issue_codes=("verification_unavailable",),
            issue_lines=("verification_unavailable",),
            record_count=0,
            reason="Audit verification could not be completed.",
            next_commands=("run april audit verify --json",),
        )

    issue_lines = tuple(
        f"{issue.code}(line {issue.line})" if issue.line is not None else issue.code
        for issue in result.issues[:12]
    )
    issue_codes = tuple(sorted({issue.code for issue in result.issues}))[:12]
    accepted = result.status in {"valid", "anchor_lagged"} and result.valid
    if accepted:
        reason = f"Audit chain accepted ({result.status}); records={result.record_count}."
        next_commands: tuple[str, ...] = ()
    elif result.status == "corrupt" or result.corrupt:
        reason = "Audit chain is corrupt; historical records remain unverified."
        next_commands = (
            "run april audit verify --json",
            'run april audit recover --reason "owner-reviewed recovery"',
        )
    elif result.status == "unavailable":
        reason = "Audit verification is unavailable; storage or credentials could not be verified."
        next_commands = ("run april audit verify --json",)
    else:
        reason = f"Audit status {result.status!r} is not accepted for operational startup."
        next_commands = ("run april audit verify --json",)
    return AuditStartupDecision(
        accepted=accepted,
        status=result.status,
        issue_codes=issue_codes,
        issue_lines=issue_lines,
        record_count=result.record_count,
        reason=reason,
        next_commands=next_commands,
    )


def verify_audit_chain(path: Path, *, anchor: AuditAnchor | None = None) -> AuditVerification:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    except OSError:
        return AuditVerification(
            status="unavailable",
            valid=False,
            corrupt=False,
            anchor_lagged=False,
            record_count=0,
            terminal_sequence=None,
            terminal_hash=None,
            issues=(AuditIssue("audit_unreadable", None, "Audit log could not be read."),),
        )
    try:
        return _verify_bytes(data, anchor=anchor)
    except (CredentialStoreError, OSError):
        return AuditVerification(
            status="unavailable",
            valid=False,
            corrupt=False,
            anchor_lagged=False,
            record_count=0,
            terminal_sequence=None,
            terminal_hash=None,
            issues=(
                AuditIssue(
                    "anchor_unavailable",
                    None,
                    "Protected audit anchor could not be read.",
                ),
            ),
        )
    except AprilError as exc:
        if exc.code != "AUDIT_ANCHOR_FAILED":
            raise
        return AuditVerification(
            status="unavailable",
            valid=False,
            corrupt=False,
            anchor_lagged=False,
            record_count=0,
            terminal_sequence=None,
            terminal_hash=None,
            issues=(
                AuditIssue(
                    "anchor_unavailable",
                    None,
                    "Protected audit anchor could not be read.",
                ),
            ),
        )


def _verify_bytes(data: bytes, *, anchor: AuditAnchor | None) -> AuditVerification:
    issues: list[AuditIssue] = []
    records: list[tuple[int, dict[str, Any]]] = []
    raw_lines = data.splitlines(keepends=True)
    if data and not data.endswith(b"\n"):
        issues.append(
            AuditIssue("malformed_json", len(raw_lines), "Final audit line is incomplete.")
        )
    for line_number, raw_line in enumerate(raw_lines, start=1):
        try:
            decoded = raw_line.decode("utf-8")
            record = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            issues.append(AuditIssue("malformed_json", line_number, "Malformed JSON record."))
            continue
        if not isinstance(record, dict):
            issues.append(AuditIssue("invalid_schema", line_number, "Record is not an object."))
            continue
        records.append((line_number, record))

    expected_sequence = 1
    previous_hash = GENESIS_HASH
    terminal_sequence: int | None = None
    terminal_hash: str | None = None
    for record_index, (line_number, record) in enumerate(records, start=1):
        if not _valid_record_schema(record):
            issues.append(AuditIssue("invalid_schema", line_number, "Record schema is invalid."))
            continue
        sequence = int(record["sequence"])
        if record_index == 1 and (sequence != 1 or record["previous_hash"] != GENESIS_HASH):
            issues.append(
                AuditIssue("missing_genesis", line_number, "Audit genesis state is missing.")
            )
        if sequence != expected_sequence:
            if sequence == expected_sequence - 1:
                code = "duplicate_sequence"
            elif sequence < expected_sequence:
                code = "reordered_record"
            else:
                code = "sequence_gap"
            issues.append(AuditIssue(code, line_number, "Audit sequence is not monotonic."))
        if record["previous_hash"] != previous_hash:
            issues.append(
                AuditIssue(
                    "incorrect_previous_hash",
                    line_number,
                    "Previous-record hash does not match.",
                )
            )
        supplied_hash = str(record["record_hash"])
        unsigned = dict(record)
        unsigned.pop("record_hash", None)
        if not secrets_compare_hash(supplied_hash, _record_hash(unsigned)):
            issues.append(
                AuditIssue("incorrect_event_hash", line_number, "Record hash does not match.")
            )
        expected_sequence = sequence + 1
        previous_hash = supplied_hash
        terminal_sequence = sequence
        terminal_hash = supplied_hash

    anchor_lagged = False
    if anchor is not None:
        anchor_value: str | None
        insecure_anchor = False
        try:
            anchor_value = anchor.get()
        except AprilError as exc:
            if exc.code == "AUDIT_ANCHOR_INVALID":
                issues.append(
                    AuditIssue("invalid_terminal_anchor", None, "Protected anchor is invalid.")
                )
                anchor_value = None
            elif exc.code == "AUDIT_ANCHOR_INSECURE":
                issues.append(
                    AuditIssue(
                        "insecure_anchor_permissions",
                        None,
                        "Protected audit anchor permissions are insecure.",
                    )
                )
                insecure_anchor = True
                anchor_value = None
            else:
                raise
        try:
            protected = _decode_anchor(anchor_value)
        except (TypeError, ValueError):
            issues.append(
                AuditIssue("invalid_terminal_anchor", None, "Protected anchor is invalid.")
            )
            protected = None
        if not insecure_anchor:
            if protected is None:
                if terminal_sequence == 1 and records:
                    anchor_lagged = True
                elif records:
                    issues.append(
                        AuditIssue(
                            "missing_terminal_anchor",
                            None,
                            "Protected terminal anchor is missing.",
                        )
                    )
            elif terminal_sequence is None:
                issues.append(
                    AuditIssue(
                        "terminal_truncation",
                        None,
                        "Audit records are missing compared with the protected anchor.",
                    )
                )
            else:
                anchor_sequence, anchor_hash = protected
                if anchor_sequence == terminal_sequence and anchor_hash == terminal_hash:
                    pass
                elif (
                    anchor_sequence == terminal_sequence - 1
                    and records
                    and records[-1][1].get("previous_hash") == anchor_hash
                ):
                    anchor_lagged = True
                elif anchor_sequence > terminal_sequence:
                    issues.append(
                        AuditIssue(
                            "terminal_truncation",
                            None,
                            "Audit records are missing compared with the protected anchor.",
                        )
                    )
                else:
                    issues.append(
                        AuditIssue(
                            "terminal_anchor_mismatch",
                            None,
                            "Protected anchor does not match the terminal record.",
                        )
                    )

    corrupt = bool(issues)
    if anchor_lagged and not issues:
        status = "anchor_lagged"
    elif corrupt:
        status = "corrupt"
    else:
        status = "valid"
    return AuditVerification(
        status=status,
        valid=not corrupt,
        corrupt=corrupt,
        anchor_lagged=anchor_lagged,
        record_count=len(records),
        terminal_sequence=terminal_sequence,
        terminal_hash=terminal_hash,
        issues=tuple(issues),
    )


def _valid_record_schema(record: dict[str, Any]) -> bool:
    expected = {
        "schema_version",
        "sequence",
        "event_id",
        "timestamp",
        "event_type",
        "payload",
        "previous_hash",
        "record_hash",
    }
    keys = set(record)
    if not expected.issubset(keys) or not (keys - expected).issubset(_COMPATIBILITY_FIELDS):
        return False
    return bool(
        record["schema_version"] == AUDIT_SCHEMA_VERSION
        and isinstance(record["sequence"], int)
        and not isinstance(record["sequence"], bool)
        and record["sequence"] > 0
        and isinstance(record["event_id"], str)
        and isinstance(record["timestamp"], str)
        and _is_utc_timestamp(record["timestamp"])
        and isinstance(record["event_type"], str)
        and record["event_type"]
        and isinstance(record["payload"], dict)
        and _is_hash(record["previous_hash"])
        and _is_hash(record["record_hash"])
    )


def _is_utc_timestamp(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return offset is not None and offset.total_seconds() == 0


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record_without_hash).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _encode_anchor(sequence: int, record_hash: str) -> str:
    return _canonical_json(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "sequence": sequence,
            "record_hash": record_hash,
        }
    )


def _decode_anchor(value: str | None) -> tuple[int, str] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != AUDIT_SCHEMA_VERSION
        or not isinstance(payload.get("sequence"), int)
        or payload["sequence"] < 1
        or not _is_hash(payload.get("record_hash"))
    ):
        raise ValueError("Invalid protected audit anchor.")
    return int(payload["sequence"]), str(payload["record_hash"])


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def secrets_compare_hash(left: str, right: str) -> bool:
    import secrets

    return secrets.compare_digest(left, right)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short audit write")
        view = view[written:]


def _flock(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _funlock(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
