from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from april_common.credentials import CredentialKey, CredentialStore
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
        except (OSError, UnicodeDecodeError) as exc:
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
            if verification.corrupt:
                raise AprilError(
                    "AUDIT_CHAIN_CORRUPT",
                    "Audit chain verification failed; append refused.",
                    500,
                    {"issues": [issue.code for issue in verification.issues]},
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


def verify_audit_chain(path: Path, *, anchor: AuditAnchor | None = None) -> AuditVerification:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    except OSError:
        return AuditVerification(
            status="corrupt",
            valid=False,
            corrupt=True,
            anchor_lagged=False,
            record_count=0,
            terminal_sequence=None,
            terminal_hash=None,
            issues=(AuditIssue("audit_unreadable", None, "Audit log could not be read."),),
        )
    result = _verify_bytes(data, anchor=anchor)
    return result


def _verify_bytes(data: bytes, *, anchor: AuditAnchor | None) -> AuditVerification:
    issues: list[AuditIssue] = []
    records: list[dict[str, Any]] = []
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
        records.append(record)

    expected_sequence = 1
    previous_hash = GENESIS_HASH
    terminal_sequence: int | None = None
    terminal_hash: str | None = None
    for line_number, record in enumerate(records, start=1):
        if not _valid_record_schema(record):
            issues.append(AuditIssue("invalid_schema", line_number, "Record schema is invalid."))
            continue
        sequence = int(record["sequence"])
        if line_number == 1 and (sequence != 1 or record["previous_hash"] != GENESIS_HASH):
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
        try:
            protected = _decode_anchor(anchor.get())
        except (AprilError, ValueError):
            issues.append(
                AuditIssue("invalid_terminal_anchor", None, "Protected anchor is invalid.")
            )
            protected = None
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
                and records[-1].get("previous_hash") == anchor_hash
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
