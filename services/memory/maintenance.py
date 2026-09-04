from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from april_common.audit import AuditLogger
from april_common.time import utc_now_iso
from services.memory.database import SQLITE_BUSY_TIMEOUT_MS, sqlite_write_fence
from services.memory.migrations import SCHEMA_VERSION

BACKUP_FORMAT_VERSION = 1
BACKUP_DATABASE_NAME = "database.sqlite3"
BACKUP_MANIFEST_NAME = "manifest.json"
LAST_BACKUP_METADATA = Path("data/backups/last-success.json")


class DatabaseMaintenanceError(RuntimeError):
    pass


class BackupCancelled(DatabaseMaintenanceError):
    pass


@dataclass(frozen=True, slots=True)
class DatabaseCheckResult:
    ok: bool
    full: bool
    available: bool
    quick_check: str
    foreign_key_consistent: bool
    foreign_key_violations: int
    schema_version: int | None
    expected_schema_version: int
    migration_consistent: bool
    journal_mode: str
    synchronous_mode: int | None
    busy_timeout_ms: int | None
    wal_exists: bool
    wal_size: int | None
    shm_exists: bool
    shm_size: int | None
    integrity_check: str | None
    last_successful_backup: dict[str, Any] | None
    checked_at: str
    failures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MigrationStatus = Literal[
    "current",
    "migration_pending",
    "migration_ahead",
    "migration_inconsistent",
    "database_corrupt",
]


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    status: MigrationStatus
    current_schema_version: int | None
    expected_schema_version: int
    failures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackupManifest:
    format_version: int
    creation_timestamp: str
    source_database_identifier: str
    schema_version: int
    database_sha256: str
    size: int
    integrity_check_result: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackupResult:
    output: Path
    manifest: BackupManifest


@dataclass(frozen=True, slots=True)
class RestoreResult:
    restored: bool
    schema_version: int
    rollback_backup: Path
    quick_check: str
    integrity_check: str


ConnectionFactory = Callable[[Path], sqlite3.Connection]
Validator = Callable[[Path], DatabaseCheckResult]


def check_database(
    database_path: Path,
    *,
    home: Path | None = None,
    full: bool = False,
    connection_factory: ConnectionFactory | None = None,
) -> DatabaseCheckResult:
    path = database_path.expanduser().resolve(strict=False)
    failures: list[str] = []
    if not path.is_file():
        return DatabaseCheckResult(
            ok=False,
            full=full,
            available=False,
            quick_check="not_run",
            foreign_key_consistent=False,
            foreign_key_violations=0,
            schema_version=None,
            expected_schema_version=SCHEMA_VERSION,
            migration_consistent=False,
            journal_mode="unavailable",
            synchronous_mode=None,
            busy_timeout_ms=None,
            wal_exists=Path(f"{path}-wal").exists(),
            wal_size=_safe_size(Path(f"{path}-wal")),
            shm_exists=Path(f"{path}-shm").exists(),
            shm_size=_safe_size(Path(f"{path}-shm")),
            integrity_check=None,
            last_successful_backup=_last_backup(home),
            checked_at=utc_now_iso(),
            failures=("database_unavailable",),
        )

    factory = connection_factory or _open_check_connection
    connection: sqlite3.Connection | None = None
    quick = "not_run"
    foreign_count = 0
    schema_version: int | None = None
    journal_mode = "unknown"
    synchronous_mode: int | None = None
    busy_timeout: int | None = None
    integrity: str | None = None
    try:
        connection = factory(path)
        quick_rows = connection.execute("PRAGMA quick_check").fetchall()
        quick = _pragma_result(quick_rows)
        if quick != "ok":
            failures.append("quick_check_failed")
        foreign_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        foreign_count = len(foreign_rows)
        if foreign_count:
            failures.append("foreign_key_violations")
        journal_row = connection.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(journal_row[0]).lower() if journal_row else "unknown"
        if journal_mode != "wal":
            failures.append("unexpected_journal_mode")
        synchronous_row = connection.execute("PRAGMA synchronous").fetchone()
        synchronous_mode = int(synchronous_row[0]) if synchronous_row else None
        if synchronous_mode != 1:
            failures.append("unexpected_synchronous_mode")
        timeout_row = connection.execute("PRAGMA busy_timeout").fetchone()
        busy_timeout = int(timeout_row[0]) if timeout_row else None
        if busy_timeout != SQLITE_BUSY_TIMEOUT_MS:
            failures.append("unexpected_busy_timeout")
        schema_version = _read_schema_version(connection)
        if schema_version != SCHEMA_VERSION:
            failures.append("migration_mismatch")
        if full:
            integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
            integrity = _pragma_result(integrity_rows)
            if integrity != "ok":
                failures.append("integrity_check_failed")
    except sqlite3.Error:
        failures.append("database_open_or_pragma_failed")
    finally:
        if connection is not None:
            connection.close()
    failures = list(dict.fromkeys(failures))
    return DatabaseCheckResult(
        ok=not failures,
        full=full,
        available=True,
        quick_check=quick,
        foreign_key_consistent=foreign_count == 0,
        foreign_key_violations=foreign_count,
        schema_version=schema_version,
        expected_schema_version=SCHEMA_VERSION,
        migration_consistent=schema_version == SCHEMA_VERSION,
        journal_mode=journal_mode,
        synchronous_mode=synchronous_mode,
        busy_timeout_ms=busy_timeout,
        wal_exists=Path(f"{path}-wal").exists(),
        wal_size=_safe_size(Path(f"{path}-wal")),
        shm_exists=Path(f"{path}-shm").exists(),
        shm_size=_safe_size(Path(f"{path}-shm")),
        integrity_check=integrity,
        last_successful_backup=_last_backup(home),
        checked_at=utc_now_iso(),
        failures=tuple(failures),
    )


def migration_plan(database_path: Path, *, home: Path | None = None) -> MigrationPlan:
    """Return a read-only, bounded description of the database migration state."""
    check = check_database(database_path, home=home, full=True)
    schema = check.schema_version
    if not check.available or check.quick_check != "ok" or not check.foreign_key_consistent:
        return MigrationPlan(
            status="database_corrupt",
            current_schema_version=schema,
            expected_schema_version=SCHEMA_VERSION,
            failures=check.failures or ("database_corrupt",),
        )
    if schema is None:
        return MigrationPlan(
            status="migration_inconsistent",
            current_schema_version=None,
            expected_schema_version=SCHEMA_VERSION,
            failures=("migration_history_missing",),
        )
    if schema > SCHEMA_VERSION:
        return MigrationPlan(
            status="migration_ahead",
            current_schema_version=schema,
            expected_schema_version=SCHEMA_VERSION,
            failures=("schema_ahead",),
        )
    if schema < SCHEMA_VERSION:
        return MigrationPlan(
            status="migration_pending",
            current_schema_version=schema,
            expected_schema_version=SCHEMA_VERSION,
            failures=("migration_pending",),
        )
    return MigrationPlan(
        status="current",
        current_schema_version=schema,
        expected_schema_version=SCHEMA_VERSION,
    )


def create_backup(
    database_path: Path,
    output: Path,
    *,
    home: Path,
    audit: AuditLogger | None = None,
    cancellation_event: threading.Event | None = None,
) -> BackupResult:
    source = database_path.expanduser().resolve(strict=False)
    target = output.expanduser().resolve(strict=False)
    try:
        if audit is not None:
            audit.write(
                {
                    "event_type": "database_backup_started",
                    "output_basename": target.name,
                }
            )
        if target.exists():
            raise DatabaseMaintenanceError("Backup output already exists.")
        with sqlite_write_fence(source):
            result = _create_backup_under_fence(
                source,
                target,
                cancellation_event=cancellation_event,
            )
        _write_last_backup(home, result)
        if audit is not None:
            audit.write(
                {
                    "event_type": "database_backup_succeeded",
                    "output_basename": target.name,
                    "schema_version": result.manifest.schema_version,
                    "size": result.manifest.size,
                }
            )
        return result
    except BaseException as exc:
        if audit is not None:
            audit.write(
                {
                    "event_type": "database_backup_failed",
                    "output_basename": target.name,
                    "reason_code": type(exc).__name__,
                }
            )
        raise


def restore_backup(
    database_path: Path,
    input_path: Path,
    *,
    home: Path,
    services_running: bool,
    audit: AuditLogger | None = None,
    validator: Validator | None = None,
) -> RestoreResult:
    active = database_path.expanduser().resolve(strict=False)
    package = input_path.expanduser().resolve(strict=False)
    try:
        if audit is not None:
            audit.write(
                {
                    "event_type": "database_restore_started",
                    "input_basename": package.name,
                }
            )
        if services_running:
            raise DatabaseMaintenanceError(
                "APRIL services are running; stop them or use the safe stop-services option."
            )
        candidate, manifest = validate_backup_package(package)
        candidate_check = check_database(candidate, home=home, full=True)
        _require_restore_candidate(candidate_check, manifest)
        validate = validator or (lambda path: check_database(path, home=home, full=True))
        rollback_root = home / "data" / "backups"
        rollback_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        rollback_target = Path(tempfile.mkdtemp(prefix="restore-rollback-", dir=rollback_root))
        rollback_target.rmdir()
        with sqlite_write_fence(active):
            if active.exists():
                rollback_result = _create_backup_under_fence(active, rollback_target)
            else:
                raise DatabaseMaintenanceError("Active database is unavailable.")
            _replace_database_from_candidate(candidate, active)
            try:
                post = validate(active)
            except BaseException as validation_error:
                _restore_rollback(rollback_result, active)
                raise DatabaseMaintenanceError(
                    "Restored database could not be reopened; rollback completed."
                ) from validation_error
            if not post.ok or post.integrity_check != "ok":
                _restore_rollback(rollback_result, active)
                raise DatabaseMaintenanceError(
                    "Restored database failed post-replacement validation; rollback completed."
                )
            _remove_sqlite_sidecars(active)
        if audit is not None:
            audit.write(
                {
                    "event_type": "database_restore_succeeded",
                    "input_basename": package.name,
                    "schema_version": manifest.schema_version,
                    "rollback_basename": rollback_target.name,
                }
            )
        return RestoreResult(
            restored=True,
            schema_version=manifest.schema_version,
            rollback_backup=rollback_target,
            quick_check="ok",
            integrity_check="ok",
        )
    except BaseException as exc:
        if audit is not None:
            audit.write(
                {
                    "event_type": "database_restore_failed",
                    "input_basename": package.name,
                    "reason_code": type(exc).__name__,
                }
            )
        raise


def validate_backup_package(package: Path) -> tuple[Path, BackupManifest]:
    if not package.is_dir():
        raise DatabaseMaintenanceError("Backup input must be an APRIL backup directory.")
    mode = stat.S_IMODE(package.stat().st_mode)
    if mode & 0o077:
        raise DatabaseMaintenanceError("Backup directory permissions are not owner-only.")
    database = package / BACKUP_DATABASE_NAME
    manifest_path = package / BACKUP_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BackupManifest(**payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise DatabaseMaintenanceError("Backup manifest is missing or malformed.") from exc
    if manifest.format_version != BACKUP_FORMAT_VERSION:
        raise DatabaseMaintenanceError("Backup format version is unsupported.")
    if manifest.schema_version > SCHEMA_VERSION:
        raise DatabaseMaintenanceError("Backup uses an unsupported future schema version.")
    if not database.is_file():
        raise DatabaseMaintenanceError("Backup database file is missing.")
    if _sha256_file(database) != manifest.database_sha256:
        raise DatabaseMaintenanceError("Backup database SHA-256 does not match its manifest.")
    if database.stat().st_size != manifest.size:
        raise DatabaseMaintenanceError("Backup database size does not match its manifest.")
    return database, manifest


def _create_backup_under_fence(
    source: Path,
    target: Path,
    *,
    cancellation_event: threading.Event | None = None,
) -> BackupResult:
    if not source.is_file():
        raise DatabaseMaintenanceError("Source database is unavailable.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    os.chmod(staging, 0o700)
    destination = staging / BACKUP_DATABASE_NAME
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        if _cancelled(cancellation_event):
            raise BackupCancelled("Backup was cancelled.")
        source_connection = sqlite3.connect(
            f"file:{source.as_posix()}?mode=ro",
            uri=True,
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
        )
        destination_connection = sqlite3.connect(destination)

        def progress(_status: int, _remaining: int, _total: int) -> None:
            if _cancelled(cancellation_event):
                raise BackupCancelled("Backup was cancelled.")

        source_connection.backup(destination_connection, pages=128, progress=progress)
        destination_connection.commit()
        destination_connection.close()
        destination_connection = None
        source_connection.close()
        source_connection = None
        os.chmod(destination, 0o600)
        if _cancelled(cancellation_event):
            raise BackupCancelled("Backup was cancelled.")
        check = check_database(destination, full=True)
        if (
            check.quick_check != "ok"
            or check.integrity_check != "ok"
            or not check.foreign_key_consistent
        ):
            raise DatabaseMaintenanceError("Backup snapshot failed integrity validation.")
        schema_version = check.schema_version
        if schema_version is None:
            raise DatabaseMaintenanceError("Backup snapshot has no migration state.")
        manifest = BackupManifest(
            format_version=BACKUP_FORMAT_VERSION,
            creation_timestamp=utc_now_iso(),
            source_database_identifier=hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
            schema_version=schema_version,
            database_sha256=_sha256_file(destination),
            size=destination.stat().st_size,
            integrity_check_result=check.integrity_check or "not_run",
        )
        manifest_path = staging / BACKUP_MANIFEST_NAME
        _write_json_file(manifest_path, manifest.to_dict(), mode=0o600)
        _fsync_file(destination)
        _fsync_directory(staging)
        if _cancelled(cancellation_event):
            raise BackupCancelled("Backup was cancelled.")
        os.replace(staging, target)
        _fsync_directory(target.parent)
        return BackupResult(output=target, manifest=manifest)
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()
        if staging.exists():
            shutil.rmtree(staging)


def _replace_database_from_candidate(candidate: Path, active: Path) -> None:
    active.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{active.name}.restore-",
        dir=active.parent,
    )
    temporary = Path(temporary_name)
    try:
        with candidate.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, 0o600)
        _remove_sqlite_sidecars(active)
        os.replace(temporary, active)
        os.chmod(active, 0o600)
        _fsync_directory(active.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _restore_rollback(rollback: BackupResult, active: Path) -> None:
    _replace_database_from_candidate(
        rollback.output / BACKUP_DATABASE_NAME,
        active,
    )
    rollback_check = check_database(active, full=True)
    if not rollback_check.ok:
        raise DatabaseMaintenanceError(
            "Restore validation failed and rollback revalidation failed."
        )


def _require_restore_candidate(
    check: DatabaseCheckResult,
    manifest: BackupManifest,
) -> None:
    if check.quick_check != "ok":
        raise DatabaseMaintenanceError("Backup candidate failed quick_check.")
    if check.integrity_check != "ok":
        raise DatabaseMaintenanceError("Backup candidate failed integrity_check.")
    if not check.foreign_key_consistent:
        raise DatabaseMaintenanceError("Backup candidate has foreign-key violations.")
    if check.schema_version != manifest.schema_version:
        raise DatabaseMaintenanceError("Backup manifest schema does not match the database.")
    if check.schema_version != SCHEMA_VERSION:
        raise DatabaseMaintenanceError("Backup schema is incompatible with this APRIL version.")


def _open_check_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    return connection


def _read_schema_version(connection: sqlite3.Connection) -> int | None:
    try:
        row = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row and row[0] is not None else None


def _pragma_result(rows: list[tuple[Any, ...]]) -> str:
    values = [str(row[0]) for row in rows if row]
    return "ok" if values == ["ok"] else "; ".join(values) or "no_result"


def _last_backup(home: Path | None) -> dict[str, Any] | None:
    if home is None:
        return None
    path = home.expanduser().resolve() / LAST_BACKUP_METADATA
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    allowed = {"creation_timestamp", "output_basename", "schema_version", "size"}
    return {key: payload[key] for key in allowed if key in payload}


def _write_last_backup(home: Path, result: BackupResult) -> None:
    path = home.expanduser().resolve() / LAST_BACKUP_METADATA
    _write_json_file(
        path,
        {
            "creation_timestamp": result.manifest.creation_timestamp,
            "output_basename": result.output.name,
            "schema_version": result.manifest.schema_version,
            "size": result.manifest.size,
        },
        mode=0o600,
    )


def _write_json_file(path: Path, payload: dict[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cancelled(event: threading.Event | None) -> bool:
    return event is not None and event.is_set()


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
