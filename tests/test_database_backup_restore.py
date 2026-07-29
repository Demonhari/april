from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from services.memory.database import Database, sqlite_write_transaction
from services.memory.maintenance import (
    BACKUP_DATABASE_NAME,
    BACKUP_MANIFEST_NAME,
    BackupCancelled,
    DatabaseMaintenanceError,
    check_database,
    create_backup,
    restore_backup,
    validate_backup_package,
)
from services.memory.migrations import SCHEMA_VERSION, run_migrations


def _prepare_database(path: Path) -> None:
    async def prepare() -> None:
        database = Database(path)
        await database.connect()
        try:
            await run_migrations(database)
            await database.execute(
                "INSERT INTO users(id, name, created_at) VALUES('u1', 'before', datetime('now'))"
            )
        finally:
            await database.close()

    asyncio.run(prepare())


def test_database_check_valid_and_migration_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    valid = check_database(path, home=tmp_path)
    assert valid.ok
    assert valid.quick_check == "ok"
    assert valid.foreign_key_consistent
    assert valid.journal_mode == "wal"
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM schema_migrations")
    mismatch = check_database(path)
    assert not mismatch.ok
    assert "migration_mismatch" in mismatch.failures


def test_database_check_reports_pragma_failures(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)

    class Cursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

        def fetchone(self) -> tuple[object, ...] | None:
            return self.rows[0] if self.rows else None

    class Proxy:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, sql: str) -> object:
            if sql == "PRAGMA quick_check":
                return Cursor([("broken page",)])
            if sql == "PRAGMA journal_mode":
                return Cursor([("delete",)])
            if sql == "PRAGMA integrity_check":
                return Cursor([("corrupt btree",)])
            return self.connection.execute(sql)

        def close(self) -> None:
            self.connection.close()

    result = check_database(
        path,
        full=True,
        connection_factory=lambda candidate: Proxy(  # type: ignore[arg-type,return-value]
            sqlite3.connect(candidate)
        ),
    )
    assert "quick_check_failed" in result.failures
    assert "unexpected_journal_mode" in result.failures
    assert "integrity_check_failed" in result.failures
    assert result.integrity_check == "corrupt btree"


def test_foreign_key_violation_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "INSERT INTO messages(id, conversation_id, role, content, created_at) "
            "VALUES('m1', 'missing', 'user', 'x', datetime('now'))"
        )
    result = check_database(path)
    assert not result.foreign_key_consistent
    assert "foreign_key_violations" in result.failures


def test_backup_concurrent_write_and_owner_only_output(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    started = threading.Event()

    def writer() -> None:
        with sqlite_write_transaction(path) as connection:
            connection.execute("UPDATE users SET name='concurrent' WHERE id='u1'")
            started.set()

    thread = threading.Thread(target=writer)
    thread.start()
    started.wait(timeout=5)
    output = tmp_path / "backup.april"
    result = create_backup(path, output, home=tmp_path)
    thread.join(timeout=5)
    assert result.output == output
    assert (output / BACKUP_DATABASE_NAME).is_file()
    assert (output / BACKUP_MANIFEST_NAME).is_file()
    assert os.stat(output).st_mode & 0o077 == 0
    candidate, manifest = validate_backup_package(output)
    assert candidate.is_file()
    assert manifest.schema_version == SCHEMA_VERSION


def test_backup_cancellation_never_publishes(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    cancellation = threading.Event()
    cancellation.set()
    output = tmp_path / "cancelled.april"
    with pytest.raises(BackupCancelled):
        create_backup(
            path,
            output,
            home=tmp_path,
            cancellation_event=cancellation,
        )
    assert not output.exists()


def test_corrupt_manifest_hash_and_future_schema_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    output = tmp_path / "backup.april"
    create_backup(path, output, home=tmp_path)
    manifest_path = output / BACKUP_MANIFEST_NAME
    manifest_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(DatabaseMaintenanceError, match="manifest"):
        validate_backup_package(output)
    create_manifest = create_backup(path, tmp_path / "second.april", home=tmp_path)
    manifest_path.write_text(
        json.dumps(create_manifest.manifest.to_dict()),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["database_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatabaseMaintenanceError, match="SHA-256"):
        validate_backup_package(output)
    manifest["database_sha256"] = create_backup.__module__  # invalid but non-secret
    manifest["schema_version"] = SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatabaseMaintenanceError, match="future schema"):
        validate_backup_package(output)


def test_restore_refuses_running_services_and_succeeds_with_wal_cleanup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    output = tmp_path / "backup.april"
    create_backup(path, output, home=tmp_path)
    with pytest.raises(DatabaseMaintenanceError, match="services are running"):
        restore_backup(
            path,
            output,
            home=tmp_path,
            services_running=True,
        )
    with sqlite_write_transaction(path) as connection:
        connection.execute("UPDATE users SET name='after' WHERE id='u1'")
    Path(f"{path}-wal").touch()
    Path(f"{path}-shm").touch()
    result = restore_backup(
        path,
        output,
        home=tmp_path,
        services_running=False,
    )
    assert result.restored
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM users WHERE id='u1'").fetchone()[0] == "before"
    assert result.rollback_backup.is_dir()


def test_post_replacement_failure_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    _prepare_database(path)
    output = tmp_path / "backup.april"
    create_backup(path, output, home=tmp_path)
    with sqlite_write_transaction(path) as connection:
        connection.execute("UPDATE users SET name='current' WHERE id='u1'")
    valid = check_database(path, home=tmp_path, full=True)
    failed = valid.__class__(**{**valid.to_dict(), "ok": False, "integrity_check": "failed"})
    with pytest.raises(DatabaseMaintenanceError, match="rollback completed"):
        restore_backup(
            path,
            output,
            home=tmp_path,
            services_running=False,
            validator=lambda _path: failed,
        )
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM users WHERE id='u1'").fetchone()[0] == "current"
