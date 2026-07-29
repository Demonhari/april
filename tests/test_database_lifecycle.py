from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path

import pytest

from services.api.dependencies import build_container
from services.memory.database import Database


def _multiprocess_writer(database_path: str, start: int, count: int) -> None:
    async def _write() -> None:
        database = Database(Path(database_path))
        await database.connect()
        try:
            for value in range(start, start + count):
                async with database.transaction() as connection:
                    await connection.execute("INSERT INTO writes(value) VALUES(?)", (value,))
        finally:
            await database.close()

    asyncio.run(_write())


async def test_connect_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "april.db")
    await database.connect()
    first = database.connection
    await database.connect()
    assert database.connection is first
    await database.close()
    assert database.is_connected is False


async def test_close_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "april.db")
    await database.connect()
    assert database.is_connected is True
    await database.close()
    await database.close()
    assert database.is_connected is False


async def test_async_context_manager_closes_connection(tmp_path: Path) -> None:
    database = Database(tmp_path / "april.db")
    async with database as handle:
        assert handle is database
        assert database.is_connected is True
        await database.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    assert database.is_connected is False


async def test_build_container_closes_database_on_assembly_failure(
    settings_tmp: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[Database] = []
    real_init = Database.__init__

    def _spy_init(self: Database, path: Path) -> None:
        real_init(self, path)
        created.append(self)

    async def _boom(database: Database) -> None:
        raise RuntimeError("migration boom")

    monkeypatch.setattr(Database, "__init__", _spy_init)
    monkeypatch.setattr("services.api.dependencies.validate_configuration", lambda home: [])
    monkeypatch.setattr("services.api.dependencies.run_migrations", _boom)

    with pytest.raises(RuntimeError, match="migration boom"):
        await build_container(settings_tmp)  # type: ignore[arg-type]

    assert created, "build_container should have created a Database"
    assert all(not db.is_connected for db in created), (
        "a failed build must not leak an open database connection"
    )


async def test_file_database_uses_required_pragmas(tmp_path: Path) -> None:
    database = Database(tmp_path / "april.db")
    await database.connect()
    try:
        assert (await database.fetchone("PRAGMA journal_mode"))[0] == "wal"
        assert (await database.fetchone("PRAGMA synchronous"))[0] == 1
        assert (await database.fetchone("PRAGMA busy_timeout"))[0] == 5000
        assert (await database.fetchone("PRAGMA foreign_keys"))[0] == 1
    finally:
        await database.close()


async def test_shared_path_write_coordination_and_rollback(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    first = Database(path)
    second = Database(path.parent / "." / path.name)
    await first.connect()
    await second.connect()
    await first.execute("CREATE TABLE writes(value INTEGER NOT NULL)")
    assert first._write_lock is second._write_lock

    async def write(database: Database, value: int) -> None:
        async with database.transaction() as connection:
            await connection.execute("INSERT INTO writes(value) VALUES(?)", (value,))
            await asyncio.sleep(0)

    await asyncio.gather(*(write(first if value % 2 else second, value) for value in range(40)))
    assert (await first.fetchone("SELECT COUNT(*) FROM writes"))[0] == 40

    async def failing_transaction() -> None:
        async with second.transaction() as connection:
            await connection.execute("INSERT INTO writes(value) VALUES(100)")
            await connection.execute("INSERT INTO writes(value) VALUES(101)")
            raise RuntimeError("rollback")

    with pytest.raises(RuntimeError, match="rollback"):
        await failing_transaction()
    assert (await first.fetchone("SELECT COUNT(*) FROM writes"))[0] == 40

    await first.close()
    await second.close()


async def test_cancellation_releases_write_coordination(tmp_path: Path) -> None:
    path = tmp_path / "april.db"
    first = Database(path)
    second = Database(path)
    await first.connect()
    await second.connect()
    await first.execute("CREATE TABLE writes(value INTEGER NOT NULL)")
    entered = asyncio.Event()
    hold = asyncio.Event()

    async def cancelled_write() -> None:
        async with first.transaction() as connection:
            await connection.execute("INSERT INTO writes(value) VALUES(1)")
            entered.set()
            await hold.wait()

    task = asyncio.create_task(cancelled_write())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with second.transaction() as connection:
        await connection.execute("INSERT INTO writes(value) VALUES(2)")
    rows = await second.fetchall("SELECT value FROM writes ORDER BY value")
    assert [row[0] for row in rows] == [2]
    await first.close()
    await second.close()


def test_cross_process_write_coordination(tmp_path: Path) -> None:
    path = tmp_path / "april.db"

    async def _prepare() -> None:
        database = Database(path)
        await database.connect()
        await database.execute("CREATE TABLE writes(value INTEGER NOT NULL)")
        await database.close()

    asyncio.run(_prepare())
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_multiprocess_writer, args=(str(path), offset, 20))
        for offset in (0, 100)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    async def _count() -> int:
        database = Database(path)
        await database.connect()
        try:
            row = await database.fetchone("SELECT COUNT(*) FROM writes")
            return int(row[0]) if row is not None else 0
        finally:
            await database.close()

    assert asyncio.run(_count()) == 40
