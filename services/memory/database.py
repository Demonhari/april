from __future__ import annotations

import asyncio
import contextvars
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

import aiosqlite

SQLITE_BUSY_TIMEOUT_MS = 5000
_WRITE_LOCKS: weakref.WeakValueDictionary[Path, asyncio.Lock] = weakref.WeakValueDictionary()
_WRITE_LOCKS_GUARD = threading.Lock()
_ACTIVE_WRITE_PATHS: contextvars.ContextVar[frozenset[Path]] = contextvars.ContextVar(
    "april_active_sqlite_write_paths",
    default=frozenset(),
)


def _is_memory_path(path: Path) -> bool:
    return str(path) == ":memory:"


def _database_key(path: Path) -> Path:
    if _is_memory_path(path):
        return path
    return path.expanduser().resolve(strict=False)


def _shared_write_lock(path: Path) -> asyncio.Lock:
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(path)
        if lock is None:
            lock = asyncio.Lock()
            _WRITE_LOCKS[path] = lock
        return lock


async def configure_aiosqlite_connection(
    connection: aiosqlite.Connection,
    *,
    memory_database: bool,
) -> None:
    """Apply APRIL's required SQLite settings before exposing a connection."""
    await connection.execute("PRAGMA foreign_keys = ON")
    cursor = await connection.execute("PRAGMA journal_mode = WAL")
    journal_mode_row = await cursor.fetchone()
    journal_mode = str(journal_mode_row[0]).lower() if journal_mode_row else ""
    if memory_database:
        if journal_mode not in {"memory", "wal"}:
            raise sqlite3.OperationalError(
                f"Unexpected journal mode for in-memory SQLite database: {journal_mode}"
            )
    elif journal_mode != "wal":
        raise sqlite3.OperationalError(f"Failed to enable SQLite WAL mode: {journal_mode}")
    await connection.execute("PRAGMA synchronous = NORMAL")
    await connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    await connection.commit()


def configure_sqlite_connection(
    connection: sqlite3.Connection,
    *,
    memory_database: bool,
) -> None:
    """Apply the same APRIL SQLite settings to a synchronous connection."""
    connection.execute("PRAGMA foreign_keys = ON")
    row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    journal_mode = str(row[0]).lower() if row else ""
    if memory_database:
        if journal_mode not in {"memory", "wal"}:
            raise sqlite3.OperationalError(
                f"Unexpected journal mode for in-memory SQLite database: {journal_mode}"
            )
    elif journal_mode != "wal":
        raise sqlite3.OperationalError(f"Failed to enable SQLite WAL mode: {journal_mode}")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    connection.commit()


def connect_sqlite(path: Path) -> sqlite3.Connection:
    """Open a synchronous APRIL database connection with required pragmas."""
    memory_database = _is_memory_path(path)
    database_path = path if memory_database else path.expanduser().resolve(strict=False)
    if not memory_database:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        configure_sqlite_connection(connection, memory_database=memory_database)
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def sqlite_write_fence(path: Path) -> Iterator[Path]:
    """Acquire APRIL's existing cross-process write fence without opening SQLite."""
    if _is_memory_path(path):
        yield path
        return

    import fcntl

    key = _database_key(path)
    lock_path = key.with_name(f"{key.name}.write.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + (SQLITE_BUSY_TIMEOUT_MS / 1000)
    try:
        while True:
            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise sqlite3.OperationalError(
                        "Timed out waiting for APRIL SQLite write coordination"
                    ) from None
                time.sleep(0.01)
        yield key
    finally:
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(file_descriptor)


@contextmanager
def sqlite_write_transaction(path: Path) -> Iterator[sqlite3.Connection]:
    """Coordinate a synchronous APRIL write transaction with async processes."""
    if _is_memory_path(path):
        memory_connection = connect_sqlite(path)
        try:
            memory_connection.execute("BEGIN IMMEDIATE")
            yield memory_connection
            memory_connection.commit()
        except BaseException:
            memory_connection.rollback()
            raise
        finally:
            memory_connection.close()
        return

    with sqlite_write_fence(path) as key:
        connection = connect_sqlite(key)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._key = _database_key(path)
        self._write_lock = _shared_write_lock(self._key)
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._connection is not None:
            return
        memory_database = _is_memory_path(self.path)
        database_path = self.path if memory_database else self._key
        if not memory_database:
            database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(database_path)
        try:
            connection.row_factory = aiosqlite.Row
            await configure_aiosqlite_connection(
                connection,
                memory_database=memory_database,
            )
        except BaseException:
            # Never leave a half-initialised connection unclosed; the aiosqlite
            # worker thread would otherwise be reported as an unclosed resource.
            await connection.close()
            raise
        self._connection = connection

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database is not connected")
        return self._connection

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> aiosqlite.Cursor:
        statement = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        if statement in {"SELECT", "EXPLAIN"} or self._key in _ACTIVE_WRITE_PATHS.get():
            return await self.connection.execute(sql, parameters)
        async with self.transaction() as connection:
            return await connection.execute(sql, parameters)

    async def fetchone(self, sql: str, parameters: tuple[Any, ...] = ()) -> aiosqlite.Row | None:
        cursor = await self.connection.execute(sql, parameters)
        return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        cursor = await self.connection.execute(sql, parameters)
        return list(await cursor.fetchall())

    async def _acquire_process_lock(self) -> int | None:
        if _is_memory_path(self.path):
            return None
        import fcntl

        lock_path = self._key.with_name(f"{self._key.name}.write.lock")
        file_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + (SQLITE_BUSY_TIMEOUT_MS / 1000)
        try:
            while True:
                try:
                    fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return file_descriptor
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise sqlite3.OperationalError(
                            "Timed out waiting for APRIL SQLite write coordination"
                        ) from None
                    await asyncio.sleep(0.01)
        except BaseException:
            os.close(file_descriptor)
            raise

    @staticmethod
    def _release_process_lock(file_descriptor: int | None) -> None:
        if file_descriptor is None:
            return
        import fcntl

        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(file_descriptor)

    @asynccontextmanager
    async def write_coordination(self) -> AsyncIterator[None]:
        if self._key in _ACTIVE_WRITE_PATHS.get():
            raise RuntimeError("Nested SQLite write transactions are not supported.")
        process_lock: int | None = None
        await self._write_lock.acquire()
        active_paths_token: contextvars.Token[frozenset[Path]] | None = None
        try:
            process_lock = await self._acquire_process_lock()
            active_paths_token = _ACTIVE_WRITE_PATHS.set(_ACTIVE_WRITE_PATHS.get() | {self._key})
            yield
        finally:
            if active_paths_token is not None:
                _ACTIVE_WRITE_PATHS.reset(active_paths_token)
            self._release_process_lock(process_lock)
            self._write_lock.release()

    @asynccontextmanager
    async def transaction_under_coordination(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run a transaction while the caller holds this database's write fence."""

        if self._key not in _ACTIVE_WRITE_PATHS.get():
            raise RuntimeError("SQLite write coordination must be acquired first.")
        transaction_started = False
        begin = asyncio.create_task(self.connection.execute("BEGIN IMMEDIATE"))
        try:
            await asyncio.shield(begin)
            transaction_started = True
        except BaseException:
            # aiosqlite work already queued on its worker cannot be cancelled.
            # Wait for BEGIN to settle and roll it back before releasing locks.
            await asyncio.shield(begin)
            transaction_started = True
            await asyncio.shield(self.connection.rollback())
            transaction_started = False
            raise
        try:
            yield self.connection
            await asyncio.shield(self.connection.commit())
        except BaseException:
            if transaction_started:
                await asyncio.shield(self.connection.rollback())
            raise

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        async with (
            self.write_coordination(),
            self.transaction_under_coordination() as connection,
        ):
            yield connection
