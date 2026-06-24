from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

import aiosqlite


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.path)
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            # Tolerate brief contention when a short-lived connection (e.g. the
            # repo indexer) writes while the main connection is open.
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.commit()
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
        cursor = await self.connection.execute(sql, parameters)
        await self.connection.commit()
        return cursor

    async def fetchone(self, sql: str, parameters: tuple[Any, ...] = ()) -> aiosqlite.Row | None:
        cursor = await self.connection.execute(sql, parameters)
        return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        cursor = await self.connection.execute(sql, parameters)
        return list(await cursor.fetchall())

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        await self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            await self.connection.rollback()
            raise
        else:
            await self.connection.commit()
