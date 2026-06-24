from __future__ import annotations

from pathlib import Path

import pytest

from services.api.dependencies import build_container
from services.memory.database import Database


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
