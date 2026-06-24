from __future__ import annotations

import types

from typer.testing import CliRunner

from apps.runner.main import app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.user_profile import UserProfileStore


async def test_users_table_has_profile_columns(settings_tmp) -> None:
    async with Database(settings_tmp.database_path) as database:
        await run_migrations(database)
        rows = await database.fetchall("PRAGMA table_info(users)")
        columns = {row[1] for row in rows}
        assert {"address", "timezone", "preferences_json", "updated_at"} <= columns


async def test_profile_crud(settings_tmp) -> None:
    async with Database(settings_tmp.database_path) as database:
        await run_migrations(database)
        store = UserProfileStore(database)
        assert await store.get() is None

        created = await store.set(
            display_name="Sam",
            preferred_address="Sam",
            timezone="UTC",
            preferences={"tone": "concise"},
        )
        assert created.display_name == "Sam"
        assert created.preferred_address == "Sam"
        assert created.timezone == "UTC"
        assert created.preferences == {"tone": "concise"}
        assert created.created_at
        assert created.updated_at
        assert await store.get() == created

        updated = await store.set(display_name="Samuel")
        assert updated.display_name == "Samuel"
        assert updated.created_at == created.created_at  # creation timestamp preserved

        assert await store.delete() is True
        assert await store.get() is None


async def test_profile_only_stores_explicit_fields(settings_tmp) -> None:
    async with Database(settings_tmp.database_path) as database:
        await run_migrations(database)
        store = UserProfileStore(database)
        profile = await store.set(display_name="Pat")
        # Nothing is inferred: optional attributes stay unset.
        assert profile.preferred_address is None
        assert profile.timezone is None
        assert profile.preferences == {}


def test_cli_profile_set_show_delete(settings_tmp, monkeypatch) -> None:
    manager = types.SimpleNamespace(home=settings_tmp.home)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    runner = CliRunner()

    created = runner.invoke(
        app, ["april", "profile", "set", "--display-name", "Sam", "--timezone", "UTC"]
    )
    assert created.exit_code == 0, created.stdout
    assert "Sam" in created.stdout

    shown = runner.invoke(app, ["april", "profile", "show"])
    assert shown.exit_code == 0
    assert "Sam" in shown.stdout

    deleted = runner.invoke(app, ["april", "profile", "delete"])
    assert deleted.exit_code == 0
    assert "True" in deleted.stdout
