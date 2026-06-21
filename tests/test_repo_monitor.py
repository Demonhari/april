from __future__ import annotations

import subprocess
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.scheduler.repo_monitor import compute_repo_activity
from tests.test_core_api import auth, make_container


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "APRIL Test")
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return _head(repo)


def _head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def _memory(settings_tmp) -> tuple[SqliteMemory, Database]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    return SqliteMemory(database), database


@pytest.mark.asyncio
async def test_dirty_files_surface_dirty_count(settings_tmp) -> None:
    repo = Path(settings_tmp.home) / "proj_dirty"
    _init_repo(repo)
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    memory, database = await _memory(settings_tmp)
    await memory.add_project(str(repo))
    activity = await compute_repo_activity(memory, persist=False)
    assert len(activity) == 1
    assert activity[0].dirty_count == 1
    assert activity[0].new_commits is False
    await database.close()


@pytest.mark.asyncio
async def test_second_commit_after_snapshot_sets_new_commits(settings_tmp) -> None:
    repo = Path(settings_tmp.home) / "proj_commits"
    _init_repo(repo)
    memory, database = await _memory(settings_tmp)
    project = await memory.add_project(str(repo))
    # First scan persists the baseline at the initial HEAD.
    first = await compute_repo_activity(memory, persist=True)
    assert first[0].new_commits is False
    # A second commit advances HEAD.
    (repo / "file.txt").write_text("hello again\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "second")
    second = await compute_repo_activity(memory, persist=False)
    assert second[0].new_commits is True
    assert (await memory.get_repo_snapshot(project.id))["head_sha"] != _head(repo)
    await database.close()


@pytest.mark.asyncio
async def test_persist_flag_controls_baseline_advance(settings_tmp) -> None:
    repo = Path(settings_tmp.home) / "proj_persist"
    head1 = _init_repo(repo)
    memory, database = await _memory(settings_tmp)
    project = await memory.add_project(str(repo))

    # persist=False leaves no snapshot behind.
    await compute_repo_activity(memory, persist=False)
    assert await memory.get_repo_snapshot(project.id) is None

    # persist=True writes the baseline.
    await compute_repo_activity(memory, persist=True)
    snapshot = await memory.get_repo_snapshot(project.id)
    assert snapshot is not None
    assert snapshot["head_sha"] == head1

    # A new commit followed by persist=False must NOT move the baseline.
    (repo / "file.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "third")
    head2 = _head(repo)
    await compute_repo_activity(memory, persist=False)
    assert (await memory.get_repo_snapshot(project.id))["head_sha"] == head1

    # persist=True advances it to the new HEAD.
    await compute_repo_activity(memory, persist=True)
    assert (await memory.get_repo_snapshot(project.id))["head_sha"] == head2
    await database.close()


@pytest.mark.asyncio
async def test_non_git_project_is_skipped(settings_tmp) -> None:
    plain = Path(settings_tmp.home) / "not_a_repo"
    plain.mkdir(parents=True, exist_ok=True)
    (plain / "file.txt").write_text("x\n", encoding="utf-8")
    memory, database = await _memory(settings_tmp)
    await memory.add_project(str(plain))
    activity = await compute_repo_activity(memory, persist=False)
    assert activity == []
    await database.close()


def _enable_repo_monitor(settings_tmp):
    return settings_tmp.model_copy(
        update={
            "scheduler": settings_tmp.scheduler.model_copy(
                update={"repo_monitor_enabled": True}
            )
        }
    )


def test_preview_includes_project_activity_when_enabled(settings_tmp) -> None:
    repo = Path(settings_tmp.home) / "proj_preview"
    _init_repo(repo)
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    container = anyio.run(make_container, _enable_repo_monitor(settings_tmp))
    client = TestClient(create_app(container))
    client.post("/projects", json={"path": str(repo)}, headers=auth(settings_tmp))

    response = client.get("/scheduler/briefing/preview", headers=auth(settings_tmp))
    assert response.status_code == 200
    body = response.json()["body"]
    assert "Project activity:" in body
    assert "uncommitted file" in body


def test_preview_has_no_activity_section_when_disabled(settings_tmp) -> None:
    repo = Path(settings_tmp.home) / "proj_disabled"
    _init_repo(repo)
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    # Default settings_tmp keeps repo_monitor_enabled False.
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    client.post("/projects", json={"path": str(repo)}, headers=auth(settings_tmp))

    response = client.get("/scheduler/briefing/preview", headers=auth(settings_tmp))
    assert response.status_code == 200
    assert "Project activity" not in response.json()["body"]
