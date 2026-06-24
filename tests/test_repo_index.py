from __future__ import annotations

from pathlib import Path

from april_common.settings import project_root
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.vector_memory import VectorMemory
from skills.code.repo_indexer import _git_head, repo_indexer
from skills.schemas import ToolResult


async def _migrate(settings) -> None:
    async with Database(settings.database_path) as database:
        await run_migrations(database)


async def _index(repo: Path, *, force: bool = False) -> ToolResult:
    return await repo_indexer(
        {"repo_path": str(repo), "project_id": None, "force_full_reindex": force}
    )


def _indexed_paths(settings) -> set[str]:
    paths: set[str] = set()
    for source in VectorMemory(settings.vector_index_path).sources(source_type="repo"):
        paths.update(source["paths"])
    return paths


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    (repo / "b.py").write_text("print('b')\n", encoding="utf-8")
    return repo


async def test_initial_index(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    result = await _index(repo)
    assert result.ok
    assert result.data["reindexed_files"] == 2
    assert result.data["chunks"] >= 2
    assert result.data["git_commit"] is None
    assert _indexed_paths(settings_tmp) == {str(repo / "a.py"), str(repo / "b.py")}


async def test_no_op_second_index(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    await _index(repo)
    result = await _index(repo)
    assert result.data["reindexed_files"] == 0
    assert result.data["skipped_files"] == 2
    assert result.data["removed_files"] == 0


async def test_changed_file_reindexes_only_it(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    await _index(repo)
    (repo / "a.py").write_text("print('a changed')\nprint('more')\n", encoding="utf-8")
    result = await _index(repo)
    assert result.data["reindexed_files"] == 1
    assert result.data["skipped_files"] == 1


async def test_added_file(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    await _index(repo)
    (repo / "c.py").write_text("print('c')\n", encoding="utf-8")
    result = await _index(repo)
    assert result.data["reindexed_files"] == 1
    assert result.data["skipped_files"] == 2
    assert str(repo / "c.py") in _indexed_paths(settings_tmp)


async def test_deleted_file_removes_chunks(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    await _index(repo)
    (repo / "b.py").unlink()
    result = await _index(repo)
    assert result.data["removed_files"] == 1
    paths = _indexed_paths(settings_tmp)
    assert str(repo / "b.py") not in paths
    assert str(repo / "a.py") in paths


async def test_renamed_file(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    await _index(repo)
    (repo / "a.py").rename(repo / "renamed.py")
    result = await _index(repo)
    assert result.data["reindexed_files"] == 1  # renamed.py is treated as new
    assert result.data["removed_files"] == 1  # the old path is removed
    paths = _indexed_paths(settings_tmp)
    assert str(repo / "renamed.py") in paths
    assert str(repo / "a.py") not in paths


async def test_non_git_directory(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    result = await _index(repo)
    assert result.data["git_commit"] is None


async def test_symlink_escape_is_not_followed(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside content\n", encoding="utf-8")
    (repo / "link.txt").symlink_to(outside)
    await _index(repo)
    assert all(not path.endswith("link.txt") for path in _indexed_paths(settings_tmp))


async def test_force_full_reindex(settings_tmp, tmp_path: Path) -> None:
    await _migrate(settings_tmp)
    repo = _make_repo(tmp_path)
    await _index(repo)
    result = await _index(repo, force=True)
    assert result.data["reindexed_files"] == 2
    assert result.data["skipped_files"] == 0


def test_git_head_detection() -> None:
    head = _git_head(project_root())
    assert head is not None
    assert len(head) == 40
