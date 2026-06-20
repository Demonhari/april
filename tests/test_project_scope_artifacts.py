from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import anyio
import pytest

from april_common.errors import PermissionDeniedError
from april_common.project_scope import (
    git_apply_check,
    git_head,
    git_staged_digest,
    git_worktree_digest,
    normalize_project_root,
    validate_patch_text,
)
from services.memory.schemas import ApprovalRecord
from services.permissions.artifacts import (
    build_git_commit_metadata,
    build_patch_text_metadata,
    verify_approval_artifact,
)


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "april@example.test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "APRIL Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def commit_all(path: Path, message: str = "initial") -> None:
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=path, check=True, capture_output=True)


def record_for(
    *,
    tool: str,
    args: dict[str, object],
    metadata: dict[str, object],
) -> ApprovalRecord:
    return ApprovalRecord(
        id="approval-1",
        tool=tool,
        args=args,
        agent="coding_agent",
        canonical_hash="hash",
        metadata=metadata,
        permission_level=3,
        risk_level="code_write",
        status="pending",
        expires_at="2099-01-01T00:00:00Z",
        created_at="2026-01-01T00:00:00Z",
    )


async def build_patch_metadata_kwargs(
    repo_path: str, patch_text: str, patch_path: str
) -> dict[str, Any]:
    return await build_patch_text_metadata(
        repo_path=repo_path,
        patch_text=patch_text,
        patch_path=patch_path,
        expected_side_effects=["Apply patch."],
    )


async def build_commit_metadata_kwargs(repo_path: str) -> dict[str, Any]:
    return await build_git_commit_metadata(repo_path=repo_path)


def test_project_scope_rejects_invalid_roots_and_patch_edges(tmp_path: Path) -> None:
    file_root = tmp_path / "not-a-dir"
    file_root.write_text("content", encoding="utf-8")
    with pytest.raises(PermissionDeniedError):
        normalize_project_root(file_root)

    with pytest.raises(PermissionDeniedError):
        validate_patch_text("", tmp_path)
    with pytest.raises(PermissionDeniedError):
        validate_patch_text("--- /dev/null\n+++ /dev/null\n", tmp_path)
    with pytest.raises(PermissionDeniedError):
        validate_patch_text("diff --git a/model.gguf b/model.gguf\n", tmp_path)

    create_patch = "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n"
    assert validate_patch_text(create_patch, tmp_path) == ["new.py"]


def test_git_scope_helpers_cover_non_git_and_apply_failures(tmp_path: Path) -> None:
    assert anyio.run(git_head, tmp_path) is None
    assert anyio.run(git_worktree_digest, tmp_path) is None
    with pytest.raises(PermissionDeniedError):
        anyio.run(git_staged_digest, tmp_path)

    init_repo(tmp_path)
    (tmp_path / "example.py").write_text("value = 1\n", encoding="utf-8")
    commit_all(tmp_path)
    assert anyio.run(git_head, tmp_path)
    assert anyio.run(git_worktree_digest, tmp_path)

    bad_patch = tmp_path / "bad.patch"
    bad_patch.write_text(
        "diff --git a/example.py b/example.py\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-missing\n"
        "+value = 2\n",
        encoding="utf-8",
    )
    ok, _stdout, stderr = anyio.run(git_apply_check, tmp_path, bad_patch)
    assert not ok
    assert stderr


def test_patch_artifact_metadata_and_verification_failures(tmp_path: Path) -> None:
    init_repo(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("value = 1\n", encoding="utf-8")
    commit_all(tmp_path)
    patch = tmp_path / "fix.patch"
    patch_text = (
        "diff --git a/example.py b/example.py\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
    )
    patch.write_text(patch_text, encoding="utf-8")
    metadata = anyio.run(
        build_patch_metadata_kwargs,
        str(tmp_path),
        patch_text,
        str(patch),
    )

    valid_record = record_for(
        tool="patch_applier",
        args={"repo_path": str(tmp_path), "patch_path": str(patch)},
        metadata=metadata,
    )
    assert anyio.run(verify_approval_artifact, valid_record) is None

    wrong_paths = dict(metadata)
    wrong_paths["affected_paths"] = ["other.py"]
    result = anyio.run(
        verify_approval_artifact,
        record_for(
            tool="patch_applier",
            args={"repo_path": str(tmp_path), "patch_path": str(patch)},
            metadata=wrong_paths,
        ),
    )
    assert result is not None
    assert "affected paths changed" in result.stderr

    wrong_repo = dict(metadata)
    wrong_repo["repo_root"] = str(tmp_path / "other")
    result = anyio.run(
        verify_approval_artifact,
        record_for(
            tool="patch_applier",
            args={"repo_path": str(tmp_path), "patch_path": str(patch)},
            metadata=wrong_repo,
        ),
    )
    assert result is not None
    assert "repository no longer matches" in result.stderr

    target.write_text("value = changed\n", encoding="utf-8")
    result = anyio.run(verify_approval_artifact, valid_record)
    assert result is not None
    assert "Repository state changed" in result.stderr


def test_git_commit_artifact_metadata_rejects_repo_mismatch(tmp_path: Path) -> None:
    init_repo(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("value = 1\n", encoding="utf-8")
    commit_all(tmp_path)
    target.write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "example.py"], cwd=tmp_path, check=True, capture_output=True)

    metadata = anyio.run(build_commit_metadata_kwargs, str(tmp_path))
    metadata["repo_root"] = str(tmp_path / "other")
    result = anyio.run(
        verify_approval_artifact,
        record_for(
            tool="git_commit",
            args={"repo_path": str(tmp_path), "message": "approved"},
            metadata=metadata,
        ),
    )
    assert result is not None
    assert "repository no longer matches" in result.stderr
