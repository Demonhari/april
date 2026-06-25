from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from april_common.errors import PermissionDeniedError
from april_common.project_scope import (
    git_apply_bytes,
    git_apply_check_bytes,
    git_head,
    git_staged_digest,
    git_staged_tree_id,
    git_worktree_digest,
    inspect_patch_bytes,
    normalize_project_root,
    sha256_bytes,
    sha256_file,
)
from april_common.settings import get_settings
from services.memory.schemas import ApprovalRecord
from skills.schemas import ToolResult

PATCH_ARTIFACT_VERSION = 1
PATCH_ARTIFACT_ID_RE = re.compile(r"^[a-f0-9]{64}$")
_REPOSITORY_LOCKS: dict[str, asyncio.Lock] = {}


async def build_patch_approval_metadata(
    *,
    repo_path: str,
    patch_path: str,
    expected_side_effects: list[str],
    project_id: str | None = None,
) -> dict[str, Any]:
    root = normalize_project_root(repo_path)
    patch_bytes = _read_patch_source(Path(patch_path), root)
    artifact = await inspect_patch_bytes(patch_bytes=patch_bytes, repo_root=root)
    stored = store_patch_artifact(patch_bytes)
    return {
        "artifact_type": "patch",
        "artifact_version": PATCH_ARTIFACT_VERSION,
        "artifact_id": stored["artifact_id"],
        "patch_sha256": artifact.patch_sha256,
        "patch_byte_length": artifact.patch_byte_length,
        "affected_paths": artifact.affected_paths,
        "project_id": project_id,
        "repo_root": artifact.repo_root,
        "repo_head": artifact.repo_head,
        "repo_state_digest": artifact.repo_state_digest,
        "expected_side_effects": expected_side_effects,
    }


async def build_patch_text_metadata(
    *,
    repo_path: str,
    patch_text: str,
    patch_path: str,
    expected_side_effects: list[str],
    project_id: str | None = None,
) -> dict[str, Any]:
    root = normalize_project_root(repo_path)
    patch_bytes = patch_text.encode("utf-8")
    artifact = await inspect_patch_bytes(patch_bytes=patch_bytes, repo_root=root)
    stored = store_patch_artifact(patch_bytes)
    return {
        "artifact_type": "patch",
        "artifact_version": PATCH_ARTIFACT_VERSION,
        "artifact_id": stored["artifact_id"],
        "patch_sha256": artifact.patch_sha256,
        "patch_byte_length": artifact.patch_byte_length,
        "affected_paths": artifact.affected_paths,
        "project_id": project_id,
        "repo_root": str(root),
        "repo_head": await git_head(root),
        "repo_state_digest": await git_worktree_digest(root),
        "expected_side_effects": expected_side_effects,
    }


async def build_git_commit_metadata(
    *, repo_path: str, message: str | None = None, project_id: str | None = None
) -> dict[str, Any]:
    root = normalize_project_root(repo_path)
    return {
        "artifact_type": "git_commit",
        "project_id": project_id,
        "repo_root": str(root),
        "repo_head": await git_head(root),
        "staged_diff_sha256": await git_staged_digest(root),
        "staged_tree_id": await git_staged_tree_id(root),
        "commit_message": message,
    }


async def verify_approval_artifact(record: ApprovalRecord) -> ToolResult | None:
    try:
        if record.tool == "patch_applier":
            return await _verify_patch(record)
        if record.tool == "git_commit":
            return await _verify_git_commit(record)
        if record.tool == "apply_log_cleanup":
            from services.permissions.cleanup import verify_log_cleanup_approval

            return await verify_log_cleanup_approval(record)
    except PermissionDeniedError as exc:
        return _failed(exc.message, details=exc.details)
    return None


async def apply_approved_patch(record: ApprovalRecord) -> ToolResult:
    async with repository_mutation_lock(str(record.args.get("repo_path", ""))):
        return await _apply_patch_locked(record)


@asynccontextmanager
async def repository_mutation_lock(repo_path: str) -> AsyncIterator[None]:
    root = normalize_project_root(repo_path)
    key = str(root)
    lock = _REPOSITORY_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _REPOSITORY_LOCKS[key] = lock
    async with lock:
        yield


async def _verify_patch(record: ApprovalRecord) -> ToolResult | None:
    result = await _verified_patch_artifact(record, run_git_check=True)
    if isinstance(result, ToolResult):
        return result
    return None


async def _apply_patch_locked(record: ApprovalRecord) -> ToolResult:
    result = await _verified_patch_artifact(record, run_git_check=True)
    if isinstance(result, ToolResult):
        return result
    root, patch_bytes, artifact = result
    code, stdout, stderr = await git_apply_bytes(root, patch_bytes)
    return ToolResult(
        ok=code == 0,
        stdout=stdout,
        stderr=stderr,
        data={
            "returncode": code,
            "artifact_id": record.metadata.get("artifact_id"),
            "patch_sha256": artifact.patch_sha256,
            "patch_byte_length": artifact.patch_byte_length,
            "affected_paths": artifact.affected_paths,
            "repo_root": str(root),
        },
        risk_level="code_write",
        permission_level=3,
    )


async def _verified_patch_artifact(
    record: ApprovalRecord, *, run_git_check: bool
) -> ToolResult | tuple[Path, bytes, Any]:
    metadata = record.metadata
    args = record.args
    repo_path = str(args.get("repo_path", ""))
    expected_repo = metadata.get("repo_root")
    expected_sha = metadata.get("patch_sha256")
    expected_length = metadata.get("patch_byte_length")
    expected_paths = sorted(str(path) for path in metadata.get("affected_paths", []))
    expected_state = metadata.get("repo_state_digest")
    artifact_id = metadata.get("artifact_id")
    if metadata.get("artifact_type") != "patch" or not artifact_id or not expected_sha:
        return _failed("Patch approval is missing immutable artifact metadata.")
    root = normalize_project_root(repo_path)
    if expected_repo and str(root) != str(expected_repo):
        return _failed("Selected repository no longer matches the approved repository.")
    try:
        patch_bytes = load_patch_artifact_bytes(str(artifact_id))
        artifact = await inspect_patch_bytes(patch_bytes=patch_bytes, repo_root=root)
    except PermissionDeniedError as exc:
        return _failed(exc.message, details=exc.details)
    if expected_sha and artifact.patch_sha256 != expected_sha:
        return _failed("Patch digest changed after approval.")
    if expected_length is not None and artifact.patch_byte_length != int(expected_length):
        return _failed("Patch byte length changed after approval.")
    if expected_paths and artifact.affected_paths != expected_paths:
        return _failed("Patch affected paths changed after approval.")
    if expected_state and artifact.repo_state_digest != expected_state:
        return _failed("Repository state changed after patch approval.")
    if not run_git_check:
        return root, patch_bytes, artifact
    ok, stdout, stderr = await git_apply_check_bytes(root, patch_bytes)
    if ok:
        return root, patch_bytes, artifact
    return ToolResult(
        ok=False,
        stdout=stdout,
        stderr=stderr or "git apply --check failed.",
        data={
            "artifact_id": artifact_id,
            "patch_sha256": artifact.patch_sha256,
            "patch_byte_length": artifact.patch_byte_length,
            "affected_paths": artifact.affected_paths,
            "repo_root": str(root),
        },
        risk_level="code_write",
        permission_level=3,
    )


async def _verify_git_commit(record: ApprovalRecord) -> ToolResult | None:
    async with repository_mutation_lock(str(record.args.get("repo_path", ""))):
        metadata = record.metadata
        repo_path = str(record.args.get("repo_path", ""))
        root = normalize_project_root(repo_path)
        expected_repo = metadata.get("repo_root")
        expected_digest = metadata.get("staged_diff_sha256")
        expected_tree = metadata.get("staged_tree_id")
        expected_message = metadata.get("commit_message")
        if expected_repo and str(root) != str(expected_repo):
            return _failed("Selected repository no longer matches the approved repository.")
        if expected_message is not None and record.args.get("message") != expected_message:
            return _failed("Git commit message changed after approval.")
        current_digest = await git_staged_digest(root)
        if expected_digest and current_digest != expected_digest:
            return _failed(
                "Staged Git diff changed after approval.",
                data={"staged_diff_sha256": current_digest},
            )
        current_tree = await git_staged_tree_id(root)
        if expected_tree and current_tree != expected_tree:
            return _failed(
                "Staged Git tree changed after approval.",
                data={"staged_tree_id": current_tree},
            )
        return None


def store_patch_artifact(patch_bytes: bytes) -> dict[str, Any]:
    artifact_id = sha256_bytes(patch_bytes)
    directory = _patch_artifact_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact_id}.patch"
    if path.exists():
        if sha256_file(path) != artifact_id:
            raise PermissionDeniedError("Patch artifact digest mismatch in APRIL artifact store.")
        return {"artifact_id": artifact_id, "path": str(path), "byte_length": len(patch_bytes)}
    temporary = directory / f".{artifact_id}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(patch_bytes)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"artifact_id": artifact_id, "path": str(path), "byte_length": len(patch_bytes)}


def load_patch_artifact_bytes(artifact_id: str) -> bytes:
    if not PATCH_ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise PermissionDeniedError("Invalid patch artifact ID.")
    path = _patch_artifact_dir() / f"{artifact_id}.patch"
    try:
        patch_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise PermissionDeniedError("Patch artifact is missing.") from exc
    if sha256_bytes(patch_bytes) != artifact_id:
        raise PermissionDeniedError("Patch artifact digest mismatch.")
    return patch_bytes


def _read_patch_source(source: Path, repo_root: Path) -> bytes:
    settings = get_settings()
    path = source.expanduser().resolve(strict=True)
    allowed_roots = (
        repo_root,
        settings.resolve_path(Path("data/patches")),
        _patch_artifact_dir(),
    )
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        raise PermissionDeniedError(
            "Patch source must be inside the selected repository or APRIL artifact store."
        )
    size = path.stat().st_size
    if size > settings.paths.max_file_write_bytes:
        raise PermissionDeniedError(
            "Patch artifact exceeds configured maximum write size.",
            {"size": size},
        )
    return path.read_bytes()


def _patch_artifact_dir() -> Path:
    return get_settings().resolve_path(Path("data/artifacts/patches"))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _failed(
    message: str,
    *,
    details: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        ok=False,
        stderr=message,
        data=data or details or {},
        risk_level="code_write",
        permission_level=3,
    )
