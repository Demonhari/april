from __future__ import annotations

from pathlib import Path
from typing import Any

from april_common.errors import PermissionDeniedError
from april_common.project_scope import (
    git_apply_check,
    git_head,
    git_staged_digest,
    git_worktree_digest,
    inspect_patch_file,
    normalize_project_root,
    sha256_file,
    validate_patch_text,
)
from services.memory.schemas import ApprovalRecord
from skills.schemas import ToolResult


async def build_patch_approval_metadata(
    *, repo_path: str, patch_path: str, expected_side_effects: list[str]
) -> dict[str, Any]:
    artifact = await inspect_patch_file(patch_path=patch_path, repo_root=repo_path)
    return {
        "artifact_type": "patch",
        "patch_sha256": artifact.patch_sha256,
        "affected_paths": artifact.affected_paths,
        "repo_root": artifact.repo_root,
        "repo_head": artifact.repo_head,
        "repo_state_digest": artifact.repo_state_digest,
        "expected_side_effects": expected_side_effects,
    }


async def build_patch_text_metadata(
    *, repo_path: str, patch_text: str, patch_path: str, expected_side_effects: list[str]
) -> dict[str, Any]:
    root = normalize_project_root(repo_path)
    affected_paths = validate_patch_text(patch_text, root)
    return {
        "artifact_type": "patch",
        "patch_sha256": sha256_file(Path(patch_path)),
        "affected_paths": affected_paths,
        "repo_root": str(root),
        "repo_head": await git_head(root),
        "repo_state_digest": await git_worktree_digest(root),
        "expected_side_effects": expected_side_effects,
    }


async def build_git_commit_metadata(*, repo_path: str) -> dict[str, Any]:
    root = normalize_project_root(repo_path)
    return {
        "artifact_type": "git_commit",
        "repo_root": str(root),
        "repo_head": await git_head(root),
        "staged_diff_sha256": await git_staged_digest(root),
    }


async def verify_approval_artifact(record: ApprovalRecord) -> ToolResult | None:
    try:
        if record.tool == "patch_applier":
            return await _verify_patch(record)
        if record.tool == "git_commit":
            return await _verify_git_commit(record)
    except PermissionDeniedError as exc:
        return _failed(exc.message, details=exc.details)
    return None


async def _verify_patch(record: ApprovalRecord) -> ToolResult | None:
    metadata = record.metadata
    args = record.args
    repo_path = str(args.get("repo_path", ""))
    patch_path = str(args.get("patch_path", ""))
    expected_repo = metadata.get("repo_root")
    expected_sha = metadata.get("patch_sha256")
    expected_paths = sorted(str(path) for path in metadata.get("affected_paths", []))
    expected_state = metadata.get("repo_state_digest")
    root = normalize_project_root(repo_path)
    if expected_repo and str(root) != str(expected_repo):
        return _failed("Selected repository no longer matches the approved repository.")
    try:
        artifact = await inspect_patch_file(patch_path=patch_path, repo_root=root)
    except PermissionDeniedError as exc:
        return _failed(exc.message, details=exc.details)
    if expected_sha and artifact.patch_sha256 != expected_sha:
        return _failed("Patch digest changed after approval.")
    if expected_paths and artifact.affected_paths != expected_paths:
        return _failed("Patch affected paths changed after approval.")
    if expected_state and artifact.repo_state_digest != expected_state:
        return _failed("Repository state changed after patch approval.")
    ok, stdout, stderr = await git_apply_check(root, patch_path)
    if not ok:
        return ToolResult(
            ok=False,
            stdout=stdout,
            stderr=stderr or "git apply --check failed.",
            data={
                "patch_sha256": artifact.patch_sha256,
                "affected_paths": artifact.affected_paths,
                "repo_root": str(root),
            },
            risk_level="code_write",
            permission_level=3,
        )
    return None


async def _verify_git_commit(record: ApprovalRecord) -> ToolResult | None:
    metadata = record.metadata
    repo_path = str(record.args.get("repo_path", ""))
    root = normalize_project_root(repo_path)
    expected_repo = metadata.get("repo_root")
    expected_digest = metadata.get("staged_diff_sha256")
    if expected_repo and str(root) != str(expected_repo):
        return _failed("Selected repository no longer matches the approved repository.")
    current_digest = await git_staged_digest(root)
    if expected_digest and current_digest != expected_digest:
        return _failed(
            "Staged Git diff changed after approval.",
            data={"staged_diff_sha256": current_digest},
        )
    return None


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
