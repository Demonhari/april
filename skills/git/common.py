from __future__ import annotations

import asyncio

from april_common.errors import PermissionDeniedError
from april_common.path_security import normalize_existing_path
from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)
from skills.filesystem.common import current_path_policy

MAX_GIT_OUTPUT = 200_000


async def run_git(
    repo_path: str,
    args: list[str],
    *,
    timeout: float = 15.0,
    cancellation_event: asyncio.Event | None = None,
) -> tuple[int, str, str]:
    policy = current_path_policy()
    repo = normalize_existing_path(repo_path, policy)
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise PermissionDeniedError("Path is not a Git repository.", {"path": str(repo)})
    result = await run_restricted_process(
        ["git", "-C", str(repo), *args],
        cwd=repo,
        category=ProcessCategory.GIT,
        timeout_seconds=timeout,
        max_stdout_bytes=MAX_GIT_OUTPUT,
        max_stderr_bytes=MAX_GIT_OUTPUT,
        resource_limit_profile=ResourceLimitProfile.COMMAND,
        cancellation_event=cancellation_event,
    )
    if result.status is ProcessStatus.TIMED_OUT:
        return 124, result.stdout, result.stderr or "Git command timed out."
    if result.status is ProcessStatus.START_FAILED:
        return 126, "", f"Git command could not start ({result.failure_code})."
    if result.status is ProcessStatus.CANCELLED:
        return 130, result.stdout, result.stderr or "Git command cancelled."
    return result.returncode or 0, result.stdout, result.stderr


def git_paths(args: dict[str, object]) -> list[str]:
    value = args.get("repo_path")
    return [str(value)] if value is not None else []
