from __future__ import annotations

import asyncio

from april_common.errors import PermissionDeniedError
from april_common.path_security import normalize_existing_path
from skills.filesystem.common import current_path_policy

MAX_GIT_OUTPUT = 200_000


async def run_git(
    repo_path: str, args: list[str], *, timeout: float = 15.0
) -> tuple[int, str, str]:
    policy = current_path_policy()
    repo = normalize_existing_path(repo_path, policy)
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise PermissionDeniedError("Path is not a Git repository.", {"path": str(repo)})
    process = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repo),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, "", "Git command timed out."
    stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_GIT_OUTPUT]
    stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_GIT_OUTPUT]
    return process.returncode or 0, stdout, stderr


def git_paths(args: dict[str, object]) -> list[str]:
    value = args.get("repo_path")
    return [str(value)] if value is not None else []
