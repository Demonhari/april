from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.git.common import git_paths, run_git
from skills.schemas import ToolDefinition, ToolResult


async def git_commit(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        message = str(args["message"])
        code, stdout, stderr = await run_git(args["repo_path"], ["commit", "-m", message])
        commit_hash = None
        if code == 0:
            hash_code, hash_stdout, hash_stderr = await run_git(
                args["repo_path"],
                ["rev-parse", "HEAD"],
            )
            if hash_code == 0:
                commit_hash = hash_stdout.strip()
            elif hash_stderr:
                stderr = f"{stderr}\n{hash_stderr}".strip()
        return ToolResult(
            ok=code == 0,
            stdout=stdout,
            stderr=stderr,
            data={"commit_hash": commit_hash} if commit_hash else {},
            risk_level="code_write",
            permission_level=3,
        )

    return await timed_tool(run, risk_level="code_write", permission_level=3)


def git_commit_definition() -> ToolDefinition:
    return ToolDefinition(
        name="git_commit",
        description="Create a local Git commit after approval.",
        permission_level=3,
        risk_level="code_write",
        confirmation_required=True,
        allowed_agents={"coding_agent"},
        executor=git_commit,
        affected_paths=git_paths,
    )
