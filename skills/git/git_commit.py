from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.git.common import git_paths, run_git
from skills.schemas import ToolDefinition, ToolResult


async def git_commit(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        message = str(args["message"])
        code, stdout, stderr = await run_git(args["repo_path"], ["commit", "-m", message])
        return ToolResult(
            ok=code == 0,
            stdout=stdout,
            stderr=stderr,
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
