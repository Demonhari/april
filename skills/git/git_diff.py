from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.git.common import git_paths, run_git
from skills.schemas import ToolDefinition, ToolResult


async def git_diff(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        code, stdout, stderr = await run_git(args["repo_path"], ["diff", "--", "."])
        return ToolResult(
            ok=code == 0,
            stdout=stdout,
            stderr=stderr,
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def git_diff_definition() -> ToolDefinition:
    return ToolDefinition(
        name="git_diff",
        description="Read Git diff.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"coding_agent"},
        executor=git_diff,
        affected_paths=git_paths,
    )
