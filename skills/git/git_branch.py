from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.git.common import git_paths, run_git
from skills.schemas import ToolDefinition, ToolResult


async def git_branch(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        code, stdout, stderr = await run_git(args["repo_path"], ["branch", "--show-current"])
        return ToolResult(
            ok=code == 0,
            stdout=stdout.strip(),
            stderr=stderr,
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def git_branch_definition() -> ToolDefinition:
    return ToolDefinition(
        name="git_branch",
        description="Read current Git branch.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"coding_agent"},
        executor=git_branch,
        affected_paths=git_paths,
    )
