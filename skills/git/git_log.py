from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.git.common import git_paths, run_git
from skills.schemas import ToolDefinition, ToolResult


async def git_log(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        limit = str(min(int(args.get("limit", 20)), 100))
        code, stdout, stderr = await run_git(args["repo_path"], ["log", "--oneline", f"-n{limit}"])
        return ToolResult(
            ok=code == 0,
            stdout=stdout,
            stderr=stderr,
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def git_log_definition() -> ToolDefinition:
    return ToolDefinition(
        name="git_log",
        description="Read recent Git history.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"coding_agent"},
        executor=git_log,
        affected_paths=git_paths,
    )
