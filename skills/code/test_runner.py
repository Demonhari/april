from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult
from skills.terminal.command_policy import run_restricted_command


async def test_runner(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        argv = list(args.get("argv", ["pytest"]))
        code, stdout, stderr = await run_restricted_command(
            argv, args["repo_path"], timeout=args.get("timeout")
        )
        return ToolResult(
            ok=code == 0,
            stdout=stdout,
            stderr=stderr,
            data={"returncode": code},
            risk_level="code_write",
            permission_level=3,
        )

    return await timed_tool(run, risk_level="code_write", permission_level=3)


def test_runner_definition() -> ToolDefinition:
    return ToolDefinition(
        name="test_runner",
        description="Run configured tests after approval.",
        permission_level=3,
        risk_level="code_write",
        confirmation_required=True,
        allowed_agents={"coding_agent"},
        executor=test_runner,
        affected_paths=lambda args: [str(args.get("repo_path", ""))],
    )
