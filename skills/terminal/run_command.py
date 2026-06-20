from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult
from skills.terminal.command_policy import run_restricted_command


async def run_command(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        code, stdout, stderr = await run_restricted_command(
            list(args["argv"]),
            args["cwd"],
            timeout=args.get("timeout"),
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


def _argument_risk(args: dict[str, Any]) -> str:
    argv = args.get("argv", [])
    if argv and str(argv[0]).endswith("open"):
        return "system_action"
    return "code_write"


def run_command_definition() -> ToolDefinition:
    return ToolDefinition(
        name="run_command",
        description="Run a configured developer command after approval.",
        permission_level=3,
        risk_level="code_write",
        confirmation_required=True,
        allowed_agents={"coding_agent", "system_action_agent"},
        executor=run_command,
        argument_risk=_argument_risk,
        affected_paths=lambda args: [str(args.get("cwd", ""))],
    )
