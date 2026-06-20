from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from april_common.effective_config import load_tools_file
from april_common.settings import get_settings
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult

OPEN_BINARY = Path("/usr/bin/open")


async def open_app(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        app_name = str(args.get("name") or args.get("app") or "").strip()
        if not app_name or "/" in app_name or "\x00" in app_name:
            return ToolResult(
                ok=False,
                stderr="Application name must be a configured plain app name.",
                risk_level="system_action",
                permission_level=4,
            )
        settings = get_settings()
        tools = load_tools_file(settings.home)
        if app_name not in tools.tools.open_app_allowlist:
            return ToolResult(
                ok=False,
                stderr="Application is not in the configured open_app allowlist.",
                risk_level="system_action",
                permission_level=4,
            )
        if sys.platform != "darwin" or not OPEN_BINARY.exists():
            return ToolResult(
                ok=False,
                stderr="open_app is only available on macOS with /usr/bin/open.",
                risk_level="system_action",
                permission_level=4,
            )
        completed = subprocess.run(
            [str(OPEN_BINARY), "-a", app_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return ToolResult(
            ok=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            data={"app": app_name},
            risk_level="system_action",
            permission_level=4,
        )

    return await timed_tool(run, risk_level="system_action", permission_level=4)


def open_app_definition() -> ToolDefinition:
    return ToolDefinition(
        name="open_app",
        description="Open a configured local macOS application after exact approval.",
        permission_level=4,
        risk_level="system_action",
        confirmation_required=True,
        allowed_agents={"system_action_agent"},
        executor=open_app,
    )
