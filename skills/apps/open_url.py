from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from april_common.effective_config import load_tools_file
from april_common.settings import get_settings
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult

OPEN_BINARY = Path("/usr/bin/open")


async def open_url(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        raw_url = str(args.get("url") or "").strip()
        parsed = urlparse(raw_url)
        settings = get_settings()
        tools = load_tools_file(settings.home)
        allowed_schemes = set(tools.tools.open_url_allowed_schemes)
        if parsed.scheme.lower() not in allowed_schemes or parsed.scheme.lower() not in {
            "http",
            "https",
        }:
            return ToolResult(
                ok=False,
                stderr="URL must use a configured http or https scheme.",
                risk_level="external_action",
                permission_level=5,
            )
        if not parsed.netloc or parsed.username or parsed.password:
            return ToolResult(
                ok=False,
                stderr="URL must include a host and must not include credentials.",
                risk_level="external_action",
                permission_level=5,
            )
        normalized_url = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path or "/",
                "",
                parsed.query,
                parsed.fragment,
            )
        )
        if sys.platform != "darwin" or not OPEN_BINARY.exists():
            return ToolResult(
                ok=False,
                stderr="open_url is only available on macOS with /usr/bin/open.",
                risk_level="external_action",
                permission_level=5,
            )
        completed = subprocess.run(
            [str(OPEN_BINARY), normalized_url],
            capture_output=True,
            text=True,
            check=False,
        )
        return ToolResult(
            ok=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            data={"url": normalized_url},
            risk_level="external_action",
            permission_level=5,
        )

    return await timed_tool(run, risk_level="external_action", permission_level=5)


def open_url_definition() -> ToolDefinition:
    return ToolDefinition(
        name="open_url",
        description="Open a normalized http/https URL after exact approval.",
        permission_level=5,
        risk_level="external_action",
        confirmation_required=True,
        allowed_agents={"system_action_agent"},
        executor=open_url,
    )
