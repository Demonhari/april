from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def open_url(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        return ToolResult(
            ok=False,
            stderr="Opening URLs is an external/system action and is disabled in the MVP executor.",
            risk_level="external_action",
            permission_level=5,
        )

    return await timed_tool(run, risk_level="external_action", permission_level=5)


def open_url_definition() -> ToolDefinition:
    return ToolDefinition(
        name="open_url",
        description="Open a URL after approval. Disabled in MVP executor.",
        permission_level=5,
        risk_level="external_action",
        confirmation_required=True,
        allowed_agents={"system_action_agent"},
        executor=open_url,
    )
