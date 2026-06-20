from __future__ import annotations

from typing import Any

from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def open_app(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        return ToolResult(
            ok=False,
            stderr="Opening applications is approval-gated and disabled in the MVP executor.",
            risk_level="system_action",
            permission_level=4,
        )

    return await timed_tool(run, risk_level="system_action", permission_level=4)


def open_app_definition() -> ToolDefinition:
    return ToolDefinition(
        name="open_app",
        description="Open a local application after approval. Disabled in MVP executor.",
        permission_level=4,
        risk_level="system_action",
        confirmation_required=True,
        allowed_agents={"system_action_agent"},
        executor=open_app,
    )
