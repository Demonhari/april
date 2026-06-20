from __future__ import annotations

from typing import Any

from april_common.settings import get_settings
from services.memory.reminder_store import ReminderStore
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def create_reminder(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        store = await ReminderStore.open(settings.database_path)
        try:
            reminder = await store.create(str(args["content"]), args.get("due_at"))
        finally:
            await store.close()
        return ToolResult(
            ok=True,
            stdout=reminder.id,
            data=reminder.model_dump(),
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def create_reminder_definition() -> ToolDefinition:
    return ToolDefinition(
        name="create_reminder",
        description="Create a local reminder record.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"general_agent"},
        executor=create_reminder,
    )
