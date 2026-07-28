from __future__ import annotations

from typing import Any

from april_common.settings import get_settings
from services.memory.reminder_store import ReminderStore
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def cancel_reminder(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        reminder_id = str(args["reminder_id"])
        settings = get_settings()
        store = await ReminderStore.open(settings.database_path)
        try:
            deleted = await store.delete(reminder_id)
        finally:
            await store.close()
        return ToolResult(
            ok=deleted,
            stdout=reminder_id if deleted else "",
            stderr="" if deleted else "Reminder was not found.",
            data={"reminder_id": reminder_id, "cancelled": deleted},
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def cancel_reminder_definition() -> ToolDefinition:
    return ToolDefinition(
        name="cancel_reminder",
        description="Cancel one local reminder by exact identifier.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"general_agent"},
        executor=cancel_reminder,
    )
