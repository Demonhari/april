from __future__ import annotations

import json
from typing import Any

from april_common.settings import get_settings
from services.memory.reminder_store import ReminderStore
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def list_reminders(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        store = await ReminderStore.open(settings.database_path)
        try:
            reminders = [reminder.model_dump() for reminder in await store.list()]
        finally:
            await store.close()
        return ToolResult(
            ok=True,
            stdout=json.dumps(reminders, indent=2),
            data={"reminders": reminders},
            risk_level="read_only",
            permission_level=1,
        )

    return await timed_tool(run, risk_level="read_only", permission_level=1)


def list_reminders_definition() -> ToolDefinition:
    return ToolDefinition(
        name="list_reminders",
        description="List local reminder records.",
        permission_level=1,
        risk_level="read_only",
        allowed_agents={"general_agent"},
        executor=list_reminders,
    )
