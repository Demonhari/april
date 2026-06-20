from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from april_common.settings import get_settings
from april_common.time import utc_now_iso
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def create_reminder(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        reminders_path = settings.resolve_path(Path("data/reminders.jsonl"))
        reminders_path.parent.mkdir(parents=True, exist_ok=True)
        reminder = {
            "id": str(uuid.uuid4()),
            "content": str(args["content"]),
            "due_at": args.get("due_at"),
            "created_at": utc_now_iso(),
        }
        with reminders_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(reminder, sort_keys=True) + "\n")
        return ToolResult(
            ok=True,
            stdout=reminder["id"],
            data=reminder,
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
