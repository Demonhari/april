from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from fastapi import Depends, FastAPI

from april_common.time import utc_now
from services.api.dependencies import ApiContainer
from services.api.schemas import ReminderCreateRequest
from services.evolution.dreamer import latest_report
from services.scheduler import compose_briefing, compute_repo_activity


def register_scheduler_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
) -> None:
    @app.get("/reminders")
    async def reminders(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "reminders": [
                reminder.model_dump() for reminder in await active.memory.list_reminders()
            ]
        }

    @app.post("/reminders")
    async def reminder_create(
        request: ReminderCreateRequest, active: ApiContainer = Depends(authorized)
    ) -> object:
        reminder = await active.memory.create_reminder(request.content, due_at=request.due_at)
        return {"reminder": reminder.model_dump()}

    @app.delete("/reminders/{reminder_id}")
    async def reminder_delete(
        reminder_id: str, active: ApiContainer = Depends(authorized)
    ) -> object:
        return {"deleted": await active.memory.delete_reminder(reminder_id)}

    @app.get("/tasks")
    async def tasks(active: ApiContainer = Depends(authorized)) -> object:
        return {"tasks": [task.model_dump() for task in await active.memory.list_tasks()]}

    @app.get("/scheduler/briefing/preview")
    async def scheduler_briefing_preview(
        active: ApiContainer = Depends(authorized),
    ) -> object:
        now = utc_now()
        until = now + timedelta(hours=24)
        repo_activity = None
        if active.settings.scheduler.repo_monitor_enabled:
            # Preview must not advance the baseline (persist=False, idempotent).
            repo_activity = await compute_repo_activity(active.memory, persist=False)
        notification = await compose_briefing(
            active.memory,
            now_iso=now.isoformat().replace("+00:00", "Z"),
            until_iso=until.isoformat().replace("+00:00", "Z"),
            repo_activity=repo_activity,
            evolution_report=latest_report(active.settings),
        )
        return notification.model_dump()
