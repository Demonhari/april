from __future__ import annotations

from services.memory.sqlite_memory import SqliteMemory
from services.scheduler.notifications import Notification

BRIEFING_TITLE = "APRIL Daily Briefing"
CLOSED_TASK_STATUSES = {"completed", "done", "cancelled"}
_MAX_TASK_TITLES = 5
_EMPTY_BODY = "Nothing scheduled. No open tasks or upcoming reminders."


def _task_title(title: str, intent: str, fallback: str) -> str:
    for candidate in (title, intent, fallback):
        stripped = candidate.strip()
        if stripped:
            return stripped
    return fallback


async def compose_briefing(memory: SqliteMemory, *, now_iso: str, until_iso: str) -> Notification:
    """Build a plain-text daily briefing Notification with no LLM or external I/O.

    Pure read-only assembly over memory: open tasks, reminders due within the window,
    and the project count. Notification-safe (no markdown) so any sink can render it.
    """
    open_tasks = [
        task for task in await memory.list_tasks() if task.status not in CLOSED_TASK_STATUSES
    ]
    upcoming = await memory.list_upcoming_reminders(now_iso, until_iso)
    project_count = len(await memory.list_projects())

    if not open_tasks and not upcoming:
        body = _EMPTY_BODY
    else:
        lines: list[str] = []
        lines.append(f"Open tasks ({len(open_tasks)}):")
        if open_tasks:
            for task in open_tasks[:_MAX_TASK_TITLES]:
                first_step = task.steps[0].title if task.steps else ""
                lines.append(f"- {_task_title(first_step, task.intent, task.id)}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Reminders due:")
        if upcoming:
            for reminder in upcoming:
                lines.append(f"- {reminder.content} (due {reminder.due_at})")
        else:
            lines.append("- none")
        lines.append("")
        lines.append(f"Projects: {project_count}")
        body = "\n".join(lines)

    return Notification(
        kind="briefing",
        title=BRIEFING_TITLE,
        body=body,
        created_at=now_iso,
    )
