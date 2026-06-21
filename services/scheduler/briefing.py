from __future__ import annotations

from services.memory.sqlite_memory import SqliteMemory
from services.scheduler.notifications import Notification
from services.scheduler.repo_monitor import RepoActivity

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


def _repo_activity_lines(repo_activity: list[RepoActivity] | None) -> list[str]:
    """Plain-text lines for projects that changed; empty when there is nothing to show."""
    if not repo_activity:
        return []
    changed: list[str] = []
    for activity in repo_activity:
        if not (activity.new_commits or activity.dirty_count > 0):
            continue
        parts: list[str] = []
        if activity.dirty_count > 0:
            noun = "file" if activity.dirty_count == 1 else "files"
            parts.append(f"{activity.dirty_count} uncommitted {noun}")
        if activity.new_commits:
            parts.append("new commits since last briefing")
        changed.append(f"- {activity.project_name}: {', '.join(parts)}")
    if not changed:
        return []
    return ["Project activity:", *changed]


async def compose_briefing(
    memory: SqliteMemory,
    *,
    now_iso: str,
    until_iso: str,
    repo_activity: list[RepoActivity] | None = None,
) -> Notification:
    """Build a plain-text daily briefing Notification with no LLM or external I/O.

    Pure read-only assembly over memory: open tasks, reminders due within the window,
    and the project count. Optionally appends a read-only project-activity section
    when repo_activity is supplied and contains changed projects (git I/O is done by
    the caller, never here). Notification-safe (no markdown) so any sink can render it.
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

    activity_lines = _repo_activity_lines(repo_activity)
    if activity_lines:
        body = body + "\n\n" + "\n".join(activity_lines)

    return Notification(
        kind="briefing",
        title=BRIEFING_TITLE,
        body=body,
        created_at=now_iso,
    )
