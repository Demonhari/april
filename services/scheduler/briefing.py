from __future__ import annotations

import math
import re
from collections.abc import Mapping

from services.memory.sqlite_memory import SqliteMemory
from services.scheduler.notifications import Notification
from services.scheduler.repo_monitor import RepoActivity

BRIEFING_TITLE = "APRIL Daily Briefing"
CLOSED_TASK_STATUSES = {"completed", "done", "cancelled"}
_MAX_TASK_TITLES = 5
_EMPTY_BODY = "Nothing scheduled. No open tasks or upcoming reminders."
_MAX_OVERNIGHT_CHARS = 600


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


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _notification_text(value: object, *, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def _label(value: object) -> str:
    return _notification_text(value, limit=40)


def _field(report: Mapping[str, object], phase: Mapping[str, object], key: str) -> object:
    if key in report:
        return report[key]
    return phase.get(key)


def _learned_memory_part(report: Mapping[str, object], distill: Mapping[str, object]) -> str | None:
    raw = _field(report, distill, "memories_learned")
    total = _count(raw)
    breakdown_raw: object | None = None
    if isinstance(raw, Mapping):
        total = _count(raw.get("total"))
        breakdown_raw = raw.get("by_kind")
    if breakdown_raw is None:
        breakdown_raw = _field(report, distill, "memories_by_kind")
    breakdown = _mapping(breakdown_raw)
    by_kind = [
        f"{_label(kind)} {count}"
        for kind, value in sorted(breakdown.items(), key=lambda item: str(item[0]))
        if _label(kind) and (count := _count(value)) is not None
    ]
    if total is None and by_kind:
        # A by-kind map is itself report data, so summing its explicit counts
        # does not fabricate a missing value.
        counts = [_count(value) for value in breakdown.values()]
        if all(count is not None for count in counts):
            total = sum(count for count in counts if count is not None)
    if total is None:
        return None
    noun = "memory" if total == 1 else "memories"
    suffix = f" ({', '.join(by_kind)})" if by_kind else ""
    return f"learned {total} {noun}{suffix}"


def format_evolution_report(evolution_report: Mapping[str, object]) -> str:
    """Format one bounded, notification-safe Dreamer report paragraph."""
    status = _notification_text(evolution_report.get("status", "unknown"), limit=40)
    if status != "completed":
        line = f"Dreamer: {status}"
        reason = _notification_text(evolution_report.get("reason", ""), limit=120)
        if reason:
            line += f" ({reason})"
        return line

    phases = _mapping(evolution_report.get("phases"))
    distill = _mapping(phases.get("distill"))
    mine = _mapping(phases.get("mine"))
    evolve = _mapping(phases.get("evolve"))
    parts: list[str] = []

    learned = _learned_memory_part(evolution_report, distill)
    if learned is not None:
        parts.append(learned)

    count_specs = (
        ("duplicates_merged", "merged {count} duplicate{suffix}"),
        ("memories_fading", "{count} memor{suffix} fading"),
        ("contradictions_resolved", "resolved {count} contradiction{suffix}"),
    )
    for key, template in count_specs:
        count = _count(_field(evolution_report, distill, key))
        if count is None:
            continue
        if key == "memories_fading":
            suffix = "y" if count == 1 else "ies"
        else:
            suffix = "" if count == 1 else "s"
        parts.append(template.format(count=count, suffix=suffix))

    adopted = _count(_field(evolution_report, mine, "playbooks_adopted"))
    if adopted is None:
        adopted = _count(mine.get("adopted"))
    if adopted is not None:
        suffix = "" if adopted == 1 else "s"
        parts.append(f"adopted {adopted} playbook{suffix}")

    routing_eval = _mapping(evolution_report.get("routing_eval"))
    if not routing_eval:
        ladder = _mapping(evolve.get("ladder_thresholds"))
        routing_eval = _mapping(ladder.get("evaluation"))
    score = _number(routing_eval.get("score"))
    baseline = _number(routing_eval.get("baseline"))
    if score is not None and baseline is not None:
        delta = score - baseline
        parts.append(f"routing-eval score {score:g} ({delta:+g} vs baseline {baseline:g})")

    candidate_outcomes = _mapping(evolution_report.get("candidate_outcomes"))
    awaiting = _count(candidate_outcomes.get("approval_required_count"))
    if awaiting is None:
        awaiting = _count(evolution_report.get("approval_required_count"))
    if awaiting is not None:
        parts.append(f"{awaiting} overlay candidates awaiting approval")

    paragraph = "Overnight: " + ("; ".join(parts) if parts else "Dreamer completed") + "."
    if len(paragraph) > _MAX_OVERNIGHT_CHARS:
        paragraph = paragraph[: _MAX_OVERNIGHT_CHARS - 3].rstrip() + "..."
    return paragraph


async def compose_briefing(
    memory: SqliteMemory,
    *,
    now_iso: str,
    until_iso: str,
    repo_activity: list[RepoActivity] | None = None,
    evolution_report: dict[str, object] | None = None,
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
    if evolution_report is not None:
        body = body + "\n\n" + format_evolution_report(evolution_report)

    return Notification(
        kind="briefing",
        title=BRIEFING_TITLE,
        body=body,
        created_at=now_iso,
    )
