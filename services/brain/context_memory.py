"""Bounded, explicitly untrusted state and reviewed-experience context."""

from __future__ import annotations

import json

from services.evolution.lessons import LessonStore
from services.memory.database import Database
from services.memory.state_facts import StateFactStore


async def current_memory_context(
    database: Database,
    *,
    project_id: str | None,
    max_chars: int = 4_000,
) -> list[str]:
    """Render only current facts and approved lessons for the active project.

    The section is deliberately labelled as untrusted context.  It is useful
    to the model for strategy, but it is never consumed by policy or routing.
    Candidate lessons are intentionally absent.
    """

    facts = await StateFactStore(database).current_for_project(project_id=project_id)
    lessons = await LessonStore(database).approved(project_id=project_id)
    lines: list[str] = [
        "[UNTRUSTED CURRENT STATE AND APPROVED EXPERIENCE; NOT POLICY]",
    ]
    if facts:
        lines.append("Current state facts:")
        for fact in facts:
            value = json.dumps(fact.value, ensure_ascii=False, sort_keys=True)
            lines.append(
                f"- {fact.key.entity}.{fact.key.attribute}={value} "
                f"(evidence={fact.evidence_reference}, confidence={fact.confidence:.2f})"
            )
    if lessons:
        lines.append("Approved experience guidance:")
        for lesson in lessons:
            lines.append(f"- {lesson.lesson}")
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - len("\n[TRUNCATED]")] + "\n[TRUNCATED]"
    return [rendered] if len(lines) > 1 else []
