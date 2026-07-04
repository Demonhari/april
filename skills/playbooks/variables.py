from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.memory.sqlite_memory import SqliteMemory
from skills.playbooks.schema import PlaybookDefinition, PlaybookStep

# Procedural variables are pure, deterministic data substitution. They can
# never introduce new tools, change permission levels, or execute anything:
# expanded steps still run one by one through ToolExecutionService and the
# PermissionEngine, so Level 3+ steps keep requiring exact-action approval.
LAST_RUN_TOKEN = "$last_run"
EACH_ACTIVE_PROJECTS_TOKEN = "$each(active_projects)"

# Bounded expansion: at most this many projects per $each step, and at most
# this many total steps after expansion.
MAX_EACH_PROJECTS = 10
MAX_EXPANDED_STEPS = 40


class PlaybookExpansionError(ValueError):
    """Raised when deterministic expansion would exceed its bounds."""


@dataclass(slots=True)
class ExpandedSteps:
    steps: list[PlaybookStep]
    notes: list[str] = field(default_factory=list)


async def expand_playbook_steps(
    playbook: PlaybookDefinition,
    *,
    memory: SqliteMemory,
) -> ExpandedSteps:
    """Resolve procedural variables into a bounded, deterministic step list.

    - ``$last_run`` in any string argument becomes a short factual summary of
      this playbook's most recent recorded run ("none" when it never ran).
    - ``$each(active_projects)`` expands its step into one step per registered
      project (ordered by created_at then id, capped), with the token replaced
      by the project path and ``project_id`` injected when absent.
    """
    last_run_summary = await _last_run_summary(memory, playbook.id)
    notes: list[str] = []
    expanded: list[PlaybookStep] = []
    projects_cache: list[Any] | None = None
    for step in playbook.steps:
        step = _substitute_step(step, LAST_RUN_TOKEN, last_run_summary)
        if not _step_mentions(step, EACH_ACTIVE_PROJECTS_TOKEN):
            expanded.append(step)
            continue
        if projects_cache is None:
            projects = await memory.list_projects()
            projects_cache = sorted(projects, key=lambda item: (item.created_at, item.id))[
                :MAX_EACH_PROJECTS
            ]
            if len(projects) > MAX_EACH_PROJECTS:
                notes.append(
                    f"$each(active_projects) capped at {MAX_EACH_PROJECTS} of "
                    f"{len(projects)} projects"
                )
        if not projects_cache:
            notes.append("$each(active_projects) expanded to zero steps (no projects)")
            continue
        for project in projects_cache:
            per_project = _substitute_step(step, EACH_ACTIVE_PROJECTS_TOKEN, project.path)
            args = dict(per_project.args)
            args.setdefault("project_id", project.id)
            expanded.append(per_project.model_copy(update={"args": args}))
    if len(expanded) > MAX_EXPANDED_STEPS:
        raise PlaybookExpansionError(
            f"playbook expansion produced {len(expanded)} steps (maximum {MAX_EXPANDED_STEPS})"
        )
    return ExpandedSteps(steps=expanded, notes=notes)


async def _last_run_summary(memory: SqliteMemory, playbook_id: str) -> str:
    runs = await memory.list_playbook_runs(playbook_id=playbook_id, limit=1)
    if not runs:
        return "none"
    run = runs[0]
    return (
        f"status={run.get('status', 'unknown')} "
        f"completed_at={run.get('completed_at') or 'never'} "
        f"steps_completed={run.get('steps_completed', 0)}"
    )


def _substitute_step(step: PlaybookStep, token: str, replacement: str) -> PlaybookStep:
    changed = False
    new_args: dict[str, Any] = {}
    for key, value in step.args.items():
        if isinstance(value, str) and token in value:
            new_args[key] = value.replace(token, replacement)
            changed = True
        else:
            new_args[key] = value
    if not changed:
        return step
    return step.model_copy(update={"args": new_args})


def _step_mentions(step: PlaybookStep, token: str) -> bool:
    return any(isinstance(value, str) and token in value for value in step.args.values())
