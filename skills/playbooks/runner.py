from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from services.memory.sqlite_memory import SqliteMemory
from services.permissions.tool_execution import ToolExecutionService
from skills.playbooks.schema import PlaybookDefinition, PlaybookStep
from skills.playbooks.variables import PlaybookExpansionError, expand_playbook_steps


@dataclass(frozen=True, slots=True)
class PlaybookStepResult:
    step_index: int
    tool: str
    status: Literal["executed", "pending_approval", "failed"]
    result: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PlaybookRunResult:
    playbook_id: str
    status: Literal["completed", "pending_approval", "failed"]
    steps_completed: int
    steps: tuple[PlaybookStepResult, ...] = field(default_factory=tuple)
    run_id: str | None = None


class PlaybookRunner:
    def __init__(
        self,
        tool_executor: ToolExecutionService,
        *,
        memory: SqliteMemory | None = None,
    ) -> None:
        self.tool_executor = tool_executor
        self.memory = memory

    async def run(
        self,
        playbook: PlaybookDefinition,
        *,
        conversation_id: str | None = None,
        project_id: str | None = None,
        actor: str = "local-user",
        source: Literal["api", "cli"] = "api",
    ) -> PlaybookRunResult:
        # Deterministic, bounded procedural-variable expansion happens before
        # this run is recorded, so $last_run refers to the *previous* run.
        # Expanded steps are data only; each still runs through the permission
        # engine below, so L3+ steps keep their exact-action approvals.
        steps: list[PlaybookStep] = list(playbook.steps)
        expansion_error: str | None = None
        if self.memory is not None:
            try:
                expansion = await expand_playbook_steps(playbook, memory=self.memory)
                steps = expansion.steps
            except PlaybookExpansionError as exc:
                expansion_error = str(exc)
        run_id = await self._start_run(playbook, conversation_id=conversation_id)
        if expansion_error is not None:
            await self._finish_run(
                run_id, status="failed", steps_completed=0, detail=expansion_error
            )
            return PlaybookRunResult(playbook.id, "failed", 0, (), run_id=run_id)
        results: list[PlaybookStepResult] = []
        completed = 0
        status: Literal["completed", "pending_approval", "failed"] = "completed"
        detail: str | None = None
        for index, step in enumerate(steps):
            outcome = await self._run_step(
                playbook,
                step,
                index=index,
                conversation_id=conversation_id,
                project_id=project_id,
                actor=actor,
                source=source,
            )
            results.append(outcome)
            if outcome.status == "pending_approval":
                status = "pending_approval"
                detail = f"step {index} ({step.tool}) awaits exact-action approval"
                break
            if outcome.status == "failed":
                status = "failed"
                detail = f"step {index} ({step.tool}) failed"
                break
            completed += 1
        await self._finish_run(run_id, status=status, steps_completed=completed, detail=detail)
        return PlaybookRunResult(playbook.id, status, completed, tuple(results), run_id=run_id)

    async def _start_run(
        self, playbook: PlaybookDefinition, *, conversation_id: str | None
    ) -> str | None:
        if self.memory is None:
            return None
        await self.memory.upsert_playbook(
            playbook_id=playbook.id,
            name=playbook.name,
            source="loader",
            status=playbook.status,
            trigger_examples=list(playbook.trigger_examples),
            steps=[step.model_dump() for step in playbook.steps],
        )
        return await self.memory.create_playbook_run(
            playbook_id=playbook.id,
            conversation_id=conversation_id,
        )

    async def _finish_run(
        self,
        run_id: str | None,
        *,
        status: str,
        steps_completed: int,
        detail: str | None,
    ) -> None:
        if run_id is None or self.memory is None:
            return
        await self.memory.finish_playbook_run(
            run_id,
            status=status,
            steps_completed=steps_completed,
            detail=detail,
        )

    async def _run_step(
        self,
        playbook: PlaybookDefinition,
        step: PlaybookStep,
        *,
        index: int,
        conversation_id: str | None,
        project_id: str | None,
        actor: str,
        source: Literal["api", "cli"],
    ) -> PlaybookStepResult:
        agent_id = step.agent_id or playbook.agent_id
        context = await self.tool_executor.context(
            request_id=str(uuid.uuid4()),
            actor=actor,
            agent_id=agent_id,
            source=source,
            conversation_id=conversation_id,
            project_id=project_id or _project_id_from_args(step.args),
        )
        outcome = await self.tool_executor.request_or_execute(
            tool=step.tool,
            args=step.args,
            context=context,
        )
        return PlaybookStepResult(
            step_index=index,
            tool=step.tool,
            status=outcome.status,
            result=outcome.result.model_dump() if outcome.result is not None else None,
            approval=outcome.approval.model_dump() if outcome.approval is not None else None,
        )


def _project_id_from_args(args: dict[str, Any]) -> str | None:
    value = args.get("project_id")
    return str(value) if value else None
