from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from services.permissions.tool_execution import ToolExecutionService
from skills.playbooks.schema import PlaybookDefinition, PlaybookStep


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


class PlaybookRunner:
    def __init__(self, tool_executor: ToolExecutionService) -> None:
        self.tool_executor = tool_executor

    async def run(
        self,
        playbook: PlaybookDefinition,
        *,
        conversation_id: str | None = None,
        project_id: str | None = None,
        actor: str = "local-user",
        source: Literal["api", "cli"] = "api",
    ) -> PlaybookRunResult:
        results: list[PlaybookStepResult] = []
        completed = 0
        for index, step in enumerate(playbook.steps):
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
                return PlaybookRunResult(
                    playbook.id,
                    "pending_approval",
                    completed,
                    tuple(results),
                )
            if outcome.status == "failed":
                return PlaybookRunResult(playbook.id, "failed", completed, tuple(results))
            completed += 1
        return PlaybookRunResult(playbook.id, "completed", completed, tuple(results))

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
