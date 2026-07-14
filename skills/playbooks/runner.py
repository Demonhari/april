from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from april_common.errors import PermissionDeniedError
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.tool_execution import ToolExecutionService
from skills.playbooks.schema import PlaybookDefinition, PlaybookStep
from skills.playbooks.variables import PlaybookExpansionError, expand_playbook_steps

RunStatus = Literal["completed", "pending_approval", "failed", "denied", "expired", "cancelled"]


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
    status: RunStatus
    steps_completed: int
    steps: tuple[PlaybookStepResult, ...] = field(default_factory=tuple)
    run_id: str | None = None


class PlaybookRunner:
    """Execute immutable playbook snapshots with durable exact-action resume."""

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
        steps: list[PlaybookStep] = list(playbook.steps)
        expansion_error: str | None = None
        if self.memory is not None:
            try:
                expansion = await expand_playbook_steps(playbook, memory=self.memory)
                steps = expansion.steps
            except PlaybookExpansionError as exc:
                expansion_error = str(exc)
        snapshot = [step.model_dump() for step in steps]
        snapshot_hash = _snapshot_hash(snapshot)
        run_id = await self._start_run(
            playbook,
            conversation_id=conversation_id,
            expanded_steps=snapshot,
            snapshot_hash=snapshot_hash,
        )
        if expansion_error is not None:
            await self._finish_run(
                run_id, status="failed", steps_completed=0, detail=expansion_error
            )
            return PlaybookRunResult(playbook.id, "failed", 0, (), run_id=run_id)
        return await self._continue_run(
            playbook_id=playbook.id,
            agent_id=playbook.agent_id,
            steps=steps,
            run_id=run_id,
            start_index=0,
            completed=0,
            states=[],
            conversation_id=conversation_id,
            project_id=project_id,
            actor=actor,
            source=source,
        )

    async def resume(
        self,
        run_id: str,
        *,
        approval_id: str,
        actor: str = "local-user",
        source: Literal["api", "cli"] = "api",
    ) -> PlaybookRunResult:
        if self.memory is None:
            raise PermissionDeniedError("Playbook resume requires durable run storage.")
        row = await self.memory.get_playbook_run(run_id)
        if row is None:
            raise PermissionDeniedError("Playbook run does not exist.")
        playbook_id = str(row["playbook_id"])
        if row["completed_at"] is not None:
            return self._stored_result(row)
        if row["status"] != "pending_approval":
            raise PermissionDeniedError(
                "Playbook run is not awaiting approval.", {"status": row["status"]}
            )
        if row["pending_approval_id"] != approval_id:
            raise PermissionDeniedError("Approval does not belong to this playbook run.")
        steps_payload = _json_list(row["expanded_steps_json"])
        if not steps_payload or _snapshot_hash(steps_payload) != row["snapshot_hash"]:
            await self._finish_run(
                run_id,
                status="failed",
                steps_completed=int(row["steps_completed"]),
                detail="Persisted playbook snapshot failed integrity validation.",
            )
            return self._stored_result((await self.memory.get_playbook_run(run_id)) or row)
        step_index = int(row["current_step_index"])
        if step_index < 0 or step_index >= len(steps_payload):
            raise PermissionDeniedError("Persisted playbook step index is invalid.")
        step = PlaybookStep.model_validate(steps_payload[step_index])
        action_hash = _action_hash(step)
        if row["pending_action_hash"] != action_hash:
            raise PermissionDeniedError("Pending playbook action digest changed.")
        approval = await self.tool_executor.approvals.get(approval_id)
        metadata = approval.metadata
        if (
            metadata.get("playbook_run_id") != run_id
            or metadata.get("playbook_step_index") != step_index
            or metadata.get("playbook_action_hash") != action_hash
        ):
            raise PermissionDeniedError("Approval metadata does not match the pending step.")
        if approval.status in {"denied", "expired"}:
            terminal: Literal["denied", "expired"] = (
                "denied" if approval.status == "denied" else "expired"
            )
            await self._finish_run(
                run_id,
                status=terminal,
                steps_completed=int(row["steps_completed"]),
                detail=f"step {step_index} approval {terminal}",
            )
            return self._stored_result((await self.memory.get_playbook_run(run_id)) or row)
        if approval.status != "pending":
            # An approved/consumed record after a process interruption is
            # ambiguous. Fail closed; never risk executing the action twice.
            await self._finish_run(
                run_id,
                status="failed",
                steps_completed=int(row["steps_completed"]),
                detail="Approval state is not safely resumable.",
            )
            return self._stored_result((await self.memory.get_playbook_run(run_id)) or row)
        outcome = await self.tool_executor.execute_approved(
            approval_id=approval_id,
            actor=actor,
            request_id=str(uuid.uuid4()),
            tool=step.tool,
            args=step.args,
        )
        states = _json_list(row["step_states_json"])
        if (
            states
            and states[-1].get("step_index") == step_index
            and states[-1].get("status") == "pending_approval"
        ):
            states.pop()
        if outcome.status != "executed":
            states.append(_state(step_index, step.tool, "failed"))
            await self.memory.update_playbook_run_progress(
                run_id,
                status="failed",
                current_step_index=step_index,
                steps_completed=int(row["steps_completed"]),
                step_states=states,
                detail=f"step {step_index} ({step.tool}) failed",
            )
            await self._finish_run(
                run_id,
                status="failed",
                steps_completed=int(row["steps_completed"]),
                detail=f"step {step_index} ({step.tool}) failed",
            )
            return self._stored_result((await self.memory.get_playbook_run(run_id)) or row)
        states.append(_state(step_index, step.tool, "executed"))
        completed = int(row["steps_completed"]) + 1
        await self.memory.update_playbook_run_progress(
            run_id,
            status="running",
            current_step_index=step_index + 1,
            steps_completed=completed,
            step_states=states,
        )
        agent_id = str(row.get("agent_id") or "general_agent")
        return await self._continue_run(
            playbook_id=playbook_id,
            agent_id=agent_id,
            steps=[PlaybookStep.model_validate(item) for item in steps_payload],
            run_id=run_id,
            start_index=step_index + 1,
            completed=completed,
            states=states,
            conversation_id=(
                str(row["conversation_id"]) if row["conversation_id"] is not None else None
            ),
            project_id=None,
            actor=actor,
            source=source,
        )

    async def mark_denied(self, run_id: str, *, approval_id: str) -> PlaybookRunResult:
        if self.memory is None:
            raise PermissionDeniedError("Playbook resume requires durable run storage.")
        row = await self.memory.get_playbook_run(run_id)
        if row is None or row["pending_approval_id"] != approval_id:
            raise PermissionDeniedError("Approval does not belong to this playbook run.")
        if row["completed_at"] is None:
            await self._finish_run(
                run_id,
                status="denied",
                steps_completed=int(row["steps_completed"]),
                detail=f"step {row['current_step_index']} approval denied",
            )
            row = (await self.memory.get_playbook_run(run_id)) or row
        return self._stored_result(row)

    async def _continue_run(
        self,
        *,
        playbook_id: str,
        agent_id: str,
        steps: list[PlaybookStep],
        run_id: str | None,
        start_index: int,
        completed: int,
        states: list[dict[str, Any]],
        conversation_id: str | None,
        project_id: str | None,
        actor: str,
        source: Literal["api", "cli"],
    ) -> PlaybookRunResult:
        results = [_step_result(item) for item in states]
        for index in range(start_index, len(steps)):
            step = steps[index]
            outcome = await self._run_step(
                playbook_id,
                agent_id,
                step,
                run_id=run_id,
                index=index,
                conversation_id=conversation_id,
                project_id=project_id,
                actor=actor,
                source=source,
            )
            results.append(outcome)
            if outcome.status == "pending_approval":
                approval_id = str((outcome.approval or {})["approval_id"])
                states.append(_state(index, step.tool, "pending_approval"))
                if run_id is not None and self.memory is not None:
                    await self.memory.update_playbook_run_progress(
                        run_id,
                        status="pending_approval",
                        current_step_index=index,
                        steps_completed=completed,
                        step_states=states,
                        detail=f"step {index} ({step.tool}) awaits exact-action approval",
                        pending_approval_id=approval_id,
                        pending_action_hash=_action_hash(step),
                    )
                return PlaybookRunResult(
                    playbook_id,
                    "pending_approval",
                    completed,
                    tuple(results),
                    run_id=run_id,
                )
            states.append(_state(index, step.tool, outcome.status))
            if outcome.status == "failed":
                if run_id is not None and self.memory is not None:
                    await self.memory.update_playbook_run_progress(
                        run_id,
                        status="failed",
                        current_step_index=index,
                        steps_completed=completed,
                        step_states=states,
                        detail=f"step {index} ({step.tool}) failed",
                    )
                await self._finish_run(
                    run_id,
                    status="failed",
                    steps_completed=completed,
                    detail=f"step {index} ({step.tool}) failed",
                )
                return PlaybookRunResult(
                    playbook_id, "failed", completed, tuple(results), run_id=run_id
                )
            completed += 1
            if run_id is not None and self.memory is not None:
                await self.memory.update_playbook_run_progress(
                    run_id,
                    status="running",
                    current_step_index=index + 1,
                    steps_completed=completed,
                    step_states=states,
                )
        await self._finish_run(run_id, status="completed", steps_completed=completed, detail=None)
        return PlaybookRunResult(playbook_id, "completed", completed, tuple(results), run_id=run_id)

    async def _start_run(
        self,
        playbook: PlaybookDefinition,
        *,
        conversation_id: str | None,
        expanded_steps: list[dict[str, Any]],
        snapshot_hash: str,
    ) -> str | None:
        if self.memory is None:
            return None
        await self.memory.upsert_playbook(
            playbook_id=playbook.id,
            name=playbook.name,
            source=playbook.source,
            status=playbook.status,
            trigger_examples=list(playbook.trigger_examples),
            steps=[step.model_dump() for step in playbook.steps],
            required_permission_level=playbook.required_permission_level or 1,
            stats=playbook.stats.model_dump(),
        )
        return await self.memory.create_playbook_run(
            playbook_id=playbook.id,
            conversation_id=conversation_id,
            expanded_steps=expanded_steps,
            snapshot_hash=snapshot_hash,
            agent_id=playbook.agent_id,
        )

    async def _finish_run(
        self,
        run_id: str | None,
        *,
        status: RunStatus,
        steps_completed: int,
        detail: str | None,
    ) -> None:
        if run_id is not None and self.memory is not None:
            await self.memory.finish_playbook_run(
                run_id,
                status=status,
                steps_completed=steps_completed,
                detail=detail,
            )

    async def _run_step(
        self,
        playbook_id: str,
        agent_id: str,
        step: PlaybookStep,
        *,
        run_id: str | None,
        index: int,
        conversation_id: str | None,
        project_id: str | None,
        actor: str,
        source: Literal["api", "cli"],
    ) -> PlaybookStepResult:
        context = await self.tool_executor.context(
            request_id=str(uuid.uuid4()),
            actor=actor,
            agent_id=step.agent_id or agent_id,
            source=source,
            conversation_id=conversation_id,
            project_id=project_id or _project_id_from_args(step.args),
        )
        approval_metadata = (
            {
                "playbook_id": playbook_id,
                "playbook_run_id": run_id,
                "playbook_step_index": index,
                "playbook_action_hash": _action_hash(step),
            }
            if run_id is not None
            else None
        )
        outcome = await self.tool_executor.request_or_execute(
            tool=step.tool,
            args=step.args,
            context=context,
            approval_metadata=approval_metadata,
        )
        return PlaybookStepResult(
            step_index=index,
            tool=step.tool,
            status=outcome.status,
            result=outcome.result.model_dump() if outcome.result is not None else None,
            approval=outcome.approval.model_dump() if outcome.approval is not None else None,
        )

    @staticmethod
    def _stored_result(row: dict[str, Any]) -> PlaybookRunResult:
        raw_status = str(row["status"])
        status = cast(
            RunStatus,
            raw_status
            if raw_status
            in {"completed", "pending_approval", "failed", "denied", "expired", "cancelled"}
            else "failed",
        )
        states = _json_list(row.get("step_states_json"))
        return PlaybookRunResult(
            playbook_id=str(row["playbook_id"]),
            status=status,
            steps_completed=int(row["steps_completed"]),
            steps=tuple(_step_result(item) for item in states),
            run_id=str(row["id"]),
        )


def _snapshot_hash(steps: list[dict[str, Any]]) -> str:
    payload = json.dumps(steps, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _action_hash(step: PlaybookStep) -> str:
    payload = json.dumps(
        {"tool": step.tool, "args": step.args, "agent_id": step.agent_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_list(value: object) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError):
        return []
    return [item for item in decoded if isinstance(item, dict)] if isinstance(decoded, list) else []


def _state(index: int, tool: str, status: str) -> dict[str, Any]:
    return {"step_index": index, "tool": tool, "status": status}


def _step_result(state: dict[str, Any]) -> PlaybookStepResult:
    status = str(state.get("status", "failed"))
    normalized = cast(
        Literal["executed", "pending_approval", "failed"],
        status if status in {"executed", "pending_approval", "failed"} else "failed",
    )
    return PlaybookStepResult(
        step_index=int(state.get("step_index", 0)),
        tool=str(state.get("tool", "unknown")),
        status=normalized,
    )


def _project_id_from_args(args: dict[str, Any]) -> str | None:
    value = args.get("project_id")
    return str(value) if value else None
