from __future__ import annotations

# Mechanical extraction keeps the original type vocabulary available.
# ruff: noqa: F401
# mypy: disable-error-code="attr-defined"
import json
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from april_common.errors import PermissionDeniedError
from april_common.text_normalization import normalize_text, word_tokens
from april_common.time import utc_now_iso
from services.brain.planner import TaskPlan, TaskStep
from services.memory.encryption import UNAVAILABLE_CONTENT, SensitiveMemoryEncryption
from services.memory.schemas import (
    Conversation,
    ConversationSummary,
    ConversationSummaryContent,
    FeedbackEventRecord,
    LexicalHit,
    MemoryContradictionRecord,
    MemoryRecord,
    Message,
    Project,
    ReminderRecord,
    SessionRecord,
    SuspendedAgentRun,
    WakeEventRecord,
)
from services.memory.sqlite_base import SqliteRepositoryBase


class AgentRunRepository(SqliteRepositoryBase):
    async def create_task_plan(self, plan: TaskPlan) -> TaskPlan:
        title = plan.steps[0].title if plan.steps else plan.intent
        await self.database.execute(
            """
            INSERT INTO tasks(
                id, title, status, conversation_id, request_id, intent, agent,
                model_id, steps_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.id,
                title,
                plan.status,
                plan.conversation_id,
                plan.request_id,
                plan.intent,
                plan.agent,
                plan.model_id,
                json.dumps([step.model_dump() for step in plan.steps], sort_keys=True),
                plan.created_at,
            ),
        )
        return plan

    async def update_task_status(self, task_id: str, status: str) -> None:
        await self.database.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))

    async def list_tasks(self) -> list[TaskPlan]:
        rows = await self.database.fetchall("SELECT * FROM tasks ORDER BY created_at DESC")
        return [self._task_plan_from_row(row) for row in rows]

    async def record_agent_run(
        self,
        *,
        conversation_id: str | None,
        agent: str,
        status: str,
        model_id: str | None,
        summary: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO agent_runs(
                id, conversation_id, agent, status, model_id, summary, metadata_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                conversation_id,
                agent,
                status,
                model_id,
                summary,
                json.dumps(metadata or {}, sort_keys=True),
                utc_now_iso(),
            ),
        )
        return run_id

    async def record_agent_iteration(
        self,
        *,
        run_id: str,
        iteration: int,
        model_id: str | None,
        state: str,
        model_output: dict[str, Any] | None = None,
        tool_request: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
        approval_id: str | None = None,
        error: str | None = None,
    ) -> str:
        iteration_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO agent_iterations(
                id, run_id, iteration, model_id, state, model_output_json,
                tool_request_json, tool_result_json, approval_id, error, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iteration_id,
                run_id,
                iteration,
                model_id,
                state,
                json.dumps(model_output or {}, sort_keys=True),
                json.dumps(tool_request or {}, sort_keys=True),
                json.dumps(tool_result or {}, sort_keys=True),
                approval_id,
                error,
                utc_now_iso(),
            ),
        )
        return iteration_id

    async def create_suspended_agent_run(
        self,
        *,
        agent_run_id: str,
        approval_id: str,
        conversation_id: str,
        project_id: str | None,
        agent: str,
        model_id: str | None,
        iteration: int,
        request_id: str,
        messages: list[dict[str, Any]],
        tool_request: dict[str, Any],
        normalized_args: dict[str, Any],
        context: dict[str, Any],
    ) -> SuspendedAgentRun:
        suspended_id = str(uuid.uuid4())
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO suspended_agent_runs(
                    id, agent_run_id, approval_id, conversation_id, project_id, agent,
                    model_id, iteration, request_id, messages_json, tool_request_json,
                    normalized_args_json, context_json, status, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'suspended', ?)
                """,
                (
                    suspended_id,
                    agent_run_id,
                    approval_id,
                    conversation_id,
                    project_id,
                    agent,
                    model_id,
                    iteration,
                    request_id,
                    json.dumps(messages, sort_keys=True),
                    json.dumps(tool_request, sort_keys=True),
                    json.dumps(normalized_args, sort_keys=True),
                    json.dumps(context, sort_keys=True),
                    now,
                ),
            )
            await conn.execute(
                "UPDATE agent_runs SET status = 'suspended' WHERE id = ?",
                (agent_run_id,),
            )
        return SuspendedAgentRun(
            id=suspended_id,
            agent_run_id=agent_run_id,
            approval_id=approval_id,
            conversation_id=conversation_id,
            project_id=project_id,
            agent=agent,
            model_id=model_id,
            iteration=iteration,
            request_id=request_id,
            messages=messages,
            tool_request=tool_request,
            normalized_args=normalized_args,
            context=context,
            status="suspended",
            created_at=now,
        )

    async def get_suspended_agent_run_by_approval(
        self, approval_id: str
    ) -> SuspendedAgentRun | None:
        row = await self.database.fetchone(
            "SELECT * FROM suspended_agent_runs WHERE approval_id = ?",
            (approval_id,),
        )
        if row is None:
            return None
        return self._suspended_run_from_row(row)

    async def mark_agent_run_resumed(self, *, approval_id: str) -> None:
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            cursor = await conn.execute(
                "SELECT agent_run_id FROM suspended_agent_runs WHERE approval_id = ?",
                (approval_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            suspended_cursor = await conn.execute(
                """
                UPDATE suspended_agent_runs
                SET status = 'resumed', resumed_at = ?
                WHERE approval_id = ? AND status = 'suspended'
                """,
                (now, approval_id),
            )
            if suspended_cursor.rowcount != 1:
                raise RuntimeError("Suspended agent run is not resumable.")
            run_cursor = await conn.execute(
                """
                UPDATE agent_runs
                SET status = 'running'
                WHERE id = ? AND status = 'suspended'
                """,
                (row["agent_run_id"],),
            )
            if run_cursor.rowcount != 1:
                raise RuntimeError("Agent run is not suspended.")

    async def mark_agent_run_completed(self, *, agent_run_id: str, status: str = "ok") -> None:
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                UPDATE agent_runs
                SET status = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, now, agent_run_id),
            )
            await conn.execute(
                """
                UPDATE suspended_agent_runs
                SET status = 'completed', completed_at = ?
                WHERE agent_run_id = ? AND status IN ('suspended', 'resumed')
                """,
                (now, agent_run_id),
            )

    async def mark_agent_run_denied(self, *, approval_id: str) -> None:
        await self._mark_suspended_terminal(
            approval_id=approval_id,
            suspended_status="denied",
            run_status="denied",
        )

    async def mark_agent_run_expired(self, *, approval_id: str) -> None:
        await self._mark_suspended_terminal(
            approval_id=approval_id,
            suspended_status="expired",
            run_status="expired",
        )

    async def mark_agent_run_failed(self, *, approval_id: str) -> None:
        await self._mark_suspended_terminal(
            approval_id=approval_id,
            suspended_status="failed",
            run_status="failed",
        )

    async def fail_suspended_agent_run(
        self,
        *,
        approval_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Close a suspended run while retaining its actual sanitized failure."""

        now = utc_now_iso()
        async with self.database.transaction() as conn:
            cursor = await conn.execute(
                "SELECT agent_run_id FROM suspended_agent_runs WHERE approval_id = ?",
                (approval_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            await conn.execute(
                """
                UPDATE suspended_agent_runs
                SET messages_json = ?, status = 'failed', completed_at = ?
                WHERE approval_id = ?
                """,
                (json.dumps(messages, sort_keys=True), now, approval_id),
            )
            await conn.execute(
                "UPDATE agent_runs SET status = 'failed', completed_at = ? WHERE id = ?",
                (now, str(row["agent_run_id"])),
            )

    async def _mark_suspended_terminal(
        self,
        *,
        approval_id: str,
        suspended_status: str,
        run_status: str,
    ) -> None:
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            cursor = await conn.execute(
                "SELECT agent_run_id FROM suspended_agent_runs WHERE approval_id = ?",
                (approval_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            await conn.execute(
                """
                UPDATE suspended_agent_runs
                SET status = ?, completed_at = ?
                WHERE approval_id = ?
                """,
                (suspended_status, now, approval_id),
            )
            await conn.execute(
                """
                UPDATE agent_runs
                SET status = ?, completed_at = ?
                WHERE id = ?
                """,
                (run_status, now, row["agent_run_id"]),
            )

    def _suspended_run_from_row(self, row: Any) -> SuspendedAgentRun:
        data = dict(row)
        data["messages"] = json.loads(data.pop("messages_json"))
        data["tool_request"] = json.loads(data.pop("tool_request_json"))
        data["normalized_args"] = json.loads(data.pop("normalized_args_json"))
        data["context"] = json.loads(data.pop("context_json"))
        return SuspendedAgentRun.model_validate(data)

    def _task_plan_from_row(self, row: Any) -> TaskPlan:
        data = dict(row)
        raw_steps = data.get("steps_json") or "[]"
        try:
            steps_data = json.loads(raw_steps)
        except json.JSONDecodeError:
            steps_data = []
        steps = [TaskStep.model_validate(step) for step in steps_data if isinstance(step, dict)]
        if not steps:
            steps = [TaskStep(index=1, title=str(data.get("title") or "Task"))]
        status = str(data.get("status") or "planned")
        if status not in {"planned", "running", "completed", "pending_approval", "error"}:
            status = "planned"
        return TaskPlan(
            id=str(data["id"]),
            conversation_id=str(data.get("conversation_id") or ""),
            request_id=str(data.get("request_id") or ""),
            intent=str(data.get("intent") or data.get("title") or "task"),
            agent=str(data.get("agent") or ""),
            model_id=str(data.get("model_id") or ""),
            steps=steps,
            status=status,
            created_at=str(data["created_at"]),
        )
