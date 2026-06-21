from __future__ import annotations

import json
import uuid
from typing import Any

from april_common.errors import PermissionDeniedError
from april_common.time import utc_now_iso
from services.brain.planner import TaskPlan, TaskStep
from services.memory.database import Database
from services.memory.schemas import (
    Conversation,
    MemoryRecord,
    Message,
    Project,
    ReminderRecord,
    SuspendedAgentRun,
)


class SqliteMemory:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def add_project(self, path: str, name: str | None = None) -> Project:
        existing = await self.get_project_by_path(path)
        if existing is not None:
            return existing
        project_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        project_name = name or path.rstrip("/").split("/")[-1] or path
        await self.database.execute(
            "INSERT INTO projects(id, path, name, created_at) VALUES(?, ?, ?, ?)",
            (project_id, path, project_name, created_at),
        )
        return Project(id=project_id, path=path, name=project_name, created_at=created_at)

    async def get_project(self, project_id: str) -> Project | None:
        row = await self.database.fetchone("SELECT * FROM projects WHERE id = ?", (project_id,))
        if row is None:
            return None
        return Project.model_validate(dict(row))

    async def get_project_by_path(self, path: str) -> Project | None:
        row = await self.database.fetchone("SELECT * FROM projects WHERE path = ?", (path,))
        if row is None:
            return None
        return Project.model_validate(dict(row))

    async def list_projects(self) -> list[Project]:
        rows = await self.database.fetchall("SELECT * FROM projects ORDER BY created_at DESC")
        return [Project.model_validate(dict(row)) for row in rows]

    async def create_memory(
        self,
        content: str,
        *,
        kind: str = "fact",
        reason: str,
        project_id: str | None = None,
    ) -> MemoryRecord:
        memory_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO memories(id, project_id, kind, content, reason, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (memory_id, project_id, kind, content, reason, created_at),
            )
            await conn.execute(
                "INSERT INTO memories_fts(id, content, reason) VALUES(?, ?, ?)",
                (memory_id, content, reason),
            )
        return MemoryRecord(
            id=memory_id,
            content=content,
            kind=kind,
            project_id=project_id,
            reason=reason,
            created_at=created_at,
        )

    async def list_memories(self) -> list[MemoryRecord]:
        rows = await self.database.fetchall("SELECT * FROM memories ORDER BY created_at DESC")
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    async def search_memories(self, query: str) -> list[MemoryRecord]:
        if query.strip() in {"", "*"}:
            return await self.list_memories()
        rows = await self.database.fetchall(
            """
            SELECT m.*
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT 20
            """,
            (query,),
        )
        if not rows:
            rows = await self.database.fetchall(
                "SELECT * FROM memories WHERE content LIKE ? OR reason LIKE ? LIMIT 20",
                (f"%{query}%", f"%{query}%"),
            )
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    async def delete_memory(self, memory_id: str) -> bool:
        async with self.database.transaction() as conn:
            await conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            cursor = await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    async def export_memories(self) -> str:
        memories = [memory.model_dump() for memory in await self.list_memories()]
        return json.dumps({"memories": memories}, indent=2)

    async def create_conversation(
        self,
        title: str | None = None,
        *,
        project_id: str | None = None,
        actor: str = "local-user",
    ) -> str:
        conversation_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO conversations(id, title, project_id, actor, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, title, project_id, actor, created_at, created_at),
        )
        return conversation_id

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = await self.database.fetchone(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        if row is None:
            return None
        return Conversation.model_validate(dict(row))

    async def ensure_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        *,
        project_id: str | None = None,
        actor: str = "local-user",
    ) -> str:
        existing = await self.get_conversation(conversation_id)
        if existing is not None:
            if existing.project_id != project_id:
                raise PermissionDeniedError(
                    "Conversation project scope cannot change.",
                    {
                        "conversation_id": conversation_id,
                        "existing_project_id": existing.project_id,
                        "requested_project_id": project_id,
                    },
                )
            return conversation_id
        now = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO conversations(id, title, project_id, actor, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, title, project_id, actor, now, now),
        )
        return conversation_id

    async def add_message(self, conversation_id: str, role: str, content: str) -> str:
        message_id = str(uuid.uuid4())
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, now),
            )
            await conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return message_id

    async def recent_messages(self, conversation_id: str, *, limit: int = 8) -> list[Message]:
        rows = await self.database.fetchall(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        messages = [Message.model_validate(dict(row)) for row in rows]
        return list(reversed(messages))

    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self.database.transaction() as conn:
            await conn.execute(
                "DELETE FROM suspended_agent_runs WHERE conversation_id = ?",
                (conversation_id,),
            )
            cursor = await conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0

    async def record_conversation_event(
        self,
        *,
        conversation_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> str:
        event_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO conversation_events(
                id, conversation_id, event_type, payload_json, created_at
            )
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                event_id,
                conversation_id,
                event_type,
                json.dumps(payload, sort_keys=True),
                utc_now_iso(),
            ),
        )
        return event_id

    async def create_reminder(self, content: str, due_at: str | None = None) -> ReminderRecord:
        reminder_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO reminders(id, content, due_at, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (reminder_id, content, due_at, created_at),
        )
        return ReminderRecord(
            id=reminder_id,
            content=content,
            due_at=due_at,
            created_at=created_at,
        )

    async def list_reminders(self) -> list[ReminderRecord]:
        rows = await self.database.fetchall("SELECT * FROM reminders ORDER BY created_at DESC")
        return [ReminderRecord.model_validate(dict(row)) for row in rows]

    async def delete_reminder(self, reminder_id: str) -> bool:
        cursor = await self.database.execute(
            "DELETE FROM reminders WHERE id = ?",
            (reminder_id,),
        )
        return cursor.rowcount > 0

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
    ) -> str:
        run_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO agent_runs(
                id, conversation_id, agent, status, model_id, summary, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, conversation_id, agent, status, model_id, summary, utc_now_iso()),
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
            await conn.execute(
                """
                UPDATE suspended_agent_runs
                SET status = 'resumed', resumed_at = ?
                WHERE approval_id = ?
                """,
                (now, approval_id),
            )
            await conn.execute(
                "UPDATE agent_runs SET status = 'running' WHERE id = ?",
                (row["agent_run_id"],),
            )

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

    async def record_tool_call(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        status: str,
        permission_level: int,
        risk_level: str,
        result: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO tool_calls(
                id, conversation_id, tool, args_json, result_json, status,
                permission_level, risk_level, created_at, completed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call_id,
                conversation_id,
                tool,
                json.dumps(args, sort_keys=True),
                json.dumps(result or {}, sort_keys=True),
                status,
                permission_level,
                risk_level,
                utc_now_iso(),
                utc_now_iso() if result is not None else None,
            ),
        )
        return call_id

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
