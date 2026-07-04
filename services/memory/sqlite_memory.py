from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from april_common.errors import PermissionDeniedError
from april_common.time import utc_now_iso
from services.brain.planner import TaskPlan, TaskStep
from services.memory.database import Database
from services.memory.schemas import (
    Conversation,
    FeedbackEventRecord,
    MemoryContradictionRecord,
    MemoryRecord,
    Message,
    Project,
    ReminderRecord,
    SessionRecord,
    SuspendedAgentRun,
    WakeEventRecord,
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
        confidence: float = 0.7,
        source: str = "user",
        expires_at: str | None = None,
        superseded_by: str | None = None,
    ) -> MemoryRecord:
        memory_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO memories(
                    id, project_id, kind, content, reason, created_at,
                    confidence, source, expires_at, superseded_by
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    project_id,
                    kind,
                    content,
                    reason,
                    created_at,
                    confidence,
                    source,
                    expires_at,
                    superseded_by,
                ),
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
            confidence=confidence,
            source=source,
            expires_at=expires_at,
            superseded_by=superseded_by,
        )

    async def get_memory(
        self, memory_id: str, *, include_inactive: bool = False
    ) -> MemoryRecord | None:
        if include_inactive:
            row = await self.database.fetchone("SELECT * FROM memories WHERE id = ?", (memory_id,))
        else:
            row = await self.database.fetchone(
                """
                SELECT * FROM memories
                WHERE id = ?
                  AND superseded_by IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (memory_id, utc_now_iso()),
            )
        if row is None:
            return None
        return MemoryRecord.model_validate(dict(row))

    async def find_duplicate_memory(
        self,
        content: str,
        *,
        kind: str,
        project_id: str | None = None,
    ) -> MemoryRecord | None:
        normalized = " ".join(content.casefold().split())
        rows = await self.database.fetchall(
            """
            SELECT * FROM memories
            WHERE kind = ? AND (project_id IS ? OR project_id = ?)
            ORDER BY created_at DESC
            """,
            (kind, project_id, project_id),
        )
        for row in rows:
            record = MemoryRecord.model_validate(dict(row))
            if " ".join(record.content.casefold().split()) == normalized:
                return record
        return None

    async def list_memories(
        self, *, project_id: str | None = None, include_inactive: bool = False
    ) -> list[MemoryRecord]:
        active_clause = (
            ""
            if include_inactive
            else "AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
        )
        params: tuple[object, ...]
        if project_id is None:
            params = (utc_now_iso(),) if not include_inactive else ()
            rows = await self.database.fetchall(
                f"""
                SELECT * FROM memories
                WHERE 1 = 1 {active_clause}
                ORDER BY created_at DESC
                """,
                params,
            )
        else:
            params = (
                (project_id, utc_now_iso()) if not include_inactive else (project_id,)
            )
            rows = await self.database.fetchall(
                f"""
                SELECT * FROM memories
                WHERE project_id = ? {active_clause}
                ORDER BY created_at DESC
                """,
                params,
            )
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    async def search_memories(
        self, query: str, *, project_id: str | None = None
    ) -> list[MemoryRecord]:
        if query.strip() in {"", "*"}:
            return await self.list_memories(project_id=project_id)
        active_sql = "superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
        now = utc_now_iso()
        if project_id is None:
            rows = await self.database.fetchall(
                """
                SELECT m.*
                FROM memories_fts f
                JOIN memories m ON m.id = f.id
                WHERE memories_fts MATCH ?
                  AND m.superseded_by IS NULL
                  AND (m.expires_at IS NULL OR m.expires_at > ?)
                ORDER BY rank
                LIMIT 20
                """,
                (query, now),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT m.*
                FROM memories_fts f
                JOIN memories m ON m.id = f.id
                WHERE memories_fts MATCH ? AND m.project_id = ?
                  AND m.superseded_by IS NULL
                  AND (m.expires_at IS NULL OR m.expires_at > ?)
                ORDER BY rank
                LIMIT 20
                """,
                (query, project_id, now),
            )
        if not rows:
            if project_id is None:
                rows = await self.database.fetchall(
                    f"""
                    SELECT * FROM memories
                    WHERE ({active_sql}) AND (content LIKE ? OR reason LIKE ?)
                    LIMIT 20
                    """,
                    (now, f"%{query}%", f"%{query}%"),
                )
            else:
                rows = await self.database.fetchall(
                    """
                    SELECT * FROM memories
                    WHERE project_id = ?
                      AND superseded_by IS NULL
                      AND (expires_at IS NULL OR expires_at > ?)
                      AND (content LIKE ? OR reason LIKE ?)
                    LIMIT 20
                    """,
                    (project_id, now, f"%{query}%", f"%{query}%"),
                )
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    async def mark_memories_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            for memory_id in dict.fromkeys(memory_ids):
                await conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1, last_used_at = ?
                    WHERE id = ?
                    """,
                    (now, memory_id),
                )

    async def refresh_memory(
        self, memory_id: str, *, confidence: float | None = None
    ) -> MemoryRecord | None:
        """Auditable duplicate-merge refresh: bump usage and keep max confidence."""
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            if confidence is None:
                await conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1, last_used_at = ?
                    WHERE id = ?
                    """,
                    (now, memory_id),
                )
            else:
                await conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1,
                        last_used_at = ?,
                        confidence = MAX(confidence, ?)
                    WHERE id = ?
                    """,
                    (now, confidence, memory_id),
                )
        return await self.get_memory(memory_id, include_inactive=True)

    async def set_memory_decay(
        self, memory_id: str, *, confidence: float, expires_at: str | None
    ) -> bool:
        """Deterministic decay update: lower confidence, optionally start fading."""
        cursor = await self.database.execute(
            "UPDATE memories SET confidence = ?, expires_at = ? WHERE id = ?",
            (confidence, expires_at, memory_id),
        )
        return cursor.rowcount > 0

    async def list_memories_by_state(
        self, state: str, *, limit: int = 100
    ) -> list[MemoryRecord]:
        """Inspect memory lifecycle states without hiding or deleting anything.

        States: ``machine`` (machine-written, still active), ``superseded``,
        ``expired`` (expires_at passed), ``fading`` (expires_at set but not yet
        reached), and ``active`` (what retrieval serves).
        """
        now = utc_now_iso()
        capped = max(1, min(limit, 500))
        clauses = {
            "machine": (
                "source != 'user' AND superseded_by IS NULL "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (now, capped),
            ),
            "superseded": ("superseded_by IS NOT NULL", (capped,)),
            "expired": ("expires_at IS NOT NULL AND expires_at <= ?", (now, capped)),
            "fading": (
                "expires_at IS NOT NULL AND expires_at > ? AND superseded_by IS NULL",
                (now, capped),
            ),
            "active": (
                "superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)",
                (now, capped),
            ),
        }
        if state not in clauses:
            raise ValueError(
                "state must be one of machine, superseded, expired, fading, active"
            )
        where, params = clauses[state]
        rows = await self.database.fetchall(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    async def supersede_memory(self, memory_id: str, *, superseded_by: str) -> bool:
        """Mark a memory as superseded without deleting the row."""
        cursor = await self.database.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL",
            (superseded_by, memory_id),
        )
        return cursor.rowcount > 0

    async def record_memory_contradiction(
        self, *, memory_id_a: str, memory_id_b: str
    ) -> MemoryContradictionRecord:
        """Flag a contradictory pair for Dreamer adjudication; both rows are kept."""
        pair_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO memory_contradictions(
                id, memory_id_a, memory_id_b, status, created_at
            )
            VALUES(?, ?, ?, 'pending', ?)
            """,
            (pair_id, memory_id_a, memory_id_b, created_at),
        )
        return MemoryContradictionRecord(
            id=pair_id,
            memory_id_a=memory_id_a,
            memory_id_b=memory_id_b,
            status="pending",
            created_at=created_at,
        )

    async def list_memory_contradictions(
        self, *, status: str | None = "pending", limit: int = 100
    ) -> list[MemoryContradictionRecord]:
        capped = max(1, min(limit, 500))
        if status is None:
            rows = await self.database.fetchall(
                "SELECT * FROM memory_contradictions ORDER BY created_at LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM memory_contradictions
                WHERE status = ?
                ORDER BY created_at
                LIMIT ?
                """,
                (status, capped),
            )
        return [MemoryContradictionRecord.model_validate(dict(row)) for row in rows]

    async def resolve_memory_contradiction(
        self, contradiction_id: str, *, resolution: str
    ) -> bool:
        cursor = await self.database.execute(
            """
            UPDATE memory_contradictions
            SET status = 'resolved', resolution = ?, resolved_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (resolution, utc_now_iso(), contradiction_id),
        )
        return cursor.rowcount > 0

    async def count_machine_memories_since(self, since_iso: str, *, source: str) -> int:
        row = await self.database.fetchone(
            "SELECT COUNT(*) AS count FROM memories WHERE source = ? AND created_at >= ?",
            (source, since_iso),
        )
        return int(row["count"]) if row is not None else 0

    async def delete_memory(self, memory_id: str) -> bool:
        async with self.database.transaction() as conn:
            await conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            cursor = await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    async def export_memories(self, *, project_id: str | None = None) -> str:
        memories = [
            memory.model_dump() for memory in await self.list_memories(project_id=project_id)
        ]
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

    async def list_due_reminders(self, now_iso: str) -> list[ReminderRecord]:
        rows = await self.database.fetchall(
            """
            SELECT *
            FROM reminders
            WHERE due_at IS NOT NULL AND fired_at IS NULL AND due_at <= ?
            ORDER BY due_at ASC
            """,
            (now_iso,),
        )
        return [ReminderRecord.model_validate(dict(row)) for row in rows]

    async def mark_reminder_fired(self, reminder_id: str, fired_at: str) -> bool:
        cursor = await self.database.execute(
            "UPDATE reminders SET fired_at = ? WHERE id = ? AND fired_at IS NULL",
            (fired_at, reminder_id),
        )
        return cursor.rowcount > 0

    async def list_upcoming_reminders(self, now_iso: str, until_iso: str) -> list[ReminderRecord]:
        rows = await self.database.fetchall(
            """
            SELECT *
            FROM reminders
            WHERE due_at IS NOT NULL AND fired_at IS NULL AND due_at <= ?
            ORDER BY due_at ASC
            """,
            (until_iso,),
        )
        return [ReminderRecord.model_validate(dict(row)) for row in rows]

    async def delete_reminder(self, reminder_id: str) -> bool:
        cursor = await self.database.execute(
            "DELETE FROM reminders WHERE id = ?",
            (reminder_id,),
        )
        return cursor.rowcount > 0

    async def get_scheduler_state(self, key: str) -> str | None:
        row = await self.database.fetchone(
            "SELECT value FROM scheduler_state WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        return str(row["value"])

    async def set_scheduler_state(self, key: str, value: str) -> None:
        await self.database.execute(
            """
            INSERT INTO scheduler_state(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, utc_now_iso()),
        )

    async def get_repo_snapshot(self, project_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone(
            "SELECT last_head_sha, last_dirty_count, updated_at "
            "FROM repo_snapshots WHERE project_id = ?",
            (project_id,),
        )
        if row is None:
            return None
        return {
            "head_sha": row["last_head_sha"],
            "dirty_count": int(row["last_dirty_count"]),
            "updated_at": str(row["updated_at"]),
        }

    async def upsert_repo_snapshot(
        self,
        project_id: str,
        head_sha: str | None,
        dirty_count: int,
        updated_at: str,
    ) -> None:
        await self.database.execute(
            """
            INSERT INTO repo_snapshots(project_id, last_head_sha, last_dirty_count, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                last_head_sha = excluded.last_head_sha,
                last_dirty_count = excluded.last_dirty_count,
                updated_at = excluded.updated_at
            """,
            (project_id, head_sha, dirty_count, updated_at),
        )

    async def upsert_playbook(
        self,
        *,
        playbook_id: str,
        name: str,
        source: str,
        status: str,
        trigger_examples: list[str],
        steps: list[dict[str, Any]],
        required_permission_level: int = 1,
    ) -> None:
        """Mirror a playbook definition into the playbooks table.

        Playbooks live as YAML/JSON under data/playbooks; this row exists so
        playbook_runs rows have a valid foreign key and stats can accumulate.
        """
        now = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO playbooks(
                id, name, source, status, trigger_examples_json, steps_json,
                required_permission_level, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                source = excluded.source,
                status = excluded.status,
                trigger_examples_json = excluded.trigger_examples_json,
                steps_json = excluded.steps_json,
                required_permission_level = excluded.required_permission_level,
                updated_at = excluded.updated_at
            """,
            (
                playbook_id,
                name,
                source,
                status,
                json.dumps(trigger_examples, sort_keys=True),
                json.dumps(steps, sort_keys=True),
                required_permission_level,
                now,
                now,
            ),
        )

    async def create_playbook_run(
        self,
        *,
        playbook_id: str,
        conversation_id: str | None = None,
        status: str = "running",
    ) -> str:
        run_id = str(uuid.uuid4())
        if conversation_id is not None and await self.get_conversation(conversation_id) is None:
            # An unknown conversation must not fail the run row's foreign key.
            conversation_id = None
        await self.database.execute(
            """
            INSERT INTO playbook_runs(
                id, playbook_id, conversation_id, status, steps_completed, created_at
            )
            VALUES(?, ?, ?, ?, 0, ?)
            """,
            (run_id, playbook_id, conversation_id, status, utc_now_iso()),
        )
        return run_id

    async def finish_playbook_run(
        self,
        run_id: str,
        *,
        status: str,
        steps_completed: int,
        detail: str | None = None,
    ) -> None:
        await self.database.execute(
            """
            UPDATE playbook_runs
            SET status = ?, steps_completed = ?, detail = ?, completed_at = ?
            WHERE id = ?
            """,
            (status, steps_completed, detail, utc_now_iso(), run_id),
        )

    async def list_playbook_runs(
        self, *, playbook_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        capped = max(1, min(limit, 500))
        if playbook_id is None:
            rows = await self.database.fetchall(
                "SELECT * FROM playbook_runs ORDER BY created_at DESC LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM playbook_runs
                WHERE playbook_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (playbook_id, capped),
            )
        return [dict(row) for row in rows]

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

    async def create_session(
        self,
        *,
        source: str,
        conversation_id: str | None,
        started_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        session_id = str(uuid.uuid4())
        now = started_at or utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO sessions(
                id, conversation_id, source, started_at, last_activity_at, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (session_id, conversation_id, source, now, now, json.dumps(metadata or {})),
        )
        return SessionRecord(
            id=session_id,
            conversation_id=conversation_id,
            source=source,
            started_at=now,
            last_activity_at=now,
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        row = await self.database.fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            return None
        return self._session_from_row(row)

    async def latest_open_session(self) -> SessionRecord | None:
        row = await self.database.fetchone(
            """
            SELECT * FROM sessions
            WHERE closed_at IS NULL
            ORDER BY last_activity_at DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        return self._session_from_row(row)

    async def list_open_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        rows = await self.database.fetchall(
            """
            SELECT * FROM sessions
            WHERE closed_at IS NULL
            ORDER BY last_activity_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        )
        return [self._session_from_row(row) for row in rows]

    async def touch_session(self, session_id: str, *, at: str | None = None) -> None:
        await self.database.execute(
            "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
            (at or utc_now_iso(), session_id),
        )

    async def close_session(self, session_id: str, *, at: str | None = None) -> bool:
        cursor = await self.database.execute(
            "UPDATE sessions SET closed_at = ? WHERE id = ? AND closed_at IS NULL",
            (at or utc_now_iso(), session_id),
        )
        return cursor.rowcount > 0

    async def list_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        rows = await self.database.fetchall(
            "SELECT * FROM sessions ORDER BY last_activity_at DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        )
        return [self._session_from_row(row) for row in rows]

    async def record_wake_event(
        self,
        *,
        session_id: str | None,
        source: str,
        score: float | None = None,
        accepted: bool = True,
        reason: str | None = None,
        transcript_present: bool = False,
        captured_at: str | None = None,
        session_hint: str | None = None,
    ) -> WakeEventRecord:
        event_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO wake_events(
                id, session_id, source, score, accepted, reason,
                transcript_present, captured_at, session_hint, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                source,
                score,
                1 if accepted else 0,
                reason,
                1 if transcript_present else 0,
                captured_at,
                session_hint,
                created_at,
            ),
        )
        return WakeEventRecord(
            id=event_id,
            session_id=session_id,
            source=source,
            score=score,
            accepted=accepted,
            reason=reason,
            transcript_present=transcript_present,
            captured_at=captured_at,
            session_hint=session_hint,
            created_at=created_at,
        )

    async def list_wake_events(
        self, *, session_id: str | None = None, limit: int = 100
    ) -> list[WakeEventRecord]:
        capped = max(1, min(limit, 500))
        if session_id is None:
            rows = await self.database.fetchall(
                "SELECT * FROM wake_events ORDER BY created_at DESC LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM wake_events
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, capped),
            )
        return [WakeEventRecord.model_validate(dict(row)) for row in rows]

    async def record_feedback_event(
        self,
        *,
        rating: Literal["good", "bad"],
        reason: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        agent_run_id: str | None = None,
    ) -> FeedbackEventRecord:
        event_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO feedback_events(
                id, session_id, conversation_id, agent_run_id, rating, reason, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, session_id, conversation_id, agent_run_id, rating, reason, created_at),
        )
        return FeedbackEventRecord(
            id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            agent_run_id=agent_run_id,
            rating=rating,
            reason=reason,
            created_at=created_at,
        )

    async def list_feedback_events(
        self, *, conversation_id: str | None = None, limit: int = 100
    ) -> list[FeedbackEventRecord]:
        capped = max(1, min(limit, 500))
        if conversation_id is None:
            rows = await self.database.fetchall(
                "SELECT * FROM feedback_events ORDER BY created_at DESC LIMIT ?",
                (capped,),
            )
        else:
            rows = await self.database.fetchall(
                """
                SELECT * FROM feedback_events
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, capped),
            )
        return [FeedbackEventRecord.model_validate(dict(row)) for row in rows]

    async def list_tool_call_summaries(
        self, *, conversation_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Sanitized tool-call summaries for reflection: no args, no results."""
        rows = await self.database.fetchall(
            """
            SELECT tool, status, risk_level, permission_level, created_at
            FROM tool_calls
            WHERE conversation_id = ?
            ORDER BY created_at
            LIMIT ?
            """,
            (conversation_id, max(1, min(limit, 200))),
        )
        return [dict(row) for row in rows]

    async def latest_agent_run_id(self, *, conversation_id: str | None = None) -> str | None:
        if conversation_id is None:
            row = await self.database.fetchone(
                "SELECT id FROM agent_runs ORDER BY created_at DESC LIMIT 1"
            )
        else:
            row = await self.database.fetchone(
                """
                SELECT id FROM agent_runs
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
        if row is None:
            return None
        return str(row["id"])

    def _session_from_row(self, row: Any) -> SessionRecord:
        data = dict(row)
        raw_metadata = data.pop("metadata_json", None) or "{}"
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            metadata = {}
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return SessionRecord.model_validate(data)

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
