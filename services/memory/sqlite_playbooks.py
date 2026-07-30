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


class PlaybookRepository(SqliteRepositoryBase):
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
        stats: dict[str, Any] | None = None,
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
                required_permission_level, stats_json, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(stats or {}, sort_keys=True),
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
        expanded_steps: list[dict[str, Any]] | None = None,
        snapshot_hash: str | None = None,
        agent_id: str = "general_agent",
    ) -> str:
        run_id = str(uuid.uuid4())
        if conversation_id is not None and await self.get_conversation(conversation_id) is None:
            # An unknown conversation must not fail the run row's foreign key.
            conversation_id = None
        await self.database.execute(
            """
            INSERT INTO playbook_runs(
                id, playbook_id, conversation_id, status, steps_completed,
                current_step_index, expanded_steps_json, snapshot_hash,
                step_states_json, agent_id, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, 0, 0, ?, ?, '[]', ?, ?, ?)
            """,
            (
                run_id,
                playbook_id,
                conversation_id,
                status,
                json.dumps(expanded_steps or [], sort_keys=True),
                snapshot_hash,
                agent_id,
                utc_now_iso(),
                utc_now_iso(),
            ),
        )
        return run_id

    async def get_playbook_run(self, run_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone("SELECT * FROM playbook_runs WHERE id = ?", (run_id,))
        return dict(row) if row is not None else None

    async def update_playbook_run_progress(
        self,
        run_id: str,
        *,
        status: str,
        current_step_index: int,
        steps_completed: int,
        step_states: list[dict[str, Any]],
        detail: str | None = None,
        pending_approval_id: str | None = None,
        pending_action_hash: str | None = None,
    ) -> None:
        await self.database.execute(
            """
            UPDATE playbook_runs
            SET status = ?, current_step_index = ?, steps_completed = ?,
                step_states_json = ?, detail = ?, pending_approval_id = ?,
                pending_action_hash = ?, updated_at = ?
            WHERE id = ? AND completed_at IS NULL
            """,
            (
                status,
                current_step_index,
                steps_completed,
                json.dumps(step_states, sort_keys=True),
                detail,
                pending_approval_id,
                pending_action_hash,
                utc_now_iso(),
                run_id,
            ),
        )

    async def finish_playbook_run(
        self,
        run_id: str,
        *,
        status: str,
        steps_completed: int,
        detail: str | None = None,
    ) -> None:
        terminal = status in {"completed", "failed", "denied", "expired", "cancelled"}
        if not terminal:
            await self.update_playbook_run_progress(
                run_id,
                status=status,
                current_step_index=steps_completed,
                steps_completed=steps_completed,
                step_states=[],
                detail=detail,
            )
            return
        now = utc_now_iso()
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM playbook_runs WHERE id = ?", (run_id,))
            run = await cursor.fetchone()
            if run is None or run["completed_at"] is not None:
                return
            from april_common.time import parse_utc_iso

            duration_ms = max(
                0.0,
                (parse_utc_iso(now) - parse_utc_iso(str(run["created_at"]))).total_seconds()
                * 1000.0,
            )
            await conn.execute(
                """
                UPDATE playbook_runs
                SET status = ?, steps_completed = ?, current_step_index = ?,
                    detail = ?, pending_approval_id = NULL,
                    pending_action_hash = NULL, updated_at = ?, completed_at = ?,
                    duration_ms = ?
                WHERE id = ? AND completed_at IS NULL
                """,
                (status, steps_completed, steps_completed, detail, now, now, duration_ms, run_id),
            )
            cursor = await conn.execute(
                "SELECT stats_json FROM playbooks WHERE id = ?", (run["playbook_id"],)
            )
            playbook = await cursor.fetchone()
            if playbook is None:
                return
            try:
                stats = json.loads(playbook["stats_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                stats = {}
            previous_runs = int(stats.get("runs", 0))
            previous_average = float(stats.get("average_duration_ms", 0.0))
            runs = previous_runs + 1
            stats.update(
                {
                    "runs": runs,
                    "success": int(stats.get("success", 0)) + (1 if status == "completed" else 0),
                    "failures": int(stats.get("failures", 0))
                    + (1 if status in {"failed", "cancelled"} else 0),
                    "denials": int(stats.get("denials", 0))
                    + (1 if status in {"denied", "expired"} else 0),
                    "steps_completed": int(stats.get("steps_completed", 0)) + steps_completed,
                    "average_duration_ms": ((previous_average * previous_runs) + duration_ms)
                    / runs,
                    "last_run": now,
                }
            )
            await conn.execute(
                "UPDATE playbooks SET stats_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(stats, sort_keys=True), now, run["playbook_id"]),
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
