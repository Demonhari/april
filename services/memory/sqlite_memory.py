from __future__ import annotations

import json
import uuid
from typing import Any

from april_common.time import utc_now_iso
from services.memory.database import Database
from services.memory.schemas import MemoryRecord, Message, Project, ReminderRecord


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

    async def create_conversation(self, title: str | None = None) -> str:
        conversation_id = str(uuid.uuid4())
        await self.database.execute(
            "INSERT INTO conversations(id, title, created_at) VALUES(?, ?, ?)",
            (conversation_id, title, utc_now_iso()),
        )
        return conversation_id

    async def ensure_conversation(self, conversation_id: str, title: str | None = None) -> str:
        await self.database.execute(
            "INSERT OR IGNORE INTO conversations(id, title, created_at) VALUES(?, ?, ?)",
            (conversation_id, title, utc_now_iso()),
        )
        return conversation_id

    async def add_message(self, conversation_id: str, role: str, content: str) -> str:
        message_id = str(uuid.uuid4())
        await self.database.execute(
            """
            INSERT INTO messages(id, conversation_id, role, content, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, utc_now_iso()),
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
        cursor = await self.database.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return cursor.rowcount > 0

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
