from __future__ import annotations

from pathlib import Path

from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.schemas import ReminderRecord
from services.memory.sqlite_memory import SqliteMemory


class ReminderStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.memory = SqliteMemory(database)

    @classmethod
    async def open(cls, database_path: Path) -> ReminderStore:
        database = Database(database_path)
        await database.connect()
        try:
            await run_migrations(database)
        except BaseException:
            await database.close()
            raise
        return cls(database)

    async def close(self) -> None:
        await self.database.close()

    async def create(self, content: str, due_at: str | None = None) -> ReminderRecord:
        return await self.memory.create_reminder(content, due_at)

    async def list_due(self, now_iso: str) -> list[ReminderRecord]:
        return await self.memory.list_due_reminders(now_iso)

    async def list(self) -> list[ReminderRecord]:
        return await self.memory.list_reminders()

    async def mark_fired(self, reminder_id: str, fired_at: str) -> bool:
        return await self.memory.mark_reminder_fired(reminder_id, fired_at)

    async def delete(self, reminder_id: str) -> bool:
        return await self.memory.delete_reminder(reminder_id)
