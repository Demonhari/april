from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from april_common.errors import PermissionDeniedError
from april_common.settings import get_settings
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.memory.writer import MemoryWriter
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


class RememberMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=10_000)
    memory_type: Literal["fact", "preference", "project", "note"] = "fact"
    project_id: str | None = None
    source_conversation_id: str | None = None
    reason: str = Field(min_length=1, max_length=500)


async def remember_memory(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        request = RememberMemoryArgs.model_validate(args)
        settings = get_settings()
        database = Database(settings.database_path)
        await database.connect()
        try:
            await run_migrations(database)
            memory = SqliteMemory(database)
            if (
                request.project_id is not None
                and await memory.get_project(request.project_id) is None
            ):
                raise PermissionDeniedError(
                    "Unknown project for project-scoped memory.",
                    {"project_id": request.project_id},
                )
            if request.source_conversation_id is not None:
                conversation = await memory.get_conversation(request.source_conversation_id)
                if conversation is None:
                    raise PermissionDeniedError(
                        "Unknown source conversation for memory write.",
                        {"conversation_id": request.source_conversation_id},
                    )
                if conversation.project_id != request.project_id:
                    raise PermissionDeniedError(
                        "Memory source conversation project scope does not match.",
                        {
                            "conversation_project_id": conversation.project_id,
                            "memory_project_id": request.project_id,
                        },
                    )
            writer = MemoryWriter(memory)
            record = await writer.write(
                request.content,
                reason=request.reason,
                memory_type=request.memory_type,
                requested_by_user=True,
                project_id=request.project_id,
            )
            return ToolResult(
                ok=True,
                stdout=f"Stored {record.kind} memory.",
                data={
                    "memory_id": record.id,
                    "memory_type": record.kind,
                    "project_id": record.project_id,
                    "content_length": len(record.content),
                },
                risk_level="safe_write",
                permission_level=2,
            )
        finally:
            await database.close()

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def remember_memory_definition() -> ToolDefinition:
    return ToolDefinition(
        name="remember_memory",
        description="Store an explicit local durable memory after policy checks.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"general_agent", "reasoning_agent", "creative_agent"},
        executor=remember_memory,
    )
