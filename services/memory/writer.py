from __future__ import annotations

from april_common.errors import PermissionDeniedError
from services.memory.policy import MemoryPolicy
from services.memory.schemas import MemoryRecord
from services.memory.sqlite_memory import SqliteMemory


class MemoryWriter:
    def __init__(self, memory: SqliteMemory, policy: MemoryPolicy | None = None) -> None:
        self.memory = memory
        self.policy = policy or MemoryPolicy()

    async def write(
        self,
        content: str,
        *,
        reason: str,
        requested_by_user: bool = False,
        project_id: str | None = None,
    ) -> MemoryRecord:
        decision = self.policy.evaluate(content, requested_by_user=requested_by_user)
        if not decision.allowed:
            raise PermissionDeniedError(decision.reason)
        return await self.memory.create_memory(
            content,
            reason=reason or decision.reason,
            project_id=project_id,
        )
