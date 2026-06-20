from __future__ import annotations

from services.memory.policy import MemoryPolicy
from services.memory.schemas import SearchResult
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory


class MemoryRetriever:
    def __init__(self, sqlite_memory: SqliteMemory, vector_memory: VectorMemory) -> None:
        self.sqlite_memory = sqlite_memory
        self.vector_memory = vector_memory
        self.policy = MemoryPolicy()

    async def hybrid_search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        lexical = [
            memory
            for memory in await self.sqlite_memory.search_memories(query)
            if not self.policy.is_sensitive(memory.content)
        ]
        vector = self.vector_memory.search(query, limit=limit)
        results: list[SearchResult] = [
            SearchResult(
                id=memory.id,
                score=1.0,
                content=memory.content,
                metadata={"kind": memory.kind, "reason": memory.reason},
            )
            for memory in lexical
        ]
        seen = {result.id for result in results}
        for result in vector:
            if result.id not in seen:
                results.append(result)
        return results[:limit]

    async def recent_memories(self, *, limit: int = 5) -> list[SearchResult]:
        memories = [
            memory
            for memory in await self.sqlite_memory.list_memories()
            if not self.policy.is_sensitive(memory.content)
        ][:limit]
        return [
            SearchResult(
                id=memory.id,
                score=1.0,
                content=memory.content,
                metadata={"kind": memory.kind, "reason": memory.reason},
            )
            for memory in memories
        ]

    def repo_chunks(
        self,
        query: str,
        *,
        project_id: str,
        limit: int = 4,
        max_chars: int = 6000,
    ) -> list[SearchResult]:
        chunks: list[SearchResult] = []
        total_chars = 0
        for result in self.vector_memory.search(query, limit=limit * 3, project_id=project_id):
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            content = result.content[:remaining]
            if not content:
                continue
            chunks.append(result.model_copy(update={"content": content}))
            total_chars += len(content)
            if len(chunks) >= limit:
                break
        return chunks
