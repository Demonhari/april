from __future__ import annotations

from services.memory.schemas import SearchResult
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory


class MemoryRetriever:
    def __init__(self, sqlite_memory: SqliteMemory, vector_memory: VectorMemory) -> None:
        self.sqlite_memory = sqlite_memory
        self.vector_memory = vector_memory

    async def hybrid_search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        lexical = await self.sqlite_memory.search_memories(query)
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
