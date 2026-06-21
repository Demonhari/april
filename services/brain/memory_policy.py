from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from services.memory.retriever import MemoryRetriever
from services.memory.schemas import Message, Project, SearchResult

MemoryAccessPolicy = Literal["none", "conversation_and_safe_memory", "project_memory"]


@dataclass(frozen=True, slots=True)
class AgentMemoryContext:
    history: list[Message] = field(default_factory=list)
    durable_memories: list[SearchResult] = field(default_factory=list)
    project_chunks: list[SearchResult] = field(default_factory=list)
    document_chunks: list[SearchResult] = field(default_factory=list)


def _is_document_intent(intent: str) -> bool:
    lowered = intent.lower()
    return any(token in lowered for token in ("read", "document", "summary"))


async def build_agent_memory_context(
    *,
    policy: str,
    history: list[Message],
    memory_retriever: MemoryRetriever | None,
    memory_queries: list[str],
    intent: str,
    message: str,
    project: Project | None,
    history_limit: int = 8,
) -> AgentMemoryContext:
    if policy == "none":
        return AgentMemoryContext()

    bounded_history = history[-history_limit:]
    durable_memories: list[SearchResult] = []
    if memory_retriever is not None:
        durable_memories = await _safe_memory_results(
            memory_retriever=memory_retriever,
            memory_queries=memory_queries,
            intent=intent,
        )

    project_chunks: list[SearchResult] = []
    if policy == "project_memory" and project is not None and memory_retriever is not None:
        project_chunks = memory_retriever.repo_chunks(
            message,
            project_id=project.id,
            limit=4,
            max_chars=6000,
        )

    document_chunks: list[SearchResult] = []
    if (
        policy == "conversation_and_safe_memory"
        and memory_retriever is not None
        and _is_document_intent(intent)
    ):
        document_chunks = memory_retriever.document_chunks(message)

    return AgentMemoryContext(
        history=bounded_history,
        durable_memories=durable_memories,
        project_chunks=project_chunks,
        document_chunks=document_chunks,
    )


async def _safe_memory_results(
    *,
    memory_retriever: MemoryRetriever,
    memory_queries: list[str],
    intent: str,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for query in memory_queries[:3]:
        for result in await memory_retriever.hybrid_search(query, limit=3):
            if result.id not in {existing.id for existing in results}:
                results.append(result)
    if not results and intent in {"planning", "normal_conversation", "direct_agent_run"}:
        results = await memory_retriever.recent_memories(limit=3)
    return results[:6]
