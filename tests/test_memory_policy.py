from __future__ import annotations

import pytest

from services.brain.memory_policy import build_agent_memory_context
from services.memory.schemas import SearchResult


class PolicyRetriever:
    def __init__(self) -> None:
        self.hybrid_queries: list[str] = []
        self.recent_calls = 0

    async def hybrid_search(
        self,
        query: str,
        *,
        limit: int,
        project_id: str | None = None,
        global_only: bool = False,
    ) -> list[SearchResult]:
        del limit, project_id, global_only
        self.hybrid_queries.append(query)
        return [
            SearchResult(
                id="explicit",
                score=1.0,
                content="The neutral test project is Amber.",
                metadata={"kind": "relationship"},
            )
        ]

    async def recent_memories(
        self,
        *,
        limit: int,
        project_id: str | None = None,
        global_only: bool = False,
    ) -> list[SearchResult]:
        del limit, project_id, global_only
        self.recent_calls += 1
        return [
            SearchResult(
                id="preference",
                score=1.0,
                content="The user prefers concise plans.",
                metadata={"kind": "preference"},
            ),
            SearchResult(
                id="unrelated",
                score=1.0,
                content="The neutral test project is Bluebird.",
                metadata={"kind": "relationship"},
            ),
        ]

    def repo_chunks(self, *args: object, **kwargs: object) -> list[SearchResult]:
        del args, kwargs
        return []

    def document_chunks(self, *args: object, **kwargs: object) -> list[SearchResult]:
        del args, kwargs
        return []


@pytest.mark.asyncio
async def test_normal_conversation_does_not_use_recent_memory() -> None:
    retriever = PolicyRetriever()
    context = await build_agent_memory_context(
        policy="conversation_and_safe_memory",
        history=[],
        memory_retriever=retriever,  # type: ignore[arg-type]
        memory_queries=[],
        intent="normal_conversation",
        message="What is your name?",
        project=None,
    )
    assert context.durable_memories == []
    assert retriever.recent_calls == 0


@pytest.mark.asyncio
async def test_planning_recent_memory_is_preference_only() -> None:
    retriever = PolicyRetriever()
    context = await build_agent_memory_context(
        policy="conversation_and_safe_memory",
        history=[],
        memory_retriever=retriever,  # type: ignore[arg-type]
        memory_queries=[],
        intent="planning",
        message="Plan my day.",
        project=None,
    )
    assert [item.id for item in context.durable_memories] == ["preference"]
    assert retriever.recent_calls == 1


@pytest.mark.asyncio
async def test_explicit_memory_query_uses_relevant_hybrid_search() -> None:
    retriever = PolicyRetriever()
    context = await build_agent_memory_context(
        policy="conversation_and_safe_memory",
        history=[],
        memory_retriever=retriever,  # type: ignore[arg-type]
        memory_queries=["test project"],
        intent="memory_lookup",
        message="What is my test project called?",
        project=None,
    )
    assert [item.id for item in context.durable_memories] == ["explicit"]
    assert retriever.hybrid_queries == ["test project"]
    assert retriever.recent_calls == 0
