from __future__ import annotations

from pathlib import Path

import pytest

from april_common.settings import ConversationContextSettings
from services.brain.memory_policy import build_agent_memory_context
from services.memory.schemas import Message, Project, SearchResult


class BudgetRetriever:
    async def hybrid_search(self, query: str, *, limit: int) -> list[SearchResult]:
        return [
            SearchResult(id="memory-1", score=1.0, content="m" * 30),
            SearchResult(id="memory-2", score=0.9, content="n" * 30),
        ][:limit]

    async def recent_memories(self, *, limit: int) -> list[SearchResult]:
        return []

    def repo_chunks(
        self, message: str, *, project_id: str, limit: int, max_chars: int
    ) -> list[SearchResult]:
        return [
            SearchResult(
                id="file-1",
                score=1.0,
                content="f" * 200,
                metadata={"path": "README.md", "start_line": 1, "end_line": 8},
            )
        ]

    def document_chunks(self, message: str) -> list[SearchResult]:
        return []


@pytest.mark.asyncio
async def test_category_budgets_are_independent_and_keep_citation_metadata(
    tmp_path: Path,
) -> None:
    history = [
        Message(
            id="u",
            conversation_id="c",
            role="user",
            content="history user",
            created_at="2026-01-01T00:00:00Z",
        ),
        Message(
            id="a",
            conversation_id="c",
            role="assistant",
            content="history assistant",
            created_at="2026-01-01T00:00:01Z",
        ),
    ]
    budgets = ConversationContextSettings(
        conversation_history_max_chars=1000,
        durable_memory_max_chars=40,
        file_document_max_chars=30,
        tool_output_max_chars=20,
    )
    context = await build_agent_memory_context(
        policy="project_memory",
        history=history,
        memory_retriever=BudgetRetriever(),  # type: ignore[arg-type]
        memory_queries=["preference"],
        intent="coding_repo_analysis",
        message="inspect",
        project=Project(
            id="p",
            path=str(tmp_path),
            name="p",
            created_at="2026-01-01T00:00:00Z",
        ),
        budgets=budgets,
    )
    assert context.history == history
    assert len(context.durable_memories) == 1
    assert context.category_character_usage["durable_memory"] == 30
    assert context.category_character_usage["file_document"] <= 30
    assert context.project_chunks[0].metadata["path"] == "README.md"
    assert context.project_chunks[0].content.endswith("[TRUNCATED]")
    assert context.category_truncated["durable_memory"] is True
    assert context.category_truncated["file_document"] is True
    assert context.category_character_usage["tool_output"] == 0


def test_character_limits_are_explicit_prebounds_not_token_budgets() -> None:
    settings = ConversationContextSettings()
    assert settings.conversation_history_max_chars == 8000
    assert settings.durable_memory_max_chars == 4000
    assert settings.file_document_max_chars == 6000
    assert settings.tool_output_max_chars == 3000
