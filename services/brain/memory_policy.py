from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from april_common.settings import ConversationContextSettings
from services.brain.conversation_context import group_persisted_conversation_turns
from services.memory.retriever import MemoryRetriever
from services.memory.schemas import Message, Project, SearchResult

MemoryAccessPolicy = Literal["none", "conversation_and_safe_memory", "project_memory"]


@dataclass(frozen=True, slots=True)
class AgentMemoryContext:
    conversation_summary: str | None = None
    history: list[Message] = field(default_factory=list)
    durable_memories: list[SearchResult] = field(default_factory=list)
    project_chunks: list[SearchResult] = field(default_factory=list)
    document_chunks: list[SearchResult] = field(default_factory=list)
    user_model: str | None = None
    category_character_usage: dict[str, int] = field(default_factory=dict)
    category_truncated: dict[str, bool] = field(default_factory=dict)


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
    conversation_summary: str | None = None,
    budgets: ConversationContextSettings | None = None,
    user_model_path: Path | None = None,
) -> AgentMemoryContext:
    active_budgets = budgets or ConversationContextSettings()
    if policy == "none":
        return AgentMemoryContext(
            conversation_summary=_bound_text(
                conversation_summary, active_budgets.rendered_summary_max_chars
            )[0]
        )

    bounded_history, history_truncated = _bound_messages(
        history,
        active_budgets.conversation_history_max_chars,
    )
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
            max_chars=active_budgets.file_document_max_chars,
        )

    document_chunks: list[SearchResult] = []
    if (
        policy == "conversation_and_safe_memory"
        and memory_retriever is not None
        and _is_document_intent(intent)
    ):
        document_chunks = memory_retriever.document_chunks(message)
    bounded_memories, memories_truncated = _bound_results(
        durable_memories, active_budgets.durable_memory_max_chars
    )
    user_model_budget = max(
        0,
        active_budgets.durable_memory_max_chars
        - sum(len(item.content) for item in bounded_memories),
    )
    user_model = _read_user_model(user_model_path, max_chars=user_model_budget)
    user_model_truncated = bool(
        user_model is not None
        and user_model_path is not None
        and _file_is_longer_than(user_model_path, user_model_budget)
    )
    bounded_projects, bounded_documents, files_truncated = _bound_file_results(
        project_chunks,
        document_chunks,
        active_budgets.file_document_max_chars,
    )
    bounded_summary, summary_truncated = _bound_text(
        conversation_summary, active_budgets.rendered_summary_max_chars
    )

    return AgentMemoryContext(
        conversation_summary=bounded_summary,
        history=bounded_history,
        durable_memories=bounded_memories,
        project_chunks=bounded_projects,
        document_chunks=bounded_documents,
        user_model=user_model,
        category_character_usage={
            "conversation_summary": len(bounded_summary or ""),
            "conversation_history": sum(len(item.content) for item in bounded_history),
            "durable_memory": sum(len(item.content) for item in bounded_memories)
            + len(user_model or ""),
            "file_document": sum(
                len(item.content) for item in [*bounded_projects, *bounded_documents]
            ),
            "tool_output": 0,
        },
        category_truncated={
            "conversation_summary": summary_truncated,
            "conversation_history": history_truncated,
            "durable_memory": memories_truncated or user_model_truncated,
            "file_document": files_truncated,
            "tool_output": False,
        },
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


def _read_user_model(path: Path | None, *, max_chars: int) -> str | None:
    if path is None or max_chars <= 0:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped[:max_chars]


def _bound_text(value: str | None, limit: int) -> tuple[str | None, bool]:
    if value is None or len(value) <= limit:
        return value, False
    marker = "\n[TRUNCATED BY CORE CHARACTER PRE-BOUND]"
    return value[: max(0, limit - len(marker))].rstrip() + marker, True


def _bound_messages(messages: list[Message], limit: int) -> tuple[list[Message], bool]:
    groups, _ = group_persisted_conversation_turns(messages)
    selected_groups = []
    used = 0
    for group in reversed(groups):
        size = group.character_count
        if used + size > limit:
            continue
        selected_groups.append(group)
        used += size
    selected = [
        message
        for group in reversed(selected_groups)
        for message in group.messages
    ]
    return selected, len(selected) != len(messages)


def _bound_results(
    results: list[SearchResult], limit: int
) -> tuple[list[SearchResult], bool]:
    selected: list[SearchResult] = []
    used = 0
    for result in results:
        if used + len(result.content) > limit:
            continue
        selected.append(result)
        used += len(result.content)
    return selected, len(selected) != len(results)


def _bound_file_results(
    projects: list[SearchResult],
    documents: list[SearchResult],
    limit: int,
) -> tuple[list[SearchResult], list[SearchResult], bool]:
    selected_projects: list[SearchResult] = []
    selected_documents: list[SearchResult] = []
    used = 0
    truncated = False
    for result, destination in [
        *((item, selected_projects) for item in projects),
        *((item, selected_documents) for item in documents),
    ]:
        remaining = limit - used
        if remaining <= 0:
            truncated = True
            continue
        if len(result.content) <= remaining:
            destination.append(result)
            used += len(result.content)
            continue
        marker = "[TRUNCATED]"
        if remaining >= len(marker):
            destination.append(
                result.model_copy(
                    update={
                        "content": (
                            result.content[: remaining - len(marker)].rstrip() + marker
                        )
                    }
                )
            )
            used = limit
        truncated = True
    return selected_projects, selected_documents, truncated


def _file_is_longer_than(path: Path, limit: int) -> bool:
    try:
        return len(path.read_text(encoding="utf-8").strip()) > limit
    except OSError:
        return False
