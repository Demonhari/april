from __future__ import annotations

import asyncio
import json
from typing import Protocol

from april_common.audit import AuditLogger
from services.memory.policy import MemoryPolicy
from services.memory.schemas import SearchResult
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory

# Stage one collects up to this many lexical+vector candidates; stage two
# (optional local rerank) narrows them to at most RERANK_LIMIT.
CANDIDATE_LIMIT = 20
RERANK_LIMIT = 5


class MemoryReranker(Protocol):
    """Stage-two reranker contract.

    Returns candidate IDs ordered best-first, or ``None`` when reranking is
    unavailable (no runtime, timeout, invalid output). Returning ``None`` — and
    never a made-up ordering — is what keeps production reranking honest.
    """

    async def rerank(
        self, query: str, candidates: list[SearchResult], *, limit: int
    ) -> list[str] | None: ...


class RuntimeMemoryReranker:
    """Scout-style relevance rerank through the local runtime.

    Sends the query and numbered candidates to the local reading model and
    expects a strict JSON object ``{"memory_ids": [...]}``. Any failure —
    runtime down, timeout, invalid JSON, unknown IDs — yields ``None`` so the
    retriever falls back to its deterministic ranking (audited). Nothing here
    fakes a rerank.
    """

    def __init__(
        self,
        runtime_client: object,
        *,
        model_id: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.runtime_client = runtime_client
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    async def rerank(
        self, query: str, candidates: list[SearchResult], *, limit: int
    ) -> list[str] | None:
        from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat

        listing = "\n".join(
            f"{index}. (id={candidate.id}) {candidate.content[:300]}"
            for index, candidate in enumerate(candidates, start=1)
        )
        prompt = (
            "Rank the local memory snippets by relevance to the query. "
            f"Return exactly one JSON object {{\"memory_ids\": [...]}} listing at "
            f"most {limit} ids, most relevant first. Use only the given ids.\n\n"
            f"Query:\n{query}\n\nMemories:\n{listing}"
        )
        try:
            response = await asyncio.wait_for(
                self.runtime_client.chat(  # type: ignore[attr-defined]
                    model_id=self.model_id,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                "You are APRIL's local memory relevance ranker. "
                                "Return exactly one JSON object and nothing else."
                            ),
                        ),
                        ChatMessage(role="user", content=prompt),
                    ],
                    options=GenerationOptions(max_output_tokens=256),
                    response_format=ResponseFormat(
                        type="json_object",
                        json_schema={
                            "type": "object",
                            "properties": {
                                "memory_ids": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["memory_ids"],
                        },
                    ),
                    request_id="memory-rerank",
                ),
                timeout=self.timeout_seconds,
            )
        except Exception:
            return None
        try:
            payload = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return None
        raw_ids = payload.get("memory_ids") if isinstance(payload, dict) else None
        if not isinstance(raw_ids, list):
            return None
        known = {candidate.id for candidate in candidates}
        ordered: list[str] = []
        for raw in raw_ids:
            memory_id = str(raw)
            if memory_id in known and memory_id not in ordered:
                ordered.append(memory_id)
        if not ordered:
            return None
        return ordered[:limit]


class MemoryRetriever:
    def __init__(
        self,
        sqlite_memory: SqliteMemory,
        vector_memory: VectorMemory,
        *,
        reranker: MemoryReranker | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self.sqlite_memory = sqlite_memory
        self.vector_memory = vector_memory
        self.reranker = reranker
        self.audit = audit
        self.policy = MemoryPolicy()

    async def hybrid_search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Two-stage retrieval: deterministic candidate collection, optional rerank.

        Stage one gathers up to ``CANDIDATE_LIMIT`` lexical and vector matches in
        deterministic order. Stage two asks the configured local reranker to pick
        the best few; when the reranker is absent or unavailable the deterministic
        order stands (and the fallback is audited). Kept memories are marked used.
        """
        candidates = await self._collect_candidates(query)
        selected = candidates[:limit]
        if self.reranker is not None and len(candidates) > 1:
            rerank_limit = min(limit, RERANK_LIMIT)
            ordered_ids = await self.reranker.rerank(query, candidates, limit=rerank_limit)
            if ordered_ids is None:
                if self.audit is not None:
                    self.audit.write(
                        {
                            "event_type": "memory_rerank_fallback",
                            "actor": "memory_retriever",
                            "detail": "local reranker unavailable; deterministic ranking used",
                            "candidate_count": len(candidates),
                        }
                    )
            else:
                by_id = {candidate.id: candidate for candidate in candidates}
                selected = [by_id[memory_id] for memory_id in ordered_ids][:rerank_limit]
        await self.sqlite_memory.mark_memories_used([result.id for result in selected])
        return selected

    async def _collect_candidates(self, query: str) -> list[SearchResult]:
        lexical_memories = [
            memory
            for memory in await self.sqlite_memory.search_memories(query)
            if not self.policy.is_sensitive(memory.content)
        ]
        try:
            vector = self.vector_memory.search(
                query, limit=CANDIDATE_LIMIT, source_type="memory"
            )
        except Exception:
            vector = []
        results: list[SearchResult] = [
            SearchResult(
                id=memory.id,
                score=1.0,
                content=memory.content,
                metadata={"kind": memory.kind, "reason": memory.reason},
            )
            for memory in lexical_memories[:CANDIDATE_LIMIT]
        ]
        seen = {result.id for result in results}
        for result in vector:
            if len(results) >= CANDIDATE_LIMIT:
                break
            if result.id in seen:
                continue
            record = await self.sqlite_memory.get_memory(result.id)
            if record is None or self.policy.is_sensitive(record.content):
                continue
            results.append(
                result.model_copy(
                    update={
                        "content": record.content,
                        "metadata": {
                            **result.metadata,
                            "kind": record.kind,
                            "reason": record.reason,
                        },
                    }
                )
            )
            seen.add(result.id)
        return results

    async def recent_memories(self, *, limit: int = 5) -> list[SearchResult]:
        memories = [
            memory
            for memory in await self.sqlite_memory.list_memories()
            if not self.policy.is_sensitive(memory.content)
        ][:limit]
        results = [
            SearchResult(
                id=memory.id,
                score=1.0,
                content=memory.content,
                metadata={"kind": memory.kind, "reason": memory.reason},
            )
            for memory in memories
        ]
        await self.sqlite_memory.mark_memories_used([result.id for result in results])
        return results

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

    def document_chunks(
        self,
        query: str,
        *,
        limit: int = 4,
        max_chars: int = 6000,
    ) -> list[SearchResult]:
        chunks: list[SearchResult] = []
        total_chars = 0
        for result in self.vector_memory.search(query, limit=limit * 3, source_type="document"):
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
