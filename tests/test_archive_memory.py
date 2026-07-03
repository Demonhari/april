from __future__ import annotations

import json

import numpy as np
import pytest

from agents.memory import ArchiveAgent, ArchiveMemoryCandidate
from april_common.audit import AuditLogger
from services.april_runtime.schemas import ChatResponse, Usage
from services.memory.archive import ArchiveMemoryWriter, ArchiveReflectionService
from services.memory.database import Database
from services.memory.embeddings import EmbeddingProvider
from services.memory.migrations import run_migrations
from services.memory.retriever import MemoryRetriever
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory
from services.wake.session_manager import SessionManager


class FakeArchiveRuntime:
    def __init__(self, content: str) -> None:
        self.content = content

    async def chat(self, **kwargs: object) -> ChatResponse:
        return ChatResponse(
            request_id="archive-test",
            model_id="april-brain",
            content=self.content,
            usage=Usage(),
        )


class FakeArchiveAgent:
    def __init__(self, candidates: list[ArchiveMemoryCandidate]) -> None:
        self.candidates = candidates
        self.transcripts: list[str] = []

    async def extract(
        self, transcript: str, *, request_id: str | None = None
    ) -> list[ArchiveMemoryCandidate]:
        self.transcripts.append(transcript)
        return self.candidates


class TinyEmbedding(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "tiny-fake"

    @property
    def dimensions(self) -> int:
        return 4

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for index, char in enumerate(text.encode("utf-8")):
            vector[index % self.dimensions] += float(char % 7)
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm


async def _memory(settings_tmp) -> tuple[Database, SqliteMemory]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


def _candidate(
    content: str,
    *,
    kind: str = "preference",
    confidence: float = 0.9,
) -> ArchiveMemoryCandidate:
    return ArchiveMemoryCandidate(
        kind=kind,  # type: ignore[arg-type]
        content=content,
        reason="explicitly stated in session",
        confidence=confidence,
    )


@pytest.mark.asyncio
async def test_archive_agent_requires_strict_json() -> None:
    valid = ArchiveAgent(
        FakeArchiveRuntime(
            json.dumps(
                {
                    "memories": [
                        {
                            "kind": "preference",
                            "content": "I prefer concise answers",
                            "reason": "user said so",
                            "confidence": 0.9,
                        }
                    ]
                }
            )
        ),
        model_id="april-brain",
    )
    assert (await valid.extract("user: remember I prefer concise answers"))[0].content == (
        "I prefer concise answers"
    )

    invalid = ArchiveAgent(FakeArchiveRuntime("remember: concise"), model_id="april-brain")
    assert await invalid.extract("user: remember I prefer concise answers") == []


@pytest.mark.asyncio
async def test_archive_writer_policy_and_vector_index(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        vector = VectorMemory(settings_tmp.vector_index_path, embedding=TinyEmbedding())
        audit = AuditLogger(settings_tmp.audit_path)
        await memory.create_memory(
            "I prefer concise answers",
            kind="preference",
            reason="existing",
        )
        await memory.create_memory("I like coffee", kind="preference", reason="existing")
        writer = ArchiveMemoryWriter(
            memory,
            vector_memory=vector,
            audit=audit,
            daily_cap=20,
        )
        outcomes = await writer.write_candidates(
            [
                _candidate("I prefer concise answers"),
                _candidate("I prefer detailed answers", confidence=0.2),
                _candidate("I not like coffee"),
                _candidate("My project is APRIL", kind="project_state"),
            ],
            source_session_id="session-1",
        )
        assert [outcome.status for outcome in outcomes] == [
            "duplicate",
            "low_confidence",
            "contradiction",
            "written",
        ]
        written = outcomes[-1].memory
        assert written is not None
        assert written.source == "archive"
        assert written.confidence == pytest.approx(0.9)
        assert vector.search("APRIL", source_type="memory")[0].id == written.id
        audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
        assert "archive_memory_written" in audit_text
        assert "archive_memory_low_confidence" in audit_text
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_archive_reflection_runs_on_session_close(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        candidate = _candidate("I prefer morning planning")
        agent = FakeArchiveAgent([candidate])
        service = ArchiveReflectionService(
            settings_tmp,
            memory=memory,
            runtime_client=FakeArchiveRuntime("{}"),  # type: ignore[arg-type]
            archive_agent=agent,  # type: ignore[arg-type]
            writer=ArchiveMemoryWriter(memory, daily_cap=10),
        )
        manager = SessionManager(
            memory,
            continuity_minutes=10,
            on_close=service.reflect_session,
        )
        conversation_id = await memory.create_conversation()
        session = await memory.create_session(source="voice", conversation_id=conversation_id)
        await memory.add_message(conversation_id, "user", "Remember I prefer morning planning.")
        assert await manager.close(session.id) is True

        memories = await memory.search_memories("morning")
        assert len(memories) == 1
        assert memories[0].content == "I prefer morning planning"
        assert "Remember I prefer morning planning" in agent.transcripts[0]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_memory_retrieval_filters_inactive_and_updates_usage(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        vector = VectorMemory(settings_tmp.vector_index_path, embedding=TinyEmbedding())
        writer = ArchiveMemoryWriter(memory, vector_memory=vector, daily_cap=10)
        outcomes = await writer.write_candidates(
            [_candidate("I prefer local-only answers")],
            source_session_id="session-1",
        )
        active = outcomes[0].memory
        assert active is not None
        await memory.create_memory(
            "I prefer expired answers",
            kind="preference",
            reason="expired",
            expires_at="2000-01-01T00:00:00Z",
        )
        await memory.create_memory(
            "I prefer superseded answers",
            kind="preference",
            reason="superseded",
            superseded_by=active.id,
        )

        retriever = MemoryRetriever(memory, vector)
        results = await retriever.hybrid_search("local answers", limit=5)
        assert [result.content for result in results] == ["I prefer local-only answers"]
        used = await memory.get_memory(active.id)
        assert used is not None
        assert used.use_count == 1
        assert used.last_used_at is not None
    finally:
        await database.close()
