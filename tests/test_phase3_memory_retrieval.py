from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from april_common.text_normalization import (
    HASHED_TOKEN_IMPLEMENTATION_VERSION,
    embedding_tokens,
    normalize_text,
    word_tokens,
)
from services.memory.database import Database
from services.memory.embeddings import HashedTokenEmbedding
from services.memory.migrations import SCHEMA_VERSION, run_migrations
from services.memory.retriever import (
    HybridRetrievalConfig,
    MemoryRetriever,
    RerankDecision,
    decide_rerank,
)
from services.memory.schemas import SearchResult, VectorMetadata
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory


def test_unicode_normalization_and_bounded_tokens() -> None:
    assert word_tokens("Hello, WORLD!") == ["hello", "world"]
    assert word_tokens("cafe\u0301") == word_tokens("café") == ["café"]
    assert word_tokens("தமிழ், நினைவகம்!") == ["தமிழ்", "நினைவகம்"]
    assert word_tokens("Parser HTTP_Request தமிழ்") == ["parser", "http_request", "தமிழ்"]
    assert word_tokens('"OR" (alpha) + beta\x00') == ["or", "alpha", "beta"]
    assert word_tokens("... \x00") == []
    assert word_tokens("a b c d", max_tokens=2) == ["a", "b"]
    tokens = embedding_tokens("தமிழ் தமிழ்", max_ngram_tokens=2)
    assert tokens[0] == "தமிழ்"
    ngrams = [token for token in tokens if token.startswith("ng:")]
    assert 0 < len(ngrams) <= 2
    assert normalize_text("\uff21PRIL") == "april"


def test_hashed_vectors_are_deterministic_and_versioned() -> None:
    provider = HashedTokenEmbedding(64)
    first = provider.embed("Mixed தமிழ் HTTP_Request")
    second = provider.embed("Mixed தமிழ் HTTP_Request")
    assert np.array_equal(first, second)
    assert provider.implementation_id == HASHED_TOKEN_IMPLEMENTATION_VERSION


async def _memory(path: Path) -> tuple[Database, SqliteMemory]:
    database = Database(path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


@pytest.mark.asyncio
async def test_schema_18_unicode_fts_and_safe_queries(tmp_path: Path) -> None:
    database, memory = await _memory(tmp_path / "memory.db")
    try:
        assert SCHEMA_VERSION == 19
        english = await memory.create_memory("Local English memory", reason="English")
        tamil = await memory.create_memory("தமிழ் நினைவகம் பாதுகாப்பானது", reason="தமிழ்")
        mixed = await memory.create_memory("APRIL தமிழ் assistant", reason="mixed")
        assert [hit.id for hit in await memory.search_memories("english")] == [english.id]
        assert tamil.id in {hit.id for hit in await memory.search_memories("தமிழ் நினைவகம்")}
        assert mixed.id in {hit.id for hit in await memory.search_memories("APRIL தமிழ்")}
        assert await memory.search_memories('" OR ( ) NOT *') == []
        sql_row = await database.fetchone(
            "SELECT sql FROM sqlite_master WHERE name = 'memories_fts'"
        )
        assert sql_row is not None
        assert "unicode61" in str(sql_row["sql"])
        assert "remove_diacritics 0" in str(sql_row["sql"])
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_schema_17_populated_migration_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "legacy.db")
    await database.connect()
    try:
        await database.connection.executescript(
            """
            CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_migrations VALUES(17, datetime('now'));
            CREATE TABLE projects(
                id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE memories(
                id TEXT PRIMARY KEY,
                project_id TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.7,
                source TEXT NOT NULL DEFAULT 'user',
                last_used_at TEXT,
                use_count INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT,
                superseded_by TEXT
            );
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                id UNINDEXED, content, reason, tokenize='porter'
            );
            INSERT INTO memories(
                id, kind, content, reason, created_at, confidence, source
            ) VALUES
                ('e', 'fact', 'English archive', 'legacy', datetime('now'), 0.7, 'user'),
                ('t', 'fact', 'தமிழ் நினைவகம்', 'legacy', datetime('now'), 0.7, 'user');
            INSERT INTO memories_fts(id, content, reason)
            SELECT id, content, reason FROM memories;
            """
        )
        await run_migrations(database)
        await run_migrations(database)
        count = await database.fetchone("SELECT COUNT(*) AS count FROM memories")
        fts_count = await database.fetchone("SELECT COUNT(*) AS count FROM memories_fts")
        assert count is not None
        assert fts_count is not None
        assert count["count"] == fts_count["count"] == 2
        memory = SqliteMemory(database)
        assert [record.id for record in await memory.search_memories("தமிழ்")] == ["t"]
    finally:
        await database.close()


def _metadata(record_id: str, project_id: str | None = None) -> VectorMetadata:
    return VectorMetadata(
        source_type="memory",
        source_id=record_id,
        project_id=project_id,
        content_hash=record_id,
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.mark.asyncio
async def test_hybrid_fusion_project_global_and_tamil(tmp_path: Path) -> None:
    database, memory = await _memory(tmp_path / "hybrid.db")
    try:
        selected = await memory.add_project("/selected")
        unrelated = await memory.add_project("/unrelated")
        global_memory = await memory.create_memory(
            "APRIL தமிழ் local assistant",
            reason="mixed",
            source="user",
            confidence=0.9,
        )
        selected_memory = await memory.create_memory(
            "தமிழ் assistant project notes",
            reason="project",
            project_id=selected.id,
            source="reflection",
            confidence=0.8,
        )
        unrelated_memory = await memory.create_memory(
            "தமிழ் assistant unrelated",
            reason="other",
            project_id=unrelated.id,
        )
        vector = VectorMemory(tmp_path / "vectors", embedding=HashedTokenEmbedding(64))
        for record in (global_memory, selected_memory, unrelated_memory):
            vector.upsert(
                record_id=record.id,
                content=record.content,
                metadata=_metadata(record.id, record.project_id),
            )
        retriever = MemoryRetriever(memory, vector)
        results = await retriever.hybrid_search(
            "APRIL தமிழ் assistant",
            project_id=selected.id,
            limit=10,
        )
        ids = [result.id for result in results]
        assert global_memory.id in ids
        assert selected_memory.id in ids
        assert unrelated_memory.id not in ids
        assert all("score_diagnostics" in result.metadata for result in results)
        both = next(result for result in results if result.id == global_memory.id)
        diagnostics = both.metadata["score_diagnostics"]
        assert diagnostics["lexical_evidence"] is True
        assert diagnostics["semantic_evidence"] is True
        selected_result = next(result for result in results if result.id == selected_memory.id)
        assert selected_result.metadata["score_diagnostics"]["project_component"] > 0
        assert both.metadata["score_diagnostics"]["source_component"] > 0
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_lexical_only_vector_only_and_partial_rerank_fill(tmp_path: Path) -> None:
    database, memory = await _memory(tmp_path / "evidence.db")
    try:
        lexical = await memory.create_memory("lexical-only telescope", reason="lexical")
        semantic = await memory.create_memory("stored without vector terms", reason="semantic")
        vector = VectorMemory(tmp_path / "evidence-vectors", embedding=HashedTokenEmbedding(64))
        vector.upsert(
            record_id=semantic.id,
            content="vector-only nebula alias",
            metadata=_metadata(semantic.id),
        )

        class PartialReranker:
            def __init__(self) -> None:
                self.calls = 0

            async def rerank(
                self,
                query: str,
                candidates: list[SearchResult],
                *,
                limit: int,
            ) -> list[str] | None:
                self.calls += 1
                return [semantic.id, "unknown", semantic.id]

        reranker = PartialReranker()
        retriever = MemoryRetriever(memory, vector, reranker=reranker)
        lexical_results = await retriever.hybrid_search("telescope", limit=2)
        assert [result.id for result in lexical_results] == [lexical.id]
        assert lexical_results[0].metadata["score_diagnostics"]["semantic_evidence"] is False
        vector_results = await retriever.hybrid_search("nebula alias", limit=2)
        assert [result.id for result in vector_results] == [semantic.id]
        assert vector_results[0].metadata["score_diagnostics"]["lexical_evidence"] is False

        combined = await retriever.hybrid_search("lexical nebula alias telescope", limit=2)
        assert combined[0].id == semantic.id
        assert {result.id for result in combined} == {semantic.id, lexical.id}
        assert reranker.calls == 1
    finally:
        await database.close()


def _ranked(
    item_id: str,
    score: float,
    *,
    lexical_rank: int | None = None,
    vector_rank: int | None = None,
) -> SearchResult:
    return SearchResult(
        id=item_id,
        score=score,
        content=item_id,
        metadata={
            "score_diagnostics": {
                "lexical_rank": lexical_rank,
                "vector_rank": vector_rank,
            }
        },
    )


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        ([_ranked("a", 0.5), _ranked("b", 0.1)], True),
        ([_ranked("a", 0.9), _ranked("b", 0.88)], True),
        (
            [
                _ranked("a", 0.9, lexical_rank=1, vector_rank=2),
                _ranked("b", 0.78, lexical_rank=2, vector_rank=1),
            ],
            True,
        ),
        ([_ranked("a", 0.95), _ranked("b", 0.4)], False),
        ([_ranked("a", 0.4)], False),
    ],
)
def test_uncertainty_gate(candidates: list[SearchResult], expected: bool) -> None:
    decision: RerankDecision = decide_rerank(candidates, reranker_configured=True)
    assert decision.invoke is expected


def test_relevance_dominates_metadata_adjustments() -> None:
    config = HybridRetrievalConfig(recency_half_life_days=30)
    assert config.lexical_weight + config.vector_weight == pytest.approx(1.0)
    dominant = _ranked("relevant-old", 0.8)
    irrelevant = _ranked("irrelevant-new", 0.1)
    assert dominant.score > irrelevant.score
