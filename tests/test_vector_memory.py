from __future__ import annotations

from services.memory.embeddings import HashedTokenEmbedding
from services.memory.schemas import VectorMetadata
from services.memory.vector_memory import VectorMemory


def metadata(content_hash: str) -> VectorMetadata:
    return VectorMetadata(
        source_type="test",
        source_id="source",
        path="a.py",
        content_hash=content_hash,
        created_at="2026-01-01T00:00:00Z",
    )


def test_deterministic_embeddings() -> None:
    embedder = HashedTokenEmbedding(32)
    assert (embedder.embed("Hello world") == embedder.embed("hello world")).all()


def test_persistence_and_similarity_search(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="animation frame timing", metadata=metadata("h1"))
    memory = VectorMemory(tmp_path)
    results = memory.search("animation")
    assert results[0].id == "1"


def test_stale_chunk_removal(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="old", content="old", metadata=metadata("old"))
    removed = memory.delete_stale_for_path("a.py", {"new"})
    assert removed == 1
