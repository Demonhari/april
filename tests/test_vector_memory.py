from __future__ import annotations

import pytest

from april_common.errors import ConfigError
from services.memory.embeddings import HashedTokenEmbedding, embedding_provider_from_config
from services.memory.schemas import VectorMetadata
from services.memory.vector_memory import VectorMemory


def metadata(content_hash: str, project_id: str | None = None) -> VectorMetadata:
    return VectorMetadata(
        source_type="test",
        source_id="source",
        project_id=project_id,
        path="a.py",
        content_hash=content_hash,
        created_at="2026-01-01T00:00:00Z",
    )


def test_deterministic_embeddings() -> None:
    embedder = HashedTokenEmbedding(32)
    assert (embedder.embed("Hello world") == embedder.embed("hello world")).all()


def test_runtime_local_embedding_fails_closed() -> None:
    with pytest.raises(ConfigError, match="runtime-local"):
        embedding_provider_from_config("runtime-local", model_id="local-embed")


def test_persistence_and_similarity_search(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="animation frame timing", metadata=metadata("h1"))
    assert (tmp_path / "records.json").exists()
    assert (tmp_path / "vectors.npy").exists()
    assert '"vector"' not in (tmp_path / "records.json").read_text(encoding="utf-8")
    memory = VectorMemory(tmp_path)
    results = memory.search("animation")
    assert results[0].id == "1"


def test_project_scoped_similarity_search(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(
        record_id="1",
        content="animation frame timing",
        metadata=metadata("h1", project_id="project-a"),
    )
    memory.upsert(
        record_id="2",
        content="animation css",
        metadata=metadata("h2", project_id="project-b"),
    )
    results = memory.search("animation", project_id="project-a")
    assert [result.id for result in results] == ["1"]


def test_stale_chunk_removal(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="old", content="old", metadata=metadata("old", project_id="a"))
    memory.upsert(record_id="other", content="old", metadata=metadata("old", project_id="b"))
    removed = memory.delete_stale_for_path("a.py", {"new"}, project_id="a")
    assert removed == 1
    assert [result.id for result in memory.search("old", project_id="b")] == ["other"]


def test_index_chunks_removes_deleted_and_changed_files(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.index_chunks(
        source_type="repo",
        source_id="repo-1",
        project_id="project-1",
        chunks=[
            ("a.py", "animation old", 1, 1),
            ("b.py", "button old", 1, 1),
        ],
    )
    first_count = memory.health()["record_count"]
    memory.index_chunks(
        source_type="repo",
        source_id="repo-1",
        project_id="project-1",
        chunks=[
            ("a.py", "animation new", 1, 1),
        ],
    )
    assert first_count == 2
    assert memory.health()["record_count"] == 1
    results = memory.search("animation", project_id="project-1")
    assert results[0].metadata["path"] == "a.py"
    assert results[0].content == "animation new"


def test_index_chunks_is_idempotent(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    chunks = [("a.py", "animation frame", 1, 1)]
    memory.index_chunks(
        source_type="repo", source_id="repo-1", project_id="project-1", chunks=chunks
    )
    memory.index_chunks(
        source_type="repo", source_id="repo-1", project_id="project-1", chunks=chunks
    )
    assert memory.health()["record_count"] == 1
