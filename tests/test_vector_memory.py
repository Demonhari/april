from __future__ import annotations

import pytest

from april_common.errors import ConfigError, ModelUnavailableError
from services.memory.embeddings import (
    HashedTokenEmbedding,
    RuntimeLocalEmbedding,
    embedding_provider_from_config,
)
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


class _FakeEmbedClient:
    def __init__(self, *, available: bool = True, dimensions: int = 8) -> None:
        self.available = available
        self.dimensions = dimensions
        self.calls: list[str] = []

    async def embed(self, text: str, *, model_id: str | None = None) -> list[float]:
        self.calls.append(text)
        if not self.available:
            raise ModelUnavailableError("embedding", "No embedding-role model is registered.")
        return [float((len(text) + index) % 5) for index in range(self.dimensions)]


def test_unknown_provider_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown memory embedding provider"):
        embedding_provider_from_config("nonsense")


def test_runtime_local_without_client_falls_back_to_hashed_token() -> None:
    provider = embedding_provider_from_config("runtime-local", model_id="local-embed")
    assert isinstance(provider, HashedTokenEmbedding)


def test_runtime_local_builds_when_embedding_model_available() -> None:
    client = _FakeEmbedClient(available=True, dimensions=8)
    provider = embedding_provider_from_config(
        "runtime-local", model_id="april-embedding", runtime_client=client
    )
    assert isinstance(provider, RuntimeLocalEmbedding)
    assert provider.name == "runtime-local"
    assert provider.dimensions == 8
    vector = provider.embed("animation frame timing")
    assert vector.shape == (8,)


def test_runtime_local_falls_back_when_no_embedding_model() -> None:
    client = _FakeEmbedClient(available=False)
    provider = embedding_provider_from_config(
        "runtime-local", model_id="april-embedding", runtime_client=client
    )
    assert isinstance(provider, HashedTokenEmbedding)


def test_runtime_local_health_reports_provider_and_dimensions(tmp_path) -> None:
    client = _FakeEmbedClient(available=True, dimensions=8)
    provider = embedding_provider_from_config(
        "runtime-local", model_id="april-embedding", runtime_client=client
    )
    memory = VectorMemory(tmp_path, embedding=provider)
    health = memory.health()
    assert health["embedding"] == "runtime-local"
    assert health["dimensions"] == 8


def test_runtime_local_fallback_is_audited(tmp_path) -> None:
    from april_common.audit import AuditLogger

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLogger(audit_path)
    client = _FakeEmbedClient(available=False)
    provider = embedding_provider_from_config(
        "runtime-local",
        model_id="april-embedding",
        runtime_client=client,
        audit=audit,
    )
    assert isinstance(provider, HashedTokenEmbedding)
    logged = audit_path.read_text(encoding="utf-8")
    assert "memory.embedding_fallback" in logged
    assert "hashed-token" in logged


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


def test_provider_dimension_mismatch_raises_actionable_error(tmp_path) -> None:
    built = VectorMemory(tmp_path, embedding=HashedTokenEmbedding(256))
    built.upsert(record_id="1", content="animation frame timing", metadata=metadata("h1"))
    reopened = VectorMemory(tmp_path, embedding=HashedTokenEmbedding(64))
    with pytest.raises(ConfigError, match="reindex"):
        reopened.search("animation")
    with pytest.raises(ConfigError, match="reindex"):
        reopened.upsert(record_id="2", content="other", metadata=metadata("h2"))
    assert reopened.health()["compatible"] is False


def test_reindex_rebuilds_under_new_provider(tmp_path) -> None:
    built = VectorMemory(tmp_path, embedding=HashedTokenEmbedding(256))
    built.upsert(record_id="1", content="animation frame timing", metadata=metadata("h1"))
    built.upsert(record_id="2", content="button layout css", metadata=metadata("h2"))

    client = _FakeEmbedClient(available=True, dimensions=8)
    runtime_local = embedding_provider_from_config(
        "runtime-local", model_id="april-embedding", runtime_client=client
    )
    switched = VectorMemory(tmp_path, embedding=runtime_local)

    progress: list[tuple[int, int]] = []
    reindexed = switched.reindex(progress=lambda done, total: progress.append((done, total)))
    assert reindexed == 2
    assert progress[-1] == (2, 2)

    health = switched.health()
    assert health["embedding"] == "runtime-local"
    assert health["dimensions"] == 8
    assert health["compatible"] is True

    results = switched.search("animation frame timing")
    assert {result.id for result in results} == {"1", "2"}


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
