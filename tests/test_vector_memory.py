from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
from pathlib import Path

import numpy as np
import pytest

from april_common.errors import ConfigError, ModelUnavailableError
from services.memory.embeddings import (
    HashedTokenEmbedding,
    RuntimeLocalEmbedding,
    embedding_provider_from_config,
)
from services.memory.schemas import VectorMetadata
from services.memory.vector_memory import (
    FORMAT_VERSION,
    GenerationValidationResult,
    VectorMemory,
)


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
    assert np.array_equal(
        embedder.embed_many(["Hello world", "another"]),
        np.stack([embedder.embed("Hello world"), embedder.embed("another")]),
    )


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
    generation = (tmp_path / "CURRENT").read_text(encoding="utf-8").strip()
    generation_path = tmp_path / "generations" / generation
    assert (generation_path / "records.json").exists()
    assert (generation_path / "vectors.npy").exists()
    assert '"vector"' not in (generation_path / "records.json").read_text(encoding="utf-8")
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


def _current_generation(path: Path) -> str:
    return (path / "CURRENT").read_text(encoding="utf-8").strip()


def _generation_path(path: Path) -> Path:
    return path / "generations" / _current_generation(path)


def _multiprocess_upsert(path: str, record_id: str) -> None:
    memory = VectorMemory(Path(path))
    memory.upsert(
        record_id=record_id,
        content=f"process {record_id}",
        metadata=metadata(record_id),
    )


def test_generation_metadata_and_hashes_are_complete(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="one", metadata=metadata("h1"))

    generation = _current_generation(tmp_path)
    directory = _generation_path(tmp_path)
    header = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert header["format_version"] == FORMAT_VERSION
    assert header["generation_id"] == generation
    assert header["provider"] == "hashed-token"
    assert header["dimensions"] == 256
    assert header["record_count"] == header["vector_count"] == 1
    assert header["failure_state"] is None
    assert (
        header["records_sha256"]
        == hashlib.sha256((directory / "records.json").read_bytes()).hexdigest()
    )
    assert (
        header["vectors_sha256"]
        == hashlib.sha256((directory / "vectors.npy").read_bytes()).hexdigest()
    )
    assert (tmp_path / "CURRENT").read_bytes().endswith(b"\n")
    assert memory.validate_generation(directory).ok


def test_mutations_publish_generations_and_noop_delete_does_not(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="one", metadata=metadata("h1"))
    first = _current_generation(tmp_path)
    memory.index_chunks(
        source_type="repo",
        source_id="repo",
        chunks=[("a.py", "chunk", 1, 1)],
    )
    second = _current_generation(tmp_path)
    assert second != first
    assert memory.delete("missing") is False
    assert _current_generation(tmp_path) == second
    assert memory.delete("1") is True
    assert _current_generation(tmp_path) != second


def test_generation_retention_keeps_active_and_one_recovery(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path, retention=2)
    for index in range(4):
        memory.upsert(
            record_id=str(index),
            content=f"content {index}",
            metadata=metadata(f"h{index}"),
        )
    generations = list((tmp_path / "generations").iterdir())
    assert len(generations) == 2
    assert (tmp_path / "generations" / _current_generation(tmp_path)).is_dir()
    assert memory.health()["previous_generation"] is not None


@pytest.mark.parametrize(
    "failure_point",
    [
        "records",
        "vectors",
        "metadata",
        "validation",
        "publication",
        "current_write",
        "current_replace",
    ],
)
def test_publication_failure_keeps_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="old", content="old value", metadata=metadata("old"))
    old_generation = _current_generation(tmp_path)

    original_text = memory._write_text_file
    original_validate = memory.validate_generation
    if failure_point in {"records", "metadata"}:
        target = f"{failure_point}.json"

        def fail_text(path: Path, content: str) -> None:
            if path.name == target:
                raise OSError("injected write failure")
            original_text(path, content)

        monkeypatch.setattr(memory, "_write_text_file", fail_text)
    elif failure_point == "vectors":
        monkeypatch.setattr(
            memory,
            "_write_vector_file",
            lambda _path, _vectors: (_ for _ in ()).throw(OSError("injected")),
        )
    elif failure_point == "validation":

        def fail_validation(
            path: Path, *, require_compatible: bool = True, published: bool = True
        ) -> GenerationValidationResult:
            if path.parent == memory.staging_path:
                return GenerationValidationResult(
                    False, path.name, ("injected_validation_failure",)
                )
            return original_validate(
                path,
                require_compatible=require_compatible,
                published=published,
            )

        monkeypatch.setattr(memory, "validate_generation", fail_validation)
    elif failure_point == "publication":
        monkeypatch.setattr(
            memory,
            "_publish_staging_directory",
            lambda _staging, _published: (_ for _ in ()).throw(OSError("injected")),
        )
    elif failure_point == "current_write":

        def fail_current_write(descriptor: int, _generation_id: str) -> None:
            import os

            os.close(descriptor)
            raise OSError("injected")

        monkeypatch.setattr(memory, "_write_current_temp_file", fail_current_write)
    elif failure_point == "current_replace":
        monkeypatch.setattr(
            memory,
            "_replace_current_pointer",
            lambda _temp: (_ for _ in ()).throw(OSError("injected")),
        )

    with pytest.raises((OSError, RuntimeError)):
        memory.upsert(record_id="new", content="new value", metadata=metadata("new"))
    assert _current_generation(tmp_path) == old_generation
    monkeypatch.undo()
    assert [result.id for result in memory.search("old value")] == ["old"]


def test_validation_detects_counts_dimensions_provider_hashes_and_non_finite(
    tmp_path: Path,
) -> None:
    mutators = {
        "record_vector_count_mismatch": lambda directory: (directory / "records.json").write_text(
            "[]\n", encoding="utf-8"
        ),
        "dimension_mismatch": lambda directory: _edit_metadata(directory, "dimensions", 12),
        "provider_mismatch": lambda directory: _edit_metadata(
            directory, "provider", "runtime-local"
        ),
        "generation_hash_mismatch": lambda directory: (directory / "records.json").write_text(
            (directory / "records.json").read_text(encoding="utf-8").replace("value", "tampered"),
            encoding="utf-8",
        ),
        "vectors_non_finite": _write_nan_vector,
    }
    for reason, mutate in mutators.items():
        path = tmp_path / reason
        memory = VectorMemory(path)
        memory.upsert(record_id="1", content="value", metadata=metadata("h"))
        directory = _generation_path(path)
        mutate(directory)
        result = memory.validate_generation(directory)
        assert reason in result.failure_reasons


def test_validation_detects_vector_file_hash_mismatch(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="value", metadata=metadata("h"))
    directory = _generation_path(tmp_path)
    vectors_path = directory / "vectors.npy"
    vectors = np.load(vectors_path, allow_pickle=False)
    vectors[0, 0] += 1.0
    with vectors_path.open("wb") as handle:
        np.save(handle, vectors)
    result = memory.validate_generation(directory)
    assert "generation_hash_mismatch" in result.failure_reasons


def _edit_metadata(directory: Path, key: str, value: object) -> None:
    path = directory / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_nan_vector(directory: Path) -> None:
    path = directory / "vectors.npy"
    vectors = np.load(path, allow_pickle=False)
    vectors[0, 0] = np.nan
    with path.open("wb") as handle:
        np.save(handle, vectors)


@pytest.mark.parametrize("pointer", ["../escape\n", "bad/name\n", "not-a-generation\n", "x"])
def test_malformed_current_uses_recovery_without_repairing(tmp_path: Path, pointer: str) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="value", metadata=metadata("h"))
    (tmp_path / "CURRENT").write_text(pointer, encoding="utf-8")

    health = memory.health()
    assert health["status"] == "degraded"
    assert health["fallback_active"] is True
    assert "current_malformed" in health["failure_reasons"]
    assert memory.search("value")[0].id == "1"
    assert (tmp_path / "CURRENT").read_text(encoding="utf-8") == pointer


def test_invalid_current_falls_back_and_explicit_repair_switches_pointer(
    tmp_path: Path,
) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="1", content="first", metadata=metadata("h1"))
    recovery = _current_generation(tmp_path)
    memory.upsert(record_id="2", content="second", metadata=metadata("h2"))
    broken = _current_generation(tmp_path)
    records_path = tmp_path / "generations" / broken / "records.json"
    records_path.write_text(
        records_path.read_text(encoding="utf-8").replace("second", "tampered"),
        encoding="utf-8",
    )

    health = memory.health()
    assert health["effective_generation"] == recovery
    assert health["active_generation"] == broken
    assert health["fallback_active"] is True
    assert "generation_hash_mismatch" in health["failure_reasons"]
    dry_run = memory.repair_index()
    assert dry_run["applied"] is False
    assert dry_run["recovery_candidate"] == recovery
    assert _current_generation(tmp_path) == broken
    applied = memory.repair_index(apply=True)
    assert applied["applied"] is True
    assert _current_generation(tmp_path) == recovery


def test_missing_current_generation_and_no_valid_generation_are_safe(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path)
    (tmp_path / "CURRENT").write_text("g-20260101T000000000000Z-aaaaaaaaaaaa\n", encoding="utf-8")
    health = memory.health()
    assert health["status"] == "not_ready"
    assert "current_generation_missing" in health["failure_reasons"]
    assert "no_valid_generation" in health["failure_reasons"]
    assert memory.repair_index()["action"] == "reindex_required"


def test_health_never_raises_or_exposes_content_for_corrupt_generation(
    tmp_path: Path,
) -> None:
    memory = VectorMemory(tmp_path)
    secret_content = "private memory body"
    memory.upsert(record_id="1", content=secret_content, metadata=metadata("h"))
    records = _generation_path(tmp_path) / "records.json"
    records.write_text("{not-json", encoding="utf-8")

    health = memory.health()

    assert health["status"] == "not_ready"
    assert "no_valid_generation" in health["failure_reasons"]
    assert secret_content not in json.dumps(health)
    assert str(tmp_path) not in json.dumps(health)


class _BatchEmbedding(HashedTokenEmbedding):
    def __init__(self, *, fail_on: str | None = None) -> None:
        super().__init__(16)
        self.fail_on = fail_on
        self.batch_calls: list[list[str]] = []

    def embed_many(self, texts: list[str]) -> np.ndarray:
        self.batch_calls.append(list(texts))
        if self.fail_on is not None and self.fail_on in texts:
            raise RuntimeError("embedding failed")
        return super().embed_many(texts)


def test_reindex_uses_batches_and_publishes_once(tmp_path: Path) -> None:
    built = VectorMemory(tmp_path)
    built.upsert_many([(str(index), f"value {index}", metadata(f"h{index}")) for index in range(5)])
    provider = _BatchEmbedding()
    memory = VectorMemory(tmp_path, embedding=provider, embed_batch_size=2)
    before = set((tmp_path / "generations").iterdir())
    progress: list[tuple[int, int]] = []
    assert memory.reindex(progress=lambda done, total: progress.append((done, total))) == 5
    after = set((tmp_path / "generations").iterdir())
    assert len(after - before) == 1
    assert [len(batch) for batch in provider.batch_calls] == [2, 2, 1]
    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert memory.health()["last_successful_reindex_at"] is not None


def test_embedding_failure_during_reindex_keeps_current(tmp_path: Path) -> None:
    built = VectorMemory(tmp_path)
    built.upsert(record_id="1", content="fail", metadata=metadata("h"))
    current = _current_generation(tmp_path)
    memory = VectorMemory(tmp_path, embedding=_BatchEmbedding(fail_on="fail"))
    with pytest.raises(RuntimeError, match="embedding failed"):
        memory.reindex()
    assert _current_generation(tmp_path) == current


def test_cancellation_during_reindex_keeps_current_generation(tmp_path: Path) -> None:
    import asyncio

    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="old", content="old value", metadata=metadata("old"))
    current = _current_generation(tmp_path)

    def cancel(_done: int, _total: int) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        memory.reindex(progress=cancel)
    assert _current_generation(tmp_path) == current
    assert memory.health()["active_generation"] == current


def test_restart_reindex_discards_abandoned_staging_before_publication(
    tmp_path: Path,
) -> None:
    memory = VectorMemory(tmp_path)
    memory.upsert(record_id="old", content="old value", metadata=metadata("old"))
    current = _current_generation(tmp_path)
    abandoned = memory.staging_path / memory._new_generation_id()
    abandoned.mkdir()
    (abandoned / "records.json").write_text("partial", encoding="utf-8")

    restarted = VectorMemory(tmp_path)
    assert restarted.health()["active_generation"] == current
    assert restarted.health()["abandoned_staging_count"] == 1
    restarted.reindex()

    assert not abandoned.exists()
    assert _current_generation(tmp_path) != current
    assert restarted.health()["abandoned_staging_count"] == 0


def test_version_two_legacy_index_is_readable_and_migrates(tmp_path: Path) -> None:
    provider = HashedTokenEmbedding(16)
    records = [{"id": "1", "content": "legacy", "metadata": metadata("h").model_dump()}]
    (tmp_path / "records.json").write_text(json.dumps(records), encoding="utf-8")
    with (tmp_path / "vectors.npy").open("wb") as handle:
        np.save(handle, provider.embed_many(["legacy"]))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "provider": "hashed-token",
                "embedding_implementation_id": provider.implementation_id,
                "dimensions": 16,
                "record_count": 1,
            }
        ),
        encoding="utf-8",
    )
    memory = VectorMemory(tmp_path, embedding=provider)
    assert memory.health()["storage_mode"] == "legacy-v2"
    assert memory.search("legacy")[0].id == "1"
    memory.upsert(record_id="2", content="new", metadata=metadata("h2"))
    assert (tmp_path / "CURRENT").is_file()
    assert (tmp_path / "records.json").is_file()
    assert memory.health()["storage_mode"] == "generation"


def test_unversioned_jsonl_legacy_index_requires_reindex(tmp_path: Path) -> None:
    provider = HashedTokenEmbedding(16)
    record = {
        "id": "1",
        "content": "legacy jsonl",
        "metadata": metadata("h").model_dump(),
        "vector": provider.embed("legacy jsonl").tolist(),
    }
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    memory = VectorMemory(tmp_path, embedding=provider)
    assert memory.health()["storage_mode"] == "legacy-jsonl"
    with pytest.raises(ConfigError, match="Refusing to mix vector spaces"):
        memory.search("legacy")


def test_failed_legacy_migration_leaves_legacy_files(tmp_path: Path, monkeypatch) -> None:
    provider = HashedTokenEmbedding(16)
    record = {
        "id": "1",
        "content": "legacy",
        "metadata": metadata("h").model_dump(),
        "vector": provider.embed("legacy").tolist(),
    }
    legacy = tmp_path / "records.jsonl"
    legacy.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "provider": provider.name,
                "dimensions": provider.dimensions,
                "embedding_implementation_id": provider.implementation_id,
            }
        ),
        encoding="utf-8",
    )
    memory = VectorMemory(tmp_path, embedding=provider)
    monkeypatch.setattr(
        memory,
        "_publish_staging_directory",
        lambda _staging, _published: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        memory.upsert(record_id="2", content="new", metadata=metadata("h2"))
    assert legacy.is_file()
    assert not (tmp_path / "CURRENT").exists()


def test_concurrent_writers_and_readers_observe_complete_generations(tmp_path: Path) -> None:
    first = VectorMemory(tmp_path)
    second = VectorMemory(tmp_path)
    first.upsert(record_id="seed", content="seed", metadata=metadata("seed"))
    errors: list[Exception] = []
    seen_counts: list[int] = []

    def writer(memory: VectorMemory, prefix: str) -> None:
        try:
            for index in range(5):
                memory.upsert(
                    record_id=f"{prefix}-{index}",
                    content=f"{prefix} {index}",
                    metadata=metadata(f"{prefix}-{index}"),
                )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def reader() -> None:
        try:
            for _index in range(20):
                health = first.health()
                assert health["record_vector_count_match"] is True
                seen_counts.append(int(health["record_count"]))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(first, "a")),
        threading.Thread(target=writer, args=(second, "b")),
        threading.Thread(target=reader),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert seen_counts
    assert first.health()["record_count"] == 11
    assert (tmp_path / "generations" / _current_generation(tmp_path)).is_dir()


def test_advisory_lock_serializes_writers_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_multiprocess_upsert, args=(str(tmp_path), record_id))
        for record_id in ("one", "two")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    memory = VectorMemory(tmp_path)
    assert memory.health()["record_count"] == 2
    assert {result.id for result in memory.search("process", limit=10)} == {"one", "two"}
