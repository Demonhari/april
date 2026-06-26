from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anyio
import pytest

from april_common.audit import AuditLogger
from april_common.errors import ConfigError, ModelUnavailableError
from april_common.settings import AprilSettings, reset_settings_cache
from services.memory.database import Database
from services.memory.factory import (
    configured_embedding_provider,
    vector_memory_from_settings,
)
from services.memory.migrations import run_migrations
from services.memory.vector_memory import VectorMemory
from skills.code.repo_indexer import repo_indexer
from skills.documents.document_indexer import document_indexer
from skills.documents.document_search import document_search


class _DeterministicRuntimeClient:
    """A drop-in RuntimeClient that returns deterministic local embeddings.

    Mirrors ``RuntimeClient(url, *, timeout=, token=)`` so it can be monkeypatched
    over the real client, and exposes the ``embed`` coroutine the runtime-local
    provider depends on. Never touches the network.
    """

    dimensions = 8
    available = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.embed_calls: list[str] = []

    async def embed(self, text: str, *, model_id: str | None = None) -> list[float]:
        self.embed_calls.append(text)
        if not self.available:
            raise ModelUnavailableError("embedding", "No embedding-role model is registered.")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) or 0.001 for i in range(self.dimensions)]


def _runtime_local(settings: AprilSettings) -> AprilSettings:
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={"embedding_provider": "runtime-local", "embedding_model_id": "april-embed"}
            )
        }
    )


def test_factory_defaults_to_hashed_token(settings_tmp) -> None:
    provider = configured_embedding_provider(settings_tmp)
    assert provider.name == "hashed-token"
    vector = vector_memory_from_settings(settings_tmp)
    assert vector.embedding.name == "hashed-token"


def test_factory_uses_runtime_local_when_configured_and_reachable(settings_tmp) -> None:
    client = _DeterministicRuntimeClient()
    provider = configured_embedding_provider(_runtime_local(settings_tmp), runtime_client=client)
    assert provider.name == "runtime-local"
    assert provider.dimensions == _DeterministicRuntimeClient.dimensions
    assert client.embed_calls  # provider probed the runtime to resolve dimensions


def test_factory_runtime_local_falls_back_to_hashed_token_with_audit(settings_tmp) -> None:
    unavailable = _DeterministicRuntimeClient()
    unavailable.available = False
    audit = AuditLogger(settings_tmp.audit_path)
    provider = configured_embedding_provider(
        _runtime_local(settings_tmp), runtime_client=unavailable, audit=audit
    )
    # Matches the existing embedding fallback policy: never a silent space switch.
    assert provider.name == "hashed-token"
    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "memory.embedding_fallback" in audit_text
    assert "runtime-local" in audit_text


def test_factory_builds_runtime_client_from_settings(settings_tmp, monkeypatch) -> None:
    # When no client is injected and runtime-local is configured, the factory must
    # build the standard April Runtime HTTP client from settings (models only ever
    # go through services/april_runtime), never a direct binding.
    monkeypatch.setattr("services.april_runtime.client.RuntimeClient", _DeterministicRuntimeClient)
    provider = configured_embedding_provider(_runtime_local(settings_tmp))
    assert provider.name == "runtime-local"


def _index_one(vector: VectorMemory, content: str) -> None:
    vector.index_chunks(
        source_type="document",
        source_id="docs",
        project_id=None,
        chunks=[("a.txt", content, 1, 1)],
    )


def test_factory_mismatched_persisted_provider_raises_clear_error(settings_tmp) -> None:
    # Persist an index under hashed-token, then reopen under runtime-local.
    _index_one(vector_memory_from_settings(settings_tmp), "animation pipeline")
    runtime_vector = vector_memory_from_settings(
        _runtime_local(settings_tmp), runtime_client=_DeterministicRuntimeClient()
    )
    with pytest.raises(ConfigError) as excinfo:
        runtime_vector.search("animation")
    message = str(excinfo.value)
    assert "different embedding configuration" in message
    assert "Refusing to mix vector spaces" in message


def test_vector_memory_from_settings_persists_configured_provider(settings_tmp) -> None:
    vector = vector_memory_from_settings(
        _runtime_local(settings_tmp), runtime_client=_DeterministicRuntimeClient()
    )
    _index_one(vector, "local note")
    header = json.loads((settings_tmp.vector_index_path / "metadata.json").read_text("utf-8"))
    assert header["provider"] == "runtime-local"
    assert header["dimensions"] == _DeterministicRuntimeClient.dimensions


def _persisted_provider(settings_tmp) -> str:
    header = json.loads((settings_tmp.vector_index_path / "metadata.json").read_text("utf-8"))
    return str(header["provider"])


def test_document_indexer_uses_configured_provider(settings_tmp, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_MEMORY_EMBEDDING_PROVIDER", "runtime-local")
    monkeypatch.setenv("APRIL_MEMORY_EMBEDDING_MODEL_ID", "april-embed")
    monkeypatch.setattr("services.april_runtime.client.RuntimeClient", _DeterministicRuntimeClient)
    reset_settings_cache()
    folder = settings_tmp.home / "docs"
    folder.mkdir()
    (folder / "guide.md").write_text("# guide\nanimation notes\n", encoding="utf-8")
    result = anyio.run(document_indexer, {"folder_path": str(folder)})
    assert result.ok
    # Real proof the skill honoured the configured provider rather than defaulting.
    assert _persisted_provider(settings_tmp) == "runtime-local"


def test_document_search_uses_configured_provider(settings_tmp, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_MEMORY_EMBEDDING_PROVIDER", "runtime-local")
    monkeypatch.setenv("APRIL_MEMORY_EMBEDDING_MODEL_ID", "april-embed")
    monkeypatch.setattr("services.april_runtime.client.RuntimeClient", _DeterministicRuntimeClient)
    reset_settings_cache()
    # Seed the index under runtime-local using the same factory the skill uses.
    vector = vector_memory_from_settings(
        _runtime_local(settings_tmp), runtime_client=_DeterministicRuntimeClient()
    )
    vector.index_chunks(
        source_type="document",
        source_id="docs",
        project_id=None,
        chunks=[(str(settings_tmp.home / "a.md"), "animation pipeline notes", 1, 2)],
    )
    result = anyio.run(document_search, {"query": "animation", "limit": 5})
    assert result.ok
    # A hashed-token default would have raised a vector-space mismatch here.
    assert result.data["chunks"]


def test_repo_indexer_uses_configured_provider(settings_tmp, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APRIL_MEMORY_EMBEDDING_PROVIDER", "runtime-local")
    monkeypatch.setenv("APRIL_MEMORY_EMBEDDING_MODEL_ID", "april-embed")
    monkeypatch.setattr("services.april_runtime.client.RuntimeClient", _DeterministicRuntimeClient)
    reset_settings_cache()

    async def _migrate() -> None:
        async with Database(settings_tmp.database_path) as database:
            await run_migrations(database)

    anyio.run(_migrate)
    repo = settings_tmp.home
    (repo / "module.py").write_text("print('animation')\n", encoding="utf-8")
    result = anyio.run(
        repo_indexer, {"repo_path": str(repo), "project_id": None, "force_full_reindex": False}
    )
    assert result.ok
    assert _persisted_provider(settings_tmp) == "runtime-local"


def test_hashed_token_provider_round_trips(settings_tmp) -> None:
    # The hashed-token provider stays the deterministic default and round-trips.
    vector = vector_memory_from_settings(settings_tmp)
    assert isinstance(vector, VectorMemory)
    vector.index_chunks(
        source_type="document",
        source_id="docs",
        project_id=None,
        chunks=[(str(settings_tmp.home / "a.md"), "animation pipeline notes", 1, 2)],
    )
    results = vector.search("animation", source_type="document")
    assert results
    assert _persisted_provider(settings_tmp) == "hashed-token"
