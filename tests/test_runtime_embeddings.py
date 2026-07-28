from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from april_common.errors import RuntimeUnavailableError
from april_common.settings import reset_settings_cache
from services.april_runtime.backend import RuntimeBackend
from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import (
    MAX_EMBED_BATCH_CHARACTERS,
    MAX_EMBED_BATCH_ITEMS,
    MAX_EMBED_ITEM_CHARACTERS,
)
from services.april_runtime.server import create_app
from services.memory.embeddings import RuntimeLocalEmbedding


def _registry(tmp_path: Path, *, with_embedding: bool = True) -> ModelRegistry:
    models: dict[str, dict] = {
        "brain": {
            "id": "april-brain",
            "name": "fake",
            "path": "missing.gguf",
            "backend": "fake",
            "role": "brain",
            "chat_format": "generic",
            "threads": 1,
            "context_size": 1024,
            "temperature": 0.2,
            "max_output_tokens": 64,
            "keep_loaded": True,
        }
    }
    if with_embedding:
        models["embedding"] = {
            "id": "april-embedding",
            "name": "fake-embedding",
            "path": "missing-embedding.gguf",
            "backend": "fake",
            "role": "embedding",
            "threads": 1,
            "context_size": 1024,
            "temperature": 0.0,
            "max_output_tokens": 1,
            "keep_loaded": True,
        }
    return ModelRegistry.from_dict({"models": models}, root=tmp_path)


def _client(tmp_path: Path, *, with_embedding: bool = True) -> TestClient:
    old_home = os.environ.get("APRIL_HOME")
    os.environ["APRIL_HOME"] = str(tmp_path)
    reset_settings_cache()
    try:
        lifecycle = ModelLifecycle(
            _registry(tmp_path, with_embedding=with_embedding), root_backend="fake"
        )
        return TestClient(create_app(lifecycle))
    finally:
        if old_home is None:
            os.environ.pop("APRIL_HOME", None)
        else:
            os.environ["APRIL_HOME"] = old_home
        reset_settings_cache()


def test_embed_returns_stable_vector(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = client.post("/runtime/embed", json={"text": "animation frame timing"})
    second = client.post("/runtime/embed", json={"text": "animation frame timing"})
    assert first.status_code == 200
    payload = first.json()
    assert payload["model_id"] == "april-embedding"
    assert payload["dimensions"] == FakeBackend.EMBEDDING_DIMENSIONS
    assert len(payload["embedding"]) == FakeBackend.EMBEDDING_DIMENSIONS
    assert first.json()["embedding"] == second.json()["embedding"]


def test_embed_without_embedding_model_returns_error(tmp_path: Path) -> None:
    client = _client(tmp_path, with_embedding=False)
    response = client.post("/runtime/embed", json={"text": "hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_embed_batch_success_order_and_limits(tmp_path: Path) -> None:
    client = _client(tmp_path)
    texts = ["first English", "தமிழ் இரண்டாவது", "third mixed தமிழ்"]
    response = client.post(
        "/runtime/embed/batch",
        json={"texts": texts, "request_id": "batch-1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "batch-1"
    assert payload["count"] == len(texts)
    assert payload["item_indices"] == [0, 1, 2]
    assert payload["dimensions"] == FakeBackend.EMBEDDING_DIMENSIONS
    assert payload["embeddings"] == [FakeBackend()._deterministic_embedding(text) for text in texts]

    assert client.post("/runtime/embed/batch", json={"texts": []}).status_code == 422
    assert (
        client.post(
            "/runtime/embed/batch",
            json={"texts": ["x"] * (MAX_EMBED_BATCH_ITEMS + 1)},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/runtime/embed/batch",
            json={"texts": ["x" * (MAX_EMBED_ITEM_CHARACTERS + 1)]},
        ).status_code
        == 422
    )
    oversized_total = ["x" * (MAX_EMBED_BATCH_CHARACTERS // 9)] * 10
    assert client.post("/runtime/embed/batch", json={"texts": oversized_total}).status_code == 422


def test_embed_batch_rejects_missing_and_wrong_role_models(tmp_path: Path) -> None:
    missing = _client(tmp_path, with_embedding=False)
    assert missing.post("/runtime/embed/batch", json={"texts": ["hello"]}).status_code == 503
    wrong = _client(tmp_path)
    response = wrong.post(
        "/runtime/embed/batch",
        json={"texts": ["hello"], "model_id": "april-brain"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


@pytest.mark.parametrize("mode", ["count", "dimensions", "non_finite"])
def test_embed_batch_rejects_malformed_backend_matrix(tmp_path: Path, mode: str) -> None:
    class BadBatchBackend(FakeBackend):
        async def embed_many(self, texts: list[str]) -> list[list[float]]:
            if mode == "count":
                return [[1.0, 0.0]]
            if mode == "dimensions":
                return [[1.0, 0.0], [1.0]]
            return [[1.0, 0.0], [float("nan"), 0.0]]

    client = _client(tmp_path)
    state = client.app.state.lifecycle.get_state("april-embedding")
    state.state = "loaded"
    state.backend = BadBatchBackend()
    response = client.post("/runtime/embed/batch", json={"texts": ["a", "b"]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_embedding_state_visible_in_health_and_models(tmp_path: Path) -> None:
    client = _client(tmp_path)
    health = client.get("/runtime/health").json()
    assert health["embedding_model_id"] == "april-embedding"
    models = client.get("/runtime/models").json()["models"]
    roles = {model["id"]: model["role"] for model in models}
    assert roles["april-embedding"] == "embedding"


async def test_default_backend_embed_raises_runtime_unavailable() -> None:
    class _BareBackend(RuntimeBackend):
        async def load(self, model):  # type: ignore[no-untyped-def]
            return None

        async def unload(self) -> None:
            return None

        async def generate(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def stream(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def tokenize(self, text: str) -> list[int]:
            return []

        async def health(self):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    with pytest.raises(RuntimeUnavailableError):
        await _BareBackend().embed("hi")


async def test_backend_batch_fallback_is_sequential() -> None:
    class SequentialBackend(RuntimeBackend):
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def load(self, model):  # type: ignore[no-untyped-def]
            return None

        async def unload(self) -> None:
            return None

        async def generate(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def stream(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def tokenize(self, text: str) -> list[int]:
            return []

        async def embed(self, text: str) -> list[float]:
            self.calls.append(text)
            return [float(len(self.calls))]

        async def health(self):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    backend = SequentialBackend()
    assert await backend.embed_many(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]
    assert backend.calls == ["a", "b", "c"]


async def test_runtime_client_embed_round_trips(tmp_path: Path, monkeypatch) -> None:
    old_home = os.environ.get("APRIL_HOME")
    os.environ["APRIL_HOME"] = str(tmp_path)
    reset_settings_cache()
    try:
        lifecycle = ModelLifecycle(_registry(tmp_path), root_backend="fake")
        app = create_app(lifecycle)
        transport = httpx.ASGITransport(app=app)
        original = httpx.AsyncClient

        def factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr("services.april_runtime.client.httpx.AsyncClient", factory)

        from services.april_runtime.client import RuntimeClient

        runtime = RuntimeClient("http://127.0.0.1:8766")
        vector = await runtime.embed("animation frame timing")
        assert len(vector) == FakeBackend.EMBEDDING_DIMENSIONS
        backend_vector = await FakeBackend().embed("animation frame timing")
        assert vector == pytest.approx(backend_vector)
        vectors = await runtime.embed_many(["first", "தமிழ்"])
        assert vectors == [
            await FakeBackend().embed("first"),
            await FakeBackend().embed("தமிழ்"),
        ]
    finally:
        if old_home is None:
            os.environ.pop("APRIL_HOME", None)
        else:
            os.environ["APRIL_HOME"] = old_home
        reset_settings_cache()


async def test_runtime_client_batch_404_falls_back_but_validation_does_not(
    monkeypatch,
) -> None:
    old_app = FastAPI()
    calls = {"single": 0}

    @old_app.post("/runtime/embed")
    async def legacy_embed(payload: dict) -> dict:
        calls["single"] += 1
        return {
            "request_id": "legacy",
            "model_id": payload.get("model_id") or "april-embedding",
            "dimensions": 2,
            "embedding": [float(calls["single"]), 0.0],
        }

    transport = httpx.ASGITransport(app=old_app)
    original = httpx.AsyncClient

    def factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr("services.april_runtime.client.httpx.AsyncClient", factory)
    from services.april_runtime.client import RuntimeClient

    runtime = RuntimeClient("http://127.0.0.1:8766")
    assert await runtime.embed_many(["a", "b"]) == [[1.0, 0.0], [2.0, 0.0]]
    assert calls["single"] == 2

    malformed_app = FastAPI()

    @malformed_app.post("/runtime/embed/batch")
    async def malformed_batch(payload: dict) -> dict:
        return {
            "request_id": payload["request_id"],
            "model_id": "april-embedding",
            "count": 2,
            "dimensions": 2,
            "embeddings": [[1.0, 0.0]],
            "item_indices": [0, 1],
        }

    malformed_transport = httpx.ASGITransport(app=malformed_app)

    def malformed_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = malformed_transport
        return original(*args, **kwargs)

    calls["single"] = 0
    monkeypatch.setattr(
        "services.april_runtime.client.httpx.AsyncClient",
        malformed_factory,
    )
    with pytest.raises(RuntimeUnavailableError, match="malformed"):
        await runtime.embed_many(["a", "b"])
    assert calls["single"] == 0


def test_runtime_local_embedding_uses_one_batch_call() -> None:
    class BatchClient:
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, text: str, *, model_id: str | None = None) -> list[float]:
            return [1.0, 0.0]

        async def embed_many(
            self,
            texts: list[str],
            *,
            model_id: str | None = None,
        ) -> list[list[float]]:
            self.calls += 1
            return [[float(index), 1.0] for index, _text in enumerate(texts)]

    client = BatchClient()
    provider = RuntimeLocalEmbedding(client, "april-embedding")
    matrix = provider.embed_many(["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert client.calls == 1
