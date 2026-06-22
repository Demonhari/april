from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from april_common.errors import RuntimeUnavailableError
from april_common.settings import reset_settings_cache
from services.april_runtime.backend import RuntimeBackend
from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.server import create_app


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
    finally:
        if old_home is None:
            os.environ.pop("APRIL_HOME", None)
        else:
            os.environ["APRIL_HOME"] = old_home
        reset_settings_cache()
