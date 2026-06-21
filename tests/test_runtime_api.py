from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from april_common.settings import reset_settings_cache
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.server import create_app


def runtime_client(tmp_path: Path) -> TestClient:
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "april-brain",
                    "name": "fake",
                    "path": "missing.gguf",
                    "backend": "fake",
                    "role": "brain",
                    "threads": 1,
                    "context_size": 1024,
                    "temperature": 0.2,
                    "max_output_tokens": 64,
                    "keep_loaded": False,
                }
            }
        },
        root=tmp_path,
    )
    old_home = os.environ.get("APRIL_HOME")
    os.environ["APRIL_HOME"] = str(tmp_path)
    reset_settings_cache()
    try:
        app = create_app(ModelLifecycle(registry, root_backend="fake"))
        return TestClient(app)
    finally:
        if old_home is None:
            os.environ.pop("APRIL_HOME", None)
        else:
            os.environ["APRIL_HOME"] = old_home
        reset_settings_cache()


def runtime_lifecycle(tmp_path: Path, *, keep_loaded: bool = False) -> ModelLifecycle:
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "april-brain",
                    "name": "fake",
                    "path": "missing.gguf",
                    "backend": "fake",
                    "role": "brain",
                    "threads": 1,
                    "context_size": 1024,
                    "temperature": 0.2,
                    "max_output_tokens": 64,
                    "keep_loaded": keep_loaded,
                }
            }
        },
        root=tmp_path,
    )
    return ModelLifecycle(registry, root_backend="fake")


def test_runtime_normal_generation(tmp_path: Path) -> None:
    client = runtime_client(tmp_path)
    response = client.post(
        "/runtime/chat",
        json={"model_id": "april-brain", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["model_id"] == "april-brain"


def test_runtime_unknown_model(tmp_path: Path) -> None:
    client = runtime_client(tmp_path)
    response = client.post(
        "/runtime/chat",
        json={"model_id": "unknown", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_runtime_lifespan_preload_and_cleanup(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path, keep_loaded=True)
    with TestClient(create_app(lifecycle)):
        assert lifecycle.list_models()[0].state == "loaded"
    assert lifecycle.list_models()[0].state == "unloaded"


def test_runtime_auth_accepts_configured_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "runtime-token")
    reset_settings_cache()
    client = runtime_client(tmp_path)
    response = client.get(
        "/runtime/health",
        headers={"Authorization": "Bearer runtime-token"},
    )
    assert response.status_code == 200
    reset_settings_cache()


def test_runtime_auth_rejects_bad_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "runtime-token")
    reset_settings_cache()
    client = runtime_client(tmp_path)
    response = client.get(
        "/runtime/health",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 403
    reset_settings_cache()


def test_runtime_auth_rejects_missing_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "runtime-token")
    reset_settings_cache()
    client = runtime_client(tmp_path)
    response = client.get("/runtime/health")
    assert response.status_code == 403
    reset_settings_cache()


def test_runtime_auth_allows_unset_token_for_local_backwards_compatibility(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("APRIL_RUNTIME_TOKEN", raising=False)
    reset_settings_cache()
    client = runtime_client(tmp_path)
    response = client.get("/runtime/health")
    assert response.status_code == 200
    reset_settings_cache()
