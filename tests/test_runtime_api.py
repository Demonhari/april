from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from april_common.settings import reset_settings_cache
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.server import create_app


@contextlib.contextmanager
def _isolated_home(tmp_path: Path) -> Iterator[None]:
    """Point APRIL_HOME at an empty tmp dir so ``create_app`` reads built-in
    defaults (no repo configs or tokens), then restore the prior state.

    ``create_app`` captures settings in its lifespan/middleware closures, so it
    is safe for this isolation to end once the app object exists.
    """
    old_home = os.environ.get("APRIL_HOME")
    os.environ["APRIL_HOME"] = str(tmp_path)
    reset_settings_cache()
    try:
        yield
    finally:
        if old_home is None:
            os.environ.pop("APRIL_HOME", None)
        else:
            os.environ["APRIL_HOME"] = old_home
        reset_settings_cache()


def runtime_client(tmp_path: Path) -> TestClient:
    """Build a runtime TestClient for ``tmp_path``.

    Always use this as a context manager (``with runtime_client(tmp_path) as
    client:``) so the FastAPI lifespan startup/shutdown runs and the model
    lifecycle is cleaned up instead of leaking an unclosed client.
    """
    registry = ModelRegistry.from_dict(
        {
            "models": {
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
                    "keep_loaded": False,
                }
            }
        },
        root=tmp_path,
    )
    with _isolated_home(tmp_path):
        return TestClient(create_app(ModelLifecycle(registry, root_backend="fake")))


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
                    "chat_format": "generic",
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
    with runtime_client(tmp_path) as client:
        response = client.post(
            "/runtime/chat",
            json={"model_id": "april-brain", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 200
    assert response.json()["model_id"] == "april-brain"


def test_runtime_unknown_model(tmp_path: Path) -> None:
    with runtime_client(tmp_path) as client:
        response = client.post(
            "/runtime/chat",
            json={"model_id": "unknown", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_runtime_health_reports_simulated_ok(tmp_path: Path) -> None:
    # A working fake runtime with a missing GGUF path is ok and clearly marked
    # simulated; the missing model id is surfaced as informational data.
    with runtime_client(tmp_path) as client:
        health = client.get("/runtime/health").json()
    assert health["status"] == "ok"
    assert health["simulated"] is True
    assert health["backend"] == "fake"
    assert health["missing_models"] == ["april-brain"]


def test_runtime_lifespan_preload_and_cleanup(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path, keep_loaded=True)
    with TestClient(create_app(lifecycle)):
        assert lifecycle.list_models()[0].state == "loaded"
    assert lifecycle.list_models()[0].state == "unloaded"


def test_runtime_context_exit_unloads_after_generation(tmp_path: Path) -> None:
    # Regression: an unclosed TestClient never runs the lifespan shutdown, so a
    # model loaded on demand stays resident. Context-managed usage must release
    # the backend on exit.
    lifecycle = runtime_lifecycle(tmp_path, keep_loaded=True)
    with _isolated_home(tmp_path), TestClient(create_app(lifecycle)) as client:
        response = client.post(
            "/runtime/chat",
            json={"model_id": "april-brain", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert lifecycle.get_state("april-brain").state == "loaded"
    assert lifecycle.get_state("april-brain").state == "unloaded"


def test_runtime_auth_accepts_configured_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "runtime-token")
    reset_settings_cache()
    with runtime_client(tmp_path) as client:
        response = client.get(
            "/runtime/health",
            headers={"Authorization": "Bearer runtime-token"},
        )
    assert response.status_code == 200
    reset_settings_cache()


def test_runtime_auth_rejects_bad_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "runtime-token")
    reset_settings_cache()
    with runtime_client(tmp_path) as client:
        response = client.get(
            "/runtime/health",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 403
    reset_settings_cache()


def test_runtime_auth_rejects_missing_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "runtime-token")
    reset_settings_cache()
    with runtime_client(tmp_path) as client:
        response = client.get("/runtime/health")
    assert response.status_code == 403
    reset_settings_cache()


def test_runtime_auth_allows_unset_token_for_local_backwards_compatibility(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("APRIL_RUNTIME_TOKEN", raising=False)
    reset_settings_cache()
    with runtime_client(tmp_path) as client:
        response = client.get("/runtime/health")
    assert response.status_code == 200
    reset_settings_cache()
