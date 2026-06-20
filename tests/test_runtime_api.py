from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
    app = create_app(ModelLifecycle(registry, root_backend="fake"))
    return TestClient(app)


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
