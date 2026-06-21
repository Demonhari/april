from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.server import create_app

AUTH = {"Authorization": "Bearer local-dev-runtime-token"}


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry.from_dict(
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


def _events(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            names.append(json.loads(line[6:])["event"])
    return names


def _event_payloads(text: str) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payloads.append(json.loads(line[6:]))
    return payloads


def test_sse_token_order(tmp_path: Path) -> None:
    client = TestClient(create_app(ModelLifecycle(_registry(tmp_path), root_backend="fake")))
    response = client.post(
        "/runtime/stream",
        json={"model_id": "april-brain", "messages": [{"role": "user", "content": "hello"}]},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert _events(response.text)[0] == "meta"
    assert _events(response.text)[-1] == "done"
    assert "token" in _events(response.text)


def test_stream_and_chat_context_budget_metadata_match(tmp_path: Path) -> None:
    client = TestClient(create_app(ModelLifecycle(_registry(tmp_path), root_backend="fake")))
    request = {
        "model_id": "april-brain",
        "messages": [
            {"role": "system", "content": "local only"},
            {"role": "user", "content": "hello"},
        ],
        "options": {"max_output_tokens": 64},
    }
    chat = client.post("/runtime/chat", json=request, headers=AUTH)
    stream = client.post("/runtime/stream", json=request, headers=AUTH)
    assert chat.status_code == 200
    assert stream.status_code == 200
    chat_budget = chat.json()["diagnostics"]["context_budget"]
    meta_payload = _event_payloads(stream.text)[0]["payload"]
    assert isinstance(meta_payload, dict)
    assert meta_payload["context_budget"] == chat_budget


def test_stream_error_event(tmp_path: Path) -> None:
    lifecycle = ModelLifecycle(
        _registry(tmp_path),
        backend_factory=lambda model: FakeBackend(fail_stream=True),
        root_backend="fake",
    )
    client = TestClient(create_app(lifecycle))
    response = client.post(
        "/runtime/stream",
        json={"model_id": "april-brain", "messages": [{"role": "user", "content": "hello"}]},
        headers=AUTH,
    )
    assert "error" in _events(response.text)
