from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.brain.fallback_router import FallbackRouter
from services.brain.reasoning_resolver import resolve_reasoning_model_id
from tests.conftest import FakeRuntimeClient
from tests.test_core_api import auth, make_container

DEEP_REASONING_PROMPTS = [
    "help me reason through the trade-offs of two architectures",
    "weigh the options and compare approaches for the cache layer",
    "what is the architectural decision here, list the pros and cons",
    "let's think deeply and do a deep dive on this design decision",
]

REPO_CODE_PROMPTS = [
    "why is the animation in this repo broken",
    "find the bug in this code",
    "inspect the repository status",
]


@pytest.mark.parametrize("prompt", DEEP_REASONING_PROMPTS)
def test_deep_reasoning_phrases_route_to_reasoning_agent(prompt: str) -> None:
    decision = FallbackRouter().route(prompt)
    assert decision.agent == "reasoning_agent"
    assert decision.intent == "deep_reasoning"
    assert decision.permission_level == 1
    assert decision.risk_level == "read_only"
    assert decision.needs_confirmation is False
    assert decision.tools_needed == []
    assert decision.model_id == "april-brain"


@pytest.mark.parametrize("prompt", REPO_CODE_PROMPTS)
def test_repo_and_code_requests_still_route_to_coding_agent(prompt: str) -> None:
    decision = FallbackRouter().route(prompt)
    assert decision.agent == "coding_agent"


def test_deep_reasoning_chat_is_available_on_brain_model(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "help me reason through the trade-offs of two architectures"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "ok"
    assert result["final_message"]
    assert result["status"] != "unavailable"

    rows = anyio.run(
        container.database.fetchall,
        "SELECT model_id FROM agent_runs WHERE agent = ? AND summary = ?",
        ("reasoning_agent", "structured agent loop"),
    )
    assert len(rows) == 1
    assert rows[0]["model_id"] == "april-brain"


def _reasoning_models_payload(tmp_path: Path) -> dict[str, Any]:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  brain:\n"
        "    id: april-brain\n"
        "    name: brain\n"
        "    path: models/brain.gguf\n"
        "    backend: fake\n"
        "    role: brain\n"
        "    threads: 4\n"
        "    context_size: 4096\n"
        "    temperature: 0.3\n"
        "    max_output_tokens: 512\n"
        "  reasoning:\n"
        "    id: april-deep\n"
        "    name: deep\n"
        "    path: models/deep.gguf\n"
        "    backend: fake\n"
        "    role: reasoning\n"
        "    threads: 4\n"
        "    context_size: 8192\n"
        "    temperature: 0.4\n"
        "    max_output_tokens: 1024\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_file(config, root=tmp_path)
    lifecycle = ModelLifecycle(registry, root_backend="fake")
    return {"models": [info.model_dump() for info in lifecycle.list_models()]}


class ReasoningUpgradeRuntimeClient(FakeRuntimeClient):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self._payload = payload

    async def models(self) -> dict[str, Any]:
        return self._payload


class FailingModelsRuntimeClient(FakeRuntimeClient):
    async def models(self) -> dict[str, Any]:
        raise RuntimeError("runtime model listing is offline")


def test_resolver_upgrades_to_registered_reasoning_model(tmp_path: Path) -> None:
    payload = _reasoning_models_payload(tmp_path)
    client = ReasoningUpgradeRuntimeClient(payload)
    resolved = anyio.run(
        lambda: resolve_reasoning_model_id(
            runtime_client=client,  # type: ignore[arg-type]
            fallback_model_id="april-brain",
        )
    )
    assert resolved == "april-deep"


def test_agent_run_uses_upgraded_reasoning_model(settings_tmp, tmp_path: Path) -> None:
    payload = _reasoning_models_payload(tmp_path)
    runtime_client = ReasoningUpgradeRuntimeClient(payload)
    container = anyio.run(make_container, settings_tmp, runtime_client)
    client = TestClient(create_app(container))
    response = client.post(
        "/agents/run",
        json={"agent": "reasoning_agent", "message": "reason through the trade-offs"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"
    rows = anyio.run(
        container.database.fetchall,
        "SELECT model_id FROM agent_runs WHERE agent = ? AND summary = ?",
        ("reasoning_agent", "structured agent loop"),
    )
    assert len(rows) == 1
    assert rows[0]["model_id"] == "april-deep"


def test_resolver_fails_safe_to_brain_when_listing_errors() -> None:
    client = FailingModelsRuntimeClient()
    resolved = anyio.run(
        lambda: resolve_reasoning_model_id(
            runtime_client=client,  # type: ignore[arg-type]
            fallback_model_id="april-brain",
        )
    )
    assert resolved == "april-brain"


def test_agent_run_completes_when_model_listing_errors(settings_tmp) -> None:
    runtime_client = FailingModelsRuntimeClient()
    container = anyio.run(make_container, settings_tmp, runtime_client)
    client = TestClient(create_app(container))
    response = client.post(
        "/agents/run",
        json={"agent": "reasoning_agent", "message": "reason through the trade-offs"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "ok"
    rows = anyio.run(
        container.database.fetchall,
        "SELECT model_id FROM agent_runs WHERE agent = ? AND summary = ?",
        ("reasoning_agent", "structured agent loop"),
    )
    assert len(rows) == 1
    assert rows[0]["model_id"] == "april-brain"
