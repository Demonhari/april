"""Agent/tool registry alignment (configs/agents.yaml is the source of truth).

The effective permission registry is derived from ``configs/agents.yaml``: a tool
is allowed for exactly the agents whose ``allowed_tools`` list it. These tests pin
the alignment fixed in that file — general_agent may create/list reminders, and
reading_agent may run document_indexer — while proving the conservative denials
still hold for agents that must not reach those tools.
"""

from __future__ import annotations

import anyio
import pytest
from fastapi.testclient import TestClient

from agents.registry import default_agent_registry
from april_common.effective_config import (
    build_agent_registry_from_config,
    build_configured_tool_registry,
    load_agents_file,
)
from april_common.errors import PermissionDeniedError
from april_common.settings import project_root
from services.api.server import create_app
from services.april_runtime.model_registry import ModelRegistry
from services.permissions.engine import PermissionEngine
from skills.registry import default_registry
from tests.test_core_api import auth, make_container


def _configured_engine() -> PermissionEngine:
    """A PermissionEngine whose registry mirrors the real configs/agents.yaml."""
    home = project_root()
    model_registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    agent_registry = build_agent_registry_from_config(
        home=home,
        model_registry=model_registry,
        tool_registry=default_registry(),
    )
    return PermissionEngine(build_configured_tool_registry(home, agent_registry))


def test_default_bundled_agent_registry_matches_active_yaml_defaults() -> None:
    """The bundled fallback factories must not drift from configs/agents.yaml."""
    home = project_root()
    yaml_agents = load_agents_file(home).agents
    bundled = {agent.name: agent.config for agent in default_agent_registry().list()}

    assert set(bundled) == set(yaml_agents)
    for name, configured in yaml_agents.items():
        agent = bundled[name]
        assert agent.description == configured.description
        assert agent.model_id == configured.model_id
        assert agent.allowed_tools == set(configured.allowed_tools)
        assert agent.blocked_tools == set(configured.blocked_tools)
        assert agent.memory_access_policy == configured.memory_access
        assert agent.maximum_tool_iterations == configured.maximum_tool_iterations
        assert agent.output_schema == configured.output_schema


# --- Task 1.1: reminder chat creates a local reminder without Level 3 approval ---
def test_reminder_chat_creates_local_reminder_without_approval(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "remind me to stand up"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    # Reminders are a Level 2 local safe_write: executed, never gated on approval.
    assert result["status"] == "ok"
    assert result.get("pending_approval") in (None, {})
    assert client.get("/approvals", headers=auth(settings_tmp)).json()["approvals"] == []
    reminders = client.get("/reminders", headers=auth(settings_tmp)).json()["reminders"]
    assert any(reminder["content"] == "stand up" for reminder in reminders)


# --- Task 1.2: /documents indexes a temp local text folder as reading_agent ---
def test_documents_endpoint_indexes_local_folder_as_reading_agent(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    folder = settings_tmp.home / "corpus"
    folder.mkdir()
    (folder / "guide.md").write_text("# local guide\nanimation pipeline notes\n", encoding="utf-8")
    (folder / "notes.txt").write_text("plain reading notes\n", encoding="utf-8")

    response = client.post(
        "/documents",
        json={"path": str(folder)},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["ok"] is True
    # document_indexer is Level 2 safe_write — no approval, real chunks indexed.
    assert result["permission_level"] == 2
    assert result["risk_level"] == "safe_write"
    assert result["data"]["chunks"] > 0
    indexed = client.get("/documents", headers=auth(settings_tmp)).json()["documents"]
    indexed_paths = {path for source in indexed for path in source["paths"]}
    assert any(path.endswith("guide.md") for path in indexed_paths)


# --- Task 1.3 & positive: document_indexer allowed for reading, denied elsewhere ---
def test_permission_engine_allows_document_indexer_only_for_reading_agent() -> None:
    engine = _configured_engine()
    allowed = engine.evaluate(
        tool="document_indexer",
        args={"folder_path": "corpus"},
        agent="reading_agent",
    )
    assert allowed.permission_level == 2
    assert allowed.confirmation_required is False

    for agent in ("general_agent", "coding_agent", "system_action_agent"):
        with pytest.raises(PermissionDeniedError):
            engine.evaluate(
                tool="document_indexer",
                args={"folder_path": "corpus"},
                agent=agent,
            )


# --- Task 1.4 & positive: create_reminder allowed for general, denied elsewhere ---
def test_permission_engine_allows_create_reminder_only_for_general_agent() -> None:
    engine = _configured_engine()
    allowed = engine.evaluate(
        tool="create_reminder",
        args={"content": "stand up"},
        agent="general_agent",
    )
    assert allowed.permission_level == 2
    assert allowed.confirmation_required is False

    for agent in ("coding_agent", "reading_agent", "system_action_agent"):
        with pytest.raises(PermissionDeniedError):
            engine.evaluate(
                tool="create_reminder",
                args={"content": "stand up"},
                agent=agent,
            )


def test_list_reminders_allowed_only_for_general_agent() -> None:
    engine = _configured_engine()
    allowed = engine.evaluate(tool="list_reminders", args={}, agent="general_agent")
    assert allowed.permission_level <= 2
    for agent in ("coding_agent", "reading_agent"):
        with pytest.raises(PermissionDeniedError):
            engine.evaluate(tool="list_reminders", args={}, agent=agent)


def test_write_tools_stay_denied_for_general_and_reading_agents() -> None:
    """The conservative blocks held: no write_file/run_command/etc. crept in."""
    engine = _configured_engine()
    for agent in ("general_agent", "reading_agent"):
        for tool in ("write_file", "run_command", "git_commit", "patch_applier", "open_url"):
            with pytest.raises(PermissionDeniedError):
                engine.evaluate(tool=tool, args={}, agent=agent)
