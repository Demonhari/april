"""Dreamer/evolution visibility in /health, /evolution/status, and readiness."""

from __future__ import annotations

import json

import anyio
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.evolution.evaluator import write_pending_eval_case
from services.evolution.write_guard import EvolutionWriteGuard
from tests.test_core_api import auth, make_container


def test_health_exposes_redacted_evolution_block(settings_tmp) -> None:
    write_pending_eval_case(settings_tmp, {"case_type": "negative_feedback", "prompt": "x"})
    guard = EvolutionWriteGuard(settings_tmp)
    guard.write_text(
        settings_tmp.evolution_path / "candidates" / "coding_agent-0.overlay.txt",
        "guidance",
    )
    guard.write_text(
        settings_tmp.evolution_path / "reports" / "run-1.json",
        json.dumps({"run_id": "run-1", "created_at": "2026-07-01T00:00:00Z"}),
    )
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))

    # /health is unauthenticated: the block must be booleans/counts/enums only.
    payload = client.get("/health").json()
    evolution = payload["evolution"]
    assert evolution["enabled"] is False
    assert evolution["kill_switch_active"] is False
    assert evolution["scheduler_enabled"] is False
    assert evolution["dreamer_last_run_date"] is None
    assert evolution["dreamer_last_report_available"] is True
    assert evolution["pending_eval_case_count"] == 1
    assert evolution["pending_write_capable_overlay_count"] == 1
    assert evolution["last_skip_reason"] == "evolution disabled"
    blob = json.dumps(payload)
    assert str(settings_tmp.home) not in blob
    assert str(settings_tmp.evolution_path) not in blob
    assert "run-1.json" not in blob  # only a boolean, never the report name


def test_health_evolution_kill_switch_reason(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)
    client.post("/evolution/off", headers=headers)

    evolution = client.get("/health").json()["evolution"]
    assert evolution["kill_switch_active"] is True
    assert evolution["last_skip_reason"] == "disabled by local kill switch"


def test_evolution_status_reports_gate_and_counts(settings_tmp) -> None:
    write_pending_eval_case(settings_tmp, {"case_type": "negative_feedback", "prompt": "x"})
    EvolutionWriteGuard(settings_tmp).write_text(
        settings_tmp.evolution_path / "reports" / "abc.json",
        json.dumps({"run_id": "abc", "created_at": "2026-07-01T00:00:00Z", "summary": "ok"}),
    )
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    status = client.get("/evolution/status", headers=auth(settings_tmp)).json()["status"]

    assert status["enabled"] is False
    assert status["scheduler_enabled"] is False
    assert status["scheduler_running"] is False
    assert status["kill_switch_active"] is False
    # Basename only — never a path.
    assert status["last_report_basename"] == "abc.json"
    assert status["pending_eval_case_count"] == 1
    assert status["pending_write_capable_overlay_count"] == 0
    assert status["current_gate_reason"] == "evolution disabled"
    blob = json.dumps(status)
    assert str(settings_tmp.evolution_path) not in blob


def test_chat_response_carries_intelligence_metadata(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "April, plan my work today."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    metadata = result["metadata"]
    assert metadata["chat_mode"] == "standard"
    assert isinstance(metadata["intelligence_rung"], int)
    assert "intelligence_reason" in metadata


def test_memory_reindex_reports_provider_and_degradation(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = client.post("/memory/reindex", json={}, headers=auth(settings_tmp)).json()
    assert payload["provider"] == "hashed-token"
    assert payload["configured_provider"] == "hashed-token"
    assert payload["dimensions"] == 256
    assert payload["index_compatible"] is True
    assert payload["fallback_active"] is False
    assert payload["degraded"] is False
