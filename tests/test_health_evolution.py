"""Dreamer/evolution visibility in authenticated status and readiness."""

from __future__ import annotations

import json

import anyio
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.evolution.evaluator import write_pending_eval_case
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.schemas import VectorMetadata
from tests.test_core_api import auth, make_container, run_one_job


def test_readiness_exposes_redacted_evolution_block(settings_tmp) -> None:
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

    payload = client.get("/readiness", headers=auth(settings_tmp)).json()
    evolution = payload["evolution"]
    assert evolution["enabled"] is False
    assert evolution["kill_switch_active"] is False
    assert evolution["scheduler_enabled"] is False
    assert evolution["pending_eval_case_count"] == 1
    assert evolution["pending_write_capable_overlay_candidate_count"] == 1
    blob = json.dumps(payload)
    assert str(settings_tmp.home) not in blob
    assert str(settings_tmp.evolution_path) not in blob
    assert "run-1.json" not in blob  # only a boolean, never the report name


def test_readiness_evolution_kill_switch_state(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)
    client.post("/evolution/off", headers=headers)

    evolution = client.get("/readiness", headers=headers).json()["evolution"]
    assert evolution["kill_switch_active"] is True


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
    response = client.post("/memory/reindex", json={}, headers=auth(settings_tmp))
    assert response.status_code == 202
    job_id = response.json()["id"]
    anyio.run(run_one_job, container)
    assert container.job_store is not None
    job = anyio.run(container.job_store.require, job_id)
    assert job.status.value == "succeeded"
    assert job.result is not None
    assert job.result["provider"] == "hashed-token"
    assert job.result["dimensions"] == 256
    assert job.result["validation_result"]["ok"] is True
    assert job.result["final_generation"]


def test_memory_repair_index_is_dry_run_unless_apply_is_requested(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    metadata = VectorMetadata(
        source_type="test",
        source_id="test",
        content_hash="h",
        created_at="2026-01-01T00:00:00Z",
    )
    container.vector_memory.upsert(record_id="1", content="first", metadata=metadata)
    recovery = (settings_tmp.vector_index_path / "CURRENT").read_text(encoding="utf-8").strip()
    container.vector_memory.upsert(record_id="2", content="second", metadata=metadata)
    broken = (settings_tmp.vector_index_path / "CURRENT").read_text(encoding="utf-8").strip()
    records = settings_tmp.vector_index_path / "generations" / broken / "records.json"
    records.write_text(
        records.read_text(encoding="utf-8").replace("second", "corrupt"),
        encoding="utf-8",
    )
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    dry_run = client.post("/memory/repair-index", headers=headers).json()
    assert dry_run["applied"] is False
    assert dry_run["recovery_candidate"] == recovery
    assert (settings_tmp.vector_index_path / "CURRENT").read_text(
        encoding="utf-8"
    ).strip() == broken

    applied = client.post("/memory/repair-index?apply=true", headers=headers).json()
    assert applied["applied"] is True
    assert (settings_tmp.vector_index_path / "CURRENT").read_text(
        encoding="utf-8"
    ).strip() == recovery
