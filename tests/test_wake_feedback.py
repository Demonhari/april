from __future__ import annotations

import anyio
from fastapi.testclient import TestClient

from agents.schemas import AgentResult
from services.api.server import create_app
from tests.test_core_api import auth, make_container


def test_wake_feedback_binds_to_latest_run_without_brain_routing(settings_tmp, monkeypatch) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    first = client.post("/wake", json={"source": "voice"}, headers=auth(settings_tmp)).json()
    conversation_id = first["conversation_id"]

    async def seed_run() -> str:
        return await container.memory.record_agent_run(
            conversation_id=conversation_id,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="recent answer",
        )

    run_id = anyio.run(seed_run)

    async def fail_chat(*args, **kwargs):
        raise AssertionError("Brain routing must not run for exact wake feedback")

    monkeypatch.setattr(container.orchestrator, "chat", fail_chat)
    response = client.post(
        "/wake",
        json={"source": "voice", "text": "that was wrong", "reason": "follow_up"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["final_message"] == "Feedback recorded."
    rows = anyio.run(
        container.database.fetchall,
        "SELECT rating, reason, agent_run_id, conversation_id FROM feedback_events",
    )
    assert [(row["rating"], row["agent_run_id"], row["conversation_id"]) for row in rows] == [
        ("bad", run_id, conversation_id)
    ]
    assert rows[0]["reason"] == "wake_feedback: that was wrong"
    assert "wake_feedback" in settings_tmp.audit_path.read_text(encoding="utf-8")


def test_wake_feedback_accepts_vocative_stripped_good_phrase(settings_tmp, monkeypatch) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    first = client.post("/wake", json={"source": "voice"}, headers=auth(settings_tmp)).json()
    conversation_id = first["conversation_id"]

    async def seed_run() -> None:
        await container.memory.record_agent_run(
            conversation_id=conversation_id,
            agent="general_agent",
            status="ok",
            model_id="april-brain",
            summary="recent answer",
        )

    anyio.run(seed_run)

    async def fail_chat(*args, **kwargs):
        raise AssertionError("Brain routing must not run for exact wake feedback")

    monkeypatch.setattr(container.orchestrator, "chat", fail_chat)
    response = client.post(
        "/wake",
        json={"source": "voice", "text": "April, good job"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    rows = anyio.run(
        container.database.fetchall,
        "SELECT rating, reason FROM feedback_events",
    )
    assert [(row["rating"], row["reason"]) for row in rows] == [("good", "wake_feedback: good job")]


def test_wake_feedback_near_miss_routes_normally(settings_tmp, monkeypatch) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    calls: list[str] = []

    async def fake_chat(message: str, **kwargs):
        calls.append(message)
        return AgentResult(status="ok", final_message="normal route")

    monkeypatch.setattr(container.orchestrator, "chat", fake_chat)
    response = client.post(
        "/wake",
        json={"source": "voice", "text": "that was almost right"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert response.json()["result"]["final_message"] == "normal route"
    assert calls == ["that was almost right"]
    rows = anyio.run(container.database.fetchall, "SELECT id FROM feedback_events")
    assert rows == []


def test_wake_feedback_no_prior_run_is_polite_noop(settings_tmp, monkeypatch) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))

    async def fail_chat(*args, **kwargs):
        raise AssertionError("Brain routing must not run for exact wake feedback")

    monkeypatch.setattr(container.orchestrator, "chat", fail_chat)
    response = client.post(
        "/wake",
        json={"source": "voice", "text": "that was perfect", "reason": "follow_up"},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["feedback_recorded"] is False
    assert "recent answer" in result["final_message"]
    rows = anyio.run(container.database.fetchall, "SELECT id FROM feedback_events")
    assert rows == []
    assert "wake_feedback_noop" in settings_tmp.audit_path.read_text(encoding="utf-8")
