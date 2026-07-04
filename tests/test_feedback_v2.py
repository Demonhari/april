from __future__ import annotations

import anyio
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.brain.feedback_classifier import classify_implicit_correction
from tests.test_core_api import auth, make_container


def test_classifier_matches_only_clear_corrections() -> None:
    assert classify_implicit_correction("That's wrong, the file is elsewhere.")
    assert classify_implicit_correction("no, that's wrong")
    assert classify_implicit_correction("Not what I asked for.")
    assert classify_implicit_correction("You misunderstood my question")
    assert classify_implicit_correction("  WRONG ANSWER  ")

    # Anything else — including phrases embedded mid-sentence — never counts.
    assert classify_implicit_correction("Is this wrong answer handling ok?") is None
    assert classify_implicit_correction("Tell me what's wrong with this code") is None
    assert classify_implicit_correction("no problem, continue") is None
    assert classify_implicit_correction("please fix the animation") is None
    assert classify_implicit_correction("") is None


def test_feedback_endpoint_good_and_bad(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    good = client.post("/feedback", json={"rating": "good"}, headers=headers)
    assert good.status_code == 200
    assert good.json()["feedback"]["rating"] == "good"

    bad = client.post(
        "/feedback", json={"rating": "bad", "reason": "too vague"}, headers=headers
    )
    assert bad.status_code == 200
    body = bad.json()["feedback"]
    assert body["rating"] == "bad"
    assert body["reason"] == "too vague"

    invalid = client.post("/feedback", json={"rating": "meh"}, headers=headers)
    assert invalid.status_code == 422


def test_denied_approval_records_negative_feedback(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)
    project = client.post(
        "/projects", json={"path": str(settings_tmp.home)}, headers=headers
    ).json()
    response = client.post(
        "/chat",
        json={"message": "Apply the fix.", "project_id": project["id"]},
        headers=headers,
    )
    approval_id = response.json()["result"]["pending_approval"]["approval_id"]
    denied = client.post("/tools/deny", json={"approval_id": approval_id}, headers=headers)
    assert denied.status_code == 200

    rows = anyio.run(
        container.database.fetchall,
        "SELECT rating, reason, agent_run_id, conversation_id FROM feedback_events",
    )
    assert len(rows) == 1
    assert rows[0]["rating"] == "bad"
    assert str(rows[0]["reason"]).startswith("approval_denied:")
    assert rows[0]["agent_run_id"] is not None
    assert rows[0]["conversation_id"] is not None


def test_implicit_correction_recorded_against_previous_run(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    first = client.post("/chat", json={"message": "plan my day"}, headers=headers)
    conversation_id = first.json()["result"]["conversation_id"]

    followup = client.post(
        "/chat",
        json={
            "message": "That's wrong, I meant tomorrow.",
            "conversation_id": conversation_id,
        },
        headers=headers,
    )
    assert followup.status_code == 200

    rows = anyio.run(
        container.database.fetchall,
        "SELECT rating, reason, conversation_id, agent_run_id FROM feedback_events",
    )
    assert len(rows) == 1
    assert rows[0]["rating"] == "bad"
    assert str(rows[0]["reason"]).startswith("implicit_correction:")
    assert rows[0]["conversation_id"] == conversation_id
    assert rows[0]["agent_run_id"] is not None


def test_normal_followups_record_no_implicit_feedback(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    first = client.post("/chat", json={"message": "plan my day"}, headers=headers)
    conversation_id = first.json()["result"]["conversation_id"]
    client.post(
        "/chat",
        json={"message": "thanks, now plan tomorrow", "conversation_id": conversation_id},
        headers=headers,
    )
    # A brand-new conversation opener can never be an implicit correction,
    # even when it starts with a correction phrase.
    client.post("/chat", json={"message": "That's wrong somehow"}, headers=headers)

    rows = anyio.run(container.database.fetchall, "SELECT id FROM feedback_events")
    assert rows == []
