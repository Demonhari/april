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

    bad = client.post("/feedback", json={"rating": "bad", "reason": "too vague"}, headers=headers)
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


def _pending_eval_paths(settings_tmp):
    pending = settings_tmp.evolution_path / "evals" / "pending"
    return sorted(pending.glob("*.yaml")) if pending.is_dir() else []


def test_explicit_bad_feedback_stages_pending_eval_case(settings_tmp) -> None:
    import yaml

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    first = client.post("/chat", json={"message": "plan my day"}, headers=headers)
    conversation_id = first.json()["result"]["conversation_id"]
    bad = client.post(
        "/feedback",
        json={"rating": "bad", "reason": "april bad", "conversation_id": conversation_id},
        headers=headers,
    )
    assert bad.status_code == 200

    staged = _pending_eval_paths(settings_tmp)
    assert len(staged) == 1
    case = yaml.safe_load(staged[0].read_text(encoding="utf-8"))
    assert case["case_type"] == "negative_feedback"
    assert case["signal"] == "explicit_feedback"
    assert case["status"] == "pending_review"
    assert case["prompt"] == "plan my day"
    assert case["bad_response_excerpt"]
    # The machine never invents the expected behaviour; review fills it in.
    assert case["expected_behavior"] is None


def test_good_feedback_stages_no_eval_case(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)
    first = client.post("/chat", json={"message": "plan my day"}, headers=headers)
    conversation_id = first.json()["result"]["conversation_id"]
    client.post(
        "/feedback",
        json={"rating": "good", "conversation_id": conversation_id},
        headers=headers,
    )
    assert _pending_eval_paths(settings_tmp) == []


def test_bad_feedback_without_context_stages_nothing(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)
    # No conversation and no open session: not enough context for an eval case.
    response = client.post("/feedback", json={"rating": "bad"}, headers=headers)
    assert response.status_code == 200
    assert _pending_eval_paths(settings_tmp) == []


def test_denied_approval_stages_pending_eval_case(settings_tmp) -> None:
    import yaml

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

    staged = _pending_eval_paths(settings_tmp)
    assert len(staged) == 1
    case = yaml.safe_load(staged[0].read_text(encoding="utf-8"))
    assert case["signal"] == "approval_denied"
    assert case["prompt"] == "Apply the fix."
    assert str(case["reason"]).startswith("approval_denied:")


def test_implicit_correction_stages_pending_eval_case(settings_tmp) -> None:
    import yaml

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    first = client.post("/chat", json={"message": "plan my day"}, headers=headers)
    conversation_id = first.json()["result"]["conversation_id"]
    client.post(
        "/chat",
        json={
            "message": "That's wrong, I meant tomorrow.",
            "conversation_id": conversation_id,
        },
        headers=headers,
    )
    staged = _pending_eval_paths(settings_tmp)
    assert len(staged) == 1
    case = yaml.safe_load(staged[0].read_text(encoding="utf-8"))
    assert case["signal"] == "implicit_correction"
    assert case["prompt"] == "plan my day"


def test_sensitive_feedback_context_is_excluded_from_eval_cases(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    first = client.post(
        "/chat",
        json={"message": "my api key is sk-test-abcdefghijklmnop, plan my day"},
        headers=headers,
    )
    conversation_id = first.json()["result"]["conversation_id"]
    bad = client.post(
        "/feedback",
        json={"rating": "bad", "reason": "unhelpful", "conversation_id": conversation_id},
        headers=headers,
    )
    assert bad.status_code == 200
    # The feedback event is still recorded, but no eval case is staged.
    rows = anyio.run(container.database.fetchall, "SELECT id FROM feedback_events")
    assert len(rows) == 1
    assert _pending_eval_paths(settings_tmp) == []
    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "feedback_eval_case_skipped" in audit_text
    assert "sk-test-abcdefghijklmnop" not in audit_text


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
