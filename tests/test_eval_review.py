"""Pending feedback-eval review lifecycle: list/show/promote/reject."""

from __future__ import annotations

import anyio
import yaml
from fastapi.testclient import TestClient

from services.api.server import create_app
from services.evolution.eval_review import (
    list_reviewed_eval_cases,
    pending_eval_dir,
    rejected_eval_dir,
    reviewed_eval_dir,
)
from services.evolution.evaluator import write_pending_eval_case
from tests.test_core_api import auth, make_container


def _stage_case(settings, *, prompt: str = "what is my timezone") -> str:
    path = write_pending_eval_case(
        settings,
        {
            "case_type": "negative_feedback",
            "signal": "explicit_feedback",
            "status": "pending_review",
            "prompt": prompt,
            "bad_response_excerpt": "wrong timezone",
            "reason": "answer ignored my timezone",
            "expected_behavior": None,
        },
    )
    return path.stem


def test_list_and_show_pending_cases(settings_tmp) -> None:
    case_id = _stage_case(settings_tmp)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    listing = client.get("/evolution/evals/pending", headers=headers)
    assert listing.status_code == 200
    items = listing.json()["pending"]
    assert len(items) == 1
    assert items[0]["case_id"] == case_id
    assert items[0]["case_type"] == "negative_feedback"
    assert items[0]["has_prompt"] is True

    show = client.get(f"/evolution/evals/pending/{case_id}", headers=headers)
    assert show.status_code == 200
    case = show.json()["case"]
    assert case["case_id"] == case_id
    assert case["prompt"] == "what is my timezone"

    missing = client.get(f"/evolution/evals/pending/{'0' * 12}", headers=headers)
    assert missing.status_code == 404


def test_promote_requires_expected_behavior_and_moves_case(settings_tmp) -> None:
    case_id = _stage_case(settings_tmp)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    # Whitespace-only expected behaviour is refused: a human must supply it.
    empty = client.post(
        "/evolution/evals/promote",
        json={"case_id": case_id, "expected_behavior": "   "},
        headers=headers,
    )
    assert empty.status_code == 400
    assert (pending_eval_dir(settings_tmp) / f"{case_id}.yaml").exists()

    promoted = client.post(
        "/evolution/evals/promote",
        json={"case_id": case_id, "expected_behavior": "Answer using the stored timezone."},
        headers=headers,
    )
    assert promoted.status_code == 200
    assert promoted.json()["promoted"]["status"] == "reviewed"

    # The case left pending and became an active reviewed case in the fence.
    assert not (pending_eval_dir(settings_tmp) / f"{case_id}.yaml").exists()
    reviewed_path = reviewed_eval_dir(settings_tmp) / f"{case_id}.yaml"
    assert reviewed_path.exists()
    assert reviewed_path.resolve().is_relative_to(settings_tmp.evolution_path.resolve())
    case = yaml.safe_load(reviewed_path.read_text(encoding="utf-8"))
    assert case["status"] == "reviewed"
    assert case["expected_behavior"] == "Answer using the stored timezone."

    # Promotion is audited and the case is now visible to the evaluator.
    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "feedback_eval_case_promoted" in audit_text
    visible = list_reviewed_eval_cases(settings_tmp)
    assert [item["case_id"] for item in visible] == [case_id]

    # Pending listing (and readiness pending counting) no longer sees it.
    assert client.get("/evolution/evals/pending", headers=headers).json()["pending"] == []


def test_reject_records_reason_and_hides_case_from_evaluator(settings_tmp) -> None:
    case_id = _stage_case(settings_tmp)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    rejected = client.post(
        "/evolution/evals/reject",
        json={"case_id": case_id, "reason": "one-off complaint, not reproducible"},
        headers=headers,
    )
    assert rejected.status_code == 200
    assert rejected.json()["rejected"]["status"] == "rejected"

    assert not (pending_eval_dir(settings_tmp) / f"{case_id}.yaml").exists()
    rejected_path = rejected_eval_dir(settings_tmp) / f"{case_id}.yaml"
    assert rejected_path.exists()
    case = yaml.safe_load(rejected_path.read_text(encoding="utf-8"))
    assert case["status"] == "rejected"
    assert case["rejection_reason"] == "one-off complaint, not reproducible"

    # Rejected cases are never fed to the evaluator.
    assert list_reviewed_eval_cases(settings_tmp) == []
    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "feedback_eval_case_rejected" in audit_text


def test_traversal_case_ids_are_rejected(settings_tmp) -> None:
    _stage_case(settings_tmp)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    for hostile in ("..", "../..", "a/../b", "a%2F..%2Fb", ".hidden", "a.yaml"):
        show = client.get(f"/evolution/evals/pending/{hostile}", headers=headers)
        assert show.status_code in {400, 404}, hostile
        promote = client.post(
            "/evolution/evals/promote",
            json={"case_id": hostile, "expected_behavior": "x"},
            headers=headers,
        )
        assert promote.status_code in {400, 404, 422}, hostile
        reject = client.post(
            "/evolution/evals/reject",
            json={"case_id": hostile, "reason": "x"},
            headers=headers,
        )
        assert reject.status_code in {400, 404, 422}, hostile
    # Nothing escaped the evolution fence.
    assert not (settings_tmp.home / "b.yaml").exists()


def test_eval_review_routes_require_authentication(settings_tmp) -> None:
    case_id = _stage_case(settings_tmp)
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))

    assert client.get("/evolution/evals/pending").status_code == 403
    assert client.get(f"/evolution/evals/pending/{case_id}").status_code == 403
    assert (
        client.post(
            "/evolution/evals/promote",
            json={"case_id": case_id, "expected_behavior": "x"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/evolution/evals/reject",
            json={"case_id": case_id, "reason": "x"},
        ).status_code
        == 403
    )
    # Nothing moved without authentication.
    assert (pending_eval_dir(settings_tmp) / f"{case_id}.yaml").exists()


def test_readiness_pending_count_excludes_reviewed_cases(settings_tmp, monkeypatch) -> None:
    import os

    from apps.runner.readiness import build_readiness_report

    promoted_id = _stage_case(settings_tmp, prompt="promoted case")
    kept_id = _stage_case(settings_tmp, prompt="still pending case")
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)
    client.post(
        "/evolution/evals/promote",
        json={"case_id": promoted_id, "expected_behavior": "Handle it properly."},
        headers=headers,
    )

    # Readiness reads env-independent configs; run it against the temp home.
    for key in list(os.environ):
        if key.startswith("APRIL_"):
            monkeypatch.delenv(key, raising=False)
    configs = settings_tmp.home / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "april.yaml").write_text(
        "environment: development\nruntime:\n  backend: fake\n", encoding="utf-8"
    )
    (configs / "models.yaml").write_text("models: {}\n", encoding="utf-8")
    report = build_readiness_report(settings_tmp.home)
    assert report.pending_eval_case_count == 1
    check = next(c for c in report.checks if c.name == "pending eval cases")
    assert "1 staged eval case(s)" in check.detail
    assert kept_id  # the remaining pending case is the unreviewed one
