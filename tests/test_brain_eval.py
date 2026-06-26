from __future__ import annotations

from typing import Any

from apps.runner.evals import (
    BrainEvalCase,
    _evaluate_case,
    load_brain_eval_cases,
    real_routing_report,
    run_fake_brain_eval,
)
from april_common.settings import project_root


def _matching_decision(case: BrainEvalCase, *, routing_method: str) -> dict[str, Any]:
    """A schema-shaped decision dict whose fields match ``case`` exactly."""
    return {
        "intent": case.expected_intent,
        "agent": case.expected_agent,
        "model_id": case.expected_model_id,
        "tools_needed": list(case.expected_tools or []),
        "permission_level": case.expected_permission_level,
        "risk_level": case.expected_risk_level,
        "needs_confirmation": case.expected_needs_confirmation,
        "routing_method": routing_method,
    }


def test_fake_eval_passes_fallback_cases() -> None:
    # The deterministic fallback router answers every fixture case; in fake mode a
    # fallback route is acceptable, so every case (including the fallback one) passes.
    results = run_fake_brain_eval(project_root())
    assert results
    assert all(result.ok for result in results)
    normal = next(result for result in results if result.id == "normal_chat")
    assert normal.actual["routing_method"] == "fallback"
    assert normal.ok is True


def test_real_mode_evaluator_fails_schema_valid_fallback() -> None:
    case = BrainEvalCase(
        id="c1",
        message="hello",
        expected_intent="normal_conversation",
        expected_agent="general_agent",
    )
    decision = _matching_decision(case, routing_method="fallback")
    # All routing fields match, the decision is schema-valid, but fallback means the
    # model JSON was unusable — a failure in real-model mode.
    result = _evaluate_case(case, decision, schema_valid=True, allow_fallback=False)
    assert result.ok is False
    assert result.routing_ok is False
    assert "fallback" in result.detail
    # The same decision is accepted in fake/fallback mode.
    fake = _evaluate_case(case, decision, schema_valid=True, allow_fallback=True)
    assert fake.ok is True


def test_real_mode_evaluator_passes_model_and_model_repair() -> None:
    case = BrainEvalCase(
        id="c1",
        message="hello",
        expected_intent="normal_conversation",
        expected_agent="general_agent",
    )
    for method in ("model", "model_repair"):
        decision = _matching_decision(case, routing_method=method)
        result = _evaluate_case(case, decision, schema_valid=True, allow_fallback=False)
        assert result.ok is True, method
        assert result.routing_ok is True


def test_real_mode_evaluator_fails_mismatched_routing_fields() -> None:
    case = BrainEvalCase(
        id="c1",
        message="hello",
        expected_intent="normal_conversation",
        expected_agent="general_agent",
    )
    decision = _matching_decision(case, routing_method="model")
    decision["agent"] = "coding_agent"  # schema-valid model route, wrong agent
    result = _evaluate_case(case, decision, schema_valid=True, allow_fallback=False)
    assert result.ok is False


def test_all_configured_routing_report_uses_real_mode() -> None:
    # ``real_routing_report`` is what the all-configured-models verifier delegates
    # to; a schema-valid fallback decision must count as a failure there.
    pass_case = BrainEvalCase(
        id="pass",
        message="hello",
        expected_intent="normal_conversation",
        expected_agent="general_agent",
    )
    fallback_case = BrainEvalCase(
        id="fail",
        message="plan",
        expected_intent="planning",
        expected_agent="general_agent",
    )
    decisions = [
        _matching_decision(pass_case, routing_method="model"),
        _matching_decision(fallback_case, routing_method="fallback"),
    ]
    report = real_routing_report([pass_case, fallback_case], decisions)
    assert report.total == 2
    assert report.passed == 1
    assert report.fallback_count == 1
    assert report.accuracy == 0.5


def test_real_routing_report_counts_model_repair() -> None:
    case = BrainEvalCase(
        id="c1",
        message="hello",
        expected_intent="normal_conversation",
        expected_agent="general_agent",
    )
    decisions = [_matching_decision(case, routing_method="model_repair")]
    report = real_routing_report([case], decisions)
    assert report.passed == 1
    assert report.model_repair_count == 1
    assert report.fallback_count == 0


def test_fixture_loads_without_real_gguf() -> None:
    # No real model is required to load or evaluate the fixture cases.
    cases = load_brain_eval_cases(project_root())
    assert any(case.id == "normal_chat" for case in cases)
    assert any(case.expected_routing_method == "fallback" for case in cases)
