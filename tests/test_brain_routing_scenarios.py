"""Brain routing returns a valid BrainDecision across the required scenarios.

These exercise the deterministic fallback route (the safety net used whenever the
model is unavailable or its JSON cannot be repaired), proving each scenario yields
a schema-valid BrainDecision with the correct agent, risk, permission level, and
approval requirement — and that the schemas are not loosened to let bad routes
through. Real-model JSON acceptance is verified separately by the GGUF verifier;
here we keep the route hermetic and exhaustive without a model.
"""

from __future__ import annotations

import pytest

from april_common.errors import RuntimeUnavailableError
from april_common.errors import ValidationError as AprilValidationError
from services.brain.parser import parse_brain_decision
from services.brain.router import BrainRouter
from services.brain.schemas import BrainDecision


class _OfflineRuntimeClient:
    """Forces the router onto its deterministic, safe fallback path."""

    async def chat(self, **kwargs: object) -> object:
        raise RuntimeUnavailableError("April Runtime is offline.", {})


async def _route(message: str) -> BrainDecision:
    decision = await BrainRouter(_OfflineRuntimeClient()).route(message)  # type: ignore[arg-type]
    # Every route must round-trip through the strict schema (never loosened).
    assert BrainDecision.model_validate(decision.model_dump()) == decision
    assert decision.routing_method == "fallback"
    return decision


@pytest.mark.asyncio
async def test_normal_planning_route_is_safe() -> None:
    decision = await _route("April, plan my work today.")
    assert decision.agent == "general_agent"
    assert decision.permission_level == 0
    assert decision.risk_level == "none"
    assert decision.needs_confirmation is False


@pytest.mark.asyncio
async def test_repo_read_only_analysis_route() -> None:
    decision = await _route("April, check why the animation in this repository is broken.")
    assert decision.agent == "coding_agent"
    assert decision.risk_level == "read_only"
    assert decision.permission_level == 1
    assert decision.needs_confirmation is False


@pytest.mark.asyncio
async def test_code_modification_route_requires_approval() -> None:
    decision = await _route("Apply the fix.")
    assert decision.agent == "coding_agent"
    assert decision.risk_level == "code_write"
    assert decision.permission_level == 3
    # Level 3+ code modification must require approval.
    assert decision.needs_confirmation is True


@pytest.mark.asyncio
async def test_reminder_creation_route() -> None:
    decision = await _route("remind me to stand up")
    assert decision.agent == "general_agent"
    assert decision.risk_level == "safe_write"
    assert decision.permission_level == 2
    assert decision.needs_confirmation is False
    assert [call.tool for call in decision.planned_tool_calls] == ["create_reminder"]


@pytest.mark.asyncio
async def test_document_reading_route() -> None:
    decision = await _route("Summarize this PDF document for me.")
    assert decision.agent == "reading_agent"
    assert decision.risk_level == "read_only"
    assert decision.permission_level == 1
    assert decision.needs_confirmation is False


@pytest.mark.asyncio
async def test_external_action_route_is_blocked_and_gated() -> None:
    decision = await _route("Deploy the app and send email to the team.")
    assert decision.risk_level == "external_action"
    assert decision.permission_level == 5
    # External actions are disabled, but the route itself must still demand approval.
    assert decision.needs_confirmation is True


# --- schema is not loosened ------------------------------------------------
def test_parser_rejects_unknown_agent() -> None:
    with pytest.raises(AprilValidationError):
        parse_brain_decision(
            '{"intent":"planning","agent":"hacker_agent","model_id":"april-brain",'
            '"tools_needed":[],"memory_queries":[],"permission_level":0,"risk_level":"none",'
            '"needs_confirmation":false,"task_steps":[],"decision_summary":"x"}'
        )


def test_parser_rejects_out_of_range_permission_level() -> None:
    with pytest.raises(AprilValidationError):
        parse_brain_decision(
            '{"intent":"planning","agent":"general_agent","model_id":"april-brain",'
            '"tools_needed":[],"memory_queries":[],"permission_level":7,"risk_level":"none",'
            '"needs_confirmation":false,"task_steps":[],"decision_summary":"x"}'
        )


def test_parser_rejects_unknown_risk_level() -> None:
    with pytest.raises(AprilValidationError):
        parse_brain_decision(
            '{"intent":"planning","agent":"general_agent","model_id":"april-brain",'
            '"tools_needed":[],"memory_queries":[],"permission_level":0,"risk_level":"nuke",'
            '"needs_confirmation":false,"task_steps":[],"decision_summary":"x"}'
        )
