from __future__ import annotations

import pytest

from services.brain.fallback_router import FallbackRouter
from services.brain.parser import parse_brain_decision, parse_with_repair

VALID = """
{"intent":"planning","agent":"general_agent","model_id":"april-brain","tools_needed":[],
"memory_queries":[],"permission_level":0,"risk_level":"none","needs_confirmation":false,
"task_steps":["Answer directly"],"decision_summary":"General response"}
"""


def test_valid_strict_json() -> None:
    decision = parse_brain_decision(VALID)
    assert decision.intent == "planning"
    assert decision.routing_method == "model"


@pytest.mark.asyncio
async def test_malformed_json_repair() -> None:
    async def repair(_: str) -> str:
        return VALID

    decision = await parse_with_repair("not json", repair)
    assert decision.routing_method == "model_repair"


def test_fallback_routing() -> None:
    decision = FallbackRouter().route("April, check why the animation in this repository is broken")
    assert decision.agent == "coding_agent"
    assert decision.permission_level == 1
