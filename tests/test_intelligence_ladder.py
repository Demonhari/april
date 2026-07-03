from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from agents.registry import default_agent_registry
from services.april_runtime.schemas import ChatMessage, ChatResponse, Usage
from services.brain.intelligence_ladder import (
    CouncilCandidate,
    IntelligenceLadder,
    score_council_candidate,
    select_best_candidate,
)
from services.brain.schemas import BrainDecision


def _decision(**updates: object) -> BrainDecision:
    data = {
        "intent": "planning",
        "agent": "general_agent",
        "model_id": "april-brain",
        "confidence": 0.82,
        "tools_needed": [],
        "planned_tool_calls": [],
        "memory_queries": [],
        "permission_level": 0,
        "risk_level": "none",
        "needs_confirmation": False,
        "task_steps": ["Answer"],
        "decision_summary": "General response",
        "routing_method": "model",
    }
    data.update(updates)
    return BrainDecision.model_validate(data)


class LadderRuntime:
    def __init__(self, *, delay: float = 0.0, reasoning_model: bool = False) -> None:
        self.delay = delay
        self.reasoning_model = reasoning_model
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> ChatResponse:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        joined = "\n".join(message.content for message in kwargs["messages"])
        lower = joined.lower()
        if "council responder role: practical" in lower:
            content = (
                "Use the local typed client because it preserves approval risk, "
                "then verify with tests and document the tradeoff."
            )
        elif "council responder role:" in lower:
            content = "Short answer."
        else:
            content = "Deep local answer with assumptions and verification limits."
        return ChatResponse(
            request_id=kwargs.get("request_id") or "r",
            model_id=kwargs["model_id"],
            content=content,
            usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
        )

    async def models(self) -> dict[str, Any]:
        if not self.reasoning_model:
            return {"models": []}
        return {
            "models": [
                {
                    "id": "april-reasoning",
                    "role": "reasoning",
                    "state": "loaded",
                }
            ]
        }


def _ladder(settings_tmp, runtime: LadderRuntime) -> IntelligenceLadder:
    return IntelligenceLadder(
        settings=settings_tmp,
        runtime_client=runtime,  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        clock=lambda: datetime(2026, 7, 3, 9, 30, tzinfo=UTC),
    )


def test_ladder_selects_reflex_deep_verified_and_council(settings_tmp) -> None:
    ladder = _ladder(settings_tmp, LadderRuntime())
    assert ladder.select(
        message="April, what time is it?",
        decision=_decision(),
        mode="standard",
    ).rung == 0
    assert ladder.select(
        message="compare options",
        decision=_decision(),
        mode="deep",
    ).rung == 3
    assert ladder.select(
        message="double check your answer",
        decision=_decision(),
        mode="standard",
    ).rung == 2
    assert ladder.select(
        message="compare options",
        decision=_decision(),
        mode="council",
    ).rung == 4
    assert ladder.select(
        message="run pytest",
        decision=_decision(
            tools_needed=["run_command"],
            permission_level=3,
            risk_level="code_write",
            needs_confirmation=True,
        ),
        mode="deep",
    ).rung == 1


def test_reflex_answer_is_deterministic_local(settings_tmp) -> None:
    ladder = _ladder(settings_tmp, LadderRuntime())
    answer = ladder.reflex_answer("what time is it")
    assert answer.startswith("Mode: reflex")
    assert "09:30 UTC" in answer


@pytest.mark.asyncio
async def test_deep_mode_uses_reasoning_model_and_hard_budget(settings_tmp) -> None:
    runtime = LadderRuntime(reasoning_model=True)
    ladder = _ladder(settings_tmp, runtime)
    result = await ladder.run_deep(
        message="compare approaches",
        prompt_messages=[ChatMessage(role="user", content="User request: compare approaches")],
        fallback_model_id="april-brain",
        request_id="r1",
    )
    assert result.status == "ok"
    assert result.model_id == "april-reasoning"
    assert result.final_message.startswith("Mode: deep")

    fast_budget = settings_tmp.model_copy(
        update={
            "deep_mode": settings_tmp.deep_mode.model_copy(update={"max_seconds": 0.01})
        }
    )
    slow = _ladder(fast_budget, LadderRuntime(delay=0.05))
    stopped = await slow.run_deep(
        message="compare approaches",
        prompt_messages=[ChatMessage(role="user", content="User request: compare approaches")],
        fallback_model_id="april-brain",
        request_id="r2",
    )
    assert stopped.status == "unavailable"
    assert "configured local budget" in stopped.final_message


@pytest.mark.asyncio
async def test_council_mode_selects_best_rubric_candidate(settings_tmp) -> None:
    runtime = LadderRuntime()
    ladder = _ladder(settings_tmp, runtime)
    result = await ladder.run_council(
        message="compare local client approaches",
        prompt_messages=[
            ChatMessage(role="user", content="User request: compare local client approaches")
        ],
        fallback_model_id="april-brain",
        request_id="r3",
    )
    assert result.status == "ok"
    assert result.final_message.startswith("Mode: council")
    assert "local typed client" in result.final_message
    assert result.metadata["selected_responder"] == "practical"


def test_council_rubric_scores_and_selects_candidate() -> None:
    weak = score_council_candidate(CouncilCandidate("weak", "Short."), question="local tests")
    strong = score_council_candidate(
        CouncilCandidate(
            "strong",
            "Use local tests because they verify the risk and preserve the tradeoff.",
        ),
        question="local tests",
    )
    assert select_best_candidate([weak, strong]).responder_id == "strong"
