"""Honest D5 evaluation: deterministic fixtures vs real-runtime evals.

Every runtime client in this file is an explicitly test-only fake used to
exercise the *gating logic*. Production code never fabricates a real-runtime
result: an unavailable/fake runtime always reports ``skipped_real_runtime``
with a blocker, never a pass.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from services.april_runtime.schemas import ChatResponse, Usage
from services.evolution.dreamer import DreamerService
from services.evolution.eval_review import promote_pending_case
from services.evolution.evaluator import (
    evaluate_overlay_candidate,
    evaluate_overlay_candidate_real_runtime,
    write_pending_eval_case,
)
from services.evolution.scheduler import EvolutionSchedulerGate
from services.evolution.versions import PromptOverlayManager
from tests.test_evolution_v2 import _enabled_settings, _memory, _permissive_governor

_OVERLAY = "Prefer concise answers grounded in local context and say what changed."


class FakeRealRuntimeClient:
    """TEST-ONLY fake of the local runtime for real-eval gating tests.

    It *claims* a real llama_cpp backend so tests can drive the pass/fail
    branches deterministically. It exists only under tests/ and is never
    importable from production code.
    """

    def __init__(
        self,
        *,
        backend: str = "llama_cpp",
        simulated: bool = False,
        healthy: bool = True,
        judge_verdict: bool = True,
        replay_response: str = "APRIL stays local-only and requires approval for high impact.",
    ) -> None:
        self.backend = backend
        self.simulated = simulated
        self.healthy = healthy
        self.judge_verdict = judge_verdict
        self.replay_response = replay_response
        self.chat_calls: list[list[Any]] = []

    async def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        return {
            "status": "ok" if self.healthy else "error",
            "backend": self.backend,
            "simulated": self.simulated,
        }

    async def chat(
        self,
        *,
        model_id: str,
        messages: Any,
        options: Any | None = None,
        response_format: Any | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        self.chat_calls.append(list(messages))
        joined = "\n".join(message.content for message in messages)
        if "local eval judge" in joined:
            content = json.dumps({"meets_expectation": self.judge_verdict})
        else:
            content = self.replay_response
        return ChatResponse(
            request_id=request_id or "eval-test",
            model_id=model_id,
            content=content,
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def test_deterministic_eval_is_labelled_as_fixture_check(settings_tmp) -> None:
    payload = evaluate_overlay_candidate(
        agent="general_agent", content=_OVERLAY, settings=settings_tmp
    ).to_payload()
    assert payload["eval_kind"] == "deterministic_fixture"
    assert payload["deterministic_fixture_passed"] is True


@pytest.mark.asyncio
async def test_real_eval_skips_without_runtime_client(settings_tmp) -> None:
    result = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=None,
    )
    assert result.status == "skipped_real_runtime"
    assert result.passed is False
    assert any("no local runtime client" in blocker for blocker in result.blockers)


@pytest.mark.asyncio
async def test_real_eval_skips_on_fake_or_unhealthy_backend(settings_tmp) -> None:
    fake_backend = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(backend="fake"),
    )
    assert fake_backend.status == "skipped_real_runtime"
    assert any("fake/simulated" in blocker for blocker in fake_backend.blockers)

    simulated = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(simulated=True),
    )
    assert simulated.status == "skipped_real_runtime"

    unhealthy = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(healthy=False),
    )
    assert unhealthy.status == "skipped_real_runtime"
    assert any("unhealthy" in blocker for blocker in unhealthy.blockers)


@pytest.mark.asyncio
async def test_real_eval_passes_and_fails_on_actual_output(settings_tmp) -> None:
    passing = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(),
    )
    assert passing.status == "real_runtime_eval_passed"
    assert passing.cases_run == passing.cases_passed > 0

    failing = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(replay_response="unrelated output"),
    )
    assert failing.status == "real_runtime_eval_failed"
    assert failing.passed is False
    assert failing.blockers


@pytest.mark.asyncio
async def test_real_eval_includes_promoted_cases_and_respects_judge(settings_tmp) -> None:
    case_id = write_pending_eval_case(
        settings_tmp,
        {"case_type": "negative_feedback", "prompt": "what is my timezone", "reason": "tz"},
    ).stem
    promote_pending_case(
        settings_tmp, case_id, expected_behavior="Answer using the stored timezone."
    )

    accepted = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(judge_verdict=True),
    )
    assert accepted.status == "real_runtime_eval_passed"

    judged_bad = await evaluate_overlay_candidate_real_runtime(
        agent="general_agent",
        content=_OVERLAY,
        settings=settings_tmp,
        runtime_client=FakeRealRuntimeClient(judge_verdict=False),
    )
    assert judged_bad.status == "real_runtime_eval_failed"
    assert any(case_id in blocker for blocker in judged_bad.blockers)


async def _seed_negative_feedback(memory) -> None:
    conversation_id = await memory.create_conversation()
    run_row_id = await memory.record_agent_run(
        conversation_id=conversation_id,
        agent="general_agent",
        status="ok",
        model_id="april-brain",
        summary="planning answer",
    )
    await memory.record_feedback_event(
        rating="bad",
        reason="answer ignored my timezone",
        conversation_id=conversation_id,
        agent_run_id=run_row_id,
    )


def _production_settings(settings_tmp, **overrides):
    enabled = _enabled_settings(settings_tmp, **overrides)
    return enabled.model_copy(update={"environment": "production"})


@pytest.mark.asyncio
async def test_production_dreamer_holds_candidates_without_real_runtime(settings_tmp) -> None:
    production = _production_settings(settings_tmp)
    database, memory = await _memory(production)
    try:
        await _seed_negative_feedback(memory)
        gate = EvolutionSchedulerGate(production, memory, governor=_permissive_governor(production))
        service = DreamerService(production, memory=memory, gate=gate, runtime_client=None)
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "completed"
        assert result.report_path is not None
        from pathlib import Path

        report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
        examine = report["phases"]["examine"]
        # Deterministic fixtures passed but no overlay was activated: the
        # candidate is pending with an explicit real-runtime blocker.
        assert examine["activated"] == []
        assert len(examine["pending_real_runtime"]) == 1
        pending = examine["pending_real_runtime"][0]
        assert pending["agent"] == "general_agent"
        assert pending["status"] == "skipped_real_runtime"
        assert "no local runtime client" in pending["reason"]
        modes = examine["eval_modes"]
        assert modes["real_runtime_required"] is True
        assert modes["deterministic_fixture_passed"] == 1
        assert modes["real_runtime_eval_skipped"] == 1
        assert modes["real_runtime_eval_passed"] == 0
        assert modes["blockers"]
        # No overlay version was written.
        manager = PromptOverlayManager(production, database)
        assert await manager.active_overlay_text("general_agent") is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_production_dreamer_activates_only_on_real_runtime_pass(settings_tmp) -> None:
    production = _production_settings(settings_tmp)
    database, memory = await _memory(production)
    try:
        await _seed_negative_feedback(memory)
        gate = EvolutionSchedulerGate(production, memory, governor=_permissive_governor(production))
        service = DreamerService(
            production,
            memory=memory,
            gate=gate,
            runtime_client=FakeRealRuntimeClient(),
        )
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "completed"
        from pathlib import Path

        report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
        examine = report["phases"]["examine"]
        assert examine["activated"] == [{"agent": "general_agent", "version": 1}]
        assert examine["pending_real_runtime"] == []
        modes = examine["eval_modes"]
        assert modes["real_runtime_eval_passed"] == 1
        assert modes["real_runtime_eval_skipped"] == 0
        evaluation = examine["evaluations"][0]
        assert evaluation["eval_kind"] == "deterministic_fixture"
        assert evaluation["real_runtime"]["status"] == "real_runtime_eval_passed"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_development_dreamer_keeps_test_only_activation(settings_tmp) -> None:
    enabled = _enabled_settings(settings_tmp)
    assert enabled.environment != "production"
    database, memory = await _memory(enabled)
    try:
        await _seed_negative_feedback(memory)
        gate = EvolutionSchedulerGate(enabled, memory, governor=_permissive_governor(enabled))
        service = DreamerService(enabled, memory=memory, gate=gate, runtime_client=None)
        result = await service.run_once(datetime(2026, 7, 3, 2, 30, tzinfo=UTC))
        assert result.status == "completed"
        from pathlib import Path

        report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
        examine = report["phases"]["examine"]
        # Development/fake-backend behaviour is unchanged: the deterministic
        # fixture pass allows the (test-only) activation.
        assert examine["activated"] == [{"agent": "general_agent", "version": 1}]
        assert examine["eval_modes"]["real_runtime_required"] is False
    finally:
        await database.close()
