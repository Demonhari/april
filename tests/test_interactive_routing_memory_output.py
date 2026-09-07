from __future__ import annotations

import pytest

from apps.runner.evals import BrainEvalCase, real_routing_report
from april_common.errors import ValidationError
from services.brain.deterministic_router import DeterministicRouter
from services.brain.fallback_router import FallbackRouter
from services.brain.parser import parse_brain_decision
from services.brain.response_handling import ReasoningStreamFilter, sanitize_model_output


@pytest.mark.parametrize(
    ("message", "agent"),
    [
        ("Hello April. Introduce yourself briefly.", "general_agent"),
        (
            "Write a small Python function that takes a list of numbers and returns only "
            "the even numbers.",
            "coding_agent",
        ),
        ("Explain in simple terms how APRIL's local model architecture works.", "general_agent"),
        ("Why is the sky blue?", "general_agent"),
        ("Explain this pasted code: return [n for n in numbers if n % 2 == 0]", "coding_agent"),
    ],
)
def test_tool_free_interactive_requests_do_not_need_repository(message: str, agent: str) -> None:
    decision = FallbackRouter().route(message)
    assert decision.agent == agent
    assert decision.tools_needed == []
    assert decision.permission_level == 0


def test_explicit_memory_is_anchored_and_keeps_relationship() -> None:
    for message in (
        "Remember that my test project is called Project Bluebird.",
        "Please save that my test project is named Project Amber.",
        "Could you make a note that my test project is known as Project Copper?",
        "I'd like you to remember that my test project is called Project Jade.",
    ):
        route = DeterministicRouter().route(message)
        assert route is not None
        assert route.decision.intent == "memory_write"
        assert route.decision.agent == "general_agent"
        assert route.decision.planned_tool_calls[0].args["memory_type"] == "relationship"
        assert "test project" in route.decision.planned_tool_calls[0].args["content"]
    assert DeterministicRouter().route("I remember that VS Code is installed.") is None
    assert DeterministicRouter().route("For example: remember that my editor is vim") is None
    sensitive = DeterministicRouter().route("Do not remember that my password is abc")
    assert sensitive is not None
    assert sensitive.decision.intent == "sensitive_content"
    assert DeterministicRouter().route('"Remember that my editor is vim."') is None


def test_recall_name_is_a_scoped_memory_lookup() -> None:
    decision = FallbackRouter().route("What is the name of my test project?")
    assert decision.intent == "memory_lookup"
    assert decision.memory_queries == ["test project"]
    assert decision.agent == "general_agent"
    assert decision.tools_needed == []


def test_archive_is_not_a_valid_interactive_route() -> None:
    with pytest.raises(ValidationError, match="Brain JSON"):
        parse_brain_decision(
            '{"intent":"normal_conversation","agent":"memory_agent",'
            '"model_id":"april-reading","permission_level":0,"risk_level":"none",'
            '"needs_confirmation":false,"decision_summary":"extract"}'
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>private plan</think>Answer.", "Answer."),
        ("<think>private plan", ""),
        ('{"answer":"literal <think> tag"}', '{"answer":"literal <think> tag"}'),
        (
            "```xml\n<think>literal example</think>\n```",
            "```xml\n<think>literal example</think>\n```",
        ),
    ],
)
def test_user_facing_output_removes_only_leading_control_reasoning(raw: str, expected: str) -> None:
    assert sanitize_model_output(raw) == expected


def test_stream_filter_handles_split_and_unclosed_reasoning_tags() -> None:
    stream = ReasoningStreamFilter()
    visible = "".join(
        stream.feed(chunk) for chunk in ["<thi", "nk>hidden", "</thi", "nk>", "answer"]
    )
    assert visible + stream.finish() == "answer"

    interrupted = ReasoningStreamFilter()
    assert (
        "".join(interrupted.feed(chunk) for chunk in ["<think>", "hidden"]) + interrupted.finish()
        == ""
    )


def test_stream_filter_keeps_whitespace_and_consecutive_reasoning_hidden() -> None:
    stream = ReasoningStreamFilter()
    output = "".join(
        stream.feed(chunk)
        for chunk in [
            " ",
            "<think>",
            "hidden",
            "</think>",
            "\n",
            "<think>",
            "more",
            "</think>",
            "Answer",
        ]
    )
    assert output + stream.finish() == "Answer"


def test_stream_filter_discards_oversized_and_cancelled_reasoning() -> None:
    stream = ReasoningStreamFilter()
    assert stream.feed("<think>") == ""
    assert stream.feed("x" * 32_769) == ""
    assert stream.feed("</think>visible") == ""
    assert stream.finish() == ""


def test_non_streaming_reasoning_blocks_are_bounded_and_consecutive() -> None:
    assert sanitize_model_output(" \ufeff<think>x</think><analysis>y</analysis>Answer") == "Answer"
    assert sanitize_model_output("<think>" + "x" * 32_769) == ""
    assert sanitize_model_output("<think>x</think><think>") == ""


def test_routing_report_separates_deterministic_from_model_and_true_fallback() -> None:
    case = BrainEvalCase(
        id="hello",
        message="hello",
        expected_intent="normal_conversation",
        expected_agent="general_agent",
    )
    base = {
        "intent": "normal_conversation",
        "agent": "general_agent",
        "model_id": "april-brain",
        "tools_needed": [],
        "permission_level": 0,
        "risk_level": "none",
        "needs_confirmation": False,
        "decision_summary": "Answer.",
        "routing_method": "fallback",
    }
    deterministic = {
        **base,
        "route_source": "deterministic",
        "route_provenance": "trusted_v1",
    }
    model = {
        **base,
        "routing_method": "model",
        "route_source": "model",
        "route_provenance": "trusted_v1",
    }
    fallback = {**base, "route_source": "fallback", "route_provenance": "trusted_v1"}
    report = real_routing_report([case, case, case], [deterministic, model, fallback])
    assert report.passed == 2
    assert report.deterministic_count == 1
    assert report.model_count == 1
    assert report.model_repair_count == 0
    assert report.fallback_count == 1
