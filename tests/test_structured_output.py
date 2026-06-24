from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from agents.base import BaseAgent
from agents.schemas import AgentConfig
from april_common.errors import ValidationError as AprilValidationError
from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ResponseFormat,
    Usage,
)
from services.brain.agent_loop import (
    AGENT_OUTPUT_ADAPTER,
    AGENT_OUTPUT_RESPONSE_FORMAT,
    AgentFinalAnswer,
    StructuredAgentLoop,
)
from services.brain.parser import parse_brain_decision, parse_with_repair
from services.brain.router import BrainRouter
from services.brain.structured_output import BRAIN_DECISION_RESPONSE_FORMAT
from tests.test_runtime_api import runtime_lifecycle

VALID_DECISION = (
    '{"intent":"planning","agent":"general_agent","model_id":"april-brain",'
    '"tools_needed":[],"memory_queries":[],"permission_level":0,'
    '"risk_level":"none","needs_confirmation":false,'
    '"task_steps":["Answer"],"decision_summary":"ok"}'
)


class ScriptedRuntimeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.response_formats: list[Any] = []

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: Any | None = None,
        response_format: Any | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        self.response_formats.append(response_format)
        content = self._responses.pop(0)
        return ChatResponse(
            request_id=request_id or "r", model_id=model_id, content=content, usage=Usage()
        )


# --- schema validation and size limits -------------------------------------


def test_response_format_size_limit_rejected() -> None:
    oversized = {
        "type": "object",
        "properties": {str(index): {"type": "string"} for index in range(6000)},
    }
    with pytest.raises(PydanticValidationError):
        ResponseFormat(type="json_object", json_schema=oversized)


def test_response_format_depth_limit_rejected() -> None:
    node: dict[str, Any] = {"type": "string"}
    for _ in range(40):
        node = {"type": "object", "properties": {"child": node}}
    with pytest.raises(PydanticValidationError):
        ResponseFormat(type="json_object", json_schema=node)


def test_modest_schema_accepted() -> None:
    response_format = ResponseFormat(
        type="json_object", json_schema={"type": "object", "properties": {"a": {"type": "string"}}}
    )
    assert response_format.json_schema is not None


# --- strict validation, repair, fallback ------------------------------------


def test_valid_schema_output_parses() -> None:
    decision = parse_brain_decision(VALID_DECISION)
    assert decision.intent == "planning"
    assert decision.agent == "general_agent"


def test_missing_required_key_rejected() -> None:
    text = (
        '{"agent":"general_agent","model_id":"april-brain","risk_level":"none",'
        '"needs_confirmation":false,"permission_level":0,"decision_summary":"x"}'
    )
    with pytest.raises(AprilValidationError):
        parse_brain_decision(text)


def test_incorrect_enum_value_rejected() -> None:
    text = VALID_DECISION.replace('"risk_level":"none"', '"risk_level":"made_up"')
    with pytest.raises(AprilValidationError):
        parse_brain_decision(text)


def test_malformed_json_rejected() -> None:
    with pytest.raises(AprilValidationError):
        parse_brain_decision("this is not json at all")


def test_extra_keys_rejected_by_agent_output() -> None:
    with pytest.raises(PydanticValidationError):
        AGENT_OUTPUT_ADAPTER.validate_python(
            {"type": "final_answer", "message": "hi", "unexpected": "x"}
        )


async def test_repair_success() -> None:
    async def repair(_: str) -> str:
        return VALID_DECISION

    decision = await parse_with_repair("not valid json", repair)
    assert decision.intent == "planning"
    assert decision.routing_method == "model_repair"


async def test_repair_failure_falls_back_to_deterministic_router() -> None:
    client = ScriptedRuntimeClient(["garbage one", "garbage two"])
    decision = await BrainRouter(client).route("show git status")  # type: ignore[arg-type]
    assert decision.routing_method == "fallback"
    assert "git_status" in decision.tools_needed
    # Even on the path that ends in fallback, the brain still requested its schema.
    assert client.response_formats[0] is BRAIN_DECISION_RESPONSE_FORMAT


# --- exact-schema requests --------------------------------------------------


async def test_brain_requests_its_exact_schema() -> None:
    client = ScriptedRuntimeClient([VALID_DECISION])
    decision = await BrainRouter(client).route("plan my day")  # type: ignore[arg-type]
    assert decision.intent == "planning"
    assert client.response_formats == [BRAIN_DECISION_RESPONSE_FORMAT]
    assert BRAIN_DECISION_RESPONSE_FORMAT.json_schema is not None


async def test_structured_agent_requests_its_output_schema() -> None:
    client = ScriptedRuntimeClient(['{"type":"final_answer","message":"done"}'])
    loop = StructuredAgentLoop(runtime_client=client, tool_executor=None, memory=None)  # type: ignore[arg-type]
    agent = BaseAgent(
        AgentConfig(
            name="coding_agent",
            description="d",
            model_id="april-coding",
            system_prompt_path="x",
            memory_access_policy="none",
            maximum_tool_iterations=3,
            system_prompt="sys",
        )
    )
    output = await loop._next_iteration(
        agent=agent,
        messages=[ChatMessage(role="user", content="hi")],
        request_id="r",
    )
    assert isinstance(output, AgentFinalAnswer)
    assert client.response_formats == [AGENT_OUTPUT_RESPONSE_FORMAT]


# --- propagation through the backend and lifecycle --------------------------


async def test_fake_backend_records_response_format() -> None:
    backend = FakeBackend()
    response_format = ResponseFormat(type="json_object", json_schema={"type": "object"})
    await backend.generate_messages(
        "plan my work today",
        messages=[ChatMessage(role="user", content="plan my work today")],
        temperature=0.0,
        max_output_tokens=8,
        response_format=response_format,
    )
    assert backend.last_response_format is response_format
    tokens = [
        token
        async for token in backend.stream_messages(
            "hello there",
            messages=[ChatMessage(role="user", content="hello there")],
            temperature=0.0,
            max_output_tokens=8,
            response_format=response_format,
        )
    ]
    assert backend.last_response_format is response_format
    assert tokens


async def test_response_format_propagates_through_generate(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path)
    response_format = ResponseFormat(type="json_object", json_schema={"type": "object"})
    await lifecycle.generate(
        ChatRequest(
            model_id="april-brain",
            messages=[ChatMessage(role="user", content="plan my work today")],
            response_format=response_format,
        )
    )
    backend = lifecycle.get_state("april-brain").backend
    assert backend is not None
    assert backend.last_response_format is response_format  # type: ignore[attr-defined]
    await lifecycle.cleanup()


async def test_streaming_structured_output_propagates(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path)
    response_format = ResponseFormat(type="json_object")
    events = [
        event
        async for event in lifecycle.stream(
            ChatRequest(
                model_id="april-brain",
                messages=[ChatMessage(role="user", content="hello there")],
                response_format=response_format,
            )
        )
    ]
    backend = lifecycle.get_state("april-brain").backend
    assert backend is not None
    assert backend.last_response_format is response_format  # type: ignore[attr-defined]
    assert any(name == "token" for name, _ in events)
    await lifecycle.cleanup()
