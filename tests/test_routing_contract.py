from __future__ import annotations

import pytest

from apps.runner.mac_report import RoutingReport, routing_report_from_results
from apps.runner.multi_model_report import _routing_axis_ok
from services.april_runtime.schemas import ChatResponse, Usage
from services.brain.model_routing import infer_model_route
from services.brain.route_contract import (
    AgentBinding,
    RouteCompiler,
    RouteContractError,
    RoutingProposal,
)
from services.brain.structured_output import ROUTING_PROPOSAL_RESPONSE_FORMAT


class ScriptedRoutingClient:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = list(responses)
        self.messages: list[str] = []

    async def chat(self, **kwargs: object) -> ChatResponse:
        messages = kwargs["messages"]
        self.messages.append("\n".join(str(item.content) for item in messages))  # type: ignore[union-attr]
        return self.responses.pop(0)


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    diagnostics: dict | None = None,
) -> ChatResponse:
    return ChatResponse(
        request_id="r",
        model_id="april-brain",
        content=content,
        finish_reason=finish_reason,  # type: ignore[arg-type]
        usage=Usage(input_tokens=8, output_tokens=4, total_tokens=12),
        diagnostics=diagnostics or {},
    )


def _proposal(operation: str = "planning") -> str:
    return (
        '{"operation":"'
        + operation
        + '","context":"conversation","tool_class":"none","confidence":0.8}'
    )


def test_routing_schema_contains_semantics_not_policy() -> None:
    schema = ROUTING_PROPOSAL_RESPONSE_FORMAT.json_schema
    assert schema is not None
    assert "operation" in schema["properties"]
    assert "agent" not in schema["properties"]
    assert "permission_level" not in schema["properties"]
    assert "route_source" not in schema["properties"]


def test_route_compiler_uses_active_binding_and_derives_policy() -> None:
    compiler = RouteCompiler(
        {
            "coding_agent": AgentBinding("configured-coding", frozenset()),
        }
    )
    decision = compiler.compile(
        RoutingProposal(operation="coding_assistance", context="pasted_text")
    )
    assert decision.model_id == "configured-coding"
    assert decision.permission_level == 0
    assert decision.risk_level == "none"
    assert decision.tools_needed == []


def test_route_compiler_rejects_tool_not_allowed_by_active_role() -> None:
    compiler = RouteCompiler({"coding_agent": AgentBinding("configured-coding", frozenset())})
    with pytest.raises(RouteContractError, match="not allowed"):
        compiler.compile(
            RoutingProposal(
                operation="repository_inspection",
                context="repository",
                tool_class="git_status",
            )
        )


def test_route_compiler_rejects_contradictory_operation_and_tool() -> None:
    with pytest.raises(RouteContractError, match="semantic operation"):
        RouteCompiler().compile(
            RoutingProposal(
                operation="normal_conversation",
                context="conversation",
                tool_class="run_command",
            )
        )


def test_repository_operation_cannot_be_tool_free() -> None:
    with pytest.raises(RouteContractError, match="repository tool"):
        RouteCompiler().compile(
            RoutingProposal(operation="repository_inspection", context="repository")
        )


@pytest.mark.asyncio
async def test_valid_json_with_length_finish_is_not_model_success() -> None:
    client = ScriptedRoutingClient([_response(_proposal(), finish_reason="length")])
    outcome = await infer_model_route(
        client,
        model_id="april-brain",
        message="plan my day",
        history=None,
        request_id="r",
        compiler=RouteCompiler(),
    )
    assert outcome.decision is None
    assert outcome.failure_code == "generation_length"
    assert outcome.repair_attempted is False


@pytest.mark.asyncio
async def test_repair_receives_request_contract_and_validation_category() -> None:
    client = ScriptedRoutingClient(
        [_response("not json"), _response(_proposal("normal_conversation"))]
    )
    outcome = await infer_model_route(
        client,
        model_id="april-brain",
        message="Explain the architecture",
        history=None,
        request_id="r",
        compiler=RouteCompiler(),
    )
    assert outcome.decision is not None
    assert outcome.route_source.value == "model_repair"  # type: ignore[union-attr]
    assert outcome.repair_attempted is True
    assert outcome.repair_succeeded is True
    assert "Explain the architecture" in client.messages[1]
    assert "semantic routing contract" in client.messages[1]
    assert "schema_rejection" in client.messages[1]


@pytest.mark.asyncio
async def test_missing_finish_evidence_is_not_clean_success() -> None:
    client = ScriptedRoutingClient(
        [_response(_proposal(), diagnostics={"finish_reason_present": False})]
    )
    outcome = await infer_model_route(
        client,
        model_id="april-brain",
        message="plan my day",
        history=None,
        request_id="r",
        compiler=RouteCompiler(),
    )
    assert outcome.decision is None
    assert outcome.failure_code == "missing_completion_evidence"


def test_summary_only_or_duplicate_reports_cannot_pass_routing_axis() -> None:
    summary_only = RoutingReport(
        total=38,
        passed=38,
        accuracy=1.0,
        schema_valid_count=38,
        provenance_verified=True,
    )
    assert _routing_axis_ok(summary_only, allow_deterministic=False) is False

    class Result:
        def __init__(self, case_id: str) -> None:
            self.id = case_id
            self.ok = True
            self.schema_valid = True
            self.routing_ok = True
            self.actual = {
                "route_source": "model",
                "route_provenance": "trusted_model_only_v1",
                "intent": "planning",
            }
            self.expected_intent = "planning"
            self.actual = dict(self.actual)

    report = routing_report_from_results(
        [Result("same"), Result("same")],
        require_trusted_provenance=True,
        expected_case_ids=["same", "other"],
    )
    assert report.duplicate_case_ids == 1
    assert report.case_set_complete is False
    assert _routing_axis_ok(report, allow_deterministic=False) is False


def test_unmarked_legacy_method_is_unknown_for_acceptance() -> None:
    class LegacyResult:
        def __init__(self) -> None:
            self.id = "legacy"
            self.ok = True
            self.schema_valid = True
            self.routing_ok = True
            self.actual = {"routing_method": "model", "intent": "planning"}
            self.expected_intent = "planning"

    report = routing_report_from_results([LegacyResult()], expected_case_ids=["legacy"])
    assert report.model_count == 0
    assert report.unknown_provenance_count == 1
    assert report.provenance_verified is False
    assert _routing_axis_ok(report, allow_deterministic=False) is False
