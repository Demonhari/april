from __future__ import annotations

from typing import get_args

import pytest

from apps.runner.mac_report import RoutingReport, routing_report_from_results
from apps.runner.multi_model_report import _routing_axis_ok
from services.april_runtime.schemas import ChatResponse, Usage
from services.brain.model_routing import infer_model_route
from services.brain.parser import parse_routing_proposal
from services.brain.route_contract import (
    AgentBinding,
    RouteCompiler,
    RouteContext,
    RouteOperation,
    RouteToolClass,
    RoutingProposal,
    build_router_system_prompt,
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
    assert "context" not in schema.get("required", [])


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


def test_route_compiler_drops_nonessential_tool_not_allowed_by_active_role() -> None:
    compiler = RouteCompiler({"coding_agent": AgentBinding("configured-coding", frozenset())})
    compiled = compiler.compile_with_diagnostics(
        RoutingProposal(
            operation="repository_inspection",
            context="repository",
            tool_class="git_status",
        )
    )
    assert compiled.decision.tools_needed == []
    assert "tool_dropped_not_allowed" in compiled.coercions


def test_route_compiler_coerces_contradictory_nonessential_tool() -> None:
    compiled = RouteCompiler().compile_with_diagnostics(
        RoutingProposal(
            operation="normal_conversation",
            context="conversation",
            tool_class="run_command",
        )
    )
    assert compiled.decision.tools_needed == []
    assert "tool_class_coerced:normal_conversation" in compiled.coercions


def test_repository_operation_without_tool_uses_canonical_pair() -> None:
    compiled = RouteCompiler().compile_with_diagnostics(
        RoutingProposal(operation="repository_inspection", context="conversation")
    )
    assert compiled.decision.tools_needed == ["git_status", "search_files"]
    assert "context_coerced:repository_inspection" in compiled.coercions
    assert "tool_class_coerced:repository_inspection" in compiled.coercions


def test_repository_inspection_uses_canonical_read_only_pair() -> None:
    decision = RouteCompiler().compile(
        RoutingProposal(
            operation="repository_inspection",
            context="repository",
            tool_class="git_status",
        )
    )
    assert decision.tools_needed == ["git_status", "search_files"]


def test_memory_lookup_without_queries_is_compilable_and_diagnostic() -> None:
    compiled = RouteCompiler().compile_with_diagnostics(RoutingProposal(operation="memory_lookup"))
    assert compiled.decision.memory_queries == []
    assert "memory_queries_defaulted" in compiled.coercions


def test_prompt_is_complete_and_compact() -> None:
    prompt = build_router_system_prompt()
    assert len(prompt) < 6_000
    assert "package_install" in prompt
    assert "external_action" in prompt
    assert "test_execution" not in prompt
    assert '"read the README" => repository_inspection' not in prompt
    assert '"read the README and summarize it" => document_reading' in prompt


def test_routing_parser_reports_bounded_schema_rejection_code() -> None:
    with pytest.raises(ValueError, match="semantic contract") as exc_info:
        parse_routing_proposal(
            '{"operation":"planning","context":"conversation","tool_class":"bad"}'
        )
    assert exc_info.value.code == "schema_rejection:tool_class"


def test_derivable_context_and_tools_are_coerced_without_policy_change() -> None:
    compiler = RouteCompiler()
    compiled = compiler.compile_with_diagnostics(
        RoutingProposal(
            operation="reminder_create",
            context="conversation",
            tool_class="none",
            requested_text="stand up",
        )
    )
    assert "context_coerced:reminder_create" in compiled.coercions
    assert "tool_class_coerced:reminder_create" in compiled.coercions
    assert compiled.decision.permission_level == 2
    assert compiled.decision.risk_level == "safe_write"


def test_document_and_code_routes_are_tool_free_at_contract_boundary() -> None:
    compiler = RouteCompiler()
    document = compiler.compile(
        RoutingProposal(
            operation="document_reading",
            context="local_document",
            tool_class="read_file",
        )
    )
    code = compiler.compile(
        RoutingProposal(
            operation="code_modification",
            context="repository",
            tool_class="patch_applier",
        )
    )
    assert document.tools_needed == []
    assert document.planned_tool_calls == []
    assert code.tools_needed == []
    assert code.planned_tool_calls == []


def test_every_operation_context_tool_coercion_preserves_policy() -> None:
    base = RouteCompiler()
    bindings = dict(base.bindings)
    bindings["general_agent"] = AgentBinding(
        "april-brain",
        bindings["general_agent"].allowed_tools | {"approve_action", "reject_action"},
    )
    compiler = RouteCompiler(bindings)
    for operation in get_args(RouteOperation):
        baseline = compiler.compile(
            RoutingProposal(
                operation=operation,
                context=_context_for(operation),
                tool_class="none",
                requested_text=(
                    "action-id" if operation in {"approval_command", "rejection_command"} else "x"
                ),
                memory_queries=["q"],
            )
        )
        for context in get_args(RouteContext):
            for tool_class in get_args(RouteToolClass):
                try:
                    decision = compiler.compile(
                        RoutingProposal(
                            operation=operation,
                            context=context,
                            tool_class=tool_class,
                            requested_text="action-id"
                            if operation in {"approval_command", "rejection_command"}
                            else "x",
                            memory_queries=["q"],
                        )
                    )
                except (ValueError, KeyError):
                    continue
                assert (decision.permission_level, decision.risk_level) == (
                    baseline.permission_level,
                    baseline.risk_level,
                )


def _context_for(operation: str) -> str:
    if operation.startswith("reminder_"):
        return "reminder"
    if operation in {"approval_command", "rejection_command"}:
        return "system"
    if operation in {"memory_lookup", "memory_write"}:
        return "memory"
    if operation in {"repository_inspection", "patch_proposal", "code_modification"}:
        return "repository"
    if operation == "document_reading":
        return "local_document"
    if operation in {"coding_assistance"}:
        return "pasted_text"
    return "conversation"


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
async def test_model_route_preserves_parser_and_compiler_coercions() -> None:
    client = ScriptedRoutingClient([_response('{"operation":"planning","confidence":85}')])
    outcome = await infer_model_route(
        client,
        model_id="april-brain",
        message="plan my day",
        history=None,
        request_id="r",
        compiler=RouteCompiler(),
    )
    assert outcome.decision is not None
    assert "confidence_normalized" in outcome.coercions
    assert "context_defaulted" in outcome.coercions


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
async def test_repair_cannot_change_a_valid_candidate_operation() -> None:
    client = ScriptedRoutingClient(
        [
            _response(
                '{"operation":"approval_command","context":"system",'
                '"tool_class":"approve_action","requested_text":"approval-1234"}'
            ),
            _response(_proposal("normal_conversation")),
        ]
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
    assert outcome.failure_code == "repair_failure"
    assert outcome.repair_rejection_code == "operation_changed"


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
