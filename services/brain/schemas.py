from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agents.schemas import InteractiveAgentName

RiskLevel = Literal[
    "none", "read_only", "safe_write", "code_write", "system_action", "external_action"
]

# Canonical fixture and policy vocabulary. Legacy spellings are accepted only
# at the compatibility boundary below and are normalized before downstream use.
CanonicalIntent = Literal[
    "normal_conversation",
    "planning",
    "coding_repo_analysis",
    "coding_assistance",
    "document_reading",
    "creative_writing",
    "deep_reasoning",
    "memory_lookup",
    "memory_write",
    "patch_proposal",
    "code_modification",
    "command_execution",
    "log_cleanup",
    "package_install",
    "external_action",
    "ambiguous_request",
    "prompt_injection",
    "path_escape_attempt",
    "sensitive_content",
    "unsupported_tool",
    "reminders",
]

_LEGACY_INTENT_ALIASES = {
    "repository_search": "coding_repo_analysis",
    "configured_test_execution": "command_execution",
    "reminder_list": "reminders",
    "reminder_cancel": "reminders",
    "reminder_create": "reminders",
    "destructive_action": "sensitive_content",
    "destructive": "sensitive_content",
    "path_escape": "path_escape_attempt",
    "unknown_tool": "unsupported_tool",
    "approval_command": "normal_conversation",
    "rejection_command": "normal_conversation",
    "direct_agent_run": "normal_conversation",
    # Benchmark fixture category aliases are normalized only after the strict
    # JSON boundary; the generated schema still exposes the canonical 20.
    "git_status": "coding_repo_analysis",
    "git_diff": "coding_repo_analysis",
    "file_reading": "document_reading",
    "file_search": "coding_repo_analysis",
    "reminder_creation": "reminders",
    "reminder_listing": "reminders",
    "patch_preparation": "patch_proposal",
    "test_execution": "command_execution",
    "approval": "normal_conversation",
    "rejection": "normal_conversation",
    "destructive_external": "external_action",
    "ambiguous_general": "ambiguous_request",
}


class PlannedToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class BrainDecision(BaseModel):
    intent: CanonicalIntent
    agent: InteractiveAgentName
    model_id: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    high_stakes: bool = False
    tools_needed: list[str] = Field(default_factory=list)
    planned_tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    memory_queries: list[str] = Field(default_factory=list)
    permission_level: int = Field(ge=0, le=5)
    risk_level: RiskLevel
    needs_confirmation: bool
    task_steps: list[str] = Field(default_factory=list, max_length=8)
    decision_summary: str
    routing_method: Literal["model", "model_repair", "fallback"] = "model"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_intent(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("intent"), str):
            value = dict(value)
            value["intent"] = _LEGACY_INTENT_ALIASES.get(value["intent"], value["intent"])
        return value

    @model_validator(mode="after")
    def validate_routing_semantics(self) -> BrainDecision:
        """Reject schema-valid routes that cannot be valid interactive plans."""
        tools = {call.tool for call in self.planned_tool_calls} | set(self.tools_needed)
        known_tools = {
            "approve_action",
            "reject_action",
            "apply_log_cleanup",
            "cancel_reminder",
            "create_note",
            "create_reminder",
            "document_indexer",
            "document_search",
            "git_branch",
            "git_commit",
            "git_diff",
            "git_log",
            "git_push",
            "git_status",
            "list_files",
            "list_reminders",
            "open_app",
            "open_url",
            "patch_applier",
            "patch_generator",
            "plan_log_cleanup",
            "read_file",
            "remember_memory",
            "repo_indexer",
            "run_command",
            "search_files",
            "search_notes",
            "test_runner",
            "write_file",
        }
        unknown = sorted(tools - known_tools)
        if unknown:
            raise ValueError(f"Unknown routing tool(s): {', '.join(unknown)}")
        if self.intent == "memory_write" and "remember_memory" not in tools:
            raise ValueError("memory_write routes must request remember_memory")
        if "remember_memory" in tools and self.intent != "memory_write":
            raise ValueError("remember_memory is only valid for memory_write")
        if self.intent == "memory_write" and self.agent != "general_agent":
            raise ValueError("interactive memory writes must use general_agent")
        if self.intent == "coding_assistance":
            if self.agent != "coding_agent" or tools:
                raise ValueError("tool-free coding assistance must use coding_agent without tools")
            if self.permission_level != 0 or self.risk_level != "none":
                raise ValueError("tool-free coding assistance must be level 0")
        return self


class RouteSource(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    MODEL_REPAIR = "model_repair"
    FALLBACK = "fallback"


class RouteResult(BaseModel):
    """Trusted routing provenance kept outside model-generated output."""

    decision: BrainDecision
    route_source: RouteSource
    raw_model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    historical_reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_confidence: float = Field(ge=0.0, le=1.0)
    reliability_sample_count: int = Field(default=0, ge=0)
    confidence_source: str
    matched_rule: str | None = None
    fallback_reason: str | None = None
    structured_output_valid: bool = True
    repair_used: bool = False
    proposal_operation: str | None = None
    proposal_context: str | None = None
    contract_fingerprint: str | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
    routing_failure_code: str | None = None

    @property
    def route_key(self) -> str:
        tool_class = (
            self.decision.planned_tool_calls[0].tool
            if self.decision.planned_tool_calls
            else (self.decision.tools_needed[0] if self.decision.tools_needed else "no_tool")
        )
        return ":".join(
            (
                self.decision.intent[:64],
                self.decision.agent,
                tool_class[:64],
            )
        )
