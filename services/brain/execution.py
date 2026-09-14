from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agents.schemas import LocalCitation, ProposedChange
from services.april_runtime.schemas import ChatMessage
from services.brain.request_context import RequestContext
from services.brain.schemas import BrainDecision, RouteResult
from services.brain.task_contract import TaskContract
from services.memory.schemas import Message


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Bounded, untrusted context reused by draft verification.

    This is deliberately separate from ``trusted_context``: history, retrieved
    sources, and tool output are evidence for the answer, never application
    authority or instructions.
    """

    history: tuple[Message, ...] = ()
    conversation_summary: str | None = None
    source_sections: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    tool_outputs: tuple[str, ...] = ()
    truncated_categories: tuple[str, ...] = ()


@dataclass(slots=True)
class PreparedTurn:
    """State passed between routing, context assembly, execution, and finalization."""

    request_id: str
    conversation_id: str
    decision: BrainDecision
    route_result: RouteResult
    agent_name: str
    model_id: str
    messages: list[ChatMessage]
    citations: list[LocalCitation] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    final_message: str | None = None
    final_status: Literal["ok", "error"] = "error"
    proposed_changes: list[ProposedChange] = field(default_factory=list)
    project_id: str | None = None
    actor: str = "local-user"
    history: list[Message] = field(default_factory=list)
    context_sections: list[str] = field(default_factory=list)
    stable_prefix: str | None = None
    task_contract: TaskContract | None = None
    request_context: RequestContext = field(default_factory=RequestContext.unknown)
    trusted_context: str | None = None
    verification_evidence: VerificationEvidence | None = None
    structured_agent: bool = False
    task_plan_id: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)
