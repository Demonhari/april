from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agents.schemas import LocalCitation, ProposedChange
from services.april_runtime.schemas import ChatMessage
from services.brain.schemas import BrainDecision, RouteResult
from services.memory.schemas import Message


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
    structured_agent: bool = False
    task_plan_id: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)
