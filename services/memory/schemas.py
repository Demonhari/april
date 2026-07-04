from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Project(BaseModel):
    id: str
    path: str
    name: str
    created_at: str


MEMORY_KINDS: tuple[str, ...] = (
    "fact",
    "preference",
    "correction",
    "project_state",
    "skill_note",
    "relationship",
    "open_loop",
    # Legacy v1 kinds kept readable so existing rows stay valid.
    "project",
    "note",
)


class MemoryRecord(BaseModel):
    id: str
    content: str
    kind: str
    project_id: str | None = None
    reason: str
    created_at: str
    confidence: float = 0.7
    source: str = "user"
    last_used_at: str | None = None
    use_count: int = 0
    expires_at: str | None = None
    superseded_by: str | None = None


class MemoryContradictionRecord(BaseModel):
    """A contradictory memory pair kept for Dreamer adjudication."""

    id: str
    memory_id_a: str
    memory_id_b: str
    status: str = "pending"
    resolution: str | None = None
    created_at: str
    resolved_at: str | None = None


class SessionRecord(BaseModel):
    id: str
    conversation_id: str | None = None
    source: str
    started_at: str
    last_activity_at: str
    closed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WakeEventRecord(BaseModel):
    id: str
    session_id: str | None = None
    source: str
    score: float | None = None
    accepted: bool = True
    reason: str | None = None
    transcript_present: bool = False
    captured_at: str | None = None
    session_hint: str | None = None
    created_at: str


class FeedbackEventRecord(BaseModel):
    id: str
    session_id: str | None = None
    conversation_id: str | None = None
    agent_run_id: str | None = None
    rating: Literal["good", "bad"]
    reason: str | None = None
    created_at: str


class Conversation(BaseModel):
    id: str
    title: str | None = None
    project_id: str | None = None
    actor: str = "local-user"
    created_at: str
    updated_at: str | None = None


class Message(BaseModel):
    id: str
    conversation_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    created_at: str


class ApprovalRecord(BaseModel):
    id: str
    tool: str
    args: dict[str, Any]
    agent: str = "general_agent"
    canonical_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    permission_level: int
    risk_level: str
    status: Literal["pending", "approved", "denied", "expired", "consumed"]
    expires_at: str
    created_at: str
    consumed_at: str | None = None


class SuspendedAgentRun(BaseModel):
    id: str
    agent_run_id: str
    approval_id: str
    conversation_id: str
    project_id: str | None = None
    agent: str
    model_id: str | None = None
    iteration: int
    request_id: str
    messages: list[dict[str, Any]]
    tool_request: dict[str, Any]
    normalized_args: dict[str, Any]
    context: dict[str, Any]
    status: Literal["suspended", "resumed", "completed", "denied", "expired", "failed"]
    created_at: str
    resumed_at: str | None = None
    completed_at: str | None = None


class ReminderRecord(BaseModel):
    id: str
    content: str
    due_at: str | None = None
    created_at: str
    fired_at: str | None = None


class TaskRecord(BaseModel):
    id: str
    title: str
    status: str
    created_at: str


class VectorMetadata(BaseModel):
    source_type: str
    source_id: str
    project_id: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content_hash: str
    created_at: str


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
