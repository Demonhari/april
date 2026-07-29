from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

# v2 provenance vocabulary for memory rows. "user" is explicit manual writes;
# the rest are machine-written (reflection = Archive session reflection,
# dream = Dreamer consolidation, import = bulk import). "archive" is the legacy
# spelling of "reflection" kept readable so existing rows stay valid.
MEMORY_SOURCES: tuple[str, ...] = ("user", "reflection", "dream", "import")
LEGACY_MEMORY_SOURCE_ALIASES: dict[str, str] = {"archive": "reflection"}


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
    content_encrypted: bool = False


class LexicalHit(BaseModel):
    """Internal, content-safe lexical ranking evidence."""

    memory: MemoryRecord
    lexical_rank: int = Field(ge=1)
    normalized_score: float = Field(ge=0.0, le=1.0)
    matched_tokens: tuple[str, ...] = ()


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


SUMMARY_ITEM_MAX_CHARS = 500
SUMMARY_GOAL_MAX_CHARS = 800
SUMMARY_LIST_MAX_ITEMS = 20


class ConversationSummaryContent(BaseModel):
    """Validated, deliberately shallow conversation-summary payload."""

    model_config = ConfigDict(extra="forbid")

    current_goal: str | None = Field(default=None, max_length=SUMMARY_GOAL_MAX_CHARS)
    important_facts: list[str] = Field(default_factory=list, max_length=SUMMARY_LIST_MAX_ITEMS)
    decisions: list[str] = Field(default_factory=list, max_length=SUMMARY_LIST_MAX_ITEMS)
    constraints: list[str] = Field(default_factory=list, max_length=SUMMARY_LIST_MAX_ITEMS)
    completed_actions: list[str] = Field(default_factory=list, max_length=SUMMARY_LIST_MAX_ITEMS)
    open_loops: list[str] = Field(default_factory=list, max_length=SUMMARY_LIST_MAX_ITEMS)

    @field_validator(
        "important_facts",
        "decisions",
        "constraints",
        "completed_actions",
        "open_loops",
    )
    @classmethod
    def bound_summary_items(cls, value: list[str]) -> list[str]:
        if any(len(item) > SUMMARY_ITEM_MAX_CHARS for item in value):
            raise ValueError(f"summary items must be at most {SUMMARY_ITEM_MAX_CHARS} characters")
        return value


class ConversationSummary(BaseModel):
    conversation_id: str
    content: ConversationSummaryContent
    through_message_id: str
    through_created_at: str
    summarized_message_count: int = Field(ge=0)
    source_hash: str = Field(min_length=64, max_length=64)
    model_id: str | None = None
    version: int = Field(ge=1)
    created_at: str
    updated_at: str


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
