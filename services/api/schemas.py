from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import AgentResult


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: str | None = None
    project_id: str | None = None
    repo_path: str | None = None
    mode: Literal["standard", "deep", "council"] = "standard"


class ChatResponse(BaseModel):
    request_id: str
    result: AgentResult


class AgentRunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structured: bool = True


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: str | None = None
    project_id: str | None = None
    repo_path: str | None = None
    options: AgentRunOptions = Field(default_factory=AgentRunOptions)


class ToolApprovalAction(BaseModel):
    approval_id: str
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class ToolRequestEnvelope(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    agent: str = "general_agent"


class MemorySearchResponse(BaseModel):
    results: list[dict[str, Any]]


class MemoryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=10_000)
    memory_type: Literal["fact", "preference", "project", "note"] = "fact"
    project_id: str | None = None
    source_conversation_id: str | None = None
    reason: str = Field(min_length=1, max_length=500)


class WakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["voice", "terminal", "desktop", "hotkey", "socket"]
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    text: str | None = Field(default=None, min_length=1, max_length=50_000)
    reason: str | None = Field(default=None, max_length=200)
    # Optional v2 fields; older clients omit them and behaviour is unchanged.
    captured_at: str | None = Field(default=None, max_length=64)
    session_hint: str | None = Field(default=None, max_length=128)


class SessionAttachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["voice", "terminal", "desktop", "hotkey", "socket"] = "terminal"


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["good", "bad"]
    reason: str | None = Field(default=None, max_length=2_000)
    conversation_id: str | None = None
    agent_run_id: str | None = None


class PlaybookRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None
    project_id: str | None = None


class EvolutionRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    version: int = Field(ge=1)


class DatasetExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class OverlayApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class ProjectCreateRequest(BaseModel):
    path: str
    name: str | None = None


class DocumentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ReminderCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=500)
    due_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    vector_index: dict[str, Any]
    voice: dict[str, Any]
    runtime_url: str
