from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import AgentResult


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: str | None = None
    project_id: str | None = None
    repo_path: str | None = None


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
