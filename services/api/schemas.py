from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.schemas import AgentResult


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    result: AgentResult


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


class HealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    vector_index: dict[str, Any]
    voice: dict[str, Any]
    runtime_url: str
