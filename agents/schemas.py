from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LocalCitation(BaseModel):
    path: str
    start_line: int | None = None
    end_line: int | None = None


class ProposedChange(BaseModel):
    path: str
    summary: str
    patch_path: str | None = None


class AgentResult(BaseModel):
    status: Literal["ok", "pending_approval", "unavailable", "error"]
    final_message: str
    conversation_id: str | None = None
    tool_requests: list[dict[str, Any]] = Field(default_factory=list)
    local_citations: list[LocalCitation] = Field(default_factory=list)
    proposed_changes: list[ProposedChange] = Field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    name: str
    description: str
    model_id: str | None
    system_prompt_path: str
    allowed_tools: set[str] = Field(default_factory=set)
    blocked_tools: set[str] = Field(default_factory=set)
    memory_access_policy: str
    maximum_tool_iterations: int
    output_schema: str = "AgentResult"
    system_prompt: str
