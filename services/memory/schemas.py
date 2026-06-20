from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Project(BaseModel):
    id: str
    path: str
    name: str
    created_at: str


class MemoryRecord(BaseModel):
    id: str
    content: str
    kind: str
    project_id: str | None = None
    reason: str
    created_at: str


class Conversation(BaseModel):
    id: str
    title: str | None = None
    created_at: str


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


class ReminderRecord(BaseModel):
    id: str
    content: str
    due_at: str | None = None
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
