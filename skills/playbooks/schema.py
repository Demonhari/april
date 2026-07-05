from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlaybookStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    agent_id: str | None = None


class PlaybookStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: int = Field(default=0, ge=0)
    success: int = Field(default=0, ge=0)
    last_run: str | None = None


class PlaybookDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    status: Literal["candidate", "active"] = "candidate"
    source: Literal["learned", "authored"] = "authored"
    required_permission_level: int | None = Field(default=None, ge=0, le=5)
    stats: PlaybookStats = Field(default_factory=PlaybookStats)
    agent_id: str = "general_agent"
    trigger_examples: list[str] = Field(default_factory=list, max_length=20)
    steps: list[PlaybookStep] = Field(min_length=1, max_length=20)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}", value):
            raise ValueError("playbook id must be a safe basename")
        return value
