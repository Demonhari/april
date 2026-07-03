from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from services.april_runtime.schemas import (
    ChatMessage,
    ChatResponse,
    GenerationOptions,
    ResponseFormat,
)

ArchiveMemoryKind = Literal[
    "fact",
    "preference",
    "correction",
    "project_state",
    "skill_note",
    "relationship",
    "open_loop",
]


class ArchiveMemoryCandidate(BaseModel):
    kind: ArchiveMemoryKind
    content: str = Field(min_length=1, max_length=1_000)
    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class ArchiveCandidateEnvelope(BaseModel):
    memories: list[ArchiveMemoryCandidate] = Field(default_factory=list, max_length=50)


class ArchiveRuntimeClient(Protocol):
    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        response_format: ResponseFormat | None = None,
        request_id: str | None = None,
    ) -> ChatResponse: ...


class ArchiveAgent:
    def __init__(
        self,
        runtime_client: ArchiveRuntimeClient,
        *,
        model_id: str,
        prompt_path: Path | None = None,
    ) -> None:
        self.runtime_client = runtime_client
        self.model_id = model_id
        self.prompt_path = prompt_path or Path(__file__).with_name("prompt.md")

    async def extract(
        self,
        transcript: str,
        *,
        request_id: str | None = None,
    ) -> list[ArchiveMemoryCandidate]:
        response = await self.runtime_client.chat(
            model_id=self.model_id,
            messages=[
                ChatMessage(role="system", content=self.prompt_path.read_text(encoding="utf-8")),
                ChatMessage(role="user", content=transcript),
            ],
            request_id=request_id,
        )
        try:
            payload = json.loads(response.content)
            envelope = ArchiveCandidateEnvelope.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return []
        return envelope.memories
