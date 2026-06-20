from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ModelRole = Literal["brain", "coding", "reading", "creative", "reasoning", "system_action"]
ModelState = Literal["unavailable", "unloaded", "loading", "loaded", "unloading", "error"]
FinishReason = Literal["stop", "length", "error", "cancelled"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class GenerationOptions(BaseModel):
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_output_tokens")
    @classmethod
    def validate_max_output_tokens(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_output_tokens must be positive")
        return value


class ChatRequest(BaseModel):
    model_id: str
    messages: list[ChatMessage]
    options: GenerationOptions = Field(default_factory=GenerationOptions)
    request_id: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    request_id: str
    model_id: str
    content: str
    finish_reason: FinishReason = "stop"
    usage: Usage
    context_truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class LoadModelRequest(BaseModel):
    model_id: str
    request_id: str | None = None


class ModelOperationResponse(BaseModel):
    request_id: str
    model_id: str
    state: ModelState
    message: str


class ModelInfo(BaseModel):
    id: str
    name: str
    role: ModelRole
    backend: str
    path: str
    state: ModelState
    keep_loaded: bool
    context_size: int
    temperature: float
    max_output_tokens: int
    last_used_at: str | None = None
    load_error: str | None = None
    generations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    missing_path: bool = False


class RuntimeHealth(BaseModel):
    status: Literal["ok", "degraded"]
    backend: str
    models: list[ModelInfo]
    missing_models: list[str] = Field(default_factory=list)
    request_id: str


class StreamEvent(BaseModel):
    request_id: str
    event: Literal["meta", "token", "usage", "done", "error"]
    timestamp: str
    model_id: str
    payload: dict[str, Any]


class ErrorResponse(BaseModel):
    error: dict[str, Any]
