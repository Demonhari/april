from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelRole = Literal[
    "brain", "coding", "reading", "creative", "reasoning", "system_action", "embedding"
]
ModelState = Literal["unavailable", "unloaded", "loading", "loaded", "unloading", "error"]
FinishReason = Literal["stop", "length", "error", "cancelled"]

# Bounds applied to caller-supplied JSON schemas before they reach the low-level
# grammar compiler. They keep an untrusted request from handing llama.cpp an
# unbounded or pathologically nested document.
MAX_RESPONSE_FORMAT_SCHEMA_BYTES = 16_384
MAX_RESPONSE_FORMAT_SCHEMA_DEPTH = 24


def _json_schema_depth(value: Any, depth: int = 0) -> int:
    if depth > MAX_RESPONSE_FORMAT_SCHEMA_DEPTH:
        return depth
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, list):
        children = value  # type: ignore[assignment]
    else:
        return depth
    return max((_json_schema_depth(child, depth + 1) for child in children), default=depth)


class ResponseFormat(BaseModel):
    """Optional structured-output constraint for a chat generation.

    ``type="json_object"`` asks the backend to emit a single JSON object. When
    ``json_schema`` is supplied the backend constrains output to that JSON Schema
    where supported, degrading to prompt-plus-validation otherwise. The schema is
    size- and depth-limited so a caller cannot hand the grammar compiler an
    unbounded document.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "json_object"] = "json_object"
    json_schema: dict[str, Any] | None = None

    @field_validator("json_schema")
    @classmethod
    def validate_json_schema(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("json_schema must be JSON-serialisable") from exc
        if len(encoded.encode("utf-8")) > MAX_RESPONSE_FORMAT_SCHEMA_BYTES:
            raise ValueError(
                f"json_schema exceeds the {MAX_RESPONSE_FORMAT_SCHEMA_BYTES}-byte limit"
            )
        if _json_schema_depth(value) > MAX_RESPONSE_FORMAT_SCHEMA_DEPTH:
            raise ValueError("json_schema nesting is too deep")
        return value


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class GenerationOptions(BaseModel):
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None

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

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float | None) -> float | None:
        if value is not None and not 0 < value <= 1:
            raise ValueError("top_p must be between 0 and 1")
        return value


class ChatRequest(BaseModel):
    model_id: str
    messages: list[ChatMessage]
    options: GenerationOptions = Field(default_factory=GenerationOptions)
    response_format: ResponseFormat | None = None
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
    diagnostics: dict[str, Any] = Field(default_factory=dict)


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
    active_requests: int = 0
    generation_errors: int = 0
    recent_latency_ms: float | None = None
    recent_tokens_per_second: float | None = None
    loaded_at: str | None = None
    unloaded_at: str | None = None
    load_duration_ms: float | None = None
    idle_unload_seconds: float | None = None
    priority: int = 0
    threads: int | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    n_gpu_layers: int | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None


class RuntimeHealth(BaseModel):
    status: Literal["ok", "degraded"]
    backend: str
    models: list[ModelInfo]
    missing_models: list[str] = Field(default_factory=list)
    request_id: str
    loaded_model_count: int = 0
    active_requests: int = 0
    generation_error_count: int = 0
    embedding_model_id: str | None = None
    lifecycle_policy: dict[str, Any] = Field(default_factory=dict)
    process_rss_bytes: int | None = None
    process_peak_rss_bytes: int | None = None
    process_memory_estimated: bool = True


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1)
    model_id: str | None = None
    request_id: str | None = None


class EmbedResponse(BaseModel):
    request_id: str
    model_id: str
    dimensions: int
    embedding: list[float]


class StreamEvent(BaseModel):
    request_id: str
    event: Literal["meta", "token", "usage", "done", "error"]
    timestamp: str
    model_id: str
    payload: dict[str, Any]


class ErrorResponse(BaseModel):
    error: dict[str, Any]
