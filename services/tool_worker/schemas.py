from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TOOL_WORKER_PROTOCOL_VERSION = 1
MAX_TOOL_WORKER_REQUEST_BYTES = 4 * 1024 * 1024
MAX_TOOL_WORKER_RESPONSE_BYTES = 512 * 1024
MAX_TOOL_WORKER_OUTPUT_BYTES = 100_000
MAX_TOOL_WORKER_TIMEOUT_SECONDS = 3600.0
MAX_TOOL_WORKER_ARGV_ITEMS = 64
MAX_TOOL_WORKER_ARGUMENT_CHARS = 8192

ToolWorkerOperation = Literal[
    "self_check",
    "cancel",
    "run_command",
    "test_runner",
    "patch_applier",
    "git_commit",
]


class ToolWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=128)
    capability: str = Field(min_length=32, max_length=256)
    operation: ToolWorkerOperation
    project_root: str = Field(min_length=1, max_length=4096)
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(gt=0, le=MAX_TOOL_WORKER_TIMEOUT_SECONDS)
    max_stdout_bytes: int = Field(ge=0, le=MAX_TOOL_WORKER_OUTPUT_BYTES)
    max_stderr_bytes: int = Field(ge=0, le=MAX_TOOL_WORKER_OUTPUT_BYTES)

    @field_validator("request_id")
    @classmethod
    def safe_request_id(cls, value: str) -> str:
        if any(char in value for char in "\r\n\x00"):
            raise ValueError("request_id contains invalid characters")
        return value


class ToolWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    request_id: str
    ok: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    status: str
    failure_code: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ToolWorkerHealth(BaseModel):
    protocol_version: int
    ready: bool
    self_check: bool
    socket_mode: str
