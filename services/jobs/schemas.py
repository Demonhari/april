from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MAX_JOB_PAYLOAD_BYTES = 16_384
MAX_JOB_RESULT_BYTES = 16_384
MAX_JOB_EVENT_CODE_CHARS = 160
MAX_JOB_EVENTS_PER_JOB = 100
MAX_JOB_LIST_LIMIT = 100
DEFAULT_JOB_LIST_LIMIT = 25
DEFAULT_LEASE_SECONDS = 30
MAX_LEASE_SECONDS = 300


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.CANCELLED,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.INTERRUPTED,
    }
)


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    job_type: str
    status: JobStatus
    payload_hash: str
    owner: str
    conversation_id: str | None = None
    project_id: str | None = None
    progress_percent: int = Field(ge=0, le=100)
    progress_code: str | None = Field(default=None, max_length=MAX_JOB_EVENT_CODE_CHARS)
    attempt_count: int = Field(ge=0)
    maximum_attempts: int = Field(ge=1)
    cancellation_requested: bool
    worker_id: str | None = None
    lease_acquired_at: str | None = None
    lease_expires_at: str | None = None
    heartbeat_at: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = Field(default=None, max_length=MAX_JOB_EVENT_CODE_CHARS)


class ClaimedJob(JobRecord):
    payload: dict[str, Any]


class JobEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    job_id: str
    event_type: str
    message_code: str | None = None
    progress_percent: int | None = Field(default=None, ge=0, le=100)
    created_at: str


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_type: str = Field(min_length=1, max_length=80)
    payload: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    approval_id: str | None = Field(default=None, min_length=1, max_length=128)


class JobListResponse(BaseModel):
    jobs: list[JobRecord]
    limit: int
    offset: int


class JobCancelResponse(BaseModel):
    job: JobRecord
    already_terminal: bool


class JobRetryResponse(BaseModel):
    job: JobRecord
    already_queued: bool
