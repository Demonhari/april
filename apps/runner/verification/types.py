from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.april_runtime.model_registry import ModelDefinition

VerifyStatus = Literal["pass", "fail", "skip", "manual"]


@dataclass(slots=True)
class VerifyCheck:
    name: str
    ok: bool
    detail: str = ""
    status: VerifyStatus | None = None

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = "pass" if self.ok else "fail"


@dataclass(slots=True)
class ModelPlanEntry:
    """One configured model and whether the multi-model verifier can exercise it.

    Pure data so the discovery/skip decision is unit-testable without a real
    runtime: a missing file, an unreadable file, a non-chat (embedding) role, or
    an absent llama-cpp-python all yield ``available=False`` with an explicit
    ``reason`` — never a silent pass.
    """

    model: ModelDefinition
    path: Path
    available: bool
    reason: str | None = None

    @property
    def path_basename(self) -> str:
        return self.path.name


class BenchmarkResult(BaseModel):
    run_index: int
    ok: bool = True
    detail: str = ""
    load_time_seconds: float = 0.0
    warm_load_time_seconds: float | None = None
    first_token_latency_seconds: float | None = None
    generation_time_seconds: float = 0.0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    unload_success: bool = False
    unload_time_seconds: float | None = None
    process_rss_bytes: int | None = None
    peak_process_rss_bytes: int | None = None
    prompt_token_count: int | None = None
    prompt_eval_duration_seconds: float | None = None
    context_size: int = 1024
    backend_settings: dict[str, Any] = Field(default_factory=dict)


class WorkflowReportCheck(BaseModel):
    name: str
    ok: bool
    status: VerifyStatus
    detail: str = ""


class WorkflowVerificationReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["workflow"] = "workflow"
    generated_at: str
    # Redacted structural config fingerprint at generation time (staleness check).
    config_fingerprint: str | None = None
    summary: str = "degraded"
    real_model_verified: bool = False
    real_model_exercised: bool = False
    checks: list[WorkflowReportCheck] = Field(default_factory=list)
    checks_failed: int = 0
    check_failures: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    max_output_tokens: int | None = None


class MissingChatResultError(RuntimeError):
    """A /chat response completed without the scored result envelope."""
