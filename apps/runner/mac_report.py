"""Machine-readable, redacted target-Mac verification report.

This module builds a JSON-serialisable acceptance report for
``run april verify --target-mac``. It is deliberately separated from the
verifier so the report shape can be unit-tested with scripted/fake data without
a real GGUF model.

Redaction is by construction: every field below is a basename, a count, a
duration, a boolean, or an eval total. No prompt content, generated text,
tokens, secrets, or absolute paths are ever included. New fields must keep that
invariant.
"""

from __future__ import annotations

import platform
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from april_common.time import utc_now_iso

ReportSummary = Literal["pass", "degraded", "fail"]


class ReportThresholds(BaseModel):
    """Optional performance gates. Structural correctness is the default
    acceptance criterion; thresholds only downgrade a structurally-passing run
    to ``degraded`` so per-Mac speed differences never hard-fail acceptance."""

    min_tokens_per_second: float | None = None
    max_load_seconds: float | None = None
    max_first_token_latency_seconds: float | None = None
    max_rss_mb: float | None = None
    min_routing_accuracy: float | None = None


# An absolute (or home-relative) filesystem path with at least two segments, e.g.
# ``/Users/hari/april/models/granite-3.3-2b-q4_k_m.gguf``. A single ``/foo`` and
# incidental text like ``read/write`` never match (two+ segments required).
_ABSOLUTE_PATH_RE = re.compile(r"~?(?:/[\w.\-]+){2,}/?")


def redact_reason(text: str | None) -> str:
    """Collapse any absolute-path-looking substring to its basename.

    Skip/fail reasons are useful with the full path in the terminal, but the
    machine-readable report must never carry directory structure. This keeps the
    report's "basenames only" invariant even when a reason embeds a path.
    """
    if not text:
        return text or ""

    def _basename(match: re.Match[str]) -> str:
        name = Path(match.group(0)).name
        return name or match.group(0)

    return _ABSOLUTE_PATH_RE.sub(_basename, text)


class EnvironmentSnapshot(BaseModel):
    generated_at: str
    os: str
    cpu_architecture: str
    python_version: str


class RealModelReport(BaseModel):
    attempted: bool = False
    model_id: str | None = None
    role: str | None = None
    path_basename: str | None = None
    quantization: str | None = None
    context_size: int | None = None
    load_success: bool = False
    load_duration_seconds: float | None = None
    chat_success: bool = False
    structured_brain_json_success: bool = False
    streaming_success: bool = False
    first_token_latency_seconds: float | None = None
    unload_success: bool = False
    output_token_count: int = 0
    tokens_per_second: float | None = None
    process_rss_bytes: int | None = None
    process_peak_rss_bytes: int | None = None


class RoutingReport(BaseModel):
    total: int = 0
    passed: int = 0
    accuracy: float = 0.0


class SkippedCheck(BaseModel):
    name: str
    reason: str


class MacVerificationReport(BaseModel):
    schema_version: int = 1
    generated_at: str
    os: str
    cpu_architecture: str
    python_version: str
    runtime_backend: str
    real_model: RealModelReport
    routing: RoutingReport | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    threshold_failures: list[str] = Field(default_factory=list)
    skipped: list[SkippedCheck] = Field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    summary: ReportSummary = "degraded"


def environment_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        generated_at=utc_now_iso(),
        os=f"{platform.system()} {platform.release()}".strip(),
        cpu_architecture=platform.machine(),
        python_version=platform.python_version(),
    )


def quantization_from_basename(basename: str | None) -> str | None:
    """Best-effort quantisation tag from a GGUF basename (e.g. ``q4_k_m``).

    Only the basename is inspected, so no path information leaks into the report.
    """
    if not basename:
        return None
    match = re.search(r"(q\d+(?:_[a-z0-9]+)*|f16|f32|bf16)", basename, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def routing_report_from_results(results: Sequence[object]) -> RoutingReport:
    total = len(results)
    passed = sum(1 for result in results if getattr(result, "ok", False))
    accuracy = round(passed / total, 4) if total else 0.0
    return RoutingReport(total=total, passed=passed, accuracy=accuracy)


def threshold_failures(real_model: RealModelReport, thresholds: ReportThresholds) -> list[str]:
    failures: list[str] = []
    tps = real_model.tokens_per_second
    min_tps = thresholds.min_tokens_per_second
    if min_tps is not None and tps is not None and tps < min_tps:
        failures.append(f"tokens_per_second {tps:.2f} below minimum {min_tps:.2f}")
    load = real_model.load_duration_seconds
    max_load = thresholds.max_load_seconds
    if max_load is not None and load is not None and load > max_load:
        failures.append(f"load_duration_seconds {load:.2f} above maximum {max_load:.2f}")
    latency = real_model.first_token_latency_seconds
    max_latency = thresholds.max_first_token_latency_seconds
    if max_latency is not None and latency is not None and latency > max_latency:
        failures.append(
            f"first_token_latency_seconds {latency:.2f} above maximum {max_latency:.2f}"
        )
    rss = real_model.process_rss_bytes
    max_rss = thresholds.max_rss_mb
    if max_rss is not None and rss is not None and rss / (1024 * 1024) > max_rss:
        failures.append(f"process_rss_mb {rss / (1024 * 1024):.1f} above maximum {max_rss:.1f}")
    return failures


def _summary(
    real_model: RealModelReport,
    *,
    checks_failed: int,
    threshold_failures_present: bool,
    require_real_model: bool,
) -> ReportSummary:
    # Honesty first: a hard check failure, or a required-but-absent real model,
    # is a fail. A run that never exercised a real model can never be "pass".
    if checks_failed > 0:
        return "fail"
    if require_real_model and not (real_model.attempted and real_model.load_success):
        return "fail"
    if not real_model.attempted:
        return "degraded"
    structural_ok = (
        real_model.load_success
        and real_model.chat_success
        and real_model.streaming_success
        and real_model.unload_success
    )
    if not structural_ok:
        return "fail"
    if threshold_failures_present:
        return "degraded"
    return "pass"


def build_mac_report(
    *,
    environment: EnvironmentSnapshot,
    runtime_backend: str,
    real_model: RealModelReport,
    routing: RoutingReport | None,
    skipped: list[SkippedCheck],
    checks_passed: int,
    checks_failed: int,
    thresholds: ReportThresholds | None = None,
    require_real_model: bool = False,
) -> MacVerificationReport:
    active_thresholds = thresholds or ReportThresholds()
    failures = threshold_failures(real_model, active_thresholds)
    # Basename-redact skip reasons so an embedded absolute path never reaches the
    # on-disk report (the terminal table still shows the full path).
    skipped = [SkippedCheck(name=item.name, reason=redact_reason(item.reason)) for item in skipped]
    summary = _summary(
        real_model,
        checks_failed=checks_failed,
        threshold_failures_present=bool(failures),
        require_real_model=require_real_model,
    )
    return MacVerificationReport(
        generated_at=environment.generated_at,
        os=environment.os,
        cpu_architecture=environment.cpu_architecture,
        python_version=environment.python_version,
        runtime_backend=runtime_backend,
        real_model=real_model,
        routing=routing,
        thresholds=active_thresholds.model_dump(exclude_none=True),
        threshold_failures=failures,
        skipped=skipped,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        summary=summary,
    )


def write_report(report: MacVerificationReport, path: Path) -> Path:
    """Write the report JSON to ``path`` (creating parents). Returns the path."""
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved
