"""Machine-readable, redacted multi-model GGUF verification report.

This module builds the acceptance report for
``run april verify --all-configured-models`` (alias ``--mac-readiness``). It is
separated from the verifier so the report shape and summary rules can be
unit-tested with scripted/fake per-model results without a real GGUF model.

Redaction is by construction, exactly like :mod:`apps.runner.mac_report`: every
field below is a basename, a count, a duration, a boolean, an accuracy, or an
eval total. No prompt content, generated text, tokens, secrets, or absolute
paths are ever included, and any path-looking skip reason is basename-redacted.
New fields must keep that invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.mac_report import (
    EnvironmentSnapshot,
    ReportSummary,
    ReportThresholds,
    RoutingReport,
    SkippedCheck,
    redact_reason,
)


class PerModelResult(BaseModel):
    """One configured model's verification outcome.

    A model that was not exercised (missing file or no llama-cpp-python) has
    ``available=False`` and a ``skipped_reason``; it can never count as passed.
    """

    model_id: str
    role: str
    backend: str
    path_basename: str | None = None
    quantization: str | None = None
    available: bool = False
    skipped_reason: str | None = None
    context_size: int | None = None
    load_success: bool = False
    load_duration_seconds: float | None = None
    chat_success: bool = False
    streaming_success: bool = False
    first_token_latency_seconds: float | None = None
    tokens_per_second: float | None = None
    output_token_count: int = 0
    unload_success: bool = False
    process_rss_bytes: int | None = None
    process_peak_rss_bytes: int | None = None
    # Brain-only structured checks (None for specialists).
    structured_brain_json_success: bool | None = None
    routing: RoutingReport | None = None
    # Specialist-only role-appropriate smoke prompt (None for the brain).
    smoke_success: bool | None = None

    @property
    def structural_ok(self) -> bool:
        return (
            self.available
            and self.load_success
            and self.chat_success
            and self.streaming_success
            and self.unload_success
        )


class SpecialistSwitchReport(BaseModel):
    """Brain-resident → coding load/unload → reading load/unload → brain usable.

    Verifies specialists can be cycled without losing the resident brain, which
    is the core multi-model lifecycle guarantee on a memory-constrained Mac.
    """

    attempted: bool = False
    brain_loaded: bool = False
    coding_loaded: bool = False
    coding_unloaded: bool = False
    reading_loaded: bool = False
    reading_unloaded: bool = False
    brain_usable_after: bool = False

    @property
    def success(self) -> bool:
        return (
            self.attempted
            and self.brain_loaded
            and self.coding_loaded
            and self.coding_unloaded
            and self.reading_loaded
            and self.reading_unloaded
            and self.brain_usable_after
        )


class MultiModelVerificationReport(BaseModel):
    schema_version: int = 1
    # Discriminator so the report viewer can classify this file without guessing.
    report_type: Literal["multi_model"] = "multi_model"
    generated_at: str
    os: str
    cpu_architecture: str
    python_version: str
    runtime_backend: str
    # True ONLY for a real backend with at least one structurally-passing model.
    # A fake/simulated run can never set this true, so simulation is never
    # mistaken for real-model verification.
    real_model_verified: bool = False
    models: list[PerModelResult] = Field(default_factory=list)
    specialist_switch: SpecialistSwitchReport | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    threshold_failures: list[str] = Field(default_factory=list)
    skipped: list[SkippedCheck] = Field(default_factory=list)
    models_attempted: int = 0
    models_available: int = 0
    models_passed: int = 0
    checks_failed: int = 0
    summary: ReportSummary = "degraded"


def per_model_threshold_failures(result: PerModelResult, thresholds: ReportThresholds) -> list[str]:
    failures: list[str] = []
    label = result.model_id
    tps = result.tokens_per_second
    min_tps = thresholds.min_tokens_per_second
    if min_tps is not None and tps is not None and tps < min_tps:
        failures.append(f"{label}: tokens_per_second {tps:.2f} below minimum {min_tps:.2f}")
    load = result.load_duration_seconds
    max_load = thresholds.max_load_seconds
    if max_load is not None and load is not None and load > max_load:
        failures.append(f"{label}: load_duration_seconds {load:.2f} above maximum {max_load:.2f}")
    latency = result.first_token_latency_seconds
    max_latency = thresholds.max_first_token_latency_seconds
    if max_latency is not None and latency is not None and latency > max_latency:
        failures.append(
            f"{label}: first_token_latency_seconds {latency:.2f} above maximum {max_latency:.2f}"
        )
    rss = result.process_rss_bytes
    max_rss = thresholds.max_rss_mb
    if max_rss is not None and rss is not None and rss / (1024 * 1024) > max_rss:
        failures.append(
            f"{label}: process_rss_mb {rss / (1024 * 1024):.1f} above maximum {max_rss:.1f}"
        )
    return failures


def _summary(
    *,
    attempted: bool,
    real_model_verified: bool,
    simulated: bool,
    checks_failed: int,
    runtime_error: bool,
    require_real_model: bool,
    threshold_failures_present: bool,
    optional_skipped: bool,
    switch_ok: bool,
) -> ReportSummary:
    # Honesty first: a runtime error or any structural failure is a fail; a
    # required-but-absent real model is a fail.
    if runtime_error or checks_failed > 0:
        return "fail"
    if require_real_model and not real_model_verified:
        return "fail"
    if not attempted:
        # No real model was exercised at all.
        return "fail" if require_real_model else "degraded"
    # A fake/simulated run is structurally fine at best — never real "pass".
    if simulated or not real_model_verified:
        return "degraded"
    if threshold_failures_present or optional_skipped or not switch_ok:
        return "degraded"
    return "pass"


def build_multi_model_report(
    *,
    environment: EnvironmentSnapshot,
    runtime_backend: str,
    results: list[PerModelResult],
    specialist_switch: SpecialistSwitchReport | None,
    extra_skipped: list[SkippedCheck] | None = None,
    thresholds: ReportThresholds | None = None,
    require_real_model: bool = False,
    runtime_error: bool = False,
) -> MultiModelVerificationReport:
    """Assemble a redacted multi-model acceptance report from per-model results.

    ``results`` lists every configured model (available or skipped). Skipped
    models keep ``available=False`` and a ``skipped_reason`` so they are reported
    as skipped, never passed.
    """
    active_thresholds = thresholds or ReportThresholds()
    simulated = runtime_backend == "fake"

    # Redact any path-looking skip reason in-place (basename only).
    for result in results:
        if result.skipped_reason:
            result.skipped_reason = redact_reason(result.skipped_reason)

    attempted = [result for result in results if result.available]
    structural_failures = sum(1 for result in attempted if not result.structural_ok)
    switch_failed = (
        specialist_switch is not None
        and specialist_switch.attempted
        and not specialist_switch.success
    )
    checks_failed = structural_failures + (1 if switch_failed else 0)

    failures: list[str] = []
    for result in attempted:
        failures.extend(per_model_threshold_failures(result, active_thresholds))

    models_passed = sum(1 for result in attempted if result.structural_ok)
    real_model_verified = (not simulated) and any(result.structural_ok for result in attempted)

    skipped = list(extra_skipped or [])
    skipped.extend(
        SkippedCheck(name=result.model_id, reason=redact_reason(result.skipped_reason))
        for result in results
        if not result.available and result.skipped_reason
    )

    summary = _summary(
        attempted=bool(attempted),
        real_model_verified=real_model_verified,
        simulated=simulated,
        checks_failed=checks_failed,
        runtime_error=runtime_error,
        require_real_model=require_real_model,
        threshold_failures_present=bool(failures),
        optional_skipped=bool(skipped),
        switch_ok=specialist_switch is None or specialist_switch.success,
    )

    return MultiModelVerificationReport(
        generated_at=environment.generated_at,
        os=environment.os,
        cpu_architecture=environment.cpu_architecture,
        python_version=environment.python_version,
        runtime_backend=runtime_backend,
        real_model_verified=real_model_verified,
        models=results,
        specialist_switch=specialist_switch,
        thresholds=active_thresholds.model_dump(exclude_none=True),
        threshold_failures=failures,
        skipped=skipped,
        models_attempted=len(attempted),
        models_available=len(attempted),
        models_passed=models_passed,
        checks_failed=checks_failed,
        summary=summary,
    )


def write_multi_model_report(report: MultiModelVerificationReport, path: Path) -> Path:
    """Write the report JSON to ``path`` (creating parents). Returns the path."""
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved
