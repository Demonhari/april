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
    smoke_schema_valid: bool | None = None
    smoke_kind: str | None = None

    @property
    def structural_ok(self) -> bool:
        return (
            self.available
            and self.load_success
            and self.chat_success
            and self.streaming_success
            and self.unload_success
        )

    def acceptance_failures(self, thresholds: ReportThresholds) -> list[str]:
        failures: list[str] = []
        label = self.model_id
        if not self.available:
            failures.append(f"{label}: model was not exercised")
            return failures

        required = {
            "load": self.load_success,
            "chat": self.chat_success,
            "streaming": self.streaming_success,
            "unload": self.unload_success,
        }
        for name, ok in required.items():
            if not ok:
                failures.append(f"{label}: {name} check failed")

        if self.role == "brain":
            if self.structured_brain_json_success is not True:
                failures.append(f"{label}: structured Brain JSON check failed")
        elif self.smoke_success is not True:
            failures.append(f"{label}: specialist role smoke check failed")
        elif self.smoke_schema_valid is False:
            failures.append(f"{label}: specialist role smoke schema check failed")
        return failures

    def acceptance_ok(self, thresholds: ReportThresholds) -> bool:
        return not self.acceptance_failures(thresholds)


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
    # Redacted structural config fingerprint at generation time (staleness check).
    config_fingerprint: str | None = None
    os: str
    cpu_architecture: str
    python_version: str
    runtime_backend: str
    # True ONLY for a real backend with the required core model set verified.
    # Partial success is reported separately so it cannot be mistaken for
    # target-Mac real-model readiness.
    real_model_verified: bool = False
    real_models_exercised: int = 0
    real_models_passed: int = 0
    any_real_model_exercised: bool = False
    any_real_model_passed: bool = False
    core_model_set_verified: bool = False
    all_available_models_verified: bool = False
    all_configured_models_verified: bool = False
    verification_level: Literal["none", "partial", "core", "all"] = "none"
    models: list[PerModelResult] = Field(default_factory=list)
    specialist_switch: SpecialistSwitchReport | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    threshold_failures: list[str] = Field(default_factory=list)
    skipped: list[SkippedCheck] = Field(default_factory=list)
    models_attempted: int = 0
    models_available: int = 0
    models_passed: int = 0
    checks_failed: int = 0
    check_failures: list[str] = Field(default_factory=list)
    summary: ReportSummary = "degraded"


def per_model_threshold_failures(result: PerModelResult, thresholds: ReportThresholds) -> list[str]:
    failures: list[str] = []
    label = result.model_id
    if result.role == "brain" and result.routing is not None and result.routing.total > 0:
        min_accuracy = thresholds.min_routing_accuracy
        if min_accuracy is not None and result.routing.accuracy < min_accuracy:
            failures.append(
                f"{label}: routing accuracy {result.routing.accuracy:.2f} "
                f"below minimum {min_accuracy:.2f}"
            )
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


def _active_thresholds(thresholds: ReportThresholds | None) -> ReportThresholds:
    active = thresholds or ReportThresholds()
    if active.min_routing_accuracy is None:
        return active.model_copy(update={"min_routing_accuracy": 0.90})
    return active


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


def _role_passes(results: list[PerModelResult], thresholds: ReportThresholds) -> dict[str, bool]:
    roles: dict[str, bool] = {}
    for result in results:
        roles[result.role] = result.acceptance_ok(thresholds)
    return roles


def _core_model_set_verified(
    *,
    results: list[PerModelResult],
    thresholds: ReportThresholds,
    simulated: bool,
) -> bool:
    if simulated:
        return False
    configured_roles = {result.role for result in results}
    role_passes = _role_passes(results, thresholds)
    if role_passes.get("brain") is not True:
        return False
    for role in ("coding", "reading"):
        if role in configured_roles and role_passes.get(role) is not True:
            return False
    return True


def _specialist_switch_ok(
    *,
    results: list[PerModelResult],
    specialist_switch: SpecialistSwitchReport | None,
) -> bool:
    specialist_roles = {"coding", "reading"}
    switch_required = any(result.role in specialist_roles for result in results)
    if not switch_required:
        return True
    return bool(specialist_switch and specialist_switch.success)


def _verification_level(
    *,
    any_real_model_passed: bool,
    core_model_set_verified: bool,
    all_configured_models_verified: bool,
) -> Literal["none", "partial", "core", "all"]:
    if all_configured_models_verified:
        return "all"
    if core_model_set_verified:
        return "core"
    if any_real_model_passed:
        return "partial"
    return "none"


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
    config_fingerprint: str | None = None,
) -> MultiModelVerificationReport:
    """Assemble a redacted multi-model acceptance report from per-model results.

    ``results`` lists every configured model (available or skipped). Skipped
    models keep ``available=False`` and a ``skipped_reason`` so they are reported
    as skipped, never passed.
    """
    active_thresholds = _active_thresholds(thresholds)
    simulated = runtime_backend == "fake"

    # Redact any path-looking skip reason in-place (basename only).
    for result in results:
        if result.skipped_reason:
            result.skipped_reason = redact_reason(result.skipped_reason)

    attempted = [result for result in results if result.available]
    check_failures: list[str] = []
    for result in attempted:
        check_failures.extend(result.acceptance_failures(active_thresholds))
    switch_failed = (
        specialist_switch is not None
        and specialist_switch.attempted
        and not specialist_switch.success
    )
    if switch_failed:
        check_failures.append("specialist switching failed")
    checks_failed = len(check_failures)

    failures: list[str] = []
    for result in attempted:
        failures.extend(per_model_threshold_failures(result, active_thresholds))

    models_passed = sum(1 for result in attempted if result.acceptance_ok(active_thresholds))
    real_models_exercised = 0 if simulated else len(attempted)
    real_models_passed = 0 if simulated else models_passed
    any_real_model_exercised = real_models_exercised > 0
    switch_ok = _specialist_switch_ok(results=results, specialist_switch=specialist_switch)
    core_model_set_verified = _core_model_set_verified(
        results=results,
        thresholds=active_thresholds,
        simulated=simulated,
    )
    any_real_model_passed = real_models_passed > 0
    real_model_verified = core_model_set_verified
    all_available_models_verified = (
        not simulated
        and bool(attempted)
        and all(result.acceptance_ok(active_thresholds) for result in attempted)
        and switch_ok
    )
    all_configured_models_verified = (
        not simulated
        and bool(results)
        and len(attempted) == len(results)
        and all(result.available for result in results)
        and all(result.acceptance_ok(active_thresholds) for result in results)
        and switch_ok
    )
    verification_level = _verification_level(
        any_real_model_passed=any_real_model_passed,
        core_model_set_verified=core_model_set_verified,
        all_configured_models_verified=all_configured_models_verified,
    )

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
        switch_ok=switch_ok,
    )

    return MultiModelVerificationReport(
        generated_at=environment.generated_at,
        config_fingerprint=config_fingerprint,
        os=environment.os,
        cpu_architecture=environment.cpu_architecture,
        python_version=environment.python_version,
        runtime_backend=runtime_backend,
        real_model_verified=real_model_verified,
        real_models_exercised=real_models_exercised,
        real_models_passed=real_models_passed,
        any_real_model_exercised=any_real_model_exercised,
        any_real_model_passed=any_real_model_passed,
        core_model_set_verified=core_model_set_verified,
        all_available_models_verified=all_available_models_verified,
        all_configured_models_verified=all_configured_models_verified,
        verification_level=verification_level,
        models=results,
        specialist_switch=specialist_switch,
        thresholds=active_thresholds.model_dump(exclude_none=True),
        threshold_failures=failures,
        skipped=skipped,
        models_attempted=len(attempted),
        models_available=len(attempted),
        models_passed=models_passed,
        checks_failed=checks_failed,
        check_failures=check_failures,
        summary=summary,
    )


def write_multi_model_report(report: MultiModelVerificationReport, path: Path) -> Path:
    """Write the report JSON to ``path`` (creating parents). Returns the path."""
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved
