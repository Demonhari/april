"""Single MacBook Pro acceptance gate for APRIL.

``run april acceptance`` is the one command an operator runs to answer: *is this
Mac actually ready?* It composes the existing, separately-unit-tested checks
rather than re-implementing them:

* configuration validation (:func:`april_common.config_validation.validate_configuration`),
* offline readiness (:func:`apps.runner.readiness.build_readiness_report`),
* deterministic fake-backend verification (:func:`apps.runner.verify.run_fake_verification`),
* optional all-configured real-model verification
  (:func:`apps.runner.verify.run_all_configured_models_verification`),
* optional live push-to-talk voice verification (injected runner), and
* optional live wake-word verification (injected runner).

It then folds the results into a single ``pass`` / ``warning`` / ``fail`` status
with copy-pasteable next actions. The report is redacted by construction — every
field is a boolean, a count, a length, a status string, a check name, or a
command. No tokens, transcripts, generated text, or absolute paths are stored,
and any path-looking detail is reduced to its basename via
:func:`apps.runner.mac_report.redact_reason`.

The orchestrator is dependency-injected and separated from the CLI so it can be
unit-tested with fake/mocked verifier functions and temporary configs, with no
GGUF, llama-cpp-python, microphone, speaker, whisper.cpp, Piper, openWakeWord,
network, or Homebrew required.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.mac_report import ReportThresholds, SkippedCheck, redact_reason
from apps.runner.readiness import ReadinessReport, build_readiness_report
from apps.runner.verify import run_all_configured_models_verification, run_fake_verification
from apps.runner.voice_live import VoiceLiveReport
from apps.runner.wake_live import WakeWordLiveReport
from april_common.config_validation import validate_configuration
from april_common.time import utc_now, utc_now_iso

FinalStatus = Literal["pass", "warning", "fail"]
# Honest ladder of what acceptance actually proved on this Mac. ``fake_sanity`` is
# the floor (fake plumbing only); each higher rung additionally requires the named
# checks to have been *requested and passed*.
AcceptanceLevel = Literal[
    "fake_sanity", "real_models", "real_models_plus_voice", "full_wake_voice"
]
ServiceMode = Literal["none", "real", "fake"]

_VERIFY_REAL = (
    "run april verify --all-configured-models --require-real-model "
    "--report data/verification/mac-readiness.json"
)
_VERIFY_VOICE_LIVE = "run april voice verify-live --report data/verification/voice-live.json"
_VERIFY_WAKE_LIVE = "run april voice verify-wake-live --report data/verification/wake-live.json"
_VOICE_DOCTOR = "run april voice doctor"
_CONFIG_VALIDATE = "run april config validate"
_REAL_NOT_REQUIRED = (
    "Real-model verification was not requested; re-run with --require-real-models "
    "to load and exercise every configured GGUF model."
)
_SANITY_ONLY = (
    "This is fake/local sanity only, not Mac readiness. Re-run with "
    "--require-real-models for real-model acceptance, or pass --allow-sanity-pass "
    "to accept a clean fake-only run as a pass."
)
_REAL_BACKEND_FAKE = (
    "Runtime backend is fake; real-model acceptance is impossible. Set "
    "runtime.backend=llama_cpp (or unset APRIL_RUNTIME_BACKEND=fake) and install "
    "pip install -e '.[runtime]'."
)


class AcceptanceFlagError(ValueError):
    """Raised for an incompatible combination of acceptance flags."""


def validate_acceptance_flags(
    *, require_real_models: bool, start_services: bool, fake_services: bool
) -> None:
    """Reject contradictory acceptance flag combinations before doing any work.

    A fake runtime can never verify real models, so ``--fake-services`` may not be
    combined with ``--require-real-models``; and fake services only make sense when
    acceptance is also starting services.
    """
    if fake_services and require_real_models:
        raise AcceptanceFlagError(
            "Cannot combine --fake-services with --require-real-models: a fake runtime "
            "cannot verify real models."
        )
    if fake_services and not start_services:
        raise AcceptanceFlagError("--fake-services requires --start-services.")


class AcceptanceEnvironment(BaseModel):
    os: str
    cpu_architecture: str
    python_version: str
    deployment: str
    llama_cpp_python_available: bool
    runtime_is_fake: bool


class FakeVerificationSummary(BaseModel):
    ran: bool = False
    checks_total: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    failures: list[str] = Field(default_factory=list)
    summary: Literal["pass", "fail", "skipped"] = "skipped"


class ReadinessSummary(BaseModel):
    real_model_ready: bool
    voice_enabled: bool
    voice_ready: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RealModelSummary(BaseModel):
    summary: str
    verification_level: str
    real_model_verified: bool
    models_attempted: int
    models_available: int
    models_passed: int
    checks_failed: int
    check_failures: list[str] = Field(default_factory=list)
    skipped: list[SkippedCheck] = Field(default_factory=list)


class VoiceLiveSummary(BaseModel):
    summary: str
    recording_success: bool
    stt_success: bool
    transcript_length: int
    tts_success: bool
    playback_user_confirmed: bool
    voice_live_verified: bool


class WakeWordLiveSummary(BaseModel):
    summary: str
    doctor_status: str
    wake_word_configured: bool
    wake_word_detected: bool
    recording_success: bool
    stt_success: bool
    transcript_length: int
    normalized_transcript_length: int
    api_success: bool
    tts_success: bool
    playback_user_confirmed: bool
    retained_debug_audio: bool
    wake_word_live_verified: bool


class ServicesSummary(BaseModel):
    requested: bool = False
    mode: ServiceMode = "none"
    started_by_acceptance: bool = False
    stopped_after_acceptance: bool = False
    api_reachable: bool = False
    runtime_reachable: bool = False
    # not_requested | already_running | ok | failed
    startup_status: str = "not_requested"
    # not_applicable | stopped | kept_running | failed
    shutdown_status: str = "not_applicable"


class AcceptanceReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["acceptance"] = "acceptance"
    generated_at: str
    environment: AcceptanceEnvironment
    runtime_backend: str
    acceptance_level: AcceptanceLevel = "fake_sanity"
    config_valid: bool
    config_errors: list[str] = Field(default_factory=list)
    requested: dict[str, bool] = Field(default_factory=dict)
    fake_verification: FakeVerificationSummary
    readiness: ReadinessSummary
    real_model_verification: RealModelSummary | None = None
    voice_live: VoiceLiveSummary | None = None
    wake_word_live: WakeWordLiveSummary | None = None
    services: ServicesSummary = Field(default_factory=ServicesSummary)
    final_status: FinalStatus = "fail"
    next_actions: list[str] = Field(default_factory=list)


def default_acceptance_report_path(home: Path) -> Path:
    """Default ``--write-report`` location, under the Git-ignored verification dir."""
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return home.expanduser() / "data" / "verification" / f"acceptance-{stamp}.json"


def write_acceptance_report(report: AcceptanceReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


def _environment(readiness: ReadinessReport) -> AcceptanceEnvironment:
    return AcceptanceEnvironment(
        os=readiness.os,
        cpu_architecture=readiness.cpu_architecture,
        python_version=readiness.python_version,
        deployment=readiness.environment,
        llama_cpp_python_available=readiness.llama_cpp_python_available,
        runtime_is_fake=readiness.runtime_is_fake,
    )


def _fake_summary(home: Path, *, config_valid: bool) -> FakeVerificationSummary:
    # Deterministic fake verification is impossible if the configuration itself
    # does not load; report it as skipped rather than crashing.
    if not config_valid:
        return FakeVerificationSummary(ran=False, summary="skipped")
    checks = run_fake_verification(home)
    total = len(checks)
    passed = sum(1 for check in checks if check.ok)
    failed = total - passed
    failures = [f"{check.name}: {redact_reason(check.detail)}" for check in checks if not check.ok]
    return FakeVerificationSummary(
        ran=True,
        checks_total=total,
        checks_passed=passed,
        checks_failed=failed,
        failures=failures,
        summary="pass" if failed == 0 else "fail",
    )


def _real_model_summary(
    home: Path,
    *,
    max_output_tokens: int,
    timeout: float,
    thresholds: ReportThresholds | None,
) -> RealModelSummary:
    verifier = run_all_configured_models_verification(
        home,
        require_real_model=True,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        thresholds=thresholds,
    )
    report = verifier.build_report()
    return RealModelSummary(
        summary=report.summary,
        verification_level=report.verification_level,
        real_model_verified=report.real_model_verified,
        models_attempted=report.models_attempted,
        models_available=report.models_available,
        models_passed=report.models_passed,
        checks_failed=report.checks_failed,
        check_failures=[redact_reason(failure) for failure in report.check_failures],
        skipped=[
            SkippedCheck(name=item.name, reason=redact_reason(item.reason))
            for item in report.skipped
        ],
    )


def _voice_summary(report: VoiceLiveReport) -> VoiceLiveSummary:
    return VoiceLiveSummary(
        summary=report.summary,
        recording_success=report.recording_success,
        stt_success=report.stt_success,
        transcript_length=report.transcript_length,
        tts_success=report.tts_success,
        playback_user_confirmed=report.playback_user_confirmed,
        voice_live_verified=report.voice_live_verified,
    )


def _wake_summary(report: WakeWordLiveReport) -> WakeWordLiveSummary:
    return WakeWordLiveSummary(
        summary=report.summary,
        doctor_status=report.doctor_status,
        wake_word_configured=report.wake_word_configured,
        wake_word_detected=report.wake_word_detected,
        recording_success=report.recording_success,
        stt_success=report.stt_success,
        transcript_length=report.transcript_length,
        normalized_transcript_length=report.normalized_transcript_length,
        api_success=report.api_success,
        tts_success=report.tts_success,
        playback_user_confirmed=report.playback_user_confirmed,
        retained_debug_audio=report.retained_debug_audio,
        wake_word_live_verified=report.wake_word_live_verified,
    )


def _fake_backend_real_summary() -> RealModelSummary:
    """Synthetic real-model failure when the runtime backend is fake.

    ``--require-real-models`` cannot be satisfied by a fake/simulated runtime, so
    we fail closed and explain the fix without spawning a pointless verifier run.
    """
    return RealModelSummary(
        summary="fail",
        verification_level="none",
        real_model_verified=False,
        models_attempted=0,
        models_available=0,
        models_passed=0,
        checks_failed=1,
        check_failures=["runtime backend is fake; real-model verification is impossible"],
    )


def _acceptance_level(
    *,
    require_real_models: bool,
    real: RealModelSummary | None,
    voice: VoiceLiveSummary | None,
    wake: WakeWordLiveSummary | None,
) -> AcceptanceLevel:
    # A higher rung is only claimed when the named checks were requested and passed.
    real_ok = require_real_models and real is not None and real.summary == "pass"
    if not real_ok:
        return "fake_sanity"
    if voice is None or voice.summary != "pass":
        return "real_models"
    if wake is None or wake.summary != "pass":
        return "real_models_plus_voice"
    return "full_wake_voice"


def _final_status(
    *,
    config_valid: bool,
    fake: FakeVerificationSummary,
    readiness: ReadinessReport,
    require_real_models: bool,
    allow_sanity_pass: bool,
    real: RealModelSummary | None,
    voice: VoiceLiveSummary | None,
    wake: WakeWordLiveSummary | None,
    services: ServicesSummary,
) -> FinalStatus:
    # Honesty first: invalid config, a failed fake run, an explicitly-requested
    # service that could not start, a fake backend when real models are required,
    # a required-but-failed real model run, or any requested live check that did
    # not pass is a hard fail.
    hard_fail = (
        not config_valid
        or (fake.ran and fake.summary == "fail")
        or (services.requested and services.startup_status == "failed")
        or (require_real_models and readiness.runtime_is_fake)
        or (require_real_models and (real is None or real.summary == "fail"))
        or (voice is not None and voice.summary != "pass")
        or (wake is not None and wake.summary != "pass")
    )
    if hard_fail:
        return "fail"
    # Fake/local sanity only: never silently look like full Mac readiness. Without
    # --require-real-models, the ceiling is a warning unless the operator opts in
    # to a sanity pass AND nothing else is even advisory.
    if not require_real_models:
        if allow_sanity_pass and not (readiness.blockers or readiness.warnings):
            return "pass"
        return "warning"
    # Real models were required and passed; readiness advisories still downgrade.
    if readiness.blockers or readiness.warnings:
        return "warning"
    return "pass"


def _next_actions(
    *,
    config_valid: bool,
    readiness: ReadinessReport,
    require_real_models: bool,
    real: RealModelSummary | None,
    voice: VoiceLiveSummary | None,
    wake: WakeWordLiveSummary | None,
    final_status: FinalStatus,
) -> list[str]:
    actions: list[str] = []

    def add(action: str) -> None:
        if action and action not in actions:
            actions.append(action)

    if not config_valid:
        add(_CONFIG_VALIDATE)
    if require_real_models and readiness.runtime_is_fake:
        add(_REAL_BACKEND_FAKE)
    # readiness already computes the canonical, redacted next commands.
    for action in readiness.next_actions:
        add(action)
    if require_real_models and real is not None and real.summary != "pass":
        add(_VERIFY_REAL)
    if not require_real_models:
        add(_REAL_NOT_REQUIRED)
        if final_status == "warning":
            add(_SANITY_ONLY)
    if voice is not None and voice.summary != "pass":
        add(_VOICE_DOCTOR)
        add(_VERIFY_VOICE_LIVE)
    if wake is not None and wake.summary != "pass":
        add(_VOICE_DOCTOR)
        add(_VERIFY_WAKE_LIVE)
    if final_status == "pass":
        add("Acceptance passed: no blockers detected on this Mac.")
    return actions


def run_acceptance(
    home: Path,
    *,
    require_real_models: bool = False,
    allow_sanity_pass: bool = False,
    max_output_tokens: int = 32,
    timeout: float = 180.0,
    thresholds: ReportThresholds | None = None,
    services: ServicesSummary | None = None,
    voice_live_runner: Callable[[], VoiceLiveReport] | None = None,
    wake_word_live_runner: Callable[[], WakeWordLiveReport] | None = None,
) -> AcceptanceReport:
    """Compose APRIL's acceptance checks into one redacted report.

    ``voice_live_runner`` / ``wake_word_live_runner`` are injected so the
    interactive live checks stay out of this pure orchestrator: the CLI passes
    closures that drive the real verifiers, and tests pass fakes. A live check is
    considered *requested* exactly when its runner is provided. ``services`` is the
    service-lifecycle record built by the caller (the CLI) when it orchestrates
    APRIL services for live checks; it is embedded verbatim and factored into the
    final status (a requested service that failed to start is a hard fail).
    """
    home = home.expanduser()
    services = services or ServicesSummary()
    errors = list(validate_configuration(home))
    config_valid = not errors
    readiness = build_readiness_report(home)

    fake = _fake_summary(home, config_valid=config_valid)

    real: RealModelSummary | None = None
    if require_real_models:
        # A fake/simulated runtime can never verify real models: fail closed
        # without spawning a verifier rather than risk mistaking fake for real.
        if readiness.runtime_is_fake:
            real = _fake_backend_real_summary()
        else:
            real = _real_model_summary(
                home,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                thresholds=thresholds,
            )

    voice: VoiceLiveSummary | None = None
    if voice_live_runner is not None:
        voice = _voice_summary(voice_live_runner())

    wake: WakeWordLiveSummary | None = None
    if wake_word_live_runner is not None:
        wake = _wake_summary(wake_word_live_runner())

    final_status = _final_status(
        config_valid=config_valid,
        fake=fake,
        readiness=readiness,
        require_real_models=require_real_models,
        allow_sanity_pass=allow_sanity_pass,
        real=real,
        voice=voice,
        wake=wake,
        services=services,
    )
    acceptance_level = _acceptance_level(
        require_real_models=require_real_models, real=real, voice=voice, wake=wake
    )
    next_actions = _next_actions(
        config_valid=config_valid,
        readiness=readiness,
        require_real_models=require_real_models,
        real=real,
        voice=voice,
        wake=wake,
        final_status=final_status,
    )

    return AcceptanceReport(
        generated_at=utc_now_iso(),
        environment=_environment(readiness),
        runtime_backend=readiness.runtime_backend,
        acceptance_level=acceptance_level,
        config_valid=config_valid,
        config_errors=[redact_reason(error) for error in errors],
        requested={
            "require_real_models": require_real_models,
            "allow_sanity_pass": allow_sanity_pass,
            "voice_live": voice_live_runner is not None,
            "wake_word_live": wake_word_live_runner is not None,
        },
        fake_verification=fake,
        readiness=ReadinessSummary(
            real_model_ready=readiness.real_model_ready,
            voice_enabled=readiness.voice_enabled,
            voice_ready=readiness.voice_ready,
            blockers=list(readiness.blockers),
            warnings=list(readiness.warnings),
        ),
        real_model_verification=real,
        voice_live=voice,
        wake_word_live=wake,
        services=services,
        final_status=final_status,
        next_actions=next_actions,
    )
