"""Real-Mac go-live proof/evidence report for APRIL.

``run april go-live`` answers one honest question: *is this Mac actually ready to
run APRIL on real local models?* It is the real-model-only sibling of
``run april acceptance`` — where acceptance can report a fake/local sanity pass,
go-live exists to prove the real thing and to make the ladder explicit:

1. fake/local plumbing works,
2. real GGUF models are installed,
3. real models load / chat / stream / unload,
4. the Brain produces strict JSON routing with the real brain model (no fallback),
5. specialist model switching works, and
6. APRIL is actually ready on this Mac.

The builder composes the existing, separately-unit-tested verification primitives
rather than re-implementing them:

* configuration validation (:func:`april_common.config_validation.validate_configuration`),
* offline readiness (:func:`apps.runner.readiness.build_readiness_report`),
* deterministic fake-backend sanity (:func:`apps.runner.verify.run_fake_verification`),
* the all-configured real-model verifier
  (:func:`apps.runner.verify.run_all_configured_models_verification`) — which itself
  exercises real load/chat/stream/unload, the strict-JSON brain routing eval, and
  specialist switching.

The report is **redacted by construction**: every field is a boolean, a count, a
status string, an enum, or a basename. No absolute paths, tokens, generated model
text, transcripts, raw prompts (only stable fixture totals), or secrets are ever
stored, and any path-looking detail is reduced to its basename via
:func:`apps.runner.mac_report.redact_reason`.

Like the acceptance orchestrator, :func:`build_go_live_report` is a pure function
over already-computed sub-reports so it can be unit-tested with fake/mocked inputs
and temporary configs — no GGUF, llama-cpp-python, microphone, speaker, network,
or Homebrew required. Go-live never records audio, never opens the microphone,
never listens for a wake word, never runs TTS, never downloads a model, never
installs a package, and never mutates configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.acceptance import FakeVerificationSummary, ServicesSummary
from apps.runner.mac_report import redact_reason
from apps.runner.multi_model_report import MultiModelVerificationReport, PerModelResult
from apps.runner.readiness import ReadinessReport
from april_common.time import utc_now, utc_now_iso

FinalStatus = Literal["pass", "warning", "fail"]
# Go-live only ever claims the real-model rung. Live voice belongs to a later
# ``full_wake_voice`` milestone and is intentionally out of scope here.
GoLiveLevel = Literal["fake_sanity", "real_models"]
# The real-model *core* is the honest "does the real GGUF path actually work"
# rung, kept separate from the *hardened* go-live rung (dev tokens, runtime-local
# embeddings, and other hardening advisories) so a working real-model path is
# never hidden behind a hardening warning.
RealModelCoreStatus = Literal["ready", "fail", "not_run"]

# Exact, copy-pasteable next commands. None of these are executed here.
_INSTALL_RUNTIME = "pip install -e '.[runtime]'"
_DOWNLOAD_MODELS = "run april model download --all-core --apply --yes"
_CONFIG_VALIDATE = "run april config validate"
_SET_BACKEND = "Set runtime.backend=llama_cpp (or unset APRIL_RUNTIME_BACKEND=fake)."
_SETUP_TOKENS = "run april setup tokens"
_GO_LIVE_AGAIN = "run april go-live --write-report --start-services"
_PASS_NOTE = "Go-live proof passed: APRIL is ready on this Mac for the real-model milestone."
_VOICE_NOTE = (
    "Voice is opt-in and disabled; it is not required for the first real-model go-live. "
    "Run `run april setup voice` then `run april voice verify-live` for the later "
    "full_wake_voice milestone."
)
_EMBED_NOTE = (
    "Runtime-local embeddings are not configured (memory.embedding_provider is the "
    "development hashed-token default)."
)
_DESKTOP_NOTE = "Desktop app is unsigned/dev-only; no signed Mac app ships in this milestone."


class GoLiveEnvironment(BaseModel):
    os: str
    cpu_architecture: str
    python_version: str
    deployment: str
    runtime_backend: str
    runtime_is_fake: bool
    llama_cpp_python_available: bool


class GoLiveReadinessSummary(BaseModel):
    real_model_ready: bool
    real_model_preflight_ready: bool = False
    voice_enabled: bool
    configured_chat_models_count: int
    configured_chat_models_present_count: int
    api_token_status: str
    runtime_token_status: str
    embedding_provider: str
    embedding_runtime_local: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GoLiveAcceptanceSummary(BaseModel):
    requested: bool = False
    ran: bool = False
    acceptance_level: GoLiveLevel = "fake_sanity"
    summary: str = "skipped"
    real_model_verified: bool = False
    verification_level: str = "none"
    models_attempted: int = 0
    models_available: int = 0
    models_passed: int = 0
    checks_failed: int = 0
    check_failures: list[str] = Field(default_factory=list)


class GoLiveRoutingSummary(BaseModel):
    report_exists: bool = False
    routing_cases_total: int = 0
    routing_cases_passed: int = 0
    routing_schema_valid_count: int = 0
    routing_failures: int = 0
    routing_fallback_count: int = 0
    model_repair_count: int = 0

    @property
    def passed_without_fallback(self) -> bool:
        return (
            self.report_exists
            and self.routing_fallback_count == 0
            and self.routing_cases_total > 0
            and self.routing_cases_passed == self.routing_cases_total
        )


class GoLiveSpecialistSummary(BaseModel):
    # Switching is only required when more than one chat model is configured (a
    # brain plus at least one specialist). With a single chat model it is N/A.
    applicable: bool = False
    attempted: bool = False
    verified: bool = False
    chat_models_count: int = 0


class GoLiveFinalStatus(BaseModel):
    status: FinalStatus = "fail"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class GoLiveReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["go_live"] = "go_live"
    generated_at: str
    # Redacted structural config fingerprint at generation time; lets readiness
    # mark this report stale when the configuration later changes. ``None`` for
    # reports generated before fingerprinting existed (timestamp-only freshness).
    config_fingerprint: str | None = None
    # --- flat summary fields (also surfaced by the reports browser) ----------
    runtime_backend: str
    llama_cpp_python_available: bool
    voice_enabled: bool
    real_model_ready: bool
    # --- core vs hardened readiness (the headline distinction) ---------------
    # ``core_real_model_ready`` is true when the real GGUF path works end to end
    # (backend/llama-cpp present, configured chat GGUFs present, real
    # load/chat/stream/unload, strict brain routing with no fallback, specialist
    # switching where applicable). It is intentionally independent of hardening.
    core_real_model_ready: bool = False
    real_model_core_status: RealModelCoreStatus = "not_run"
    # ``hardened_go_live_ready`` additionally requires the hardening rung: no
    # default/placeholder/blank tokens, runtime-local embeddings, and no other
    # hardening blockers. Hardening advisories never hide the core result.
    hardened_go_live_ready: bool = False
    hardening_warnings: list[str] = Field(default_factory=list)
    hardening_blockers: list[str] = Field(default_factory=list)
    configured_chat_models_count: int
    configured_chat_models_present_count: int
    acceptance_level: GoLiveLevel = "fake_sanity"
    real_model_verified: bool = False
    models_attempted: int = 0
    models_passed: int = 0
    routing_cases_total: int = 0
    routing_cases_passed: int = 0
    routing_fallback_count: int = 0
    specialist_switching_verified: bool = False
    final_status: FinalStatus = "fail"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    # --- structured detail ---------------------------------------------------
    environment: GoLiveEnvironment
    readiness: GoLiveReadinessSummary
    acceptance: GoLiveAcceptanceSummary
    routing: GoLiveRoutingSummary
    specialist: GoLiveSpecialistSummary
    services: ServicesSummary = Field(default_factory=ServicesSummary)
    final: GoLiveFinalStatus


def default_go_live_report_path(home: Path) -> Path:
    """Default ``--write-report`` location, under the Git-ignored verification dir."""
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return home.expanduser() / "data" / "verification" / f"go-live-{stamp}.json"


def write_go_live_report(report: GoLiveReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


def _chat_models(readiness: ReadinessReport) -> list[object]:
    """Configured real chat GGUF models (llama_cpp backend, non-embedding role)."""
    return [
        model
        for model in readiness.models
        if model.backend == "llama_cpp" and model.role != "embedding"
    ]


def _brain_routing(multi_model: MultiModelVerificationReport | None) -> GoLiveRoutingSummary:
    if multi_model is None:
        return GoLiveRoutingSummary()
    brain: PerModelResult | None = next(
        (model for model in multi_model.models if model.role == "brain"), None
    )
    routing = brain.routing if brain is not None else None
    if routing is None or routing.total == 0:
        return GoLiveRoutingSummary()
    return GoLiveRoutingSummary(
        report_exists=True,
        routing_cases_total=routing.total,
        routing_cases_passed=routing.passed,
        routing_schema_valid_count=routing.schema_valid_count,
        routing_failures=routing.failures,
        routing_fallback_count=routing.fallback_count,
        model_repair_count=routing.model_repair_count,
    )


def _specialist_summary(
    multi_model: MultiModelVerificationReport | None, *, chat_count: int
) -> GoLiveSpecialistSummary:
    applicable = chat_count > 1
    switch = multi_model.specialist_switch if multi_model is not None else None
    return GoLiveSpecialistSummary(
        applicable=applicable,
        attempted=bool(switch and switch.attempted),
        verified=bool(switch and switch.success),
        chat_models_count=chat_count,
    )


def _acceptance_summary(
    multi_model: MultiModelVerificationReport | None, *, requested: bool
) -> GoLiveAcceptanceSummary:
    if multi_model is None:
        return GoLiveAcceptanceSummary(requested=requested, ran=False)
    level: GoLiveLevel = "real_models" if multi_model.summary == "pass" else "fake_sanity"
    return GoLiveAcceptanceSummary(
        requested=requested,
        ran=True,
        acceptance_level=level,
        summary=multi_model.summary,
        real_model_verified=multi_model.real_model_verified,
        verification_level=multi_model.verification_level,
        models_attempted=multi_model.models_attempted,
        models_available=multi_model.models_available,
        models_passed=multi_model.models_passed,
        checks_failed=multi_model.checks_failed,
        check_failures=[redact_reason(failure) for failure in multi_model.check_failures],
    )


def _blockers(
    *,
    config_valid: bool,
    fake: FakeVerificationSummary,
    readiness: ReadinessReport,
    chat_count: int,
    chat_present_count: int,
    real_requested: bool,
    acceptance: GoLiveAcceptanceSummary,
    routing: GoLiveRoutingSummary,
    specialist: GoLiveSpecialistSummary,
    services: ServicesSummary,
) -> list[str]:
    """Hard-fail reasons. Go-live is a real-model proof: an environment that cannot
    support real models is a fail, not a warning. Voice is never a blocker."""
    blockers: list[str] = []
    if not config_valid:
        blockers.append("configuration invalid")
    if fake.ran and fake.summary == "fail":
        blockers.append("fake/local plumbing sanity failed")
    if readiness.runtime_is_fake:
        blockers.append("runtime backend is fake (not llama_cpp)")
    if not readiness.llama_cpp_python_available:
        blockers.append("llama-cpp-python is not installed")
    if chat_count == 0:
        blockers.append("no chat GGUF models are configured")
    elif chat_present_count < chat_count:
        blockers.append("required GGUF model file(s) missing")
    # Real-model proof failures (only meaningful once real verification ran).
    if real_requested and acceptance.ran:
        if acceptance.summary == "fail" or acceptance.checks_failed > 0:
            blockers.append("real model load/chat/stream/unload failed")
        if routing.routing_fallback_count > 0:
            blockers.append("brain routing fell back during real-model routing eval")
        if specialist.applicable and specialist.attempted and not specialist.verified:
            blockers.append("specialist switching failed")
    # Service lifecycle: only when the operator asked us to manage main services.
    if services.requested and (
        services.startup_status == "failed" or services.shutdown_status == "failed"
    ):
        blockers.append("services failed to start or stop cleanly")
    return blockers


def _core_ready(
    *,
    config_valid: bool,
    fake: FakeVerificationSummary,
    readiness: ReadinessReport,
    chat_count: int,
    chat_present_count: int,
    real_requested: bool,
    acceptance: GoLiveAcceptanceSummary,
    routing: GoLiveRoutingSummary,
    specialist: GoLiveSpecialistSummary,
    services: ServicesSummary,
    blockers: list[str],
) -> bool:
    """The real-model *core* gate: does the real GGUF path actually work?

    This is intentionally independent of hardening (dev tokens, runtime-local
    embeddings). It is the honest answer to "real model core: ready?" so a working
    real-model path is never downgraded by a hardening advisory. Voice being
    disabled is never part of this gate.
    """
    if blockers:
        return False
    services_ok = (not services.requested) or services.startup_status in {"ok", "already_running"}
    return (
        config_valid
        and not (fake.ran and fake.summary == "fail")
        and not readiness.runtime_is_fake
        and readiness.llama_cpp_python_available
        and chat_count > 0
        and chat_present_count == chat_count
        and real_requested
        and acceptance.ran
        and acceptance.summary == "pass"
        and acceptance.real_model_verified
        and routing.passed_without_fallback
        and ((not specialist.applicable) or specialist.verified)
        and services_ok
    )


def _hardening_warnings(
    *,
    readiness: ReadinessReport,
    embedding_runtime_local: bool,
) -> list[str]:
    """Hardening advisories that hold a working real-model core at ``warning``.

    These are the dev/placeholder/blank-token advisories already surfaced by
    readiness plus the runtime-local embeddings requirement. They are never
    allowed to mask the core result; they only gate the *hardened* rung.
    """
    warnings: list[str] = [redact_reason(warning) for warning in readiness.warnings]
    if not embedding_runtime_local and _EMBED_NOTE not in warnings:
        warnings.append(_EMBED_NOTE)
    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            unique.append(warning)
    return unique


def _warnings(
    *,
    readiness: ReadinessReport,
    real_requested: bool,
    acceptance: GoLiveAcceptanceSummary,
    embedding_runtime_local: bool,
) -> list[str]:
    """Non-fatal advisories. Surfaced regardless of final status; gating ones
    (dev tokens, non-local embeddings) hold a real-passing run at ``warning``."""
    warnings: list[str] = []
    if not (real_requested and acceptance.ran and acceptance.summary == "pass"):
        warnings.append("real-model acceptance was not requested or could not run")
    # readiness.warnings already carries dev/placeholder token advisories, redacted.
    for warning in readiness.warnings:
        warnings.append(redact_reason(warning))
    if not embedding_runtime_local:
        warnings.append(_EMBED_NOTE)
    if not readiness.voice_enabled:
        warnings.append(_VOICE_NOTE)
    warnings.append(_DESKTOP_NOTE)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            unique.append(warning)
    return unique


def _next_actions(
    *,
    final_status: FinalStatus,
    blockers: list[str],
    readiness: ReadinessReport,
    chat_count: int,
    chat_present_count: int,
    acceptance: GoLiveAcceptanceSummary,
    routing: GoLiveRoutingSummary,
) -> list[str]:
    actions: list[str] = []

    def add(action: str) -> None:
        if action and action not in actions:
            actions.append(action)

    if "configuration invalid" in blockers:
        add(_CONFIG_VALIDATE)
    if readiness.runtime_is_fake:
        add(_SET_BACKEND)
    if not readiness.llama_cpp_python_available:
        add(_INSTALL_RUNTIME)
    if chat_count == 0 or chat_present_count < chat_count:
        add(_DOWNLOAD_MODELS)
    if acceptance.ran and acceptance.summary == "fail":
        add(_GO_LIVE_AGAIN)
    if routing.report_exists and routing.routing_fallback_count > 0:
        add(_GO_LIVE_AGAIN)
    if readiness.warnings:
        add(_SETUP_TOKENS)
    if final_status == "pass":
        add(_PASS_NOTE)
    elif final_status == "warning":
        add(_GO_LIVE_AGAIN)
    return actions


def build_go_live_report(
    *,
    home: Path,
    config_valid: bool,
    config_errors: Sequence[str],
    readiness: ReadinessReport,
    fake_verification: FakeVerificationSummary,
    real_requested: bool,
    multi_model: MultiModelVerificationReport | None,
    embedding_provider: str,
    embedding_model_id: str | None,
    services: ServicesSummary | None = None,
    config_fingerprint: str | None = None,
) -> GoLiveReport:
    """Fold APRIL's real-model verification primitives into one redacted go-live
    proof report. Pure over its inputs so it is unit-testable with mocked sub-reports.

    ``multi_model`` is the report from
    :func:`apps.runner.verify.run_all_configured_models_verification` run with
    ``require_real_model=True`` (and routing evals enabled); ``None`` means the real
    verification was not run (a warning, never a silent pass). ``config_errors`` are
    redacted before embedding so an absolute path can never leak.
    """
    services = services or ServicesSummary()
    chat_models = _chat_models(readiness)
    chat_count = len(chat_models)
    chat_present_count = sum(1 for model in chat_models if getattr(model, "path_exists", False))
    embedding_runtime_local = embedding_provider == "runtime-local" and bool(embedding_model_id)

    acceptance = _acceptance_summary(multi_model, requested=real_requested)
    routing = _brain_routing(multi_model)
    specialist = _specialist_summary(multi_model, chat_count=chat_count)

    blockers = _blockers(
        config_valid=config_valid,
        fake=fake_verification,
        readiness=readiness,
        chat_count=chat_count,
        chat_present_count=chat_present_count,
        real_requested=real_requested,
        acceptance=acceptance,
        routing=routing,
        specialist=specialist,
        services=services,
    )
    warnings = _warnings(
        readiness=readiness,
        real_requested=real_requested,
        acceptance=acceptance,
        embedding_runtime_local=embedding_runtime_local,
    )

    # A configuration that does not even load is reported with its (redacted) errors
    # folded into the blockers so the on-disk report is self-explanatory. Done
    # before status derivation so the core/hardened gates see the full blocker set.
    if not config_valid and config_errors:
        for error in config_errors:
            redacted = redact_reason(error)
            if redacted not in blockers:
                blockers.append(redacted)

    # --- core real-model readiness (independent of hardening) ----------------
    core_real_model_ready = _core_ready(
        config_valid=config_valid,
        fake=fake_verification,
        readiness=readiness,
        chat_count=chat_count,
        chat_present_count=chat_present_count,
        real_requested=real_requested,
        acceptance=acceptance,
        routing=routing,
        specialist=specialist,
        services=services,
        blockers=blockers,
    )
    if core_real_model_ready:
        real_model_core_status: RealModelCoreStatus = "ready"
    elif blockers:
        real_model_core_status = "fail"
    else:
        real_model_core_status = "not_run"

    # --- hardened go-live readiness (core + hardening advisories) -------------
    hardening_warnings = _hardening_warnings(
        readiness=readiness,
        embedding_runtime_local=embedding_runtime_local,
    )
    hardening_blockers: list[str] = []
    hardened_go_live_ready = (
        core_real_model_ready and not hardening_warnings and not hardening_blockers
    )

    if blockers:
        final_status: FinalStatus = "fail"
    elif hardened_go_live_ready:
        final_status = "pass"
    else:
        final_status = "warning"

    next_actions = _next_actions(
        final_status=final_status,
        blockers=blockers,
        readiness=readiness,
        chat_count=chat_count,
        chat_present_count=chat_present_count,
        acceptance=acceptance,
        routing=routing,
    )

    real_model_ready = (
        acceptance.summary == "pass" and acceptance.real_model_verified and final_status != "fail"
    )

    return GoLiveReport(
        generated_at=utc_now_iso(),
        config_fingerprint=config_fingerprint,
        runtime_backend=readiness.runtime_backend,
        llama_cpp_python_available=readiness.llama_cpp_python_available,
        voice_enabled=readiness.voice_enabled,
        real_model_ready=real_model_ready,
        core_real_model_ready=core_real_model_ready,
        real_model_core_status=real_model_core_status,
        hardened_go_live_ready=hardened_go_live_ready,
        hardening_warnings=hardening_warnings,
        hardening_blockers=hardening_blockers,
        configured_chat_models_count=chat_count,
        configured_chat_models_present_count=chat_present_count,
        acceptance_level=acceptance.acceptance_level,
        real_model_verified=acceptance.real_model_verified,
        models_attempted=acceptance.models_attempted,
        models_passed=acceptance.models_passed,
        routing_cases_total=routing.routing_cases_total,
        routing_cases_passed=routing.routing_cases_passed,
        routing_fallback_count=routing.routing_fallback_count,
        specialist_switching_verified=specialist.verified,
        final_status=final_status,
        blockers=blockers,
        warnings=warnings,
        next_actions=next_actions,
        environment=GoLiveEnvironment(
            os=readiness.os,
            cpu_architecture=readiness.cpu_architecture,
            python_version=readiness.python_version,
            deployment=readiness.environment,
            runtime_backend=readiness.runtime_backend,
            runtime_is_fake=readiness.runtime_is_fake,
            llama_cpp_python_available=readiness.llama_cpp_python_available,
        ),
        readiness=GoLiveReadinessSummary(
            real_model_ready=real_model_ready,
            real_model_preflight_ready=readiness.real_model_preflight_ready,
            voice_enabled=readiness.voice_enabled,
            configured_chat_models_count=chat_count,
            configured_chat_models_present_count=chat_present_count,
            api_token_status=readiness.api_token_status,
            runtime_token_status=readiness.runtime_token_status,
            embedding_provider=embedding_provider,
            embedding_runtime_local=embedding_runtime_local,
            blockers=list(readiness.blockers),
            warnings=[redact_reason(warning) for warning in readiness.warnings],
        ),
        acceptance=acceptance,
        routing=routing,
        specialist=specialist,
        services=services,
        final=GoLiveFinalStatus(
            status=final_status,
            blockers=blockers,
            warnings=warnings,
            next_actions=next_actions,
        ),
    )
