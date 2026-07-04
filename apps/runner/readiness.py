"""Offline, redacted real-model readiness report for the target Mac.

``run april readiness`` answers one question without starting any service,
downloading any model, or installing any package: *what is still missing before
APRIL can run real local models (and, optionally, real local voice)?*

It is deliberately inert and safe:

* It only reads ``configs/*.yaml`` and settings/env, plus ``importlib`` spec
  lookups and ``Path.exists`` probes. It never loads a model, never opens the
  microphone, never reaches the network, and never mutates anything.
* Every emitted field is redacted by construction: model/voice paths collapse to
  basenames, tokens are reported as ``configured``/``default-development``/
  ``missing`` (never the value), and skip/blocker details run through
  :func:`apps.runner.mac_report.redact_reason` so an embedded absolute path is
  reduced to its basename.
* It prints *actionable commands only* — it tells you the exact command to run,
  it does not run it.

The report builder is separated from the CLI so it can be unit-tested against
temporary configs with no real GGUF, binary, or device present.
"""

from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.mac_report import redact_reason
from april_common.errors import ConfigError
from april_common.settings import (
    KNOWN_DEFAULT_API_TOKENS,
    KNOWN_DEFAULT_RUNTIME_TOKENS,
    PLACEHOLDER_API_TOKENS,
    PLACEHOLDER_RUNTIME_TOKENS,
    AprilSettings,
    load_settings,
)
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import ModelRegistry

CheckStatus = Literal["ok", "warning", "blocker", "skipped"]

# Exact, copy-pasteable next commands. None of these are executed here.
_INSTALL_RUNTIME = "pip install -e '.[runtime]'"
_SETUP_MODELS = "run april setup models"
_SETUP_VOICE = "run april setup voice"
_SETUP_TOKENS = "run april setup tokens"
_SETUP_EMBEDDINGS = "run april setup embeddings --model /absolute/path/to/embedding.gguf --apply"
_VERIFY_REAL = (
    "run april verify --all-configured-models --require-real-model "
    "--report data/verification/mac-readiness.json"
)
_VERIFY_VOICE = "run april voice verify-live --report data/verification/voice-live.json"
_VERIFY_WAKE = "run april voice verify-wake-live --report data/verification/wake-live.json"

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ReadinessCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    action: str | None = None


class VoiceArtifact(BaseModel):
    name: str
    configured: bool
    exists: bool
    basename: str | None = None


class ReadinessModel(BaseModel):
    id: str
    role: str
    backend: str
    path_basename: str | None
    path_exists: bool


class ReadinessReport(BaseModel):
    schema_version: int = 1
    generated_at: str
    os: str
    cpu_architecture: str
    python_version: str
    runtime_backend: str
    runtime_is_fake: bool
    llama_cpp_python_available: bool
    environment: str
    voice_enabled: bool
    # Offline readiness never proves a real GGUF or live voice path. These stay
    # false until populated from actual verification reports elsewhere.
    real_model_ready: bool = False
    voice_ready: bool = False
    # Preflight means the local prerequisites appear present; it is still not
    # proof that load/chat/stream/unload or live voice succeeded.
    real_model_preflight_ready: bool = False
    voice_preflight_ready: bool = False
    models: list[ReadinessModel] = Field(default_factory=list)
    voice_artifacts: list[VoiceArtifact] = Field(default_factory=list)
    api_token_status: str = "missing"
    runtime_token_status: str = "missing"
    speaker_gate: str = "off"
    speaker_gate_supported: bool = False
    daemon_status: str = "unknown"
    daemon_details_available: bool = False
    sentinel_live_status: str = "not_verified"
    checks: list[ReadinessCheck] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


def _token_status(value: str | None, defaults: set[str], placeholders: set[str]) -> str:
    if not value:
        return "missing"
    if value in placeholders:
        return "placeholder-insecure"
    if value in defaults:
        return "default-development"
    return "configured"


def _voice_artifact(
    settings: AprilSettings, name: str, path: Path | None, *, enabled: bool, required: bool = True
) -> tuple[VoiceArtifact, ReadinessCheck]:
    if path is None:
        artifact = VoiceArtifact(name=name, configured=False, exists=False, basename=None)
        status: CheckStatus = "blocker" if enabled and required else "skipped"
        detail = "Not configured." if enabled else "Voice disabled; not configured."
        if enabled and not required:
            status = "warning"
            detail = "Not configured; wake-word live verification remains unavailable."
        return artifact, ReadinessCheck(
            name=f"voice: {name}",
            status=status,
            detail=detail,
            action=_SETUP_VOICE if enabled and required else None,
        )
    resolved = settings.resolve_path(path)
    exists = resolved.exists()
    artifact = VoiceArtifact(name=name, configured=True, exists=exists, basename=resolved.name)
    if exists:
        return artifact, ReadinessCheck(name=f"voice: {name}", status="ok", detail=resolved.name)
    status = "blocker" if enabled and required else "warning"
    return artifact, ReadinessCheck(
        name=f"voice: {name}",
        status=status,
        detail=redact_reason(f"Missing: {resolved}"),
        action=_SETUP_VOICE if required else None,
    )


def build_readiness_report(home: Path) -> ReadinessReport:
    root = home.expanduser().resolve()
    checks: list[ReadinessCheck] = []

    try:
        settings = load_settings(root=root)
    except ConfigError as exc:
        # A broken config blocks everything; report it honestly rather than crash.
        return ReadinessReport(
            generated_at=utc_now_iso(),
            os=f"{platform.system()} {platform.release()}".strip(),
            cpu_architecture=platform.machine(),
            python_version=platform.python_version(),
            runtime_backend="unknown",
            runtime_is_fake=False,
            llama_cpp_python_available=importlib.util.find_spec("llama_cpp") is not None,
            environment="unknown",
            voice_enabled=False,
            checks=[
                ReadinessCheck(
                    name="configuration load",
                    status="blocker",
                    detail=redact_reason(str(exc)),
                    action="run april config validate",
                )
            ],
            blockers=["configuration load"],
            next_actions=["run april config validate"],
        )

    backend = settings.runtime.backend
    runtime_is_fake = backend != "llama_cpp"
    llama_available = importlib.util.find_spec("llama_cpp") is not None

    # --- runtime backend -----------------------------------------------------
    if runtime_is_fake:
        checks.append(
            ReadinessCheck(
                name="runtime backend",
                status="blocker",
                detail=f"Backend is '{backend}' (fake/simulated), not 'llama_cpp'.",
                action="Set APRIL_RUNTIME_BACKEND=llama_cpp (or runtime.backend in april.yaml).",
            )
        )
    else:
        checks.append(ReadinessCheck(name="runtime backend", status="ok", detail="llama_cpp"))

    # --- llama-cpp-python extra ---------------------------------------------
    if llama_available:
        checks.append(
            ReadinessCheck(name="llama-cpp-python", status="ok", detail="import spec found")
        )
    else:
        checks.append(
            ReadinessCheck(
                name="llama-cpp-python",
                status="blocker",
                detail="Optional runtime extra is not installed.",
                action=_INSTALL_RUNTIME,
            )
        )

    # --- configured GGUF model files ----------------------------------------
    models: list[ReadinessModel] = []
    missing_models: list[str] = []
    try:
        registry = ModelRegistry.from_file(root / "configs" / "models.yaml", root=root)
    except ConfigError as exc:
        checks.append(
            ReadinessCheck(
                name="model registry",
                status="blocker",
                detail=redact_reason(str(exc)),
                action="run april config validate",
            )
        )
        registry = None

    if registry is not None:
        for model in registry.list():
            path = model.resolved_path(registry.root)
            exists = path.exists()
            models.append(
                ReadinessModel(
                    id=model.id,
                    role=model.role,
                    backend=model.backend,
                    path_basename=path.name,
                    path_exists=exists,
                )
            )
            if model.backend == "llama_cpp" and not exists:
                missing_models.append(model.id)
        if missing_models:
            checks.append(
                ReadinessCheck(
                    name="configured GGUF model files",
                    status="blocker",
                    detail="Missing model files: " + ", ".join(sorted(missing_models)),
                    action=_SETUP_MODELS,
                )
            )
        elif any(model.backend == "llama_cpp" for model in registry.list()):
            checks.append(
                ReadinessCheck(
                    name="configured GGUF model files",
                    status="ok",
                    detail="All configured llama_cpp model files are present.",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    name="configured GGUF model files",
                    status="warning",
                    detail="No llama_cpp model is configured; only fake models exist.",
                    action=_SETUP_MODELS,
                )
            )

    # --- runtime-local embeddings ------------------------------------------
    if settings.memory.embedding_provider == "runtime-local":
        embedding_model_id = settings.memory.embedding_model_id
        if not embedding_model_id:
            checks.append(
                ReadinessCheck(
                    name="runtime-local embedding model",
                    status="blocker",
                    detail=(
                        "runtime-local embeddings are configured but no embedding model id is set."
                    ),
                    action=_SETUP_EMBEDDINGS,
                )
            )
        elif registry is None or not registry.exists(embedding_model_id):
            checks.append(
                ReadinessCheck(
                    name="runtime-local embedding model",
                    status="blocker",
                    detail=f"Embedding model id is not registered: {embedding_model_id}",
                    action=_SETUP_EMBEDDINGS,
                )
            )
        else:
            embedding_model = registry.get(embedding_model_id)
            embedding_path = embedding_model.resolved_path(registry.root)
            checks.append(
                ReadinessCheck(
                    name="runtime-local embedding model",
                    status="ok" if embedding_path.exists() else "blocker",
                    detail=(
                        f"Registered embedding model {embedding_model_id} exists."
                        if embedding_path.exists()
                        else f"Missing embedding model file: {embedding_path.name}"
                    ),
                    action=None if embedding_path.exists() else _SETUP_EMBEDDINGS,
                )
            )
    else:
        checks.append(
            ReadinessCheck(
                name="runtime-local embedding model",
                status="skipped",
                detail="memory.embedding_provider is hashed-token; no embedding GGUF required.",
            )
        )

    # --- loopback-only binding ------------------------------------------------
    non_loopback = [
        f"{name}={host}"
        for name, host in (("api.host", settings.api.host), ("runtime.host", settings.runtime.host))
        if host not in _LOOPBACK_HOSTS
    ]
    if non_loopback:
        checks.append(
            ReadinessCheck(
                name="loopback-only binding",
                status="blocker",
                detail="Non-loopback bind address configured: " + ", ".join(sorted(non_loopback)),
                action="Set api.host and runtime.host to 127.0.0.1 (APRIL is loopback-only).",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="loopback-only binding",
                status="ok",
                detail="API and runtime bind to loopback only.",
            )
        )

    # --- hashed-token embeddings in production --------------------------------
    if (
        settings.memory.embedding_provider == "hashed-token"
        and settings.environment == "production"
    ):
        checks.append(
            ReadinessCheck(
                name="embedding provider hardening",
                status="warning",
                detail=(
                    "hashed-token embeddings are active in a production environment; "
                    "semantic memory is degraded and hardened go-live holds at warning."
                ),
                action=_SETUP_EMBEDDINGS,
            )
        )

    # --- development tokens --------------------------------------------------
    api_status = _token_status(settings.api.token, KNOWN_DEFAULT_API_TOKENS, PLACEHOLDER_API_TOKENS)
    runtime_status = _token_status(
        settings.runtime.token, KNOWN_DEFAULT_RUNTIME_TOKENS, PLACEHOLDER_RUNTIME_TOKENS
    )
    token_statuses = {api_status, runtime_status}
    if "placeholder-insecure" in token_statuses:
        # The .env.example placeholders are not secret. They are fine to discover
        # locally but must be replaced before any non-local exposure; never "ok".
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="Insecure placeholder tokens from .env.example are still active.",
                action=_SETUP_TOKENS,
            )
        )
    elif "default-development" in token_statuses:
        # Default tokens are fine for local development; they must be rotated
        # before any non-local exposure. A warning, not a hard model blocker.
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="Default development tokens are still active.",
                action=_SETUP_TOKENS,
            )
        )
    elif "missing" in token_statuses:
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="A loopback token is not configured.",
                action=_SETUP_TOKENS,
            )
        )
    else:
        checks.append(ReadinessCheck(name="api/runtime tokens", status="ok", detail="configured"))

    # --- voice artifacts (optional) -----------------------------------------
    voice_enabled = settings.voice.enabled
    voice_specs = (
        ("whisper.cpp binary", settings.voice.whisper_binary_path, True),
        ("whisper model", settings.voice.whisper_model_path, True),
        ("piper binary", settings.voice.piper_binary_path, True),
        ("piper voice model", settings.voice.piper_model_path, True),
        ("wake-word model", settings.voice.wake_word_model_path, False),
    )
    voice_artifacts: list[VoiceArtifact] = []
    for name, voice_path, required in voice_specs:
        artifact, check = _voice_artifact(
            settings, name, voice_path, enabled=voice_enabled, required=required
        )
        voice_artifacts.append(artifact)
        checks.append(check)

    wake_enabled = settings.wake.enabled
    checks.append(
        ReadinessCheck(
            name="speaker gate",
            # With wake enabled this is an explicit production limitation, not
            # a silent skip: no local speaker verification exists in this build.
            status="warning" if wake_enabled else "skipped",
            detail=(
                "speaker_gate is off; local speaker verification is unsupported in this build."
                + (" Anyone near the microphone can wake APRIL." if wake_enabled else "")
            ),
        )
    )
    if wake_enabled and not settings.voice.effective_wake_word_model_paths:
        checks.append(
            ReadinessCheck(
                name="wake-word ONNX model",
                status="blocker",
                detail="wake.enabled is on but no wake-word model path is configured.",
                action=_SETUP_VOICE,
            )
        )
    sentinel_live_status = _sentinel_live_status(root)
    if sentinel_live_status == "verified":
        sentinel_check_status: CheckStatus = "ok"
        sentinel_detail = "Latest wake-word live report verified the Sentinel pipeline."
    elif wake_enabled:
        # Wake is enabled but never live-validated on this machine: warn.
        sentinel_check_status = "warning"
        sentinel_detail = (
            "wake.enabled is on but the Sentinel pipeline has no live validation "
            "record on this Mac."
        )
    else:
        sentinel_check_status = "skipped"
        sentinel_detail = "Sentinel live pipeline has not been verified on this Mac."
    checks.append(
        ReadinessCheck(
            name="Sentinel live verification",
            status=sentinel_check_status,
            detail=sentinel_detail,
            action=None if sentinel_live_status == "verified" else _VERIFY_WAKE,
        )
    )

    # --- evolution/scheduler wiring -------------------------------------------
    if settings.evolution.enabled and not settings.scheduler.enabled:
        checks.append(
            ReadinessCheck(
                name="evolution scheduling",
                status="warning",
                detail=(
                    "evolution.enabled is on but scheduler.enabled is off; the nightly "
                    "Dreamer will never run automatically."
                ),
                action="Set scheduler.enabled: true (or run the Dreamer manually).",
            )
        )
    if settings.scheduler.enabled and settings.evolution.enabled:
        kill_switch = settings.evolution_path / "DISABLED"
        if kill_switch.exists():
            checks.append(
                ReadinessCheck(
                    name="evolution kill switch",
                    status="warning",
                    detail="Scheduler and evolution are enabled but the local kill switch "
                    "file is present; Dreamer runs are blocked.",
                    action="Remove data/evolution/DISABLED to re-enable nightly evolution.",
                )
            )

    # --- unreviewed evolution artifacts ---------------------------------------
    pending_overlays = _pending_write_capable_overlay_count(settings)
    if pending_overlays:
        checks.append(
            ReadinessCheck(
                name="prompt overlay review",
                status="warning",
                detail=(
                    f"{pending_overlays} overlay candidate(s) for write-capable agents "
                    "are stored and may await review (already-applied candidates are "
                    "listed precisely by the API)."
                ),
                action="run april evolve overlays pending",
            )
        )
    pending_evals = _pending_eval_case_count(settings)
    if pending_evals:
        checks.append(
            ReadinessCheck(
                name="pending eval cases",
                status="warning",
                detail=f"{pending_evals} staged eval case(s) have not been reviewed.",
                action="Review data/evolution/evals/pending and promote or delete cases.",
            )
        )
    daemon_status_payload = _daemon_status(settings)
    daemon_status = str(daemon_status_payload.get("status", "unknown"))
    daemon_details_available = bool(daemon_status_payload.get("details_available", False))
    checks.append(
        ReadinessCheck(
            name="daemon detailed status",
            status="ok" if daemon_details_available else "warning",
            detail=(
                f"details available; daemon status={daemon_status}"
                if daemon_details_available
                else f"details unavailable; daemon status={daemon_status}"
            ),
            action="run april daemon status" if not daemon_details_available else None,
        )
    )

    # --- aggregate -----------------------------------------------------------
    blockers = [check.name for check in checks if check.status == "blocker"]
    warnings = [check.name for check in checks if check.status == "warning"]
    # Voice readiness is its own axis; model blockers are the voice "voice:" rows.
    model_blockers = [name for name in blockers if not name.startswith("voice:")]
    voice_blockers = [name for name in blockers if name.startswith("voice:")]
    real_model_preflight_ready = not model_blockers
    voice_preflight_ready = voice_enabled and not voice_blockers

    checks.append(
        ReadinessCheck(
            name="real-model verification",
            status="skipped",
            detail="Offline readiness did not load/chat/stream/unload a GGUF model.",
            action=_VERIFY_REAL,
        )
    )
    checks.append(
        ReadinessCheck(
            name="live voice verification",
            status="skipped",
            detail=(
                "Offline readiness did not run microphone/STT/TTS playback verification."
                if voice_enabled
                else "Voice disabled; live verification not requested."
            ),
            action=_VERIFY_VOICE if voice_enabled else None,
        )
    )

    next_actions: list[str] = []
    for check in checks:
        if check.action and check.action not in next_actions:
            next_actions.append(check.action)
    # Always end with the authoritative real verification command.
    if _VERIFY_REAL not in next_actions:
        next_actions.append(_VERIFY_REAL)
    if voice_enabled and _VERIFY_VOICE not in next_actions:
        next_actions.append(_VERIFY_VOICE)
    if voice_enabled and _VERIFY_WAKE not in next_actions:
        next_actions.append(_VERIFY_WAKE)

    return ReadinessReport(
        generated_at=utc_now_iso(),
        os=f"{platform.system()} {platform.release()}".strip(),
        cpu_architecture=platform.machine(),
        python_version=platform.python_version(),
        runtime_backend=backend,
        runtime_is_fake=runtime_is_fake,
        llama_cpp_python_available=llama_available,
        environment=settings.environment,
        voice_enabled=voice_enabled,
        real_model_preflight_ready=real_model_preflight_ready,
        voice_preflight_ready=voice_preflight_ready,
        models=models,
        voice_artifacts=voice_artifacts,
        api_token_status=api_status,
        runtime_token_status=runtime_status,
        speaker_gate=settings.wake.speaker_gate,
        speaker_gate_supported=False,
        daemon_status=daemon_status,
        daemon_details_available=daemon_details_available,
        sentinel_live_status=sentinel_live_status,
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        next_actions=next_actions,
    )


def _pending_write_capable_overlay_count(settings: AprilSettings) -> int:
    """File-only count of stored overlay candidates for write-capable agents.

    Readiness stays inert (no DB connection), so this may include candidates a
    later approval already applied; the check wording says so and points at the
    precise API listing.
    """
    candidates_dir = settings.evolution_path / "candidates"
    if not candidates_dir.is_dir():
        return 0
    write_capable = {"coding_agent", "system_action_agent"}
    count = 0
    for path in candidates_dir.glob("*.overlay.txt"):
        agent = path.name.rsplit("-", 1)[0]
        if agent in write_capable:
            count += 1
    return count


def _pending_eval_case_count(settings: AprilSettings) -> int:
    pending_dir = settings.evolution_path / "evals" / "pending"
    if not pending_dir.is_dir():
        return 0
    return sum(1 for path in pending_dir.glob("*.yaml") if path.is_file())


def _daemon_status(settings: AprilSettings) -> dict[str, object]:
    try:
        from apps.daemon.apriald import read_daemon_status

        return read_daemon_status(settings)
    except Exception:
        return {"status": "unknown", "details_available": False}


def _sentinel_live_status(home: Path) -> str:
    latest: tuple[float, bool] | None = None
    verification_dir = home / "data" / "verification"
    for path in verification_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("report_type") != "wake_word_live":
            continue
        if payload.get("pipeline") != "sentinel":
            continue
        order = path.stat().st_mtime
        verified = bool(payload.get("wake_word_live_verified", False))
        if latest is None or order > latest[0]:
            latest = (order, verified)
    if latest is None:
        return "not_verified"
    return "verified" if latest[1] else "failed"
