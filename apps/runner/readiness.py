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
_VERIFY_REAL = (
    "run april verify --all-configured-models --require-real-model "
    "--report data/verification/mac-readiness.json"
)
_VERIFY_VOICE = "run april voice verify-live --report data/verification/voice-live.json"


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
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        next_actions=next_actions,
    )
