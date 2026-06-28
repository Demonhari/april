"""Guided local Mac activation wizard for APRIL.

``run april setup mac-activation`` gives one command that validates the intended
local GGUF model set and (optionally) the local voice tools, applies the config
only when ``--apply`` is supplied, and can chain straight into a real-model
acceptance run. It is a thin, redacted orchestration layer over the existing,
separately-tested ``setup_model_set`` / ``setup_voice_stack`` helpers and
``run_acceptance`` — it never downloads models, installs packages, runs ``sudo``
or Homebrew, records audio, or touches the network.

The orchestrator is dependency-injected and separated from the CLI so it can be
unit-tested with mocked validators and a mocked acceptance runner, with no GGUF,
llama-cpp-python, microphone, speaker, whisper.cpp, Piper, openWakeWord, or
network required. The on-disk report is redacted by construction: only basenames,
booleans, counts, status strings, and commands — never tokens, transcripts,
generated text, or absolute paths.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from apps.runner.acceptance import AcceptanceReport
from apps.runner.mac_report import redact_reason
from apps.runner.model_tools import setup_model_set, setup_voice_stack
from april_common.errors import ConfigError
from april_common.time import utc_now, utc_now_iso

ActivationStatus = Literal["validated", "applied", "failed"]

# Injection points mirror the real helper signatures so tests can pass fakes.
ModelSetup = Callable[..., dict[str, Any]]
VoiceSetup = Callable[..., dict[str, Any]]
AcceptanceRunner = Callable[[], AcceptanceReport]

_MODEL_ROLES = ("brain", "coding", "reading")
_VOICE_REQUIRED = ("whisper_binary", "whisper_model", "piper_binary", "piper_model")


class ModelActivationEntry(BaseModel):
    role: str
    basename: str


class ModelsActivationSummary(BaseModel):
    requested: bool = False
    validated: bool = False
    applied: bool = False
    entries: list[ModelActivationEntry] = Field(default_factory=list)
    backup_basename: str | None = None
    error: str | None = None


class VoiceArtifactEntry(BaseModel):
    name: str
    basename: str | None = None


class VoiceActivationSummary(BaseModel):
    skipped: bool = False
    requested: bool = False
    validated: bool = False
    applied: bool = False
    enabled: bool = False
    artifacts: list[VoiceArtifactEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class AcceptanceLink(BaseModel):
    ran: bool = False
    final_status: str | None = None
    acceptance_level: str | None = None
    runtime_backend: str | None = None
    real_model_summary: str | None = None
    skipped_reason: str | None = None


class MacActivationReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["mac_activation"] = "mac_activation"
    generated_at: str
    mode: Literal["dry_run", "apply"]
    models: ModelsActivationSummary
    voice: VoiceActivationSummary
    acceptance: AcceptanceLink = Field(default_factory=AcceptanceLink)
    final_status: ActivationStatus
    next_actions: list[str] = Field(default_factory=list)


def default_activation_report_path(home: Path) -> Path:
    """Default ``--write-report`` location, under the Git-ignored verification dir."""
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return home.expanduser() / "data" / "verification" / f"mac-activation-{stamp}.json"


def write_activation_report(report: MacActivationReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


def _model_summary(
    *,
    home: Path,
    model_paths: dict[str, Path | None],
    apply: bool,
    model_setup: ModelSetup,
) -> ModelsActivationSummary:
    supplied = {role: path for role, path in model_paths.items() if path is not None}
    summary = ModelsActivationSummary(requested=bool(supplied))
    if not supplied:
        summary.error = "Supply at least one of --brain / --coding / --reading."
        return summary
    # Pre-validate the suffix locally so a non-GGUF path fails with a clear message
    # even before the shared validator runs (which also checks existence).
    for role, path in supplied.items():
        if path.suffix.lower() != ".gguf":
            summary.error = f"{role} model must be a local .gguf file: {path.name}"
            return summary
    try:
        result = model_setup(
            home=home,
            role_paths={role: model_paths.get(role) for role in _MODEL_ROLES},
            apply=apply,
        )
    except ConfigError as exc:
        summary.error = redact_reason(str(exc))
        return summary
    summary.validated = True
    summary.applied = bool(result.get("applied"))
    backup = result.get("backup_basename")
    summary.backup_basename = str(backup) if backup else None
    for entry in result.get("entries", []):
        basename = entry.get("source_basename") or entry.get("registered_basename") or ""
        summary.entries.append(
            ModelActivationEntry(role=str(entry.get("role")), basename=str(basename))
        )
    return summary


def _voice_summary(
    *,
    home: Path,
    voice_paths: dict[str, Path | None],
    skip_voice: bool,
    apply: bool,
    voice_setup: VoiceSetup,
) -> VoiceActivationSummary:
    if skip_voice:
        return VoiceActivationSummary(skipped=True)
    missing = [name for name in _VOICE_REQUIRED if voice_paths.get(name) is None]
    summary = VoiceActivationSummary(requested=True)
    if missing:
        summary.error = (
            "Voice activation needs --whisper-binary, --whisper-model, --piper-binary, "
            "and --piper-model (or pass --skip-voice)."
        )
        return summary
    try:
        # Voice paths are configured but never enabled by the wizard; enabling stays
        # an explicit, separate `setup voice --apply --enable` step (no surprises).
        result = voice_setup(
            home=home,
            whisper_binary=voice_paths["whisper_binary"],
            whisper_model=voice_paths["whisper_model"],
            piper_binary=voice_paths["piper_binary"],
            piper_model=voice_paths["piper_model"],
            wake_word_model=voice_paths.get("wake_word_model"),
            apply=apply,
            enable=False,
        )
    except ConfigError as exc:
        summary.error = redact_reason(str(exc))
        return summary
    summary.validated = True
    summary.applied = bool(result.get("applied"))
    summary.enabled = bool(result.get("voice_enabled"))
    for artifact in result.get("artifacts", []):
        summary.artifacts.append(
            VoiceArtifactEntry(
                name=str(artifact.get("name")),
                basename=artifact.get("basename") or None,
            )
        )
    summary.warnings = [redact_reason(str(warning)) for warning in result.get("warnings", [])]
    return summary


def _acceptance_link(
    *,
    apply: bool,
    run_acceptance_after: bool,
    blocked: bool,
    acceptance_runner: AcceptanceRunner | None,
) -> AcceptanceLink:
    if not run_acceptance_after:
        return AcceptanceLink(ran=False)
    if not apply:
        return AcceptanceLink(ran=False, skipped_reason="Acceptance runs only after --apply.")
    if blocked:
        return AcceptanceLink(
            ran=False, skipped_reason="Activation validation failed; acceptance not run."
        )
    if acceptance_runner is None:  # pragma: no cover - CLI always supplies a runner
        return AcceptanceLink(ran=False, skipped_reason="No acceptance runner available.")
    report = acceptance_runner()
    real = report.real_model_verification
    return AcceptanceLink(
        ran=True,
        final_status=report.final_status,
        acceptance_level=report.acceptance_level,
        runtime_backend=report.runtime_backend,
        real_model_summary=real.summary if real is not None else None,
    )


def _next_actions(
    *,
    apply: bool,
    models: ModelsActivationSummary,
    voice: VoiceActivationSummary,
    acceptance: AcceptanceLink,
    final_status: ActivationStatus,
) -> list[str]:
    actions: list[str] = []

    def add(action: str) -> None:
        if action and action not in actions:
            actions.append(action)

    if models.error:
        add(
            "run april setup mac-activation --brain /absolute/path/brain.gguf "
            "--coding /absolute/path/coding.gguf --reading /absolute/path/reading.gguf --dry-run"
        )
    if voice.error:
        add(
            "run april setup mac-activation ... --whisper-binary /absolute/path/whisper "
            "--whisper-model /absolute/path/model.bin --piper-binary /absolute/path/piper "
            "--piper-model /absolute/path/voice.onnx --dry-run  (or add --skip-voice)"
        )
    if final_status == "validated":
        add("Re-run with --apply to write the validated configuration.")
    if final_status == "applied":
        add("pip install -e '.[runtime]'")
        if not acceptance.ran:
            add(
                "run april acceptance --require-real-models --write-report  "
                "(or add --run-acceptance next time)"
            )
        if voice.applied and not voice.enabled:
            add(
                "Voice paths configured but OFF. Enable with "
                "run april setup voice ... --apply --enable, then "
                "run april voice verify-live --report data/verification/voice-live.json"
            )
    return actions


def run_mac_activation(
    home: Path,
    *,
    model_paths: dict[str, Path | None],
    voice_paths: dict[str, Path | None],
    skip_voice: bool = False,
    apply: bool = False,
    run_acceptance_after: bool = False,
    model_setup: ModelSetup | None = None,
    voice_setup: VoiceSetup | None = None,
    acceptance_runner: AcceptanceRunner | None = None,
) -> MacActivationReport:
    """Validate (and optionally apply) the local model + voice activation.

    Dry-run by default. ``model_setup`` / ``voice_setup`` / ``acceptance_runner``
    are injected so the wizard can be exercised without real GGUF files, voice
    binaries, or a real acceptance run. The setup helpers default to the real,
    module-level functions, resolved at call time so they stay monkeypatchable.
    """
    home = home.expanduser()
    models = _model_summary(
        home=home,
        model_paths=model_paths,
        apply=apply,
        model_setup=model_setup or setup_model_set,
    )
    voice = _voice_summary(
        home=home,
        voice_paths=voice_paths,
        skip_voice=skip_voice,
        apply=apply,
        voice_setup=voice_setup or setup_voice_stack,
    )

    blocked = bool(models.error) or bool(voice.error)
    acceptance = _acceptance_link(
        apply=apply,
        run_acceptance_after=run_acceptance_after,
        blocked=blocked,
        acceptance_runner=acceptance_runner,
    )

    if blocked:
        final_status: ActivationStatus = "failed"
    elif apply:
        final_status = "applied"
    else:
        final_status = "validated"

    next_actions = _next_actions(
        apply=apply, models=models, voice=voice, acceptance=acceptance, final_status=final_status
    )

    return MacActivationReport(
        generated_at=utc_now_iso(),
        mode="apply" if apply else "dry_run",
        models=models,
        voice=voice,
        acceptance=acceptance,
        final_status=final_status,
        next_actions=next_actions,
    )
