"""Guided, transactional local Mac activation wizard for APRIL.

``run april setup mac-activation`` gives one command that validates the intended
local GGUF model set and (optionally) the local voice tools, applies the config
only when ``--apply`` is supplied, and can chain straight into a real-model (and
optionally live voice/wake-word) acceptance run. It is a thin, redacted
orchestration layer over the existing, separately-tested ``setup_model_set`` /
``setup_voice_stack`` helpers and the acceptance runner — it never downloads
models, installs packages, runs ``sudo`` or Homebrew, records audio, or touches
the network.

Apply is **transactional**: every supplied model and voice path is validated
first, nothing is written if validation fails, and if a later apply step fails the
previous config files are restored automatically (unless ``--no-rollback`` is set
for debugging). The on-disk report is redacted by construction: only basenames,
booleans, counts, status strings, and commands — never tokens, transcripts,
generated text, or absolute paths.

The orchestrator is dependency-injected and separated from the CLI so it can be
unit-tested with mocked validators and a mocked acceptance runner, with no GGUF,
llama-cpp-python, microphone, speaker, whisper.cpp, Piper, openWakeWord, or
network required.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
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
_MODELS_CONFIG = ("configs", "models.yaml")
_VOICE_CONFIG = ("configs", "april.yaml")


class ActivationFlagError(ValueError):
    """Raised for an incompatible combination of activation flags."""


def validate_activation_flags(
    *,
    apply: bool,
    dry_run: bool,
    skip_voice: bool,
    enable_voice: bool,
    run_acceptance_after: bool,
    acceptance_voice_live: bool,
    acceptance_wake_word_live: bool,
    start_services: bool,
    fake_services: bool,
) -> None:
    """Reject contradictory activation flag combinations before doing any work."""
    if apply and dry_run:
        raise ActivationFlagError("Use either --apply or --dry-run, not both.")
    if enable_voice and skip_voice:
        raise ActivationFlagError("Cannot combine --enable-voice with --skip-voice.")
    live = acceptance_voice_live or acceptance_wake_word_live
    if live and not run_acceptance_after:
        raise ActivationFlagError(
            "--acceptance-voice-live/--acceptance-wake-word-live require --run-acceptance."
        )
    if live and skip_voice:
        raise ActivationFlagError("Live voice acceptance cannot be combined with --skip-voice.")
    if live and not enable_voice:
        raise ActivationFlagError("Live voice acceptance requires --enable-voice.")
    # Activation acceptance is always real-model, so fake services can never satisfy it.
    if fake_services and run_acceptance_after:
        raise ActivationFlagError(
            "--fake-services cannot verify real-model acceptance (--run-acceptance)."
        )
    if fake_services and not start_services:
        raise ActivationFlagError("--fake-services requires --start-services.")


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
    enabled_requested: bool = False
    enabled_after_apply: bool = False
    artifacts: list[VoiceArtifactEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class TransactionSummary(BaseModel):
    requested: bool = False
    backup_created: bool = False
    backup_basename: str | None = None
    committed: bool = False
    rolled_back: bool = False
    # not_applicable | not_needed | restored | failed | skipped
    rollback_status: str = "not_applicable"
    rollback_reason: str | None = None


class AcceptanceLink(BaseModel):
    ran: bool = False
    final_status: str | None = None
    acceptance_level: str | None = None
    runtime_backend: str | None = None
    real_model_summary: str | None = None
    voice_live_summary: str | None = None
    wake_word_live_summary: str | None = None
    services_startup: str | None = None
    skipped_reason: str | None = None


class MacActivationReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["mac_activation"] = "mac_activation"
    generated_at: str
    mode: Literal["dry_run", "apply"]
    models: ModelsActivationSummary
    voice: VoiceActivationSummary
    transaction: TransactionSummary = Field(default_factory=TransactionSummary)
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
    enable_voice: bool,
    voice_setup: VoiceSetup,
) -> VoiceActivationSummary:
    if skip_voice:
        return VoiceActivationSummary(skipped=True)
    missing = [name for name in _VOICE_REQUIRED if voice_paths.get(name) is None]
    summary = VoiceActivationSummary(requested=True, enabled_requested=enable_voice)
    if missing:
        summary.error = (
            "Voice activation needs --whisper-binary, --whisper-model, --piper-binary, "
            "and --piper-model (or pass --skip-voice)."
        )
        return summary
    try:
        # Voice is enabled only when --apply AND --enable-voice are both supplied,
        # and only after every required artifact validates inside setup_voice_stack.
        result = voice_setup(
            home=home,
            whisper_binary=voice_paths["whisper_binary"],
            whisper_model=voice_paths["whisper_model"],
            piper_binary=voice_paths["piper_binary"],
            piper_model=voice_paths["piper_model"],
            wake_word_model=voice_paths.get("wake_word_model"),
            apply=apply,
            enable=enable_voice and apply,
        )
    except ConfigError as exc:
        summary.error = redact_reason(str(exc))
        return summary
    summary.validated = True
    summary.applied = bool(result.get("applied"))
    summary.enabled = bool(result.get("voice_enabled"))
    summary.enabled_after_apply = bool(result.get("voice_enabled"))
    for artifact in result.get("artifacts", []):
        summary.artifacts.append(
            VoiceArtifactEntry(
                name=str(artifact.get("name")),
                basename=artifact.get("basename") or None,
            )
        )
    summary.warnings = [redact_reason(str(warning)) for warning in result.get("warnings", [])]
    return summary


@dataclass
class _ConfigBackup:
    directory: Path
    # config path -> whether it existed at snapshot time
    entries: dict[Path, bool] = field(default_factory=dict)


def _config_paths_to_protect(home: Path, *, skip_voice: bool) -> list[Path]:
    paths = [home.joinpath(*_MODELS_CONFIG)]
    if not skip_voice:
        paths.append(home.joinpath(*_VOICE_CONFIG))
    return paths


def _snapshot_config(home: Path, paths: list[Path]) -> _ConfigBackup:
    stamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    directory = home / ".april_tmp" / f"mac-activation-backup-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    backup = _ConfigBackup(directory=directory)
    for path in paths:
        existed = path.exists()
        backup.entries[path] = existed
        if existed:
            shutil.copy2(path, directory / path.name)
    return backup


def _restore_config(backup: _ConfigBackup) -> bool:
    try:
        for path, existed in backup.entries.items():
            if existed:
                shutil.copy2(backup.directory / path.name, path)
            elif path.exists():
                path.unlink()
        return True
    except OSError:
        return False


def _apply_transaction(
    *,
    home: Path,
    model_paths: dict[str, Path | None],
    voice_paths: dict[str, Path | None],
    skip_voice: bool,
    enable_voice: bool,
    model_setup: ModelSetup,
    voice_setup: VoiceSetup,
    no_rollback: bool,
) -> tuple[ModelsActivationSummary, VoiceActivationSummary, TransactionSummary]:
    backup = _snapshot_config(home, _config_paths_to_protect(home, skip_voice=skip_voice))
    transaction = TransactionSummary(
        requested=True, backup_created=True, backup_basename=backup.directory.name
    )

    models = _model_summary(home=home, model_paths=model_paths, apply=True, model_setup=model_setup)
    voice = VoiceActivationSummary(skipped=True)
    failure_reason = models.error
    if not failure_reason and not skip_voice:
        voice = _voice_summary(
            home=home,
            voice_paths=voice_paths,
            skip_voice=False,
            apply=True,
            enable_voice=enable_voice,
            voice_setup=voice_setup,
        )
        failure_reason = voice.error

    if failure_reason:
        transaction.committed = False
        if no_rollback:
            # Debug-only: leave the partial state in place but report it honestly.
            transaction.rollback_status = "skipped"
            transaction.rollback_reason = redact_reason(failure_reason)
        else:
            restored = _restore_config(backup)
            transaction.rolled_back = restored
            transaction.rollback_status = "restored" if restored else "failed"
            transaction.rollback_reason = redact_reason(failure_reason)
    else:
        transaction.committed = True
        transaction.rollback_status = "not_needed"
    return models, voice, transaction


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
            ran=False, skipped_reason="Activation apply failed/rolled back; acceptance not run."
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
        voice_live_summary=report.voice_live.summary if report.voice_live is not None else None,
        wake_word_live_summary=(
            report.wake_word_live.summary if report.wake_word_live is not None else None
        ),
        services_startup=(report.services.startup_status if report.services.requested else None),
    )


def _next_actions(
    *,
    apply: bool,
    models: ModelsActivationSummary,
    voice: VoiceActivationSummary,
    transaction: TransactionSummary,
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
    if transaction.rolled_back:
        add("Apply failed and was rolled back; fix the reported path and re-run with --apply.")
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
                "run april setup mac-activation ... --enable-voice --apply, then "
                "run april voice verify-live --report data/verification/voice-live.json"
            )
        if voice.enabled:
            add("Voice is ON. Verify it with run april voice verify-live and verify-wake-live.")
    return actions


def run_mac_activation(
    home: Path,
    *,
    model_paths: dict[str, Path | None],
    voice_paths: dict[str, Path | None],
    skip_voice: bool = False,
    apply: bool = False,
    enable_voice: bool = False,
    run_acceptance_after: bool = False,
    no_rollback: bool = False,
    model_setup: ModelSetup | None = None,
    voice_setup: VoiceSetup | None = None,
    acceptance_runner: AcceptanceRunner | None = None,
) -> MacActivationReport:
    """Validate, then transactionally apply, the local model + voice activation.

    Dry-run by default. ``model_setup`` / ``voice_setup`` / ``acceptance_runner``
    are injected so the wizard can be exercised without real GGUF files, voice
    binaries, or a real acceptance run; they default to the real, module-level
    functions resolved at call time so they stay monkeypatchable.
    """
    home = home.expanduser()
    model_setup = model_setup or setup_model_set
    voice_setup = voice_setup or setup_voice_stack

    # Phase 1 — validate everything up front. This never writes config.
    models = _model_summary(
        home=home, model_paths=model_paths, apply=False, model_setup=model_setup
    )
    voice = _voice_summary(
        home=home,
        voice_paths=voice_paths,
        skip_voice=skip_voice,
        apply=False,
        enable_voice=enable_voice,
        voice_setup=voice_setup,
    )
    validation_blocked = bool(models.error) or bool(voice.error)
    transaction = TransactionSummary(requested=apply)

    # Phase 2 — apply transactionally only when validation passed.
    if apply and not validation_blocked:
        models, voice, transaction = _apply_transaction(
            home=home,
            model_paths=model_paths,
            voice_paths=voice_paths,
            skip_voice=skip_voice,
            enable_voice=enable_voice,
            model_setup=model_setup,
            voice_setup=voice_setup,
            no_rollback=no_rollback,
        )

    apply_failed = apply and (bool(models.error) or bool(voice.error))
    blocked = validation_blocked or apply_failed

    if blocked:
        final_status: ActivationStatus = "failed"
    elif apply:
        final_status = "applied"
    else:
        final_status = "validated"

    acceptance = _acceptance_link(
        apply=apply,
        run_acceptance_after=run_acceptance_after,
        blocked=blocked,
        acceptance_runner=acceptance_runner,
    )

    next_actions = _next_actions(
        apply=apply,
        models=models,
        voice=voice,
        transaction=transaction,
        acceptance=acceptance,
        final_status=final_status,
    )

    return MacActivationReport(
        generated_at=utc_now_iso(),
        mode="apply" if apply else "dry_run",
        models=models,
        voice=voice,
        transaction=transaction,
        acceptance=acceptance,
        final_status=final_status,
        next_actions=next_actions,
    )
