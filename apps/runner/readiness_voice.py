from __future__ import annotations

import json
from pathlib import Path

from apps.runner.mac_report import redact_reason
from apps.runner.readiness_models import (
    _SETUP_VOICE,
    CheckStatus,
    ReadinessCheck,
    VoiceArtifact,
)
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.report_freshness import freshness_from_payload
from april_common.settings import AprilSettings


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
        freshness = freshness_from_payload(
            payload,
            report_type="wake_word_live",
            current_fingerprint=config_fingerprint_digest(home),
            basename=path.name,
        )
        verified = bool(
            payload.get("evidence_mode") == "real_hardware"
            and payload.get("wake_word_live_verified", False)
            and not freshness.stale
        )
        if latest is None or order > latest[0]:
            latest = (order, verified)
    if latest is None:
        return "not_verified"
    return "verified" if latest[1] else "failed"


def _voice_live_status(home: Path) -> str:
    return _single_live_report_status(
        home,
        basename="voice-live.json",
        report_type="voice_live",
        verified_field="voice_live_verified",
    )


def _voice_conversation_live_status(home: Path) -> str:
    return _single_live_report_status(
        home,
        basename="voice-conversation-live.json",
        report_type="voice_conversation_live",
        verified_field="voice_conversation_live_verified",
    )


def _speaker_live_status(home: Path) -> str:
    return _single_live_report_status(
        home,
        basename="speaker-live.json",
        report_type="speaker_live",
        verified_field="speaker_live_verified",
    )


def _speaker_gate_check(
    settings: AprilSettings,
    *,
    wake_enabled: bool,
) -> tuple[bool, str, ReadinessCheck]:
    speaker_soft = settings.wake.speaker_gate == "soft"
    speaker_model_path = settings.wake.speaker_verifier_model_path
    speaker_model_configured = speaker_model_path is not None
    speaker_model_exists = bool(
        speaker_model_path is not None and settings.resolve_path(speaker_model_path).is_file()
    )
    if speaker_soft and speaker_model_exists:
        from services.wake.speaker import onnxruntime_importable

        speaker_runtime_available = onnxruntime_importable()
    else:
        speaker_runtime_available = False
    supported = bool(
        speaker_soft
        and speaker_model_configured
        and speaker_model_exists
        and speaker_runtime_available
    )
    live_status = _speaker_live_status(settings.home)
    if supported and live_status == "verified":
        detail = (
            "speaker_gate=soft has a configured local ONNX model, ONNX Runtime, "
            "and a fresh successful real-hardware speaker-live report."
        )
    elif supported:
        detail = (
            "speaker_gate=soft has its model and runtime configured, but a fresh "
            "real-hardware speaker-live report is still required."
        )
    elif speaker_soft and not speaker_model_configured:
        detail = (
            "speaker_gate=soft is configured without "
            "wake.speaker_verifier_model_path; Sentinel degrades to off with one audited "
            "warning. Follow scripts/speaker_verifier/README.md."
        )
    elif speaker_soft and not speaker_model_exists:
        detail = (
            "wake.speaker_verifier_model_path does not name an existing local file; "
            "Sentinel degrades to off with one audited warning. Follow "
            "scripts/speaker_verifier/README.md."
        )
    elif speaker_soft:
        detail = (
            "The optional onnxruntime dependency is not importable; Sentinel degrades "
            "to off with one audited warning. Install APRIL's voice extra and follow "
            "scripts/speaker_verifier/README.md."
        )
    else:
        detail = (
            "speaker_gate is off. `april voice enroll` records local samples but does "
            "not enable soft mode by itself. Configure wake.speaker_verifier_model_path "
            "as described in scripts/speaker_verifier/README.md before enabling it."
        )
    check = ReadinessCheck(
        name="speaker gate",
        status=(
            "skipped"
            if not speaker_soft
            else ("ok" if supported and live_status == "verified" else "blocker")
        ),
        detail=(
            detail
            + " The speaker gate is a convenience filter, never a security boundary."
            + (" Anyone near the microphone can wake APRIL." if wake_enabled else "")
        ),
        action=(
            "run april voice verify-speaker-live --report data/verification/speaker-live.json"
            if speaker_soft and live_status != "verified"
            else None
        ),
    )
    return supported, live_status, check


def _single_live_report_status(
    home: Path,
    *,
    basename: str,
    report_type: str,
    verified_field: str,
) -> str:
    path = home / "data" / "verification" / basename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "not_verified"
    if not isinstance(payload, dict) or payload.get("report_type") != report_type:
        return "not_verified"
    verified = (
        payload.get("evidence_mode") == "real_hardware"
        and payload.get(verified_field) is True
        and not freshness_from_payload(
            payload,
            report_type=report_type,
            current_fingerprint=config_fingerprint_digest(home),
            basename=path.name,
        ).stale
    )
    return "verified" if verified else "failed"
