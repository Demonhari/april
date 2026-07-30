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


def _voice_conversation_live_status(home: Path) -> str:
    path = home / "data" / "verification" / "voice-conversation-live.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "not_verified"
    if not isinstance(payload, dict) or payload.get("report_type") != "voice_conversation_live":
        return "not_verified"
    verified = (
        payload.get("evidence_mode") == "real_hardware"
        and payload.get("voice_conversation_live_verified") is True
        and not freshness_from_payload(
            payload,
            report_type="voice_conversation_live",
            current_fingerprint=config_fingerprint_digest(home),
            basename=path.name,
        ).stale
    )
    return "verified" if verified else "failed"
