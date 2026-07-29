from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel

from april_common.config_fingerprint import config_fingerprint_digest
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.voice.microphone import Microphone, SoundDeviceMicrophone, read_pcm_wav
from services.wake.speaker import (
    MAX_SPEAKER_AUDIO_SAMPLES,
    SPEAKER_MATCH_THRESHOLD,
    OnnxSpeakerVerifier,
    SpeakerVerifier,
    onnxruntime_importable,
)

SPEAKER_REPORT_TTL_DAYS = 7


class SpeakerLiveReport(BaseModel):
    schema_version: int = 1
    report_type: str = "speaker_live"
    generated_at: str
    expires_after_days: int = SPEAKER_REPORT_TTL_DAYS
    config_fingerprint: str
    model_basename: str | None
    model_available: bool
    onnxruntime_available: bool
    enrollment_count: int
    rejected_fixture_count: int
    threshold: float
    fresh_similarity: float | None = None
    accepted_fixture_min_similarity: float | None = None
    rejected_fixture_max_similarity: float | None = None
    false_accept_fixture_passed: bool | None = None
    false_reject_fixture_passed: bool | None = None
    fresh_sample_passed: bool = False
    debug_audio_retained: bool = False
    speaker_live_verified: bool = False


Confirm = Callable[[str], bool]


async def run_speaker_live_verification(
    *,
    settings: AprilSettings,
    confirm_capture: Confirm,
    microphone: Microphone | None = None,
    verifier: SpeakerVerifier | None = None,
    retain_debug_audio: bool = False,
    report_path: Path | None = None,
) -> SpeakerLiveReport:
    model = (
        settings.resolve_path(settings.wake.speaker_verifier_model_path)
        if settings.wake.speaker_verifier_model_path is not None
        else None
    )
    model_available = bool(model is not None and model.is_file())
    runtime_available = onnxruntime_importable() if verifier is None else True
    enrollment = _reviewed_wavs(settings.home / "data" / "voice_profiles")
    rejected = _reviewed_wavs(settings.home / "data" / "voice_profiles" / "rejected")
    report = SpeakerLiveReport(
        generated_at=utc_now_iso(),
        config_fingerprint=config_fingerprint_digest(settings.home),
        model_basename=model.name if model is not None else None,
        model_available=model_available,
        onnxruntime_available=runtime_available,
        enrollment_count=len(enrollment),
        rejected_fixture_count=len(rejected),
        threshold=SPEAKER_MATCH_THRESHOLD,
        debug_audio_retained=retain_debug_audio,
    )
    if not model_available or not runtime_available or not enrollment:
        _write_optional(report, report_path)
        return report
    if not confirm_capture(
        "Capture a fresh local speaker-verification sample now? Raw audio is deleted by default."
    ):
        _write_optional(report, report_path)
        return report

    assert model is not None
    active_verifier = verifier or OnnxSpeakerVerifier(model)
    cache = settings.audio_cache_path
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    capture = cache / f"speaker-live-{uuid.uuid4().hex}.wav"
    mic = microphone or SoundDeviceMicrophone(
        device=settings.voice.input_device,
        max_seconds=5.0,
    )
    try:
        recorded = await mic.record_push_to_talk(capture)
        pcm = read_pcm_wav(recorded, max_frames=MAX_SPEAKER_AUDIO_SAMPLES)
        report.fresh_similarity = _bounded_score(active_verifier.score(enrollment, pcm))
        report.fresh_sample_passed = report.fresh_similarity >= SPEAKER_MATCH_THRESHOLD
        accepted_scores = [
            _bounded_score(
                active_verifier.score(
                    enrollment,
                    read_pcm_wav(path, max_frames=MAX_SPEAKER_AUDIO_SAMPLES),
                )
            )
            for path in enrollment
        ]
        report.accepted_fixture_min_similarity = min(accepted_scores)
        report.false_reject_fixture_passed = all(
            score >= SPEAKER_MATCH_THRESHOLD for score in accepted_scores
        )
        if rejected:
            rejected_scores = [
                _bounded_score(
                    active_verifier.score(
                        enrollment,
                        read_pcm_wav(path, max_frames=MAX_SPEAKER_AUDIO_SAMPLES),
                    )
                )
                for path in rejected
            ]
            report.rejected_fixture_max_similarity = max(rejected_scores)
            report.false_accept_fixture_passed = all(
                score < SPEAKER_MATCH_THRESHOLD for score in rejected_scores
            )
        report.speaker_live_verified = bool(
            report.fresh_sample_passed
            and report.false_reject_fixture_passed
            and report.false_accept_fixture_passed is not False
        )
    finally:
        if not retain_debug_audio:
            with contextlib.suppress(OSError):
                capture.unlink()
    _write_optional(report, report_path)
    return report


def enable_soft_speaker_gate(settings: AprilSettings, report_path: Path) -> None:
    from april_common.report_freshness import freshness_from_payload

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = SpeakerLiveReport.model_validate(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Speaker verification report is missing or invalid.") from exc
    freshness = freshness_from_payload(
        payload,
        report_type="speaker_live",
        current_fingerprint=config_fingerprint_digest(settings.home),
        basename=report_path.name,
    )
    if not report.speaker_live_verified or freshness.stale:
        raise ValueError("A fresh successful speaker-live report is required.")
    _set_speaker_gate(settings.home / "configs" / "april.yaml", "soft")


def disable_speaker_gate(settings: AprilSettings) -> None:
    _set_speaker_gate(settings.home / "configs" / "april.yaml", "off")


def write_speaker_live_report(report: SpeakerLiveReport, path: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _reviewed_wavs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    root = directory.resolve()
    reviewed: list[Path] = []
    for candidate in sorted(directory.glob("*.wav")):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate.is_symlink() or not resolved.is_file():
            continue
        read_pcm_wav(resolved, max_frames=MAX_SPEAKER_AUDIO_SAMPLES)
        reviewed.append(resolved)
    return reviewed


def _bounded_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Speaker verifier returned an invalid similarity.")
    return score


def _write_optional(report: SpeakerLiveReport, path: Path | None) -> None:
    if path is not None:
        write_speaker_live_report(report, path)


def _set_speaker_gate(config_path: Path, value: str) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    wake = data.setdefault("wake", {})
    if not isinstance(wake, dict):
        raise ValueError("wake configuration must be a mapping.")
    wake["speaker_gate"] = value
    descriptor, name = tempfile.mkstemp(prefix=f".{config_path.name}.", dir=config_path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)
