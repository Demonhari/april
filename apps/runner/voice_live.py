from __future__ import annotations

import contextlib
import platform
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from apps.runner.mac_report import redact_reason
from april_common.errors import RuntimeUnavailableError
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.voice.audio_player import AudioPlayer, SoundDeviceAudioPlayer
from services.voice.conversation_loop import NoSpeechDetected, require_usable_transcript
from services.voice.health import query_audio_devices, voice_doctor
from services.voice.microphone import Microphone, SoundDeviceMicrophone
from services.voice.speech_to_text import SpeechToText, WhisperCppSpeechToText
from services.voice.text_to_speech import PiperTextToSpeech, TextToSpeech


class VoiceLiveSkippedCheck(BaseModel):
    name: str
    reason: str


class VoiceLiveReport(BaseModel):
    schema_version: int = 1
    report_type: str = "voice_live"
    # Only the non-injected CLI path may produce real-hardware evidence. Tests
    # can exercise the complete pipeline, but cannot lift production readiness.
    evidence_mode: Literal["real_hardware", "injected_test", "unverified"] = "unverified"
    # ``generated_at`` is the canonical timestamp the Desktop/report viewer reads,
    # matching the other verification reports; ``timestamp`` is retained for
    # backward compatibility with any already-written voice-live reports. When only
    # ``timestamp`` is supplied, ``generated_at`` mirrors it (see validator below).
    generated_at: str = ""
    timestamp: str = ""
    platform: str
    sounddevice_available: bool
    input_device_count: int
    output_device_count: int
    whisper_binary_available: bool
    whisper_model_available: bool
    piper_binary_available: bool
    piper_model_available: bool
    wake_word_model_available: bool
    recording_success: bool = False
    stt_success: bool = False
    transcript_length: int = 0
    transcription_user_confirmed: bool = False
    tts_success: bool = False
    playback_user_confirmed: bool = False
    temp_audio_retained: bool = False
    skipped: list[VoiceLiveSkippedCheck] = Field(default_factory=list)
    summary: str = "degraded"
    # True only when the full live loop genuinely passed (all five checks confirmed
    # AND summary == "pass"). A degraded/failed/skipped run can never set this true,
    # so a voice report can never be mistaken for a verified live voice pass.
    voice_live_verified: bool = False
    voice_timing: dict[str, float] | None = None

    @model_validator(mode="after")
    def _mirror_timestamp(self) -> VoiceLiveReport:
        # Keep the two timestamps consistent: fill whichever was omitted from the
        # other so older callers (timestamp only) and the report viewer agree.
        if not self.generated_at and self.timestamp:
            self.generated_at = self.timestamp
        elif not self.timestamp and self.generated_at:
            self.timestamp = self.generated_at
        elif not self.generated_at and not self.timestamp:
            now = utc_now_iso()
            self.generated_at = now
            self.timestamp = now
        return self


Confirm = Callable[[str], bool]
TranscriptObserver = Callable[[str], None]


def _resolved(settings: AprilSettings, path: Path | None) -> Path | None:
    return None if path is None else settings.resolve_path(path)


def _available(settings: AprilSettings, path: Path | None) -> bool:
    resolved = _resolved(settings, path)
    return resolved is not None and resolved.exists()


def _initial_report(
    settings: AprilSettings,
    *,
    evidence_mode: Literal["real_hardware", "injected_test"],
) -> VoiceLiveReport:
    devices = query_audio_devices()
    now = utc_now_iso()
    return VoiceLiveReport(
        evidence_mode=evidence_mode,
        generated_at=now,
        timestamp=now,
        platform=f"{platform.system()} {platform.release()}".strip(),
        sounddevice_available=bool(devices.get("sounddevice_installed")),
        input_device_count=len(devices.get("input_devices", [])),
        output_device_count=len(devices.get("output_devices", [])),
        whisper_binary_available=_available(settings, settings.voice.whisper_binary_path),
        whisper_model_available=_available(settings, settings.voice.whisper_model_path),
        piper_binary_available=_available(settings, settings.voice.piper_binary_path),
        piper_model_available=_available(settings, settings.voice.piper_model_path),
        wake_word_model_available=_available(settings, settings.voice.wake_word_model_path),
    )


def _finalize_summary(report: VoiceLiveReport) -> None:
    full_pass = (
        report.recording_success
        and report.stt_success
        and report.transcription_user_confirmed
        and report.tts_success
        and report.playback_user_confirmed
    )
    if full_pass:
        report.summary = "pass"
    elif report.recording_success or report.stt_success or report.tts_success:
        report.summary = "degraded"
    else:
        report.summary = "fail" if not report.skipped else "degraded"
    # Live-verified requires the full pass AND a "pass" summary; never set on a
    # degraded, failed, or skipped run.
    report.voice_live_verified = (
        full_pass and report.summary == "pass" and report.evidence_mode == "real_hardware"
    )


def voice_live_failure_reasons(report: VoiceLiveReport) -> list[str]:
    """Return safe, typed explanations for a non-passing voice-live report.

    These messages are derived from report booleans and the already-redacted
    skipped-check reasons. They intentionally do not include transcripts,
    paths, device names, or raw adapter errors.
    """

    reasons: list[str] = []
    if not report.recording_success:
        reasons.append("Recording was not successful.")
    if not report.stt_success:
        reasons.append("Speech-to-text was not successful.")
    if not report.transcription_user_confirmed:
        reasons.append("Transcription was not confirmed.")
    if not report.tts_success:
        reasons.append("Text-to-speech was not successful.")
    if not report.playback_user_confirmed:
        reasons.append("Playback was not confirmed.")
    for skipped in report.skipped:
        reasons.append(f"{skipped.name}: {redact_reason(skipped.reason)}")
    return reasons


def write_voice_live_report(report: VoiceLiveReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


async def run_voice_live_verification(
    *,
    settings: AprilSettings,
    confirm_recording: Confirm,
    confirm_transcription: Confirm,
    confirm_playback: Confirm,
    seconds: float = 3.0,
    retain_debug_audio: bool = False,
    microphone: Microphone | None = None,
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
    player: AudioPlayer | None = None,
    transcript_observer: TranscriptObserver | None = None,
    timing_observer: Callable[[dict[str, float]], None] | None = None,
    report_path: Path | None = None,
) -> VoiceLiveReport:
    # Runs doctor first for operator guidance, but the report stores only safe
    # counts/booleans, never device names, transcripts, or filesystem paths.
    voice_doctor(settings)
    evidence_mode: Literal["real_hardware", "injected_test"] = (
        "injected_test"
        if any(component is not None for component in (microphone, stt, tts, player))
        else "real_hardware"
    )
    report = _initial_report(settings, evidence_mode=evidence_mode)
    retain_audio = retain_debug_audio or settings.voice.retain_debug_audio
    report.temp_audio_retained = retain_audio
    if not confirm_recording("Record a short push-to-talk sample now?"):
        report.skipped.append(
            VoiceLiveSkippedCheck(name="recording", reason="user denied recording")
        )
        _finalize_summary(report)
        if report_path is not None:
            write_voice_live_report(report, report_path)
        return report

    settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
    stem = f"voice-live-{uuid.uuid4().hex}"
    input_path = settings.audio_cache_path / f"{stem}-input.wav"
    output_path = settings.audio_cache_path / f"{stem}-piper.wav"
    created_paths = [input_path, output_path]

    mic = microphone or SoundDeviceMicrophone(
        device=settings.voice.input_device,
        max_seconds=seconds,
    )
    speech = stt or WhisperCppSpeechToText(
        _resolved(settings, settings.voice.whisper_binary_path),
        _resolved(settings, settings.voice.whisper_model_path),
    )
    synthesizer = tts or PiperTextToSpeech(
        _resolved(settings, settings.voice.piper_binary_path),
        _resolved(settings, settings.voice.piper_model_path),
    )
    audio_player = player or SoundDeviceAudioPlayer(device=settings.voice.output_device)

    try:
        capture_started = time.monotonic()
        recorded_path = await mic.record_push_to_talk(input_path)
        capture_finished = time.monotonic()
        stt_started = capture_finished
        report.recording_success = recorded_path.exists()
        transcript = require_usable_transcript(await speech.transcribe(recorded_path))
        api_started = time.monotonic()
        report.stt_success = True
        report.transcript_length = len(transcript)
        if transcript_observer is not None:
            transcript_observer(transcript)
        report.transcription_user_confirmed = confirm_transcription(
            "Was the transcription correct? The report stores only transcript length."
        )
        api_finished = time.monotonic()
        tts_started = api_finished
        spoken_path = await synthesizer.synthesize("APRIL voice verification.", output_path)
        player_started = time.monotonic()
        report.tts_success = spoken_path.exists()
        await audio_player.play(spoken_path)
        report.voice_timing = {
            "capture_ms": (capture_finished - capture_started) * 1000.0,
            "stt_ms": (api_started - stt_started) * 1000.0,
            "api_ms": 0.0,
            "tts_ms": (player_started - tts_started) * 1000.0,
            "time_to_first_audio_ms": (player_started - capture_finished) * 1000.0,
        }
        report.voice_timing["api_ms"] = (api_finished - api_started) * 1000.0
        if timing_observer is not None:
            timing_observer(dict(report.voice_timing))
        report.playback_user_confirmed = confirm_playback("Did you hear the playback?")
    except KeyboardInterrupt:
        report.skipped.append(VoiceLiveSkippedCheck(name="voice-live", reason="interrupted"))
    except RuntimeUnavailableError as exc:
        report.skipped.append(
            VoiceLiveSkippedCheck(name="voice-live", reason=redact_reason(exc.message))
        )
    except NoSpeechDetected as exc:
        report.skipped.append(VoiceLiveSkippedCheck(name="transcription", reason=str(exc)))
    finally:
        if not retain_audio:
            for path in created_paths:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    _finalize_summary(report)
    if report_path is not None:
        write_voice_live_report(report, report_path)
    return report
