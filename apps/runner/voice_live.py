from __future__ import annotations

import contextlib
import platform
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from april_common.errors import RuntimeUnavailableError
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.voice.audio_player import AudioPlayer, SoundDeviceAudioPlayer
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
    timestamp: str
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


Confirm = Callable[[str], bool]


def _resolved(settings: AprilSettings, path: Path | None) -> Path | None:
    return None if path is None else settings.resolve_path(path)


def _available(settings: AprilSettings, path: Path | None) -> bool:
    resolved = _resolved(settings, path)
    return resolved is not None and resolved.exists()


def _initial_report(settings: AprilSettings) -> VoiceLiveReport:
    devices = query_audio_devices()
    return VoiceLiveReport(
        timestamp=utc_now_iso(),
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
    if (
        report.recording_success
        and report.stt_success
        and report.transcription_user_confirmed
        and report.tts_success
        and report.playback_user_confirmed
    ):
        report.summary = "pass"
    elif report.recording_success or report.stt_success or report.tts_success:
        report.summary = "degraded"
    else:
        report.summary = "fail" if not report.skipped else "degraded"


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
    report_path: Path | None = None,
) -> VoiceLiveReport:
    # Runs doctor first for operator guidance, but the report stores only safe
    # counts/booleans, never device names, transcripts, or filesystem paths.
    voice_doctor(settings)
    report = _initial_report(settings)
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
        recorded_path = await mic.record_push_to_talk(input_path)
        report.recording_success = recorded_path.exists()
        transcript = await speech.transcribe(recorded_path)
        report.stt_success = True
        report.transcript_length = len(transcript)
        report.transcription_user_confirmed = confirm_transcription(
            "Was the transcription correct? The report stores only transcript length."
        )
        spoken_path = await synthesizer.synthesize("APRIL voice verification.", output_path)
        report.tts_success = spoken_path.exists()
        await audio_player.play(spoken_path)
        report.playback_user_confirmed = confirm_playback("Did you hear the playback?")
    except KeyboardInterrupt:
        report.skipped.append(VoiceLiveSkippedCheck(name="voice-live", reason="interrupted"))
    except RuntimeUnavailableError as exc:
        report.skipped.append(VoiceLiveSkippedCheck(name="voice-live", reason=exc.message))
    finally:
        if not retain_audio:
            for path in created_paths:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    _finalize_summary(report)
    if report_path is not None:
        write_voice_live_report(report, report_path)
    return report
