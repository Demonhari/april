"""Live wake-word ("April") verification for the target Mac.

``run april voice verify-wake-live`` exercises the *real* wake-word path, not
just push-to-talk: it opens the microphone, waits for the spoken wake word,
captures the following utterance, transcribes it with whisper.cpp, normalises the
wake word out of the transcript, calls the Core API ``/voice/input`` endpoint,
synthesizes the reply with Piper, and asks the operator to confirm playback.

Like :mod:`apps.runner.voice_live`, the verifier is dependency-injected so the
full path can be unit-tested with fake microphone/detector/STT/TTS/player/API
pieces — no real microphone, speaker, whisper.cpp, Piper, openWakeWord, or
network is required by the tests. The on-disk report is redacted by
construction: it stores only booleans, counts, lengths, and a status string —
never transcript text, device names, tokens, or filesystem paths.

The capture step reuses the real :class:`WakeWordConversationLoop` capture logic
(pre-roll, VAD end-pointing, wake/utterance timeouts, and guaranteed frame-source
close) by wrapping the loop's microphone and detector with thin observers, so the
verification confirms the same code the live ``voice listen`` loop runs.
"""

from __future__ import annotations

import contextlib
import platform
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from apps.cli.client import ApiOfflineError, AprilApiClient
from april_common.errors import RuntimeUnavailableError
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.voice.audio_player import AudioPlayer
from services.voice.conversation_loop import (
    VoiceTimeout,
    WakeWordConversationLoop,
    normalize_transcript,
)
from services.voice.health import voice_doctor
from services.voice.microphone import Microphone, aclose_frame_source
from services.voice.speech_to_text import SpeechToText
from services.voice.text_to_speech import TextToSpeech
from services.voice.wake_word import WakeWordDetector

# Given the normalized command text, return APRIL's spoken reply. Injected so the
# /voice/input call can be faked in tests; the default implementation calls the
# local Core API over loopback HTTP.
WakeApiCaller = Callable[[str], Awaitable[str]]
Confirm = Callable[[str], bool]


class WakeWordLiveSkippedCheck(BaseModel):
    name: str
    reason: str


class WakeWordLiveReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["wake_word_live"] = "wake_word_live"
    generated_at: str = ""
    timestamp: str = ""
    platform: str = ""
    doctor_status: str = "unknown"
    wake_word_configured: bool = False
    wake_word_detected: bool = False
    recording_success: bool = False
    stt_success: bool = False
    transcript_length: int = 0
    normalized_transcript_length: int = 0
    api_success: bool = False
    tts_success: bool = False
    playback_user_confirmed: bool = False
    retained_debug_audio: bool = False
    skipped: list[WakeWordLiveSkippedCheck] = Field(default_factory=list)
    summary: Literal["pass", "fail"] = "fail"
    # True only when every wake-word stage genuinely passed AND the operator
    # confirmed playback. A skipped/failed run can never set this true, so a
    # wake-word report can never be mistaken for a verified live wake-word pass.
    wake_word_live_verified: bool = False

    @model_validator(mode="after")
    def _fill_timestamps(self) -> WakeWordLiveReport:
        if not self.generated_at and self.timestamp:
            self.generated_at = self.timestamp
        elif not self.timestamp and self.generated_at:
            self.timestamp = self.generated_at
        elif not self.generated_at and not self.timestamp:
            now = utc_now_iso()
            self.generated_at = now
            self.timestamp = now
        if not self.platform:
            self.platform = f"{platform.system()} {platform.release()}".strip()
        return self


class _FrameCountingMicrophone(Microphone):
    """Wrap a microphone to count frames pulled and guarantee inner close.

    The inner frame source is always closed when the wrapper generator is closed
    (``aclose``), including on the loop's break/timeout/cancellation paths, so the
    real microphone stream is released on every exit.
    """

    def __init__(self, inner: Microphone) -> None:
        self._inner = inner
        self.frames_read = 0

    async def record_push_to_talk(self, output_path: Path) -> Path:
        return await self._inner.record_push_to_talk(output_path)

    async def frames(self) -> AsyncIterator[bytes]:
        source = self._inner.frames()
        try:
            async for frame in source:
                self.frames_read += 1
                yield frame
        finally:
            await aclose_frame_source(source)


class _DetectionObserver:
    """Wrap a wake-word detector to record whether it ever fired."""

    def __init__(self, inner: WakeWordDetector) -> None:
        self._inner = inner
        self.detected = False

    def available(self) -> bool:
        return bool(self._inner.available())

    def detect(self, frame: bytes) -> bool:
        fired = bool(self._inner.detect(frame))  # type: ignore[attr-defined]
        if fired:
            self.detected = True
        return fired

    def reset(self) -> None:
        reset = getattr(self._inner, "reset", None)
        if callable(reset):
            reset()


class _UnusedApiClient:
    """Placeholder API client for the loop; the verifier calls the API itself."""

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        raise RuntimeUnavailableError(
            "The wake-word verifier calls /voice/input through its own injected caller."
        )


def _finalize_summary(report: WakeWordLiveReport) -> None:
    full_pass = (
        report.wake_word_configured
        and report.wake_word_detected
        and report.recording_success
        and report.stt_success
        and report.transcript_length > 0
        and report.api_success
        and report.tts_success
        and report.playback_user_confirmed
    )
    report.summary = "pass" if full_pass else "fail"
    report.wake_word_live_verified = full_pass


def write_wake_word_live_report(report: WakeWordLiveReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


def _build_loop(
    *,
    settings: AprilSettings,
    microphone: Microphone | None,
    detector: WakeWordDetector | None,
    stt: SpeechToText | None,
    tts: TextToSpeech | None,
    player: AudioPlayer | None,
    wake_wait_seconds: float | None,
    utterance_max_seconds: float | None,
    conversation_id: str | None,
) -> WakeWordConversationLoop:
    loop = WakeWordConversationLoop(
        api_client=_UnusedApiClient(),
        microphone=microphone,
        stt=stt,
        tts=tts,
        player=player,
        detector=detector,  # type: ignore[arg-type]
        conversation_id=conversation_id,
    )
    # Use the verifier's settings (with any timeout overrides) for capture so the
    # audio cache, wake-wait, and utterance bounds match the explicit CLI options.
    voice_overrides: dict[str, Any] = {}
    if wake_wait_seconds is not None:
        voice_overrides["wake_wait_seconds"] = wake_wait_seconds
    if utterance_max_seconds is not None:
        voice_overrides["utterance_max_seconds"] = utterance_max_seconds
    if voice_overrides:
        loop.settings = settings.model_copy(
            update={"voice": settings.voice.model_copy(update=voice_overrides)}
        )
    else:
        loop.settings = settings
    return loop


async def _default_api_caller(settings: AprilSettings, conversation_id: str, message: str) -> str:
    client = AprilApiClient(
        f"http://{settings.api.host}:{settings.api.port}",
        settings.api.token,
        timeout=settings.runtime.request_timeout_seconds,
    )
    response = await client.post(
        "/voice/input", {"message": message, "conversation_id": conversation_id}
    )
    result = response.get("result") if isinstance(response, dict) else None
    final_message = result.get("final_message") if isinstance(result, dict) else None
    return str(final_message) if final_message else ""


async def run_wake_word_live_verification(
    *,
    settings: AprilSettings,
    confirm_microphone: Confirm,
    confirm_playback: Confirm,
    wake_wait_seconds: float | None = None,
    utterance_max_seconds: float | None = None,
    retain_debug_audio: bool = False,
    microphone: Microphone | None = None,
    detector: WakeWordDetector | None = None,
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
    player: AudioPlayer | None = None,
    api_caller: WakeApiCaller | None = None,
    conversation_id: str | None = None,
    report_path: Path | None = None,
) -> WakeWordLiveReport:
    """Verify the live wake-word path end to end with injectable dependencies.

    Always closes the microphone/audio frame source on success, failure, timeout,
    cancellation, or ``KeyboardInterrupt`` (the capture loop closes the source in
    its own ``finally``; this function additionally removes temporary audio unless
    ``retain_debug_audio`` is set). Returns a redacted :class:`WakeWordLiveReport`.
    """
    doctor = voice_doctor(settings)
    report = WakeWordLiveReport(doctor_status=str(doctor.get("status", "unknown")))
    retain_audio = retain_debug_audio or settings.voice.retain_debug_audio
    report.retained_debug_audio = retain_audio

    # A configured wake-word model path is required: push-to-talk needs none, but
    # wake-word listening cannot be verified without one.
    if settings.voice.wake_word_model_path is None:
        report.skipped.append(
            WakeWordLiveSkippedCheck(
                name="wake-word model",
                reason="No wake-word model is configured (voice.wake_word_model_path).",
            )
        )
        _finalize_summary(report)
        if report_path is not None:
            write_wake_word_live_report(report, report_path)
        return report
    report.wake_word_configured = True

    # The microphone is only opened after explicit operator confirmation.
    if not confirm_microphone('Open the microphone and listen for the wake word "April" now?'):
        report.skipped.append(
            WakeWordLiveSkippedCheck(name="microphone", reason="user declined microphone access")
        )
        _finalize_summary(report)
        if report_path is not None:
            write_wake_word_live_report(report, report_path)
        return report

    loop = _build_loop(
        settings=settings,
        microphone=microphone,
        detector=detector,
        stt=stt,
        tts=tts,
        player=player,
        wake_wait_seconds=wake_wait_seconds,
        utterance_max_seconds=utterance_max_seconds,
        conversation_id=conversation_id,
    )
    counting_mic = _FrameCountingMicrophone(loop.microphone)
    loop.microphone = counting_mic
    observed_detector = _DetectionObserver(loop.detector)
    loop.detector = observed_detector  # type: ignore[assignment]

    if not observed_detector.available():
        report.skipped.append(
            WakeWordLiveSkippedCheck(
                name="wake-word model",
                reason="Configured wake-word model file is missing or unreadable.",
            )
        )
        _finalize_summary(report)
        if report_path is not None:
            write_wake_word_live_report(report, report_path)
        return report

    settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
    stem = f"wake-live-{uuid.uuid4().hex}"
    utterance_path = settings.audio_cache_path / f"{stem}-utterance.wav"
    reply_path = settings.audio_cache_path / f"{stem}-reply.wav"
    created_paths = [utterance_path, reply_path]

    try:
        captured = await loop._capture_wake_utterance(utterance_path)
        report.recording_success = captured.exists()
        transcript = await loop.stt.transcribe(captured)
        report.transcript_length = len(transcript)
        report.stt_success = bool(transcript.strip())
        normalized = normalize_transcript(transcript, wake_word="april")
        report.normalized_transcript_length = len(normalized)
        if not normalized:
            raise ValueError("Normalized transcript was empty.")
        caller = api_caller
        if caller is None:
            answer = await _default_api_caller(settings, loop.conversation_id, normalized)
        else:
            answer = await caller(normalized)
        report.api_success = bool(answer)
        # Synthesize APRIL's reply; fall back to a fixed phrase only so Piper is
        # still exercised when the API returned an empty (but non-error) answer.
        spoken = await loop.tts.synthesize(answer or "APRIL wake-word verification.", reply_path)
        report.tts_success = spoken.exists()
        await loop.player.play(spoken)
        report.playback_user_confirmed = confirm_playback("Did you hear APRIL's spoken response?")
    except KeyboardInterrupt:
        report.skipped.append(WakeWordLiveSkippedCheck(name="wake-word-live", reason="interrupted"))
    except VoiceTimeout:
        report.skipped.append(
            WakeWordLiveSkippedCheck(
                name="wake word",
                reason="No wake word was detected before the wake-word timeout.",
            )
        )
    except (RuntimeUnavailableError, ApiOfflineError) as exc:
        reason = getattr(exc, "message", None) or str(exc)
        report.skipped.append(WakeWordLiveSkippedCheck(name="wake-word-live", reason=str(reason)))
    except ValueError as exc:
        report.skipped.append(WakeWordLiveSkippedCheck(name="transcript", reason=str(exc)))
    finally:
        report.wake_word_detected = observed_detector.detected
        if not retain_audio:
            for path in created_paths:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    _finalize_summary(report)
    if report_path is not None:
        write_wake_word_live_report(report, report_path)
    return report
