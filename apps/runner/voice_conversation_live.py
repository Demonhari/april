"""Redacted two-turn production voice conversation verification."""

from __future__ import annotations

import asyncio
import contextlib
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from april_common.errors import RuntimeUnavailableError
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.voice.health import voice_doctor

EvidenceMode = Literal["real_hardware", "injected_test"]
Confirm = Callable[[str], bool]


class VoiceConversationTurn(BaseModel):
    wake_detected: bool = False
    speech_started: bool = False
    stop_reason: str = "no_speech"
    captured_duration_ms: int = 0
    speech_duration_ms: int = 0
    trailing_silence_ms: int = 0
    endpoint_latency_ms: int | None = None
    minimum_duration_met: bool = False
    transcript_length: int = 0
    stt_success: bool = False
    api_success: bool = False
    tts_success: bool = False
    playback_started: bool = False
    playback_completed: bool = False


class VoiceConversationLiveReport(BaseModel):
    report_type: Literal["voice_conversation_live"] = "voice_conversation_live"
    schema_version: int = 1
    generated_at: str = Field(default_factory=utc_now_iso)
    platform: str = Field(
        default_factory=lambda: f"{platform.system()} {platform.release()}".strip()
    )
    evidence_mode: EvidenceMode
    voice_stack_available: bool = False
    wake_word_detected: bool = False
    turn_count: int = 0
    same_conversation: bool = False
    barge_in_attempted: bool = False
    barge_in_detected: bool = False
    barge_in_action: Literal["stop", "duck"] = "stop"
    barge_in_stop_latency_ms: int | None = None
    two_turns_completed: bool = False
    follow_up_opened: bool = False
    temporary_audio_retained: bool = False
    summary: Literal["pass", "fail", "degraded"] = "fail"
    voice_conversation_live_verified: bool = False
    turns: list[VoiceConversationTurn] = Field(default_factory=list, max_length=2)
    warning_codes: list[str] = Field(default_factory=list, max_length=16)


def write_voice_conversation_live_report(report: VoiceConversationLiveReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


async def run_voice_conversation_live_verification(
    *,
    settings: AprilSettings,
    confirm_microphone: Confirm,
    sentinel: Any | None = None,
    delivery: Any | None = None,
    report_path: Path | None = None,
    timeout_seconds: float = 180.0,
    retain_debug_audio: bool = False,
) -> VoiceConversationLiveReport:
    """Run two turns; injected evidence can exercise logic but never verify hardware."""
    evidence_mode: EvidenceMode = "injected_test" if sentinel is not None else "real_hardware"
    report = VoiceConversationLiveReport(
        evidence_mode=evidence_mode,
        barge_in_action=settings.voice.barge_in_action,
        temporary_audio_retained=retain_debug_audio or settings.voice.retain_debug_audio,
    )
    doctor = voice_doctor(settings)
    report.voice_stack_available = bool(
        doctor.get("full_voice_loop_ready") and settings.voice.enabled and settings.wake.enabled
    )
    if not confirm_microphone(
        "Open the microphone for a real two-turn wake, endpoint, playback, and barge-in test?"
    ):
        report.warning_codes.append("microphone_not_authorized")
        return _finish(report, report_path)

    if sentinel is None:
        try:
            if retain_debug_audio and not settings.voice.retain_debug_audio:
                settings = settings.model_copy(
                    update={"voice": settings.voice.model_copy(update={"retain_debug_audio": True})}
                )
            sentinel, delivery = _build_production_verifier(settings)
        except RuntimeUnavailableError:
            report.warning_codes.append("voice_stack_unavailable")
            return _finish(report, report_path)
    else:
        report.voice_stack_available = True
        delivery = delivery or getattr(sentinel, "deliver", None)

    try:
        await asyncio.wait_for(sentinel.run(), timeout=timeout_seconds)
    except KeyboardInterrupt:
        report.warning_codes.append("operator_interrupted")
    except TimeoutError:
        report.warning_codes.append("two_turn_timeout")
        sentinel.stop()
        with contextlib.suppress(Exception):
            await sentinel.response_coordinator.shutdown()
    except RuntimeUnavailableError:
        report.warning_codes.append("voice_stack_unavailable")
    except Exception:
        report.warning_codes.append("voice_pipeline_failed")

    metrics = list(getattr(sentinel, "completed_endpoint_metrics", []))[:2]
    lengths = list(getattr(sentinel, "accepted_transcript_lengths", []))[:2]
    stage_map = getattr(delivery, "generation_stages", {}) if delivery is not None else {}
    stages = [stage_map[key] for key in sorted(stage_map)][:2]
    for index, endpoint in enumerate(metrics):
        stage = stages[index] if index < len(stages) else set()
        transcript_length = lengths[index] if index < len(lengths) else 0
        report.turns.append(
            VoiceConversationTurn(
                wake_detected=True,
                speech_started=endpoint.speech_started,
                stop_reason=endpoint.stop_reason,
                captured_duration_ms=endpoint.captured_duration_ms,
                speech_duration_ms=endpoint.speech_duration_ms,
                trailing_silence_ms=endpoint.trailing_silence_ms,
                endpoint_latency_ms=endpoint.endpoint_latency_ms,
                minimum_duration_met=endpoint.minimum_duration_met,
                transcript_length=transcript_length,
                stt_success=transcript_length > 0,
                api_success="api_success" in stage,
                tts_success="tts_success" in stage,
                playback_started="playback_started" in stage,
                playback_completed="playback_completed" in stage,
            )
        )
    report.turn_count = len(report.turns)
    report.wake_word_detected = report.turn_count >= 1
    coordinator = sentinel.response_coordinator
    barge_reasons = {
        reason
        for reason in getattr(coordinator, "interrupt_reasons", [])
        if reason not in {"shutdown", "muted", "stopped", "superseded"}
    }
    report.barge_in_attempted = report.turn_count >= 2
    report.barge_in_detected = bool(barge_reasons)
    report.barge_in_stop_latency_ms = getattr(coordinator, "last_barge_in_latency_ms", None)
    conversation_ids = list(getattr(delivery, "conversation_ids", []))
    session_ids = list(getattr(delivery, "session_ids", []))
    report.same_conversation = (
        len(conversation_ids) >= 2 and len(set(conversation_ids[:2])) == 1
    ) or (len(session_ids) >= 2 and len(set(session_ids[:2])) == 1)
    report.two_turns_completed = (
        report.turn_count == 2
        and all(turn.stt_success and turn.api_success and turn.tts_success for turn in report.turns)
        and report.turns[0].playback_started
        and report.turns[1].playback_completed
    )
    report.follow_up_opened = getattr(sentinel, "_follow_up_until", None) is not None
    return _finish(report, report_path)


def _finish(
    report: VoiceConversationLiveReport, report_path: Path | None
) -> VoiceConversationLiveReport:
    real_pass = (
        report.evidence_mode == "real_hardware"
        and report.voice_stack_available
        and report.two_turns_completed
        and report.same_conversation
        and report.barge_in_attempted
        and report.barge_in_detected
        and report.follow_up_opened
    )
    report.voice_conversation_live_verified = real_pass
    report.summary = "pass" if real_pass else ("degraded" if report.turn_count else "fail")
    if report_path is not None:
        write_voice_conversation_live_report(report, report_path)
    return report


def _build_production_verifier(settings: AprilSettings) -> tuple[Any, Any]:
    from services.voice.audio_player import SoundDeviceAudioPlayer
    from services.voice.microphone import SoundDeviceMicrophone
    from services.voice.speech_to_text import WhisperCppSpeechToText
    from services.voice.text_to_speech import PiperTextToSpeech
    from services.wake.confirmer import SttConfirmer
    from services.wake.sentinel import ApiWakeDelivery, MuteSwitch, Sentinel, build_scorers

    voice = settings.voice
    if (
        not settings.voice.enabled
        or not settings.wake.enabled
        or not voice.effective_wake_word_model_paths
        or voice.effective_transcription_whisper_binary_path is None
        or voice.effective_transcription_whisper_model_path is None
        or voice.piper_binary_path is None
        or voice.piper_model_path is None
    ):
        raise RuntimeUnavailableError("Complete local voice artifacts are not configured.")
    stt = WhisperCppSpeechToText(
        settings.resolve_path(voice.effective_transcription_whisper_binary_path),
        settings.resolve_path(voice.effective_transcription_whisper_model_path),
    )
    tts = PiperTextToSpeech(
        settings.resolve_path(voice.piper_binary_path),
        settings.resolve_path(voice.piper_model_path),
    )
    player = SoundDeviceAudioPlayer(device=voice.output_device)
    delivery = ApiWakeDelivery(
        base_url=f"http://{settings.api.host}:{settings.api.port}",
        token=settings.api.token or "",
        settings=settings,
        tts=tts,
        player=player,
    )
    confirmer = (
        SttConfirmer(
            stt,
            audio_cache_path=settings.audio_cache_path,
            strict_address=settings.wake.strict_address,
            retain_debug_audio=voice.retain_debug_audio,
        )
        if settings.wake.confirm_with_stt
        else None
    )
    sentinel = Sentinel(
        settings=settings,
        microphone=SoundDeviceMicrophone(device=voice.input_device),
        scorers=build_scorers(settings),
        deliver=delivery,
        confirmer=confirmer,
        transcriber=stt,
        player=player,
        mute=MuteSwitch(settings.mute_flag_path),
    )

    original_complete = delivery.on_assistant_response_complete

    def stop_after_second_response() -> None:
        if original_complete is not None:
            original_complete()
        if sentinel.accepted_wakes >= 2:
            sentinel.stop()

    delivery.on_assistant_response_complete = stop_after_second_response
    return sentinel, delivery
