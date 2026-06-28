from __future__ import annotations

from pathlib import Path

import pytest

from apps.runner.voice_live import (
    VoiceLiveReport,
    run_voice_live_verification,
    write_voice_live_report,
)
from services.voice.audio_player import FakeAudioPlayer
from services.voice.microphone import Microphone
from services.voice.speech_to_text import FakeSpeechToText
from services.voice.text_to_speech import FakeTextToSpeech


class WritingMicrophone(Microphone):
    def __init__(self) -> None:
        self.called = False

    async def record_push_to_talk(self, output_path: Path) -> Path:
        self.called = True
        output_path.write_bytes(b"fake wav bytes")
        return output_path


class InterruptingMicrophone(Microphone):
    async def record_push_to_talk(self, output_path: Path) -> Path:
        output_path.write_bytes(b"partial")
        raise KeyboardInterrupt


def _minimal_voice_report(**timestamps: str) -> VoiceLiveReport:
    return VoiceLiveReport(
        **timestamps,
        platform="Darwin 25",
        sounddevice_available=True,
        input_device_count=1,
        output_device_count=1,
        whisper_binary_available=True,
        whisper_model_available=True,
        piper_binary_available=True,
        piper_model_available=True,
        wake_word_model_available=False,
    )


def test_voice_live_report_accepts_generated_at_only() -> None:
    report = _minimal_voice_report(generated_at="2026-06-26T00:00:00Z")
    assert report.timestamp == "2026-06-26T00:00:00Z"
    assert report.generated_at == report.timestamp


def test_voice_live_report_accepts_timestamp_only() -> None:
    report = _minimal_voice_report(timestamp="2026-06-26T00:00:00Z")
    assert report.generated_at == "2026-06-26T00:00:00Z"
    assert report.generated_at == report.timestamp


def test_voice_live_report_missing_timestamps_are_filled() -> None:
    report = _minimal_voice_report()
    assert report.generated_at
    assert report.timestamp == report.generated_at


def test_write_voice_live_report_includes_both_timestamps(tmp_path: Path) -> None:
    report = _minimal_voice_report(generated_at="2026-06-26T00:00:00Z")
    out = tmp_path / "voice-live.json"
    write_voice_live_report(report, out)
    text = out.read_text(encoding="utf-8")
    assert '"generated_at": "2026-06-26T00:00:00Z"' in text
    assert '"timestamp": "2026-06-26T00:00:00Z"' in text


@pytest.mark.anyio
async def test_voice_live_success_path_uses_fakes(settings_tmp) -> None:
    microphone = WritingMicrophone()
    report = await run_voice_live_verification(
        settings=settings_tmp,
        confirm_recording=lambda _message: True,
        confirm_transcription=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=microphone,
        stt=FakeSpeechToText("hello april"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    assert microphone.called is True
    assert report.summary == "pass"
    assert report.recording_success is True
    assert report.stt_success is True
    assert report.transcript_length == len("hello april")
    assert report.tts_success is True
    assert report.playback_user_confirmed is True
    # A fully-confirmed pass is the only state that sets voice_live_verified true.
    assert report.voice_live_verified is True
    assert report.generated_at == report.timestamp
    assert report.generated_at


@pytest.mark.anyio
async def test_voice_live_confirmation_required(settings_tmp) -> None:
    microphone = WritingMicrophone()
    report = await run_voice_live_verification(
        settings=settings_tmp,
        confirm_recording=lambda _message: False,
        confirm_transcription=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=microphone,
        stt=FakeSpeechToText(),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    assert microphone.called is False
    assert report.summary == "degraded"
    assert report.skipped[0].reason == "user denied recording"
    # A degraded run is never live-verified.
    assert report.voice_live_verified is False


@pytest.mark.anyio
async def test_voice_live_temp_audio_deleted_by_default(settings_tmp) -> None:
    await run_voice_live_verification(
        settings=settings_tmp,
        confirm_recording=lambda _message: True,
        confirm_transcription=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=WritingMicrophone(),
        stt=FakeSpeechToText(),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    assert list(settings_tmp.audio_cache_path.glob("voice-live-*")) == []


@pytest.mark.anyio
async def test_voice_live_retains_audio_only_with_explicit_flag(settings_tmp) -> None:
    report = await run_voice_live_verification(
        settings=settings_tmp,
        confirm_recording=lambda _message: True,
        confirm_transcription=lambda _message: True,
        confirm_playback=lambda _message: True,
        retain_debug_audio=True,
        microphone=WritingMicrophone(),
        stt=FakeSpeechToText(),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    assert report.temp_audio_retained is True
    retained = list(settings_tmp.audio_cache_path.glob("voice-live-*"))
    assert retained


@pytest.mark.anyio
async def test_voice_live_report_redacts_transcript(settings_tmp, tmp_path: Path) -> None:
    report_path = tmp_path / "voice-live.json"
    observed: list[str] = []
    report = await run_voice_live_verification(
        settings=settings_tmp,
        confirm_recording=lambda _message: True,
        confirm_transcription=lambda _message: False,
        confirm_playback=lambda _message: True,
        microphone=WritingMicrophone(),
        stt=FakeSpeechToText("secret transcript words"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        transcript_observer=observed.append,
        report_path=report_path,
    )
    text = report_path.read_text(encoding="utf-8")
    assert observed == ["secret transcript words"]
    assert "secret transcript words" not in text
    assert "transcript_length" in text
    assert settings_tmp.api.token not in text
    # Transcription was not confirmed → not a full pass → never live-verified.
    assert report.voice_live_verified is False
    assert '"voice_live_verified": false' in text


@pytest.mark.anyio
async def test_voice_live_interrupt_cleans_up_temp_audio(settings_tmp) -> None:
    report = await run_voice_live_verification(
        settings=settings_tmp,
        confirm_recording=lambda _message: True,
        confirm_transcription=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=InterruptingMicrophone(),
        stt=FakeSpeechToText(),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    assert report.summary == "degraded"
    assert report.skipped[0].reason == "interrupted"
    assert list(settings_tmp.audio_cache_path.glob("voice-live-*")) == []
