from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from apps.runner.wake_live import (
    WakeWordLiveReport,
    run_wake_word_live_verification,
    write_wake_word_live_report,
)
from april_common.settings import AprilSettings
from services.voice.audio_player import FakeAudioPlayer
from services.voice.microphone import FakeMicrophone
from services.voice.speech_to_text import FakeSpeechToText
from services.voice.text_to_speech import FakeTextToSpeech

SILENCE = b"\x00\x00" * 160  # a 10 ms mono 16 kHz frame of silence


class ScriptedWake:
    """A wake-word detector that fires on a chosen frame index (or never)."""

    def __init__(self, *, fire_on: int | None) -> None:
        self.fire_on = fire_on
        self.calls = 0

    def available(self) -> bool:
        return True

    def detect(self, frame: bytes) -> bool:
        index = self.calls
        self.calls += 1
        return self.fire_on is not None and index == self.fire_on

    def reset(self) -> None:
        pass


class UnavailableWake(ScriptedWake):
    def available(self) -> bool:
        return False


class InfiniteMicrophone:
    """Yields silence forever; records whether the frame source was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def record_push_to_talk(self, output_path: Path) -> Path:  # pragma: no cover - unused
        return output_path

    async def frames(self) -> AsyncIterator[bytes]:
        try:
            while True:
                await asyncio.sleep(0)
                yield SILENCE
        finally:
            self.closed = True


class RecordingConfirm:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, _message: str) -> bool:
        self.calls += 1
        return self.answer


def _configured_settings(
    settings: AprilSettings, tmp_path: Path, *, configured: bool = True
) -> AprilSettings:
    voice_update: dict[str, object] = {"enabled": True}
    if configured:
        model = tmp_path / "wake.onnx"
        model.write_bytes(b"x")
        voice_update["wake_word_model_path"] = model
    return settings.model_copy(update={"voice": settings.voice.model_copy(update=voice_update)})


async def _api_caller(message: str) -> str:
    return "Acceptance reply."


@pytest.mark.anyio
async def test_wake_live_passes_with_all_fakes(settings_tmp, tmp_path: Path) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4)
    report = await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=mic,
        detector=ScriptedWake(fire_on=0),
        stt=FakeSpeechToText("April, plan my day"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
    )
    assert report.summary == "pass"
    assert report.wake_word_configured is True
    assert report.wake_word_detected is True
    assert report.recording_success is True
    assert report.stt_success is True
    assert report.transcript_length == len("April, plan my day")
    # The wake word is normalized out of the transcript before the API call.
    assert report.normalized_transcript_length == len("plan my day")
    assert report.api_success is True
    assert report.tts_success is True
    assert report.playback_user_confirmed is True
    assert report.wake_word_live_verified is True
    # Temporary audio is cleaned up by default.
    assert list(settings.audio_cache_path.glob("wake-live-*")) == []


@pytest.mark.anyio
async def test_wake_live_fails_when_no_wake_word_model_configured(
    settings_tmp, tmp_path: Path
) -> None:
    settings = _configured_settings(settings_tmp, tmp_path, configured=False)
    confirm_mic = RecordingConfirm(True)
    report = await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=confirm_mic,
        confirm_playback=lambda _message: True,
        microphone=FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4),
        detector=ScriptedWake(fire_on=0),
        stt=FakeSpeechToText("hello"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
    )
    assert report.summary == "fail"
    assert report.wake_word_configured is False
    assert report.wake_word_live_verified is False
    # The microphone is never opened (confirmation not even requested) without a model.
    assert confirm_mic.calls == 0
    assert any(
        skip.name == "wake-word model" and "configured" in skip.reason.lower()
        for skip in report.skipped
    )


@pytest.mark.anyio
async def test_wake_live_fails_when_model_file_missing(settings_tmp, tmp_path: Path) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    report = await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4),
        detector=UnavailableWake(fire_on=0),
        stt=FakeSpeechToText("hello"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
    )
    assert report.summary == "fail"
    assert report.wake_word_configured is True
    assert report.wake_word_detected is False
    assert any("missing" in skip.reason.lower() for skip in report.skipped)


@pytest.mark.anyio
async def test_wake_live_no_wake_word_detected_fails_cleanly(settings_tmp, tmp_path: Path) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    report = await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4),
        detector=ScriptedWake(fire_on=None),
        stt=FakeSpeechToText("hello"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
    )
    assert report.summary == "fail"
    assert report.wake_word_detected is False
    assert report.recording_success is False
    assert report.api_success is False


@pytest.mark.anyio
async def test_wake_live_deletes_temp_audio_by_default(settings_tmp, tmp_path: Path) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    report = await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4),
        detector=ScriptedWake(fire_on=0),
        stt=FakeSpeechToText("April hi"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
    )
    assert report.retained_debug_audio is False
    assert list(settings.audio_cache_path.glob("wake-live-*")) == []


@pytest.mark.anyio
async def test_wake_live_retains_audio_only_with_explicit_flag(
    settings_tmp, tmp_path: Path
) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    report = await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=lambda _message: True,
        confirm_playback=lambda _message: True,
        retain_debug_audio=True,
        microphone=FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4),
        detector=ScriptedWake(fire_on=0),
        stt=FakeSpeechToText("April hi"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
    )
    assert report.retained_debug_audio is True
    assert list(settings.audio_cache_path.glob("wake-live-*"))


@pytest.mark.anyio
async def test_wake_live_report_redacts_transcript_and_token(
    settings_tmp, tmp_path: Path
) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    out = tmp_path / "wake-live.json"
    await run_wake_word_live_verification(
        settings=settings,
        confirm_microphone=lambda _message: True,
        confirm_playback=lambda _message: True,
        microphone=FakeMicrophone(tmp_path / "u.wav", frames=[SILENCE] * 4),
        detector=ScriptedWake(fire_on=0),
        stt=FakeSpeechToText("April secret transcript words"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        api_caller=_api_caller,
        report_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert "secret transcript words" not in text
    assert "transcript_length" in text
    assert settings.api.token not in text


@pytest.mark.anyio
async def test_wake_live_cancellation_closes_frame_source(settings_tmp, tmp_path: Path) -> None:
    settings = _configured_settings(settings_tmp, tmp_path)
    mic = InfiniteMicrophone()
    task = asyncio.create_task(
        run_wake_word_live_verification(
            settings=settings,
            confirm_microphone=lambda _message: True,
            confirm_playback=lambda _message: True,
            microphone=mic,
            detector=ScriptedWake(fire_on=None),
            stt=FakeSpeechToText("April hi"),
            tts=FakeTextToSpeech(),
            player=FakeAudioPlayer(),
            api_caller=_api_caller,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The microphone frame source is closed even when the run is cancelled.
    assert mic.closed is True


def test_write_wake_word_live_report_round_trips(tmp_path: Path) -> None:
    report = WakeWordLiveReport(summary="fail")
    out = tmp_path / "nested" / "wake.json"
    written = write_wake_word_live_report(report, out)
    assert written.exists()
    assert '"report_type": "wake_word_live"' in written.read_text(encoding="utf-8")
