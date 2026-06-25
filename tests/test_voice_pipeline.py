from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from services.voice.audio_player import FakeAudioPlayer
from services.voice.conversation_loop import (
    PushToTalkLoop,
    VoiceTimeout,
    WakeWordConversationLoop,
    interactive_capture_strategy,
)
from services.voice.microphone import FakeMicrophone, SoundDeviceMicrophone, write_pcm_wav
from services.voice.push_to_talk import PushToTalkSession
from services.voice.speech_to_text import FakeSpeechToText
from services.voice.text_to_speech import FakeTextToSpeech
from services.voice.wake_word import WAKE_WORD_FRAME_SAMPLES, OpenWakeWordDetector

CHUNK_SAMPLES = 160  # a typical 10 ms microphone block at 16 kHz


def _chunk(value: int = 0) -> bytes:
    return (np.full(CHUNK_SAMPLES, value, dtype=np.int16)).tobytes()


class RecordingModel:
    def __init__(self, *, score: float) -> None:
        self.score = score
        self.predict_calls: list[np.ndarray] = []
        self.reset_calls = 0

    def predict(self, window: np.ndarray) -> dict[str, float]:
        self.predict_calls.append(window)
        return {"april": self.score}

    def reset(self) -> None:
        self.reset_calls += 1


def _detector_with(model: RecordingModel, tmp_path: Path, **kwargs: object) -> OpenWakeWordDetector:
    model_file = tmp_path / "wake.onnx"
    model_file.write_bytes(b"x")
    detector = OpenWakeWordDetector(model_file, **kwargs)  # type: ignore[arg-type]
    detector._model = model
    return detector


class ScriptedWake:
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


class InfiniteMicrophone:
    def __init__(self) -> None:
        self.closed = False

    async def frames(self) -> AsyncIterator[bytes]:
        try:
            while True:
                await asyncio.sleep(0)
                yield _chunk(0)
        finally:
            self.closed = True


def _wake_loop(settings_tmp, microphone, detector, tmp_path: Path) -> WakeWordConversationLoop:
    return WakeWordConversationLoop(
        api_client=_FakeApi(),  # type: ignore[arg-type]
        microphone=microphone,
        stt=FakeSpeechToText("April, hello"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        detector=detector,  # type: ignore[arg-type]
    )


class _FakeApi:
    async def post(self, path: str, payload: dict[str, str]) -> dict[str, object]:
        return {"result": {"final_message": "ok"}}


def _read_wav_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


# --- wake word: int16, aggregation, cooldown, reset -------------------------


def test_wake_detector_converts_bytes_to_int16(tmp_path: Path) -> None:
    model = RecordingModel(score=0.0)
    detector = _detector_with(model, tmp_path)
    samples = np.arange(WAKE_WORD_FRAME_SAMPLES, dtype=np.int16)
    detector.detect(samples.tobytes())
    assert len(model.predict_calls) == 1
    window = model.predict_calls[0]
    assert window.dtype == np.int16
    np.testing.assert_array_equal(window, samples)


def test_wake_detector_aggregates_small_frames(tmp_path: Path) -> None:
    model = RecordingModel(score=0.0)
    detector = _detector_with(model, tmp_path)
    # 160-sample blocks; a prediction must not run until a full 1280-sample window.
    for _ in range(7):
        assert detector.detect(_chunk(1)) is False
    assert model.predict_calls == []
    detector.detect(_chunk(1))  # 8 * 160 == 1280
    assert len(model.predict_calls) == 1


def test_wake_detector_never_receives_raw_bytes(tmp_path: Path) -> None:
    model = RecordingModel(score=0.9)
    detector = _detector_with(model, tmp_path)
    detector.detect((b"\x10\x00") * WAKE_WORD_FRAME_SAMPLES)
    assert all(isinstance(call, np.ndarray) for call in model.predict_calls)


def test_wake_detector_cooldown_debounces(tmp_path: Path) -> None:
    now = [0.0]
    model = RecordingModel(score=0.9)
    detector = _detector_with(
        model, tmp_path, threshold=0.5, cooldown_seconds=2.0, clock=lambda: now[0]
    )
    window = b"\x00\x00" * WAKE_WORD_FRAME_SAMPLES
    assert detector.detect(window) is True
    now[0] = 1.0  # still within cooldown
    assert detector.detect(window) is False
    now[0] = 3.0  # cooldown elapsed
    assert detector.detect(window) is True


def test_wake_detector_reset_clears_buffer_and_model(tmp_path: Path) -> None:
    model = RecordingModel(score=0.0)
    detector = _detector_with(model, tmp_path)
    detector.detect(_chunk(1))  # 160 samples buffered, no prediction yet
    detector.reset()
    assert model.reset_calls == 1
    for _ in range(7):
        detector.detect(_chunk(1))  # only 1120 samples since the reset
    assert model.predict_calls == []


# --- timeouts, pre-roll, cancellation ---------------------------------------


async def test_no_wake_timeout(settings_tmp, tmp_path: Path) -> None:
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[_chunk(0)] * 5)
    loop = _wake_loop(settings_tmp, mic, ScriptedWake(fire_on=None), tmp_path)
    times = iter([0.0, 1000.0])
    with pytest.raises(VoiceTimeout):
        await loop._capture_wake_utterance(tmp_path / "out.wav", clock=lambda: next(times))


async def test_utterance_timeout_starts_after_wake(settings_tmp, tmp_path: Path) -> None:
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[_chunk(5000)] * 50)
    loop = _wake_loop(settings_tmp, mic, ScriptedWake(fire_on=0), tmp_path)
    # clock: wake_deadline calc, utterance_deadline calc, then a jump past it.
    times = iter([0.0, 0.0, 1000.0])
    out = await loop._capture_wake_utterance(tmp_path / "out.wav", clock=lambda: next(times))
    # Pre-roll (the wake frame) plus exactly one post-wake frame before timeout.
    assert len(_read_wav_frames(out)) == 2 * CHUNK_SAMPLES * 2  # 2 frames * samples * 2 bytes


async def test_pre_roll_is_preserved(settings_tmp, tmp_path: Path) -> None:
    a, b, c = _chunk(1), _chunk(2), _chunk(3)
    silence = _chunk(0)
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[a, b, c, silence, silence, silence])
    loop = _wake_loop(settings_tmp, mic, ScriptedWake(fire_on=2), tmp_path)
    out = await loop._capture_wake_utterance(tmp_path / "out.wav")
    captured = _read_wav_frames(out)
    # The two frames captured before the wake word fired are retained.
    assert captured.startswith(a + b + c)


async def test_capture_cancellation_closes_frame_source(settings_tmp, tmp_path: Path) -> None:
    mic = InfiniteMicrophone()
    loop = _wake_loop(settings_tmp, mic, ScriptedWake(fire_on=None), tmp_path)
    task = asyncio.create_task(loop._capture_wake_utterance(tmp_path / "out.wav"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mic.closed is True


# --- microphone overflow + push-to-talk -------------------------------------


async def test_microphone_overflow_policy_drops_oldest() -> None:
    mic = SoundDeviceMicrophone()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
    mic._enqueue_frame(queue, b"a")
    mic._enqueue_frame(queue, b"b")
    mic._enqueue_frame(queue, b"c")  # overflow: drop oldest
    assert mic.dropped_frames == 1
    assert queue.qsize() == 2
    assert queue.get_nowait() == b"b"
    assert queue.get_nowait() == b"c"


async def test_push_to_talk_stop_ends_capture(tmp_path: Path) -> None:
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[_chunk(1)] * 100)
    session = PushToTalkSession(mic, max_seconds=30.0)
    session.request_stop()
    out = await session.capture(tmp_path / "ptt.wav")
    assert session.stop_reason == "stopped"
    assert len(_read_wav_frames(out)) == CHUNK_SAMPLES * 2  # one frame captured


async def test_push_to_talk_respects_max_frames(tmp_path: Path) -> None:
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[_chunk(1)] * 100)
    session = PushToTalkSession(mic, max_seconds=30.0)
    out = await session.capture(tmp_path / "ptt.wav", max_frames=3)
    assert session.stop_reason == "max_frames"
    assert len(_read_wav_frames(out)) == 3 * CHUNK_SAMPLES * 2


async def test_push_to_talk_max_duration_safety(tmp_path: Path) -> None:
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[_chunk(1)] * 100)
    now = iter([0.0, 1000.0])  # deadline calc, then a jump past it on the first frame
    session = PushToTalkSession(mic, max_seconds=5.0, clock=lambda: next(now))
    out = await session.capture(tmp_path / "ptt.wav")
    assert session.stop_reason == "max_duration"
    assert len(_read_wav_frames(out)) == CHUNK_SAMPLES * 2


async def test_push_to_talk_cancellation_closes_source(tmp_path: Path) -> None:
    mic = InfiniteMicrophone()
    session = PushToTalkSession(mic, max_seconds=30.0)
    task = asyncio.create_task(session.capture(tmp_path / "ptt.wav"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mic.closed is True


class ClosableFiniteMicrophone:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.closed = False

    async def frames(self) -> AsyncIterator[bytes]:
        try:
            for frame in self._frames:
                await asyncio.sleep(0)
                yield frame
        finally:
            self.closed = True


async def test_push_to_talk_empty_capture_errors(tmp_path: Path) -> None:
    mic = FakeMicrophone(tmp_path / "u.wav", frames=[])
    session = PushToTalkSession(mic, max_seconds=5.0)
    with pytest.raises(ValueError, match="no audio"):
        await session.capture(tmp_path / "ptt.wav")


async def test_push_to_talk_source_ended_closes_frame_source(tmp_path: Path) -> None:
    mic = ClosableFiniteMicrophone([_chunk(1)] * 3)
    session = PushToTalkSession(mic, max_seconds=30.0)
    out = await session.capture(tmp_path / "ptt.wav")
    assert session.stop_reason == "source_ended"
    assert mic.closed is True
    assert len(_read_wav_frames(out)) == 3 * CHUNK_SAMPLES * 2


async def test_interactive_capture_stops_on_second_enter(tmp_path: Path) -> None:
    mic = InfiniteMicrophone()
    reads = iter(["", ""])  # first Enter starts, second Enter stops
    prompts: list[str] = []
    capture = interactive_capture_strategy(
        mic,
        max_seconds=30.0,
        read_line=lambda: next(reads),
        announce=prompts.append,
    )
    out = await capture(tmp_path / "ptt.wav")
    assert out.exists()
    assert mic.closed is True
    assert any("start" in p.lower() for p in prompts)
    assert any("stop" in p.lower() for p in prompts)


def _ptt_loop(settings_tmp, *, capture=None, microphone=None) -> PushToTalkLoop:
    return PushToTalkLoop(
        api_client=_FakeApi(),  # type: ignore[arg-type]
        microphone=microphone,
        stt=FakeSpeechToText("April hello"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        capture=capture,
    )


async def test_ptt_loop_interactive_runs_and_deletes_temp_audio(
    settings_tmp, tmp_path: Path
) -> None:
    mic = InfiniteMicrophone()
    reads = iter(["", ""])
    capture = interactive_capture_strategy(
        mic, max_seconds=5.0, read_line=lambda: next(reads), announce=lambda _m: None
    )
    loop = _ptt_loop(settings_tmp, capture=capture, microphone=mic)
    answer = await loop.run_once()
    assert answer == "ok"
    assert mic.closed is True
    # Temporary capture audio is cleaned up by default.
    assert list(settings_tmp.audio_cache_path.glob("*.wav")) == []


async def test_ptt_loop_retains_debug_audio_when_enabled(settings_tmp, tmp_path: Path) -> None:
    mic = InfiniteMicrophone()
    reads = iter(["", ""])
    capture = interactive_capture_strategy(
        mic, max_seconds=5.0, read_line=lambda: next(reads), announce=lambda _m: None
    )
    loop = _ptt_loop(settings_tmp, capture=capture, microphone=mic)
    loop.settings = loop.settings.model_copy(
        update={"voice": loop.settings.voice.model_copy(update={"retain_debug_audio": True})}
    )
    await loop.run_once()
    # The captured WAV is retained for debugging.
    assert any(settings_tmp.audio_cache_path.glob("*.wav"))


async def test_ptt_loop_fixed_duration_seconds_compatibility(settings_tmp, tmp_path: Path) -> None:
    # --seconds path: the loop uses the microphone's fixed-duration recorder.
    recorded = tmp_path / "recorded.wav"
    write_pcm_wav(recorded, [_chunk(1)] * 4)
    mic = FakeMicrophone(recorded)
    loop = _ptt_loop(settings_tmp, microphone=mic)
    answer = await loop.run_once()
    assert answer == "ok"
