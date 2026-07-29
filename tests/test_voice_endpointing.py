from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from april_common.settings import VoiceSettings
from services.voice.endpointing import (
    PcmFormat,
    UtteranceEndpointDetector,
    capture_streamed_utterance,
    pcm_frame_duration_ms,
)


def _frame(level: int, *, samples: int = 160) -> bytes:
    return np.full(samples, level, dtype=np.int16).tobytes()


QUIET = _frame(100)
SPEECH = _frame(8_000)
SILENCE = _frame(0)


def _detector(**updates: object) -> UtteranceEndpointDetector:
    values: dict[str, object] = {
        "onset_frames": 3,
        "endpoint_silence_ms": 650,
        "minimum_utterance_ms": 300,
        "noise_calibration_ms": 300,
        "hangover_ms": 100,
    }
    values.update(updates)
    return UtteranceEndpointDetector(**values)  # type: ignore[arg-type]


def _feed(detector: UtteranceEndpointDetector, frames: list[bytes]) -> str | None:
    result = None
    for frame in frames:
        result = detector.process(frame)
        if result is not None:
            break
    return result


def test_frame_duration_comes_from_pcm_shape() -> None:
    assert pcm_frame_duration_ms(SILENCE, PcmFormat()) == pytest.approx(10.0)
    stereo = PcmFormat(sample_rate=8_000, channels=2, sample_width=2)
    assert pcm_frame_duration_ms(b"\x00\x00" * 160, stereo) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="Malformed"):
        pcm_frame_duration_ms(b"\x00", PcmFormat())


def test_immediate_speech_is_detected_during_calibration() -> None:
    detector = _detector()
    _feed(detector, [SPEECH] * 3)
    assert detector.state == "in_speech"


def test_quiet_calibration_is_robust_to_one_loud_outlier() -> None:
    detector = _detector(onset_frames=2)
    _feed(detector, [QUIET] * 10 + [SPEECH] + [QUIET] * 20)
    assert detector.calibrated_noise_floor < 0.01
    assert detector.effective_energy_threshold >= detector.energy_threshold
    assert detector.state in {"calibrating", "waiting_for_speech"}


def test_onset_requires_consecutive_frames() -> None:
    detector = _detector()
    _feed(detector, [SPEECH, SPEECH, SILENCE, SPEECH, SPEECH])
    assert detector.state != "in_speech"
    detector.process(SPEECH)
    assert detector.state == "in_speech"


@pytest.mark.parametrize("pause_ms", [100, 300, 500])
def test_short_pause_does_not_end_default_utterance(pause_ms: int) -> None:
    detector = _detector()
    assert _feed(detector, [SPEECH] * 30 + [SILENCE] * (pause_ms // 10)) is None
    assert detector.complete is False


def test_default_650ms_silence_ends_without_added_hangover() -> None:
    detector = _detector()
    reason = _feed(detector, [SPEECH] * 30 + [SILENCE] * 65)
    assert reason == "end_of_speech"
    metrics = detector.metrics()
    assert metrics.trailing_silence_ms == 650
    assert metrics.endpoint_latency_ms == 650


def test_resumed_speech_resets_endpoint_silence() -> None:
    detector = _detector()
    assert _feed(detector, [SPEECH] * 30 + [SILENCE] * 50 + [SPEECH]) is None
    assert detector.trailing_silence_ms == 0
    assert _feed(detector, [SILENCE] * 64) is None
    assert detector.process(SILENCE) == "end_of_speech"


def test_minimum_no_speech_source_end_and_max_duration_reasons() -> None:
    too_short = _detector()
    assert _feed(too_short, [SPEECH] * 10 + [SILENCE] * 65) == "too_short"
    no_speech = _detector()
    assert no_speech.finish().stop_reason == "no_speech"
    source_ended = _detector()
    _feed(source_ended, [SPEECH] * 30)
    assert source_ended.finish().stop_reason == "source_ended"
    maximum = _detector(max_duration_seconds=0.4, minimum_utterance_ms=100)
    assert _feed(maximum, [SPEECH] * 40) == "max_duration"
    silent_maximum = _detector(max_duration_seconds=0.4, minimum_utterance_ms=100)
    assert _feed(silent_maximum, [SILENCE] * 40) == "no_speech"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_invalid_threshold_is_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match=r"energy_threshold|finite"):
        _detector(energy_threshold=invalid)


def test_voice_settings_legacy_onset_alias_and_explicit_precedence() -> None:
    legacy = VoiceSettings(vad_required_frames=7)
    explicit = VoiceSettings(vad_required_frames=7, vad_onset_frames=4)
    assert legacy.vad_onset_frames == 7
    assert explicit.vad_onset_frames == 4
    assert VoiceSettings().endpoint_silence_ms == 650


def test_voice_settings_reject_impossible_turn_timing() -> None:
    with pytest.raises(ValueError, match="minimum_utterance"):
        VoiceSettings(minimum_utterance_ms=1_000, utterance_max_seconds=0.5)
    with pytest.raises(ValueError, match="hangover"):
        VoiceSettings(vad_hangover_ms=700, endpoint_silence_ms=650)


class _Source:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            for frame in self.frames:
                await asyncio.sleep(0)
                yield frame
        finally:
            self.closed = True

    async def aclose(self) -> None:
        self.closed = True


async def test_shared_capture_preserves_order_preroll_and_closes_owned_source() -> None:
    pre = _frame(1)
    source = _Source([SPEECH] * 30 + [SILENCE] * 65)
    result = await capture_streamed_utterance(
        source,
        endpoint_detector=_detector(),
        pre_roll=[pre],
    )
    assert result.frames[0] == pre
    assert list(result.frames[1:31]) == [SPEECH] * 30
    assert source.closed is True
    assert result.stop_reason == "end_of_speech"


async def test_shared_capture_does_not_close_shared_source() -> None:
    source = _Source([SPEECH] * 30 + [SILENCE] * 65)
    result = await capture_streamed_utterance(
        source,
        endpoint_detector=_detector(),
        close_source=False,
    )
    assert result.stop_reason == "end_of_speech"
    assert source.closed is False


async def test_shared_capture_returns_distinct_mute_and_stop() -> None:
    muted = await capture_streamed_utterance(
        _Source([SILENCE] * 2),
        endpoint_detector=_detector(),
        muted=lambda: True,
    )
    stopped = await capture_streamed_utterance(
        _Source([SILENCE] * 2),
        endpoint_detector=_detector(),
        stop_requested=lambda: True,
    )
    assert muted.stop_reason == "muted"
    assert stopped.stop_reason == "stopped"


class _InfiniteSource(_Source):
    def __init__(self) -> None:
        super().__init__([])

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate_forever()

    async def _iterate_forever(self) -> AsyncIterator[bytes]:
        try:
            while True:
                await asyncio.sleep(0)
                yield SILENCE
        finally:
            self.closed = True


async def test_capture_cancellation_closes_owned_source() -> None:
    source = _InfiniteSource()
    task = asyncio.create_task(capture_streamed_utterance(source, endpoint_detector=_detector()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed is True


async def test_capture_max_duration_bounds_accumulation() -> None:
    source = _InfiniteSource()
    result = await capture_streamed_utterance(
        source,
        endpoint_detector=_detector(
            max_duration_seconds=0.4,
            minimum_utterance_ms=100,
        ),
    )
    assert result.stop_reason == "no_speech"
    assert sum(len(frame) for frame in result.frames) <= 16_000 * 2 * 0.4
