from __future__ import annotations

import contextlib
import math
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from services.voice.microphone import aclose_frame_source
from services.voice.vad import pcm16le_rms

if TYPE_CHECKING:
    from april_common.settings import VoiceSettings

EndpointState = Literal[
    "calibrating",
    "waiting_for_speech",
    "in_speech",
    "ending",
    "complete",
]
EndpointStopReason = Literal[
    "end_of_speech",
    "max_duration",
    "source_ended",
    "no_speech",
    "too_short",
    "muted",
    "stopped",
]

_MAX_EFFECTIVE_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class PcmFormat:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.sample_width != 2:
            raise ValueError("Endpointing currently requires signed 16-bit PCM.")


@dataclass(frozen=True, slots=True)
class EndpointMetrics:
    stop_reason: EndpointStopReason
    frame_count: int
    speech_frame_count: int
    captured_duration_ms: int
    speech_duration_ms: int
    trailing_silence_ms: int
    endpoint_latency_ms: int | None
    calibrated_noise_floor: float
    effective_energy_threshold: float
    speech_started: bool
    minimum_duration_met: bool


@dataclass(frozen=True, slots=True)
class CapturedUtterance:
    frames: tuple[bytes, ...]
    metrics: EndpointMetrics
    stop_reason: EndpointStopReason


def pcm_frame_duration_ms(frame: bytes, pcm_format: PcmFormat) -> float:
    """Return PCM duration from bytes and reject partial/malformed samples."""
    frame_width = pcm_format.channels * pcm_format.sample_width
    if not frame or len(frame) % frame_width:
        raise ValueError("Malformed PCM frame length.")
    samples_per_channel = len(frame) / frame_width
    return samples_per_channel * 1_000.0 / pcm_format.sample_rate


class UtteranceEndpointDetector:
    """Calibrated energy onset plus sustained-silence utterance endpointing.

    The state machine never stores audio. Callers own frame retention and files.
    Hangover affects only the visible ``ending`` transition: silence is measured
    from its first frame, so it is not added on top of the configured endpoint.
    """

    def __init__(
        self,
        *,
        pcm_format: PcmFormat | None = None,
        energy_threshold: float = 0.01,
        onset_frames: int = 3,
        endpoint_silence_ms: int = 650,
        minimum_utterance_ms: int = 300,
        noise_calibration_ms: int = 300,
        noise_threshold_multiplier: float = 2.5,
        noise_threshold_margin: float = 0.002,
        hangover_ms: int = 100,
        max_duration_seconds: float = 15.0,
        initial_speech: bool = False,
        initial_speech_duration_ms: float = 0.0,
    ) -> None:
        numeric = (
            energy_threshold,
            noise_threshold_multiplier,
            noise_threshold_margin,
            max_duration_seconds,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Endpoint numeric settings must be finite.")
        if not 0.0 < energy_threshold <= _MAX_EFFECTIVE_THRESHOLD:
            raise ValueError("energy_threshold is outside the safe normalized range.")
        if onset_frames < 1:
            raise ValueError("onset_frames must be positive.")
        if not 300 <= endpoint_silence_ms <= 2_000:
            raise ValueError("endpoint_silence_ms is outside the safe range.")
        if minimum_utterance_ms < 0:
            raise ValueError("minimum_utterance_ms cannot be negative.")
        if noise_calibration_ms < 0:
            raise ValueError("noise_calibration_ms cannot be negative.")
        if not 1.0 <= noise_threshold_multiplier <= 10.0:
            raise ValueError("noise_threshold_multiplier is outside the safe range.")
        if not 0.0 <= noise_threshold_margin <= 0.1:
            raise ValueError("noise_threshold_margin is outside the safe range.")
        if not 0 <= hangover_ms <= endpoint_silence_ms:
            raise ValueError("hangover_ms cannot exceed endpoint_silence_ms.")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        if minimum_utterance_ms > max_duration_seconds * 1_000:
            raise ValueError("minimum utterance cannot exceed maximum duration.")

        self.pcm_format = pcm_format or PcmFormat()
        self.energy_threshold = energy_threshold
        self.onset_frames = onset_frames
        self.endpoint_silence_ms = endpoint_silence_ms
        self.minimum_utterance_ms = minimum_utterance_ms
        self.noise_calibration_ms = noise_calibration_ms
        self.noise_threshold_multiplier = noise_threshold_multiplier
        self.noise_threshold_margin = noise_threshold_margin
        self.hangover_ms = hangover_ms
        self.max_duration_ms = max_duration_seconds * 1_000.0

        self.state: EndpointState = "in_speech" if initial_speech else "calibrating"
        self.frame_count = 0
        self.speech_frame_count = onset_frames if initial_speech else 0
        self.audio_duration_ms = 0.0
        self.speech_duration_ms = initial_speech_duration_ms if initial_speech else 0.0
        self.trailing_silence_ms = 0.0
        self._onset_run: list[float] = []
        self._noise_samples: list[float] = []
        self._noise_duration_ms = 0.0
        self.calibrated_noise_floor = 0.0
        self.effective_energy_threshold = energy_threshold
        self._complete_reason: EndpointStopReason | None = None
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._silence_started_at: float | None = None

    @property
    def speech_started(self) -> bool:
        return self.state in {"in_speech", "ending", "complete"} and (
            self.speech_frame_count > 0 or self.speech_duration_ms > 0
        )

    @property
    def complete(self) -> bool:
        return self.state == "complete"

    def process(self, frame: bytes, *, now: float | None = None) -> EndpointStopReason | None:
        if self.complete:
            raise RuntimeError("Cannot process audio after endpoint completion.")
        duration_ms = pcm_frame_duration_ms(frame, self.pcm_format)
        energy = pcm16le_rms(frame)
        if not math.isfinite(energy):
            raise ValueError("PCM energy must be finite.")

        self.frame_count += 1
        self.audio_duration_ms += duration_ms
        if now is not None:
            if self._first_frame_at is None:
                self._first_frame_at = now
            self._last_frame_at = now
        if self.state in {"calibrating", "waiting_for_speech"}:
            # Only quiet candidates update calibration. A loud immediate first
            # frame can therefore start onset instead of poisoning the floor.
            if energy < self.effective_energy_threshold:
                self._update_calibration(energy, duration_ms)
                self._onset_run.clear()
            else:
                self._onset_run.append(duration_ms)
                if len(self._onset_run) >= self.onset_frames:
                    self.state = "in_speech"
                    onset_duration = sum(self._onset_run[-self.onset_frames :])
                    self.speech_frame_count += self.onset_frames
                    self.speech_duration_ms += onset_duration
                    self._onset_run.clear()
            if self.state != "in_speech":
                self.state = (
                    "calibrating"
                    if self._noise_duration_ms < self.noise_calibration_ms
                    else "waiting_for_speech"
                )
            return self._finish_for_maximum(duration_ms)

        # Hysteresis accepts a slightly lower level as continued speech, but
        # onset always uses the full effective threshold.
        continuation_threshold = max(
            self.energy_threshold * 0.8,
            self.effective_energy_threshold * 0.8,
        )
        if energy >= continuation_threshold:
            self.state = "in_speech"
            self.trailing_silence_ms = 0.0
            self._silence_started_at = None
            self.speech_frame_count += 1
            self.speech_duration_ms += duration_ms
            return self._finish_for_maximum(duration_ms)

        if self._silence_started_at is None:
            self._silence_started_at = now
        self.trailing_silence_ms += duration_ms
        if now is not None and self._silence_started_at is not None:
            self.trailing_silence_ms = max(
                self.trailing_silence_ms,
                (now - self._silence_started_at) * 1_000 + duration_ms,
            )
        self.state = "ending" if self.trailing_silence_ms >= self.hangover_ms else "in_speech"
        if self.trailing_silence_ms >= self.endpoint_silence_ms:
            reason: EndpointStopReason = (
                "end_of_speech"
                if self.speech_duration_ms >= self.minimum_utterance_ms
                else "too_short"
            )
            return self._finish(reason)
        return self._finish_for_maximum(duration_ms)

    def finish(self, reason: EndpointStopReason = "source_ended") -> EndpointMetrics:
        if not self.complete:
            if reason in {"source_ended", "max_duration"}:
                if not self.speech_started:
                    reason = "no_speech"
                elif self.speech_duration_ms < self.minimum_utterance_ms:
                    reason = "too_short"
            self._finish(reason)
        return self.metrics()

    def metrics(self) -> EndpointMetrics:
        if self._complete_reason is None:
            raise RuntimeError("Endpoint metrics are available only after completion.")
        minimum_met = self.speech_duration_ms >= self.minimum_utterance_ms
        latency = (
            round(self.trailing_silence_ms)
            if self._complete_reason == "end_of_speech"
            else None
        )
        return EndpointMetrics(
            stop_reason=self._complete_reason,
            frame_count=self.frame_count,
            speech_frame_count=self.speech_frame_count,
            captured_duration_ms=round(self._captured_duration_ms(0.0)),
            speech_duration_ms=round(self.speech_duration_ms),
            trailing_silence_ms=round(self.trailing_silence_ms),
            endpoint_latency_ms=latency,
            calibrated_noise_floor=round(self.calibrated_noise_floor, 6),
            effective_energy_threshold=round(self.effective_energy_threshold, 6),
            speech_started=self.speech_started,
            minimum_duration_met=minimum_met,
        )

    def _update_calibration(self, energy: float, duration_ms: float) -> None:
        if self._noise_duration_ms >= self.noise_calibration_ms:
            return
        # The time bound also bounds memory even for unusually tiny frames.
        self._noise_samples.append(min(energy, self.energy_threshold))
        self._noise_duration_ms += duration_ms
        if len(self._noise_samples) > 2_000:
            self._noise_samples = self._noise_samples[-2_000:]
        ordered = sorted(self._noise_samples)
        percentile_index = max(0, int((len(ordered) - 1) * 0.25))
        self.calibrated_noise_floor = ordered[percentile_index]
        calibrated = (
            self.calibrated_noise_floor * self.noise_threshold_multiplier
            + self.noise_threshold_margin
        )
        self.effective_energy_threshold = min(
            _MAX_EFFECTIVE_THRESHOLD,
            max(self.energy_threshold, calibrated),
        )

    def _finish(self, reason: EndpointStopReason) -> EndpointStopReason:
        self.state = "complete"
        self._complete_reason = reason
        return reason

    def _captured_duration_ms(self, current_frame_ms: float) -> float:
        if self._first_frame_at is None or self._last_frame_at is None:
            return self.audio_duration_ms
        elapsed = (self._last_frame_at - self._first_frame_at) * 1_000 + current_frame_ms
        return max(self.audio_duration_ms, elapsed)

    def _finish_for_maximum(self, current_frame_ms: float) -> EndpointStopReason | None:
        if self._captured_duration_ms(current_frame_ms) < self.max_duration_ms:
            return None
        if not self.speech_started:
            return self._finish("no_speech")
        if self.speech_duration_ms < self.minimum_utterance_ms:
            return self._finish("too_short")
        return self._finish("max_duration")


def endpoint_detector_from_settings(
    settings: VoiceSettings,
    *,
    pcm_format: PcmFormat | None = None,
    onset_frames: int | None = None,
    initial_speech: bool = False,
    initial_speech_duration_ms: float = 0.0,
) -> UtteranceEndpointDetector:
    return UtteranceEndpointDetector(
        pcm_format=pcm_format,
        energy_threshold=settings.vad_energy_threshold,
        onset_frames=onset_frames or settings.vad_onset_frames,
        endpoint_silence_ms=settings.endpoint_silence_ms,
        minimum_utterance_ms=settings.minimum_utterance_ms,
        noise_calibration_ms=settings.noise_calibration_ms,
        noise_threshold_multiplier=settings.noise_threshold_multiplier,
        noise_threshold_margin=settings.noise_threshold_margin,
        hangover_ms=settings.vad_hangover_ms,
        max_duration_seconds=settings.utterance_max_seconds,
        initial_speech=initial_speech,
        initial_speech_duration_ms=initial_speech_duration_ms,
    )


async def capture_streamed_utterance(
    frame_source: AsyncIterator[bytes],
    *,
    endpoint_detector: UtteranceEndpointDetector,
    pre_roll: Sequence[bytes] = (),
    stop_requested: Callable[[], bool] | None = None,
    muted: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    close_source: bool = True,
) -> CapturedUtterance:
    """Capture exactly one bounded utterance from a stream.

    ``close_source=False`` is required when a caller, such as Sentinel, owns and
    continues consuming a shared iterator.
    """
    frames = list(pre_roll)
    total_bytes = sum(map(len, frames))
    max_pcm_bytes = int(
        endpoint_detector.max_duration_ms
        / 1_000
        * endpoint_detector.pcm_format.sample_rate
        * endpoint_detector.pcm_format.channels
        * endpoint_detector.pcm_format.sample_width
    )
    # Pre-roll is intentionally bounded to the same hard audio ceiling.
    while total_bytes > max_pcm_bytes and frames:
        total_bytes -= len(frames.pop(0))
    started_at = clock()
    reason: EndpointStopReason | None = None
    try:
        async for frame in frame_source:
            if muted is not None and muted():
                reason = "muted"
                break
            if stop_requested is not None and stop_requested():
                reason = "stopped"
                break
            now = clock()
            result = endpoint_detector.process(frame, now=now)
            frames.append(frame)
            total_bytes += len(frame)
            while total_bytes > max_pcm_bytes and frames:
                total_bytes -= len(frames.pop(0))
            elapsed_ms = (now - started_at) * 1_000
            if result is not None:
                reason = result
                break
            if elapsed_ms >= endpoint_detector.max_duration_ms:
                reason = "max_duration"
                break
        if reason is None:
            reason = "source_ended"
        metrics = endpoint_detector.finish(reason)
        return CapturedUtterance(tuple(frames), metrics, metrics.stop_reason)
    finally:
        if close_source:
            with contextlib.suppress(Exception):
                await aclose_frame_source(frame_source)
