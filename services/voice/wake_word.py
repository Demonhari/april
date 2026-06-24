from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

import numpy as np

from april_common.errors import RuntimeUnavailableError

# openWakeWord expects 80 ms windows of mono 16 kHz 16-bit PCM (1280 samples).
WAKE_WORD_SAMPLE_RATE = 16_000
WAKE_WORD_FRAME_SAMPLES = 1280


class WakeWordDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def available(self) -> bool:
        return self.model_path is not None and self.model_path.exists()


class OpenWakeWordDetector(WakeWordDetector):
    def __init__(
        self,
        model_path: Path | None,
        *,
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
        frame_samples: int = WAKE_WORD_FRAME_SAMPLES,
        sample_rate: int = WAKE_WORD_SAMPLE_RATE,
        channels: int = 1,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        super().__init__(model_path)
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.frame_samples = frame_samples
        self.sample_rate = sample_rate
        self.channels = channels
        self._clock = clock
        self._model: object | None = None
        # Aggregation buffer of int16 samples between predictions.
        self._buffer = np.empty(0, dtype=np.int16)
        self._last_detection_at: float | None = None

    def _load(self) -> object:
        if self.model_path is None:
            raise RuntimeUnavailableError("Wake-word model path is not configured.")
        if not self.model_path.exists():
            raise RuntimeUnavailableError(f"Wake-word model is missing: {self.model_path}")
        if self._model is None:
            try:
                from openwakeword.model import Model
            except ImportError as exc:
                raise RuntimeUnavailableError(
                    "openWakeWord is not installed. Install APRIL voice extras to use wake word."
                ) from exc
            self._model = Model(wakeword_models=[str(self.model_path)])
        return self._model

    def reset(self) -> None:
        """Clear aggregation/cooldown state at a conversation boundary."""
        self._buffer = np.empty(0, dtype=np.int16)
        self._last_detection_at = None
        model = self._model
        model_reset = getattr(model, "reset", None)
        if callable(model_reset):
            model_reset()

    def detect(self, frame: bytes) -> bool:
        if len(frame) % 2 != 0:
            raise RuntimeUnavailableError("Wake-word frames must be 16-bit PCM.")
        if self.channels != 1 or self.sample_rate != WAKE_WORD_SAMPLE_RATE:
            raise RuntimeUnavailableError(
                "Wake word requires mono, 16 kHz, 16-bit signed PCM audio."
            )
        model = self._load()
        # Raw microphone bytes are converted to int16 samples; the openWakeWord
        # model is only ever called with a numpy int16 array, never raw bytes.
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size:
            self._buffer = np.concatenate((self._buffer, samples))
        detected = False
        # Emit one prediction per fully aggregated 80 ms window.
        while self._buffer.size >= self.frame_samples:
            window = self._buffer[: self.frame_samples]
            self._buffer = self._buffer[self.frame_samples :]
            if self._predict_window(model, window):
                detected = True
        return detected

    def _predict_window(self, model: object, window: np.ndarray) -> bool:
        # Debounce: a single utterance must not re-trigger during the cooldown.
        if self._in_cooldown():
            return False
        prediction = model.predict(window)  # type: ignore[attr-defined]
        if not isinstance(prediction, dict):
            return False
        if any(float(score) >= self.threshold for score in prediction.values()):
            self._last_detection_at = self._clock()
            return True
        return False

    def _in_cooldown(self) -> bool:
        if self._last_detection_at is None or self.cooldown_seconds <= 0:
            return False
        return (self._clock() - self._last_detection_at) < self.cooldown_seconds
