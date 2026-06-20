from __future__ import annotations

from pathlib import Path

from april_common.errors import RuntimeUnavailableError


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
    ) -> None:
        super().__init__(model_path)
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._model: object | None = None

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

    def detect(self, frame: bytes) -> bool:
        if len(frame) % 2 != 0:
            raise RuntimeUnavailableError("Wake-word frames must be 16-bit PCM.")
        model = self._load()
        prediction = model.predict(frame)  # type: ignore[attr-defined]
        if not isinstance(prediction, dict):
            return False
        return any(float(score) >= self.threshold for score in prediction.values())
