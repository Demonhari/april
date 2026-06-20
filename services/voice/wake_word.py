from __future__ import annotations

from pathlib import Path


class WakeWordDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def available(self) -> bool:
        return self.model_path is not None and self.model_path.exists()
