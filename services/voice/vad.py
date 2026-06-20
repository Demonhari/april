from __future__ import annotations

import audioop


class VoiceActivityDetector:
    def __init__(self, *, energy_threshold: float = 0.01, required_frames: int = 3) -> None:
        self.energy_threshold = energy_threshold
        self.required_frames = required_frames
        self._speech_frames = 0

    def is_speech(self, frame: bytes) -> bool:
        if not frame:
            self._speech_frames = 0
            return False
        rms = audioop.rms(frame, 2) / 32768.0
        if rms >= self.energy_threshold:
            self._speech_frames += 1
        else:
            self._speech_frames = 0
        return self._speech_frames >= self.required_frames
