from __future__ import annotations

import math
import struct

_INT16_SCALE = 32768.0


def pcm16le_rms(frame: bytes) -> float:
    """Return normalized RMS for signed 16-bit little-endian PCM."""
    if not frame:
        return 0.0
    if len(frame) % 2:
        raise ValueError("VoiceActivityDetector requires signed 16-bit PCM frames.")

    total = 0
    samples = 0
    for (sample,) in struct.iter_unpack("<h", frame):
        total += sample * sample
        samples += 1
    return math.sqrt(total / samples) / _INT16_SCALE


class VoiceActivityDetector:
    def __init__(self, *, energy_threshold: float = 0.01, required_frames: int = 3) -> None:
        self.energy_threshold = energy_threshold
        self.required_frames = required_frames
        self._speech_frames = 0

    def is_speech(self, frame: bytes) -> bool:
        if not frame:
            self._speech_frames = 0
            return False
        rms = pcm16le_rms(frame)
        if rms >= self.energy_threshold:
            self._speech_frames += 1
        else:
            self._speech_frames = 0
        return self._speech_frames >= self.required_frames
