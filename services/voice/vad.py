from __future__ import annotations


class VoiceActivityDetector:
    def is_speech(self, frame: bytes) -> bool:
        return bool(frame)
