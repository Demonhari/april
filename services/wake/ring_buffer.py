from __future__ import annotations

from collections import deque


class AudioRingBuffer:
    """Bounded pre-roll buffer of raw PCM frames.

    Keeps roughly ``seconds`` of the most recent audio so the onset of a wake
    phrase (spoken while detection was still confirming) is never lost. Frames
    are opaque byte chunks; capacity is enforced by total byte size, so variable
    frame sizes are safe.
    """

    def __init__(
        self,
        *,
        seconds: float,
        sample_rate: int = 16_000,
        bytes_per_sample: int = 2,
        channels: int = 1,
    ) -> None:
        if seconds <= 0:
            raise ValueError("ring buffer seconds must be positive")
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
        self.capacity_bytes = int(seconds * sample_rate * bytes_per_sample * channels)
        self._frames: deque[bytes] = deque()
        self._total_bytes = 0

    def append(self, frame: bytes) -> None:
        if not frame:
            return
        self._frames.append(frame)
        self._total_bytes += len(frame)
        while self._total_bytes > self.capacity_bytes and len(self._frames) > 1:
            dropped = self._frames.popleft()
            self._total_bytes -= len(dropped)

    def snapshot(self) -> list[bytes]:
        """Return buffered frames oldest-first without consuming them."""
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def duration_seconds(self) -> float:
        divisor = self.sample_rate * self.bytes_per_sample * self.channels
        return self._total_bytes / divisor if divisor else 0.0
