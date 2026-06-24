from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from april_common.errors import RuntimeUnavailableError
from services.voice.microphone import Microphone, aclose_frame_source, write_pcm_wav

# Hard upper bound regardless of configuration, so a stuck "start" can never
# capture unboundedly.
MAX_PUSH_TO_TALK_SECONDS = 300.0


class PushToTalkSession:
    """A bounded, explicitly controlled push-to-talk capture.

    Unlike the fixed-duration helper, capture runs until the first of:
    an explicit ``request_stop()``, the ``max_seconds`` safety limit, the frame
    source ending, or task cancellation. The frame source is always closed on
    exit so the microphone stream is released.
    """

    def __init__(
        self,
        microphone: Microphone,
        *,
        sample_rate: int = 16_000,
        max_seconds: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not 0 < max_seconds <= MAX_PUSH_TO_TALK_SECONDS:
            raise RuntimeUnavailableError("Voice recording duration is outside safe bounds.")
        self.microphone = microphone
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self._clock = clock
        self._stop = asyncio.Event()
        self.started = False
        self.stop_reason: str | None = None

    def request_stop(self) -> None:
        """Explicit stop (the release half of press-and-hold)."""
        self._stop.set()

    async def capture(self, output_path: Path, *, max_frames: int | None = None) -> Path:
        self.started = True
        self.stop_reason = None
        collected: list[bytes] = []
        deadline = self._clock() + self.max_seconds
        frame_source = self.microphone.frames()
        try:
            async for frame in frame_source:
                collected.append(frame)
                if self._stop.is_set():
                    self.stop_reason = "stopped"
                    break
                if max_frames is not None and len(collected) >= max_frames:
                    self.stop_reason = "max_frames"
                    break
                if self._clock() >= deadline:
                    self.stop_reason = "max_duration"
                    break
            else:
                self.stop_reason = self.stop_reason or "source_ended"
        finally:
            # Release the microphone stream on every exit path, including
            # cancellation.
            await aclose_frame_source(frame_source)
        if not collected:
            raise ValueError("Push-to-talk capture produced no audio.")
        return write_pcm_wav(output_path, collected, sample_rate=self.sample_rate)
