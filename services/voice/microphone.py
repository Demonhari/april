from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from april_common.errors import RuntimeUnavailableError


class Microphone:
    async def record_push_to_talk(self, output_path: Path) -> Path:
        raise RuntimeUnavailableError(
            "Real microphone recording requires optional voice dependencies "
            "and explicit CLI invocation."
        )

    async def frames(self) -> AsyncIterator[bytes]:
        raise RuntimeUnavailableError(
            "Streaming microphone frames require optional voice dependencies "
            "and explicit CLI invocation."
        )
        yield b""


class SoundDeviceMicrophone(Microphone):
    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 16_000,
        channels: int = 1,
        max_seconds: float = 30.0,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_seconds = max_seconds

    async def record_push_to_talk(self, output_path: Path) -> Path:
        if self.max_seconds <= 0 or self.max_seconds > 300:
            raise RuntimeUnavailableError("Voice recording duration is outside safe bounds.")
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "sounddevice is not installed. Install APRIL voice extras to record audio."
            ) from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def record() -> Any:
            frames = int(self.sample_rate * self.max_seconds)
            try:
                audio = sd.rec(
                    frames,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    device=self.device,
                )
                sd.wait()
                return audio
            except Exception as exc:  # pragma: no cover - depends on host audio stack
                raise RuntimeUnavailableError(
                    "Microphone recording failed. On macOS, check microphone permissions.",
                    {"cause": str(exc)},
                ) from exc

        audio = await asyncio.to_thread(record)
        pcm = np.asarray(audio, dtype=np.int16)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm.tobytes())
        return output_path

    async def frames(self) -> AsyncIterator[bytes]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "sounddevice is not installed. Install APRIL voice extras to stream audio."
            ) from exc
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

        def callback(indata: Any, frames: int, time: Any, status: Any) -> None:
            del frames, time, status
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

        def open_stream() -> Any:
            return sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=160,
                device=self.device,
                callback=callback,
            )

        stream = await asyncio.to_thread(open_stream)
        with stream:
            while True:
                yield await queue.get()


class FakeMicrophone(Microphone):
    def __init__(self, audio_path: Path, *, frames: list[bytes] | None = None) -> None:
        self.audio_path = audio_path
        self._frames = frames or []

    async def record_push_to_talk(self, output_path: Path) -> Path:
        return self.audio_path

    async def frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            await asyncio.sleep(0)
            yield frame
