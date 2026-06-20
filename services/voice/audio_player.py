from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np

from april_common.errors import RuntimeUnavailableError


class AudioPlayer:
    async def play(self, audio_path: Path) -> None:
        return None


class SoundDeviceAudioPlayer(AudioPlayer):
    def __init__(self, *, device: str | int | None = None) -> None:
        self.device = device

    async def play(self, audio_path: Path) -> None:
        try:
            with wave.open(str(audio_path), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
        except wave.Error as exc:
            raise RuntimeUnavailableError("Audio player requires a valid WAV file.") from exc
        if sample_width != 2:
            raise RuntimeUnavailableError("Audio player supports 16-bit PCM WAV files.")
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape((-1, channels))
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "sounddevice is not installed. Install APRIL voice extras to play audio."
            ) from exc

        def play() -> None:
            sd.play(audio, samplerate=sample_rate, device=self.device)
            sd.wait()

        await asyncio.to_thread(play)


class FakeAudioPlayer(AudioPlayer):
    async def play(self, audio_path: Path) -> None:
        return None
