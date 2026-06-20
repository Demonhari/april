from __future__ import annotations

from pathlib import Path


class AudioPlayer:
    async def play(self, audio_path: Path) -> None:
        return None


class FakeAudioPlayer(AudioPlayer):
    async def play(self, audio_path: Path) -> None:
        return None
