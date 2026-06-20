from __future__ import annotations

from pathlib import Path

from april_common.errors import RuntimeUnavailableError


class Microphone:
    async def record_push_to_talk(self, output_path: Path) -> Path:
        raise RuntimeUnavailableError(
            "Real microphone recording requires optional voice dependencies "
            "and explicit CLI invocation."
        )


class FakeMicrophone(Microphone):
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path

    async def record_push_to_talk(self, output_path: Path) -> Path:
        return self.audio_path
