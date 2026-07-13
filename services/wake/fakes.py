from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path

from services.voice.audio_player import AudioPlayer
from services.voice.microphone import Microphone
from services.wake.schemas import WakeEvent

# Test-only fakes for the wake subsystem. Production code never imports these;
# they exist so wake behaviour is fully testable without a microphone, an
# openWakeWord model, whisper.cpp, or any audio hardware.


class FakeFrameMicrophone(Microphone):
    """Yields scripted PCM frames and records stream open/close lifecycle."""

    def __init__(self, frames: Iterable[bytes]) -> None:
        self._frames = list(frames)
        self.opened_streams = 0
        self.released = False

    async def frames(self) -> AsyncIterator[bytes]:
        self.opened_streams += 1
        self.released = False
        try:
            for frame in self._frames:
                await asyncio.sleep(0)
                yield frame
        finally:
            self.released = True


class ScriptedScorer:
    """Wake scorer that replays a fixed score sequence (then 0.0 forever)."""

    def __init__(self, scores: Iterable[float]) -> None:
        self._scores = list(scores)
        self._index = 0
        self.reset_calls = 0

    def score(self, frame: bytes) -> float:
        if self._index >= len(self._scores):
            return 0.0
        value = self._scores[self._index]
        self._index += 1
        return value

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSpeakerVerifier:
    """Deterministic test-only speaker verifier with a fixed local score."""

    def __init__(self, score: float) -> None:
        self.fixed_score = score
        self.calls: list[tuple[tuple[Path, ...], bytes]] = []

    def score(self, enrollment: Sequence[Path], utterance: bytes) -> float:
        self.calls.append((tuple(enrollment), utterance))
        return self.fixed_score


class RecordingAudit:
    """In-memory audit sink for wake tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def write(self, payload: dict[str, object]) -> None:
        self.records.append(dict(payload))


class RecordingAudioPlayer(AudioPlayer):
    """AudioPlayer fake that records play/stop/duck calls."""

    def __init__(self) -> None:
        self.played: list[Path] = []
        self.stop_calls = 0
        self.duck_calls = 0

    async def play(self, audio_path: Path) -> None:
        self.played.append(audio_path)

    async def stop(self) -> None:
        self.stop_calls += 1

    async def duck(self) -> None:
        self.duck_calls += 1


class RecordingDelivery:
    """Wake delivery target that stores every delivered event."""

    def __init__(self) -> None:
        self.events: list[WakeEvent] = []

    async def __call__(self, event: WakeEvent) -> None:
        self.events.append(event)


class ManualClock:
    """Deterministic monotonic clock for cooldown/follow-up tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
