from __future__ import annotations

import asyncio
from pathlib import Path

from services.voice.audio_player import AudioPlayer
from services.wake.response_coordinator import ResponseCoordinator
from services.wake.schemas import WakeEvent


class _Player(AudioPlayer):
    def __init__(self) -> None:
        self.stop_calls = 0
        self.duck_calls = 0

    async def play(self, audio_path: Path) -> None:
        del audio_path

    async def stop(self) -> None:
        self.stop_calls += 1

    async def duck(self) -> None:
        self.duck_calls += 1


class _BlockingDelivery:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.played: list[int] = []
        self.release = asyncio.Event()

    async def __call__(self, event: WakeEvent) -> None:
        del event

    async def deliver_generation(
        self,
        event: WakeEvent,
        *,
        generation: int,
        is_current,
        set_state,
    ) -> None:
        del event
        self.started.append(generation)
        await self.release.wait()
        if is_current(generation):
            set_state("speaking", generation)
            self.played.append(generation)


async def test_new_response_supersedes_old_and_stale_generation_cannot_play() -> None:
    player = _Player()
    delivery = _BlockingDelivery()
    coordinator = ResponseCoordinator(deliver=delivery, player=player, action="stop")
    first = await coordinator.submit(WakeEvent(source="voice"))
    await asyncio.sleep(0)
    second = await coordinator.submit(WakeEvent(source="voice"))
    await asyncio.sleep(0)
    delivery.release.set()
    await coordinator.drain()
    assert first != second
    assert delivery.played == [second]
    assert player.stop_calls == 1
    assert coordinator.active is False


async def test_duck_action_remains_distinct_and_shutdown_collects_task() -> None:
    player = _Player()
    delivery = _BlockingDelivery()
    coordinator = ResponseCoordinator(deliver=delivery, player=player, action="duck")
    await coordinator.submit(WakeEvent(source="voice"))
    await asyncio.sleep(0)
    await coordinator.interrupt(reason="wake_word")
    await coordinator.shutdown()
    assert player.duck_calls == 1
    assert player.stop_calls == 0
    assert coordinator.active is False


async def test_delivery_failure_is_collected_as_safe_metadata() -> None:
    records: list[dict[str, object]] = []

    async def fail(event: WakeEvent) -> None:
        del event
        raise RuntimeError("sensitive response")

    coordinator = ResponseCoordinator(
        deliver=fail,
        player=None,
        action="stop",
        audit=records.append,
    )
    await coordinator.submit(WakeEvent(source="voice"))
    await coordinator.drain()
    assert records == [
        {
            "event_type": "voice_response_failed",
            "generation": 1,
            "error_type": "RuntimeError",
        }
    ]
    assert "sensitive response" not in str(records)
