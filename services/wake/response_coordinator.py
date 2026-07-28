from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from services.voice.audio_player import AudioPlayer
from services.wake.schemas import WakeEvent

ResponseState = Literal[
    "idle",
    "listening",
    "capturing",
    "thinking",
    "speaking",
    "interrupted",
    "degraded",
]


class GenerationDelivery(Protocol):
    async def deliver_generation(
        self,
        event: WakeEvent,
        *,
        generation: int,
        is_current: Callable[[int], bool],
        set_state: Callable[[ResponseState, int], None],
    ) -> None: ...


class ResponseCoordinator:
    """Own one bounded response task and prevent stale playback."""

    def __init__(
        self,
        *,
        deliver: Callable[[WakeEvent], Awaitable[None]],
        player: AudioPlayer | None,
        action: Literal["stop", "duck"],
        on_state: Callable[[ResponseState], None] | None = None,
        on_complete: Callable[[], None] | None = None,
        audit: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.deliver = deliver
        self.player = player
        self.action = action
        self.on_state = on_state
        self.on_complete = on_complete
        self.audit = audit
        self.clock = clock
        self.generation = 0
        self.active_task: asyncio.Task[None] | None = None
        self.playback_active = False
        self.playback_started_at: float | None = None
        self.interrupt_count = 0
        self.interrupt_reasons: list[str] = []
        self.last_interrupt_latency_ms: int | None = None
        self.last_barge_in_latency_ms: int | None = None

    @property
    def active(self) -> bool:
        return self.active_task is not None and not self.active_task.done()

    def is_current(self, generation: int) -> bool:
        return generation == self.generation

    async def interrupt(self, *, reason: str) -> bool:
        if not self.active and not self.playback_active:
            return False
        self.generation += 1
        action_started = self.clock()
        task = self.active_task
        self.active_task = None
        if task is not None and not task.done():
            task.cancel()
        if self.player is not None:
            try:
                if self.action == "duck":
                    await self.player.duck()
                else:
                    await self.player.stop()
            except Exception as exc:
                self._audit(
                    {
                        "event_type": "voice_playback_control_failed",
                        "generation": self.generation,
                        "action": self.action,
                        "error_type": type(exc).__name__,
                    }
                )
        self.playback_active = False
        self.playback_started_at = None
        self.interrupt_count += 1
        self.interrupt_reasons.append(reason)
        self.last_interrupt_latency_ms = max(0, round((self.clock() - action_started) * 1_000))
        if reason not in {"shutdown", "muted", "stopped", "superseded"}:
            self.last_barge_in_latency_ms = self.last_interrupt_latency_ms
        self._state("interrupted")
        self._audit(
            {
                "event_type": "voice_response_interrupted",
                "generation": self.generation,
                "reason": reason,
            }
        )
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return True

    async def submit(self, event: WakeEvent) -> int:
        if self.active or self.playback_active:
            await self.interrupt(reason="superseded")
        self.generation += 1
        generation = self.generation
        task = asyncio.create_task(
            self._run(event, generation),
            name=f"sentinel-response-{generation}",
        )
        self.active_task = task
        return generation

    async def shutdown(self) -> None:
        await self.interrupt(reason="shutdown")
        task = self.active_task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.active_task = None

    async def drain(self) -> None:
        task = self.active_task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _run(self, event: WakeEvent, generation: int) -> None:
        if callable(getattr(self.deliver, "deliver_generation", None)):
            self._set_generation_state("thinking", generation)
        completed = False
        try:
            generation_delivery = getattr(self.deliver, "deliver_generation", None)
            if callable(generation_delivery):
                await generation_delivery(
                    event,
                    generation=generation,
                    is_current=self.is_current,
                    set_state=self._set_generation_state,
                )
            else:
                await self.deliver(event)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._audit(
                {
                    "event_type": "voice_response_failed",
                    "generation": generation,
                    "error_type": type(exc).__name__,
                }
            )
            self._set_generation_state("degraded", generation)
        finally:
            if self.is_current(generation):
                self.playback_active = False
                self.playback_started_at = None
                self.active_task = None
                self._state("listening")
                if completed and self.on_complete is not None:
                    self.on_complete()

    def _set_generation_state(self, state: ResponseState, generation: int) -> None:
        if not self.is_current(generation):
            return
        self.playback_active = state == "speaking"
        if state == "speaking":
            self.playback_started_at = self.clock()
        else:
            self.playback_started_at = None
        self._state(state)

    def _state(self, state: ResponseState) -> None:
        if self.on_state is not None:
            self.on_state(state)

    def _audit(self, payload: dict[str, Any]) -> None:
        if self.audit is not None:
            self.audit(payload)
