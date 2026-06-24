from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator

import pytest

from services.april_runtime.stream_pump import pump_token_stream


async def _wait_event(event: threading.Event, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("producer thread did not finish in time")
        await asyncio.sleep(0.01)


class FloodProducer:
    """Yields effectively forever and never checks cancellation itself.

    This isolates the pump's own backpressure + cancellation: the only thing that
    can stop it is the queue refusing more items once cancelled.
    """

    def __init__(self, total: int = 1_000_000) -> None:
        self.total = total
        self.produced = 0
        self.finished = threading.Event()

    def __call__(self, _is_cancelled: Callable[[], bool]) -> Iterator[str]:
        try:
            for index in range(self.total):
                self.produced += 1
                yield f"tok{index} "
        finally:
            self.finished.set()


async def test_pump_delivers_all_tokens_then_completes() -> None:
    def producer(_is_cancelled: Callable[[], bool]) -> Iterator[str]:
        yield "a "
        yield "b "
        yield "c"

    tokens = [token async for token in pump_token_stream(producer)]
    assert tokens == ["a ", "b ", "c"]


async def test_pump_propagates_producer_exception() -> None:
    def producer(_is_cancelled: Callable[[], bool]) -> Iterator[str]:
        yield "a "
        raise RuntimeError("boom")

    gen = pump_token_stream(producer)
    assert await gen.__anext__() == "a "
    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()
    await gen.aclose()


async def test_cancellation_returns_within_bounded_timeout() -> None:
    producer = FloodProducer()
    gen = pump_token_stream(producer, max_queue=8, poll_seconds=0.02)
    tokens = [await gen.__anext__() for _ in range(3)]
    assert len(tokens) == 3
    # Closing the consumer must return promptly even though the producer would
    # otherwise generate forever.
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    await _wait_event(producer.finished)
    assert producer.produced < producer.total


async def test_full_queue_cannot_deadlock_generation() -> None:
    producer = FloodProducer()
    gen = pump_token_stream(producer, max_queue=4, poll_seconds=0.02)
    await gen.__anext__()  # start the producer thread
    # Let the bounded queue fill; backpressure must stop unbounded production.
    await asyncio.sleep(0.15)
    assert producer.produced <= 50  # bounded, not a million
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    await _wait_event(producer.finished)


async def test_cooperative_cancellation_via_is_cancelled() -> None:
    seen: list[int] = []

    def producer(is_cancelled: Callable[[], bool]) -> Iterator[str]:
        for index in range(1000):
            if is_cancelled():
                return
            seen.append(index)
            yield f"{index} "

    gen = pump_token_stream(producer, max_queue=4, poll_seconds=0.02)
    await gen.__anext__()
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    assert len(seen) < 1000
