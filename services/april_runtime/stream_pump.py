from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from collections.abc import AsyncIterator, Callable, Iterable

# A bounded queue gives backpressure without unbounded memory growth. The poll
# interval bounds how long a stalled publish can ignore a cancellation request.
DEFAULT_MAX_QUEUE = 64
DEFAULT_PUBLISH_POLL_SECONDS = 0.05
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 5.0

# Sentinel marking the end of the stream on the queue.
_DONE = object()

# `make_iterator` is handed an `is_cancelled` predicate so a producer (e.g. the
# llama.cpp generation loop) can wire stopping criteria and halt promptly.
MakeIterator = Callable[[Callable[[], bool]], Iterable[str]]


async def pump_token_stream(
    make_iterator: MakeIterator,
    *,
    max_queue: int = DEFAULT_MAX_QUEUE,
    poll_seconds: float = DEFAULT_PUBLISH_POLL_SECONDS,
    cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
) -> AsyncIterator[str]:
    """Bridge a blocking, thread-run token iterator to a cancellation-safe stream.

    Design goals (see Task E):

    * Bounded queue: the producer experiences backpressure when the consumer
      lags, but a stalled publish is interrupted within ``poll_seconds`` instead
      of blocking the generation thread forever on a full queue.
    * Cooperative cancellation: a single ``threading.Event`` is the per-generation
      cancel signal. It is set whenever the consumer stops (normal break,
      exception, ``aclose``/disconnect, or task cancellation) and the producer is
      always joined within ``cleanup_timeout``.
    * Exceptions raised by the producer are re-raised to the consumer so the
      caller can convert them into the standard API error structure.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=max_queue)
    cancel_event = threading.Event()

    def publish(item: object) -> bool:
        # Runs on the producer thread. Returns False once cancelled so the
        # producer stops generating instead of blocking on a full queue.
        if cancel_event.is_set():
            return False
        future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while True:
            if cancel_event.is_set():
                future.cancel()
                return False
            try:
                future.result(timeout=poll_seconds)
                return True
            except concurrent.futures.TimeoutError:
                continue
            except concurrent.futures.CancelledError:
                return False

    def produce() -> None:
        try:
            for token in make_iterator(cancel_event.is_set):
                if not publish(token):
                    break
        except Exception as exc:  # surfaced to the consumer, never swallowed
            publish(exc)
        finally:
            # No-op if already cancelled (publish short-circuits), so this never
            # blocks the producer thread during teardown.
            publish(_DONE)

    task = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        cancel_event.set()
        await _join_producer(task, cleanup_timeout)


async def _join_producer(task: asyncio.Task[None], timeout: float) -> None:
    # The producer observes the cancel flag and returns quickly; shield it so the
    # llama.cpp worker still unwinds cleanly even if the consumer is being
    # cancelled, but never wait longer than `timeout`.
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    except Exception:
        # The producer reports its own failures through the queue; a failure here
        # is teardown noise and must not mask the consumer's exit path.
        pass
