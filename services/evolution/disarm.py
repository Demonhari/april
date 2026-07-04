"""Disarmed execution context for Dreamer phases.

While a Dreamer phase (replay/distill/mine/evolve/examine) is running, normal
tool execution must be impossible: the nightly dream analyses local data and
writes only inside the evolution fence — it never acts. The context variable is
task-local, so a Dreamer run inside the shared Core API process disarms only
its own execution path, never concurrent user requests.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

_DISARMED_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "april_dreamer_disarmed_phase", default=None
)


def active_disarmed_phase() -> str | None:
    """The Dreamer phase currently disarming tool execution, if any."""
    return _DISARMED_PHASE.get()


@contextlib.contextmanager
def disarmed_execution(phase: str) -> Iterator[None]:
    """Mark the current task as disarmed for the duration of a Dreamer phase."""
    token = _DISARMED_PHASE.set(phase)
    try:
        yield
    finally:
        _DISARMED_PHASE.reset(token)
