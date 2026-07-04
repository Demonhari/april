from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WakeSource = Literal["voice", "terminal", "desktop", "hotkey", "socket"]

WAKE_SOURCES: tuple[str, ...] = ("voice", "terminal", "desktop", "hotkey", "socket")


class WakeEvent(BaseModel):
    """The local JSON wake event format shared by every wake surface.

    ``text`` is an optional already-transcribed/typed command. It is treated as a
    user message downstream (never as instructions to bypass policy) and is never
    persisted into the wake_events table — only its presence is recorded.

    ``captured_at`` and ``session_hint`` are optional v2 additions and stay
    backward compatible: older senders simply omit them. ``captured_at`` is the
    sender's capture timestamp (ISO 8601); ``session_hint`` lets a surface ask to
    join a specific still-open session. Both are advisory: the hint only joins a
    session that is genuinely open, and never bypasses routing or permissions.
    """

    model_config = ConfigDict(extra="forbid")

    source: WakeSource
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    text: str | None = Field(default=None, max_length=50_000)
    reason: str | None = Field(default=None, max_length=200)
    accepted: bool = True
    captured_at: str | None = Field(default=None, max_length=64)
    session_hint: str | None = Field(default=None, max_length=128)


class WakeResolution(BaseModel):
    """Outcome of routing a wake event through the session manager."""

    session_id: str
    conversation_id: str | None
    joined_existing: bool
    wake_event_id: str
