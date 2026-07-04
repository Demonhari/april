from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WakeSource = Literal["voice", "terminal", "desktop", "hotkey", "socket"]

WAKE_SOURCES: tuple[str, ...] = ("voice", "terminal", "desktop", "hotkey", "socket")


def normalize_transcript_alias(payload: object) -> object:
    """Fold the v2 ``transcript`` payload key into the canonical ``text`` field.

    Senders may use either name; internally only ``text`` exists. A payload
    carrying both keys with different values is ambiguous and rejected rather
    than silently preferring one.
    """
    if not isinstance(payload, dict) or "transcript" not in payload:
        return payload
    payload = dict(payload)
    transcript = payload.pop("transcript")
    text = payload.get("text")
    if text is not None and transcript is not None and text != transcript:
        raise ValueError("wake payload has conflicting 'text' and 'transcript' values")
    if payload.get("text") is None:
        payload["text"] = transcript
    return payload


class WakeEvent(BaseModel):
    """The local JSON wake event format shared by every wake surface.

    ``text`` is an optional already-transcribed/typed command. It is treated as a
    user message downstream (never as instructions to bypass policy) and is never
    persisted into the wake_events table — only its presence is recorded.
    ``transcript`` is accepted as a backward/forward-compatible payload alias
    for ``text`` and is normalized away at validation time.

    ``captured_at`` and ``session_hint`` are optional v2 additions and stay
    backward compatible: older senders simply omit them. ``captured_at`` is the
    sender's capture timestamp (ISO 8601); ``session_hint`` lets a surface ask to
    join a specific still-open session. Both are advisory: the hint only joins a
    session that is genuinely open, and never bypasses routing or permissions.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _accept_transcript_alias(cls, payload: object) -> object:
        return normalize_transcript_alias(payload)

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
