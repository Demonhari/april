"""Bounded handling of model control/reasoning text at the chat boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

_OPENING_TAG = re.compile(r"^\s*<(think|thinking|analysis)\b[^>]*>", re.IGNORECASE)
_CLOSING_TAG = re.compile(r"</(?:think|thinking|analysis)\s*>", re.IGNORECASE)
_MAX_REASONING_CHARS = 32_768


def sanitize_model_output(content: str) -> str:
    """Remove a leading hidden-reasoning block, preserving user-requested text.

    Only a control block at the beginning of a model answer is treated as
    hidden reasoning. Tags in ordinary prose, code fences, JSON, or literal-tag
    examples remain untouched.
    """
    text = content.lstrip("\ufeff")
    match = _OPENING_TAG.match(text)
    if match is None:
        return content.strip()
    close = _CLOSING_TAG.search(text, match.end(), match.end() + _MAX_REASONING_CHARS)
    if close is None:
        # An interrupted/truncated reasoning-only response has no user answer.
        return ""
    return text[close.end() :].lstrip()


@dataclass(slots=True)
class ReasoningStreamFilter:
    """Incrementally suppress a split leading reasoning block."""

    _buffer: str = ""
    _state: str = "pending"

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._state == "visible":
            return chunk
        self._buffer += chunk
        if len(self._buffer) > _MAX_REASONING_CHARS:
            self._state = "visible"
            visible = self._buffer
            self._buffer = ""
            return visible
        if self._state == "pending":
            opening = _OPENING_TAG.match(self._buffer)
            if opening is None:
                # Hold only enough prefix to decide whether a tag is split.
                stripped = self._buffer.lstrip("\ufeff")
                if len(stripped) < 16 and "<" in stripped:
                    return ""
                self._state = "visible"
                visible = self._buffer
                self._buffer = ""
                return visible
            self._state = "reasoning"
            self._buffer = self._buffer[opening.end() :]
        if self._state == "reasoning":
            close = _CLOSING_TAG.search(self._buffer)
            if close is None:
                self._buffer = self._buffer[-_MAX_REASONING_CHARS:]
                return ""
            visible = self._buffer[close.end() :]
            self._buffer = ""
            self._state = "visible"
            return visible.lstrip()
        return ""

    def finish(self) -> str:
        if self._state == "visible":
            return self._buffer
        # Unclosed leading reasoning is intentionally not exposed.
        return ""
