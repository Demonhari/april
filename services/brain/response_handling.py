"""Bounded handling of model control/reasoning text at the chat boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

_OPENING_TAG = re.compile(r"^\s*\ufeff?\s*<(think|thinking|analysis)\b[^>]*>", re.IGNORECASE)
_CLOSING_TAG = re.compile(r"</(?:think|thinking|analysis)\s*>", re.IGNORECASE)
_OPENING_PREFIX = re.compile(r"^\s*\ufeff?\s*<(?:think|thinking|analysis)\b[^>]*$", re.IGNORECASE)
_MAX_REASONING_CHARS = 32_768
_MAX_CONTROL_BLOCKS = 8
_OPENING_NAMES = ("think", "thinking", "analysis")


def _is_possible_opening_prefix(value: str) -> bool:
    """Keep a chunk boundary before a possible control tag undecided."""

    candidate = value.lstrip()
    if candidate.startswith("\ufeff"):
        candidate = candidate[1:].lstrip()
    candidate = candidate.casefold()
    if not candidate.startswith("<"):
        return False
    body = candidate[1:]
    return any(name.startswith(body) for name in _OPENING_NAMES)


def sanitize_model_output(content: str) -> str:
    """Remove a leading hidden-reasoning block, preserving user-requested text.

    Only a control block at the beginning of a model answer is treated as
    hidden reasoning. Tags in ordinary prose, code fences, JSON, or literal-tag
    examples remain untouched.
    """
    text = content
    blocks = 0
    while blocks < _MAX_CONTROL_BLOCKS:
        match = _OPENING_TAG.match(text)
        if match is None:
            return text
        close = _CLOSING_TAG.search(text, match.end(), match.end() + _MAX_REASONING_CHARS)
        if close is None:
            # An interrupted/truncated reasoning-only response has no user answer.
            return ""
        text = text[close.end() :]
        blocks += 1
    # A model that emits an unbounded chain of control blocks has not produced a
    # trustworthy answer. Never expose the remainder as if it were visible text.
    return "" if _OPENING_TAG.match(text) is not None else text


@dataclass(slots=True)
class ReasoningStreamFilter:
    """Incrementally suppress a split leading reasoning block."""

    _buffer: str = ""
    _state: str = "pending"
    _reasoning_chars: int = 0
    _control_blocks: int = 0

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._state == "aborted":
            return ""
        if self._state == "visible":
            return chunk
        self._buffer += chunk
        while True:
            if self._state == "pending":
                opening = _OPENING_TAG.match(self._buffer)
                if opening is None:
                    # Keep whitespace and a possible split opening tag pending.
                    # Whitespace must not make the filter permanently visible.
                    candidate = self._buffer.lstrip("\ufeff")
                    if (
                        not candidate.strip()
                        or _OPENING_PREFIX.match(candidate) is not None
                        or _is_possible_opening_prefix(candidate)
                    ):
                        return ""
                    self._state = "visible"
                    visible = self._buffer
                    self._buffer = ""
                    return visible
                self._state = "reasoning"
                self._reasoning_chars = len(self._buffer) - opening.end()
                self._buffer = self._buffer[opening.end() :]

            if self._state == "reasoning":
                close = _CLOSING_TAG.search(self._buffer)
                if close is None:
                    self._reasoning_chars = len(self._buffer)
                    if self._reasoning_chars > _MAX_REASONING_CHARS:
                        self._state = "aborted"
                        self._buffer = ""
                    return ""
                self._buffer = self._buffer[close.end() :]
                self._state = "pending"
                self._reasoning_chars = 0
                self._control_blocks += 1
                if self._control_blocks > _MAX_CONTROL_BLOCKS:
                    self._state = "aborted"
                    self._buffer = ""
                    return ""
                # Re-enter pending so consecutive leading control blocks remain
                # hidden instead of becoming visible after the first close.
                continue
            return ""

    def finish(self) -> str:
        if self._state == "visible":
            return self._buffer
        if self._state == "pending" and _OPENING_TAG.match(self._buffer) is None:
            if _OPENING_PREFIX.match(self._buffer) is not None or _is_possible_opening_prefix(
                self._buffer
            ):
                return ""
            visible = self._buffer
            self._buffer = ""
            return visible
        # Unclosed leading reasoning, oversized control text, and cancellation
        # are intentionally not exposed.
        return ""
