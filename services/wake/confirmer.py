from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from april_common.text import edit_distance, normalized_edit_distance
from services.voice.microphone import write_pcm_wav
from services.voice.speech_to_text import SpeechToText

__all__ = [
    "Confirmation",
    "SttConfirmer",
    "canonicalize_wake_word",
    "edit_distance",
    "is_addressed",
    "mentions_wake_word",
    "normalized_edit_distance",
    "strip_vocative",
]

_GREETINGS = ("hey", "hi", "hello", "ok", "okay", "yo")

# Fuzzy matching accepts close STT mishearings of the wake word (apryl, avril,
# aprill, "a pril"). The threshold allows one edit on a five-letter word, and the
# first letter must match so unrelated words can never drift into a wake.
_FUZZY_MAX_NORMALIZED_DISTANCE = 0.25
_FUZZY_MIN_WAKE_WORD_LENGTH = 4


def _fuzzy_wake_match(token: str, wake_word: str) -> bool:
    if not token:
        return False
    if token == wake_word:
        return True
    if token[0] != wake_word[0] or len(wake_word) < _FUZZY_MIN_WAKE_WORD_LENGTH:
        return False
    return normalized_edit_distance(token, wake_word) <= _FUZZY_MAX_NORMALIZED_DISTANCE


def _split_token(token: str) -> tuple[str, str, str]:
    """Split a whitespace token into (leading punct, casefolded core, trailing punct)."""
    start, end = 0, len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    return token[:start], token[start:end].casefold(), token[end:]


def canonicalize_wake_word(text: str, *, wake_word: str = "april") -> str:
    """Rewrite fuzzy wake-word tokens to the canonical form.

    Single tokens within one edit ("apryl", "avril", "aprill") and adjacent
    split pairs ("a pril") become the canonical wake word, preserving
    surrounding punctuation, so the exact-match address patterns keep working.
    """
    tokens = text.split()
    result: list[str] = []
    index = 0
    while index < len(tokens):
        prefix, core, suffix = _split_token(tokens[index])
        if _fuzzy_wake_match(core, wake_word):
            if core == wake_word:
                # Exact (case-insensitive) matches keep their original casing;
                # the address patterns are case-insensitive anyway.
                result.append(tokens[index])
            else:
                result.append(f"{prefix}{wake_word}{suffix}")
            index += 1
            continue
        if index + 1 < len(tokens):
            _prefix2, core2, suffix2 = _split_token(tokens[index + 1])
            if (
                core
                and core2
                and not _fuzzy_wake_match(core2, wake_word)
                and _fuzzy_wake_match(core + core2, wake_word)
            ):
                result.append(f"{prefix}{wake_word}{suffix2}")
                index += 2
                continue
        result.append(tokens[index])
        index += 1
    return " ".join(result)


def _leading_pattern(wake_word: str) -> re.Pattern[str]:
    greetings = "|".join(_GREETINGS)
    return re.compile(
        rf"^\s*(?:(?:{greetings})[\s,]+)?{re.escape(wake_word)}\b[\s,.:;!?-]*",
        flags=re.IGNORECASE,
    )


def _trailing_pattern(wake_word: str) -> re.Pattern[str]:
    return re.compile(
        rf"[\s,]*\b{re.escape(wake_word)}\b[\s,.:;!?]*$",
        flags=re.IGNORECASE,
    )


def strip_vocative(text: str, *, wake_word: str = "april", fuzzy: bool = True) -> str:
    """Remove a vocative wake-word address without touching semantic uses.

    "april, restart the runtime" -> "restart the runtime"
    "apryl restart the runtime" -> "restart the runtime" (fuzzy STT variant)
    "can you check my repo april" -> "can you check my repo"
    "hey april what's on today" -> "what's on today"
    "add april to the meeting notes" -> unchanged (mid-sentence, semantic)
    """
    normalized = " ".join(text.split())
    if not normalized:
        return normalized
    if fuzzy:
        normalized = canonicalize_wake_word(normalized, wake_word=wake_word)
    stripped = _leading_pattern(wake_word).sub("", normalized, count=1)
    stripped = _trailing_pattern(wake_word).sub("", stripped, count=1)
    # An empty result means the utterance was only the address itself; callers
    # treat that as "awake, awaiting a command".
    return stripped.strip()


def mentions_wake_word(text: str, *, wake_word: str = "april", fuzzy: bool = True) -> bool:
    if fuzzy:
        text = canonicalize_wake_word(" ".join(text.split()), wake_word=wake_word)
    return bool(re.search(rf"\b{re.escape(wake_word)}\b", text, flags=re.IGNORECASE))


def is_addressed(
    text: str,
    *,
    wake_word: str = "april",
    strict: bool = False,
    fuzzy: bool = True,
) -> bool:
    """Whether the transcript addresses APRIL.

    Non-strict mode accepts any mention of the wake word. Strict mode requires a
    vocative (leading or trailing) address, rejecting semantic mid-sentence uses.
    Fuzzy matching (on by default) canonicalizes close STT mishearings first;
    strict mode still applies its positional rules to the canonical form.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if fuzzy:
        normalized = canonicalize_wake_word(normalized, wake_word=wake_word)
    if not strict:
        return mentions_wake_word(normalized, wake_word=wake_word, fuzzy=False)
    return bool(
        _leading_pattern(wake_word).match(normalized)
        or _trailing_pattern(wake_word).search(normalized)
    )


@dataclass(frozen=True, slots=True)
class Confirmation:
    accepted: bool
    transcript: str
    command: str
    reason: str


class SttConfirmer:
    """Stage-two wake confirmation from Sentinel-owned audio.

    The confirmer only ever reads frames handed to it (the Sentinel ring buffer
    snapshot); it never opens a microphone stream of its own. Frames are written
    to a capture WAV in the audio cache, transcribed with the local STT, and the
    transcript is checked for an address to APRIL.
    """

    def __init__(
        self,
        stt: SpeechToText,
        *,
        audio_cache_path: Path,
        wake_word: str = "april",
        strict_address: bool = False,
        sample_rate: int = 16_000,
        retain_debug_audio: bool = False,
    ) -> None:
        self.stt = stt
        self.audio_cache_path = audio_cache_path
        self.wake_word = wake_word
        self.strict_address = strict_address
        self.sample_rate = sample_rate
        self.retain_debug_audio = retain_debug_audio

    async def confirm(self, frames: Sequence[bytes]) -> Confirmation:
        if not frames:
            return Confirmation(False, "", "", "no audio captured")
        capture_path = self.audio_cache_path / f"wake-confirm-{uuid.uuid4()}.wav"
        write_pcm_wav(capture_path, list(frames), sample_rate=self.sample_rate)
        try:
            transcript = await self.stt.transcribe(capture_path)
        finally:
            if not self.retain_debug_audio:
                capture_path.unlink(missing_ok=True)
        transcript = " ".join(transcript.split())
        if not transcript:
            return Confirmation(False, "", "", "empty transcript")
        if not is_addressed(transcript, wake_word=self.wake_word, strict=self.strict_address):
            return Confirmation(False, transcript, "", "transcript does not address APRIL")
        command = strip_vocative(transcript, wake_word=self.wake_word)
        return Confirmation(True, transcript, command, "stt confirmed")
