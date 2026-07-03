from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from services.voice.microphone import write_pcm_wav
from services.voice.speech_to_text import SpeechToText

_GREETINGS = ("hey", "hi", "hello", "ok", "okay", "yo")


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


def strip_vocative(text: str, *, wake_word: str = "april") -> str:
    """Remove a vocative wake-word address without touching semantic uses.

    "april, restart the runtime" -> "restart the runtime"
    "can you check my repo april" -> "can you check my repo"
    "hey april what's on today" -> "what's on today"
    "add april to the meeting notes" -> unchanged (mid-sentence, semantic)
    """
    normalized = " ".join(text.split())
    if not normalized:
        return normalized
    stripped = _leading_pattern(wake_word).sub("", normalized, count=1)
    stripped = _trailing_pattern(wake_word).sub("", stripped, count=1)
    # An empty result means the utterance was only the address itself; callers
    # treat that as "awake, awaiting a command".
    return stripped.strip()


def mentions_wake_word(text: str, *, wake_word: str = "april") -> bool:
    return bool(re.search(rf"\b{re.escape(wake_word)}\b", text, flags=re.IGNORECASE))


def is_addressed(text: str, *, wake_word: str = "april", strict: bool = False) -> bool:
    """Whether the transcript addresses APRIL.

    Non-strict mode accepts any mention of the wake word. Strict mode requires a
    vocative (leading or trailing) address, rejecting semantic mid-sentence uses.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if not strict:
        return mentions_wake_word(normalized, wake_word=wake_word)
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
