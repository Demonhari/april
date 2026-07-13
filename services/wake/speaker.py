from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

SPEAKER_MATCH_THRESHOLD = 0.5


class SpeakerVerifier(Protocol):
    """Local convenience filter for an accepted wake, never authentication.

    Implementations compare operator-owned enrollment WAV files with bounded
    16-bit mono PCM from the wake ring buffer and return a score in ``[0, 1]``.
    The score may suppress an accidental wake, but it must never grant access,
    lower a permission level, or serve as an identity/security boundary.
    """

    def score(self, enrollment: Sequence[Path], utterance: bytes) -> float: ...
