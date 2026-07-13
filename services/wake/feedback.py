from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from services.wake.confirmer import strip_vocative

FeedbackRating = Literal["good", "bad"]


@dataclass(frozen=True, slots=True)
class WakeFeedback:
    rating: FeedbackRating
    phrase: str


_FEEDBACK_PHRASES: dict[str, FeedbackRating] = {
    "that was wrong": "bad",
    "that was bad": "bad",
    "that was perfect": "good",
    "that was right": "good",
    "good job": "good",
}


def classify_wake_feedback(text: str, *, wake_word: str = "april") -> WakeFeedback | None:
    """Exact allowlist classifier for wake feedback verbs.

    This is intentionally narrower than wake-word confirmation: no fuzzy
    matching, no semantic paraphrase matching, and no model call.
    """
    stripped = strip_vocative(text, wake_word=wake_word, fuzzy=False)
    normalized = _normalize(stripped)
    rating = _FEEDBACK_PHRASES.get(normalized)
    if rating is None:
        return None
    return WakeFeedback(rating=rating, phrase=normalized)


def _normalize(text: str) -> str:
    normalized = text.strip().casefold()
    normalized = re.sub(r"[?.!]+$", "", normalized)
    return " ".join(normalized.split())
