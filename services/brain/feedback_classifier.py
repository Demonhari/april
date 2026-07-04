from __future__ import annotations

# Conservative, deterministic implicit-correction detection.
#
# Only unambiguous correction openers count. This deliberately misses many
# real corrections (recall is sacrificed for precision) so a normal message
# can never be misfiled as negative feedback. No model is involved: the
# classifier is a fixed prefix list over a normalized message.
_CORRECTION_PREFIXES: tuple[str, ...] = (
    "no, that's wrong",
    "no, that is wrong",
    "no, thats wrong",
    "that's wrong",
    "that is wrong",
    "thats wrong",
    "that's incorrect",
    "that is incorrect",
    "thats incorrect",
    "that's not right",
    "that is not right",
    "thats not right",
    "not what i asked",
    "that's not what i asked",
    "that is not what i asked",
    "thats not what i asked",
    "you misunderstood",
    "wrong answer",
)


def classify_implicit_correction(message: str) -> str | None:
    """Return the matched correction marker, or None when not clearly one.

    Matching is prefix-only on a whitespace/case-normalized message, so a
    sentence that merely *contains* one of the phrases later on never counts.
    """
    normalized = " ".join(message.strip().casefold().split())
    for prefix in _CORRECTION_PREFIXES:
        if normalized.startswith(prefix):
            return prefix
    return None
