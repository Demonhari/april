from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Deterministic Levenshtein distance for short local strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance scaled to [0, 1] by the longer string's length."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return edit_distance(a, b) / longest
