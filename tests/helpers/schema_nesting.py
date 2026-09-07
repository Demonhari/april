from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REPETITION_KEYS = frozenset({"maxLength", "minLength", "maxItems", "minItems"})


def max_repetition_nesting(schema: Mapping[str, Any] | Sequence[Any]) -> int:
    """Estimate the largest repetition bound in a JSON schema tree."""

    if isinstance(schema, Mapping):
        values = [
            value
            for key, value in schema.items()
            if key in REPETITION_KEYS and isinstance(value, int) and not isinstance(value, bool)
        ]
        children = [
            max_repetition_nesting(value)
            for value in schema.values()
            if isinstance(value, (Mapping, list, tuple))
        ]
        return max([*values, *children], default=0)
    return max(
        (
            max_repetition_nesting(value)
            for value in schema
            if isinstance(value, (Mapping, list, tuple))
        ),
        default=0,
    )
