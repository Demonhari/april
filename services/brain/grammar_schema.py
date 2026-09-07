from __future__ import annotations

_GRAMMAR_STRIPPED_KEYS = frozenset(
    {
        "maxLength",
        "minLength",
        "maxItems",
        "minItems",
        "pattern",
        "format",
        "title",
        "description",
        "default",
    }
)


def grammar_safe_json_schema(schema: dict) -> dict:
    """Return a grammar-compiler-safe copy of a JSON schema."""

    def strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip(child)
                for key, child in value.items()
                if key not in _GRAMMAR_STRIPPED_KEYS
            }
        if isinstance(value, list):
            return [strip(child) for child in value]
        return value

    result = strip(schema)
    if not isinstance(result, dict):
        raise TypeError("JSON schema must be an object")
    return result
