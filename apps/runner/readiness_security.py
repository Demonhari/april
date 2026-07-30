from __future__ import annotations


def _token_status(value: str | None, defaults: set[str], placeholders: set[str]) -> str:
    if not value:
        return "missing"
    if value in placeholders:
        return "placeholder-insecure"
    if value in defaults:
        return "default-development"
    return "configured"
