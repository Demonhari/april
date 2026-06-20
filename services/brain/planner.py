from __future__ import annotations


def concise_steps(summary: str) -> list[str]:
    return [part.strip() for part in summary.split(".") if part.strip()][:5]
