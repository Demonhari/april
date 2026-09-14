"""Compact structured tool evidence for context windows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactToolEvidence:
    action: str
    argument_digest: str
    repository_state_digest: str | None
    exit_status: int | None
    result_status: str
    important_result: str
    output_digest: str
    truncated: bool
    retained_reference: str | None = None


def compact_tool_evidence(
    *,
    action: str,
    argument_digest: str,
    repository_state_digest: str | None,
    exit_status: int | None,
    result_status: str,
    output: str,
    max_important_chars: int = 800,
    retained_reference: str | None = None,
) -> CompactToolEvidence:
    """Keep status/error signal and a digest, never an arbitrary first prefix."""
    bounded = max(80, max_important_chars)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    important = [
        line
        for line in lines
        if any(marker in line.casefold() for marker in ("error", "fail", "traceback", "warning"))
    ]
    if not important and lines:
        important = lines[-2:]
    text = "\n".join(important)
    truncated = len(text) > bounded or len(output) > bounded
    if len(text) > bounded:
        text = text[-bounded:].lstrip()
    return CompactToolEvidence(
        action=action,
        argument_digest=argument_digest,
        repository_state_digest=repository_state_digest,
        exit_status=exit_status,
        result_status=result_status,
        important_result=text,
        output_digest=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        truncated=truncated,
        retained_reference=retained_reference,
    )
