"""Freshness / staleness verdicts for APRIL's redacted verification reports.

A report is *fresh* when it is recent enough for its kind **and** was generated
under the current configuration. Staleness is computed from two signals, neither
of which requires Git:

1. **Age** — a per-type time-to-live (e.g. real-model/go-live/workflow reports go
   stale after 7 days; live-voice/wake reports after 30 days).
2. **Config fingerprint** — if the report embedded a redacted
   :mod:`april_common.config_fingerprint` digest and it differs from the current
   one, the report is stale even if it is recent ("config changed after it was
   generated"). Reports with no embedded fingerprint fall back to age-only.

Everything here is redaction-safe: it consumes already-redacted report payloads /
summaries and emits only timestamps, integer ages, booleans, and short reasons.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from april_common.time import parse_utc_iso, utc_now

# Per-type staleness time-to-live in days. Types not listed default to ``None``
# (age never makes them stale; only a fingerprint mismatch can).
STALE_TTL_DAYS: dict[str, int] = {
    "multi_model": 7,
    "target_mac": 7,
    "go_live": 7,
    "workflow": 7,
    "acceptance": 7,
    "voice_live": 30,
    "wake_word_live": 30,
}


class ReportFreshness(BaseModel):
    """Redacted freshness verdict for a single report."""

    basename: str | None = None
    report_type: str | None = None
    generated_at: str | None = None
    age_seconds: int | None = None
    age_human: str | None = None
    ttl_days: int | None = None
    stale: bool = False
    stale_reason: str | None = None
    config_fingerprint_matches: bool | None = None


def _age_human(age_seconds: int) -> str:
    if age_seconds < 60:
        return f"{age_seconds}s"
    minutes = age_seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


def compute_freshness(
    *,
    report_type: str | None,
    generated_at: str | None,
    config_fingerprint: str | None,
    current_fingerprint: str | None,
    basename: str | None = None,
    now: datetime | None = None,
) -> ReportFreshness:
    """Compute a freshness verdict from already-redacted report fields.

    ``config_fingerprint`` is the digest embedded in the report (or ``None`` for
    older reports); ``current_fingerprint`` is the live config digest (or ``None``
    when settings cannot be loaded). A mismatch marks the report stale; an absent
    embedded fingerprint falls back to age-only freshness.
    """
    reference = now or utc_now()
    age_seconds: int | None = None
    if generated_at:
        try:
            parsed = parse_utc_iso(generated_at)
        except ValueError:
            parsed = None
        if parsed is not None:
            age_seconds = max(0, int((reference - parsed).total_seconds()))

    ttl_days = STALE_TTL_DAYS.get(report_type or "")
    stale = False
    reason: str | None = None

    fingerprint_matches: bool | None = None
    if config_fingerprint is not None and current_fingerprint is not None:
        fingerprint_matches = config_fingerprint == current_fingerprint

    # Age-based staleness first (only when a TTL is defined and age is known).
    if ttl_days is not None and age_seconds is not None and age_seconds > ttl_days * 86_400:
        stale = True
        reason = f"older than {ttl_days} days"

    # Fingerprint mismatch always wins as the stronger, clearer signal.
    if fingerprint_matches is False:
        stale = True
        reason = "config changed after it was generated"

    return ReportFreshness(
        basename=basename,
        report_type=report_type,
        generated_at=generated_at,
        age_seconds=age_seconds,
        age_human=_age_human(age_seconds) if age_seconds is not None else None,
        ttl_days=ttl_days,
        stale=stale,
        stale_reason=reason,
        config_fingerprint_matches=fingerprint_matches,
    )


def freshness_from_payload(
    payload: dict[str, Any],
    *,
    report_type: str,
    current_fingerprint: str | None,
    basename: str | None = None,
    now: datetime | None = None,
) -> ReportFreshness:
    """Freshness verdict for a parsed report payload (uses its embedded fields)."""
    generated_at = payload.get("generated_at") or payload.get("timestamp")
    embedded = payload.get("config_fingerprint")
    return compute_freshness(
        report_type=report_type,
        generated_at=str(generated_at) if generated_at else None,
        config_fingerprint=str(embedded) if isinstance(embedded, str) and embedded else None,
        current_fingerprint=current_fingerprint,
        basename=basename,
        now=now,
    )
