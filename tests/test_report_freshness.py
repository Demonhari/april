from __future__ import annotations

from datetime import timedelta

from april_common.report_freshness import (
    compute_freshness,
    freshness_from_payload,
)
from april_common.time import utc_now


def _iso(delta_days: float) -> str:
    return (utc_now() - timedelta(days=delta_days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_recent_matching_report_is_fresh() -> None:
    fresh = compute_freshness(
        report_type="go_live",
        generated_at=_iso(1),
        config_fingerprint="abc123",
        current_fingerprint="abc123",
    )
    assert fresh.stale is False
    assert fresh.stale_reason is None
    assert fresh.config_fingerprint_matches is True
    assert fresh.age_human is not None


def test_old_report_is_stale_by_ttl() -> None:
    fresh = compute_freshness(
        report_type="go_live",
        generated_at=_iso(10),
        config_fingerprint="abc123",
        current_fingerprint="abc123",
    )
    assert fresh.stale is True
    assert fresh.stale_reason == "older than 7 days"


def test_voice_live_has_longer_ttl() -> None:
    # 10 days old is fine for voice-live (30-day TTL) but stale for go-live.
    voice = compute_freshness(
        report_type="voice_live",
        generated_at=_iso(10),
        config_fingerprint=None,
        current_fingerprint=None,
    )
    assert voice.stale is False
    assert voice.ttl_days == 30


def test_fingerprint_mismatch_makes_recent_report_stale() -> None:
    fresh = compute_freshness(
        report_type="multi_model",
        generated_at=_iso(0.1),
        config_fingerprint="old-digest",
        current_fingerprint="new-digest",
    )
    assert fresh.stale is True
    assert fresh.stale_reason == "config changed after it was generated"
    assert fresh.config_fingerprint_matches is False


def test_missing_embedded_fingerprint_falls_back_to_age_only() -> None:
    # No embedded fingerprint: a recent report is still fresh (age-only).
    fresh = compute_freshness(
        report_type="workflow",
        generated_at=_iso(1),
        config_fingerprint=None,
        current_fingerprint="new-digest",
    )
    assert fresh.config_fingerprint_matches is None
    assert fresh.stale is False


def test_freshness_from_payload_reads_embedded_fields() -> None:
    payload = {
        "report_type": "go_live",
        "generated_at": _iso(0.5),
        "config_fingerprint": "deadbeef",
    }
    fresh = freshness_from_payload(
        payload,
        report_type="go_live",
        current_fingerprint="deadbeef",
        basename="go-live-x.json",
    )
    assert fresh.basename == "go-live-x.json"
    assert fresh.stale is False
    assert fresh.config_fingerprint_matches is True
