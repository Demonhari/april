"""Read-only browser for APRIL's local verification reports.

``run april reports`` lists, shows, and cleans the redacted JSON reports under
``data/verification`` (acceptance, mac-activation, voice-live, wake-word-live,
multi-model, fake-soak). It is read-only by default: ``reports clean`` is dry-run
unless ``--apply`` and never deletes anything outside ``data/verification``.

The reports themselves are already redacted by construction, but this module
re-projects each one through a strict allowlist (type, timestamp, status, level,
backend, service status, next actions) so the browser can never surface a token,
transcript, generated text, raw audio path, or absolute filesystem path even if a
new raw field is added to a report later.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from apps.runner.mac_report import redact_reason
from april_common.time import parse_utc_iso, utc_now

KNOWN_REPORT_TYPES = (
    "acceptance",
    "go_live",
    "mac_activation",
    "voice_live",
    "wake_word_live",
    "multi_model",
    "fake_soak",
)
ALL_REPORT_TYPES = (*KNOWN_REPORT_TYPES, "unknown")

# report_type field value -> browser type. Reports without a report_type are
# classified heuristically below.
_TYPE_ALIASES = {
    "acceptance": "acceptance",
    "go_live": "go_live",
    "mac_activation": "mac_activation",
    "voice_live": "voice_live",
    "wake_word_live": "wake_word_live",
    "multi_model": "multi_model",
    "soak": "fake_soak",
    "fake_soak": "fake_soak",
}


class ReportSummary(BaseModel):
    basename: str
    report_type: str
    generated_at: str | None = None
    status: str | None = None
    acceptance_level: str | None = None
    runtime_backend: str | None = None
    services: str | None = None
    next_actions: list[str] = Field(default_factory=list)


class ReportListing(BaseModel):
    directory: str
    count: int
    reports: list[ReportSummary] = Field(default_factory=list)


class CleanCandidate(BaseModel):
    basename: str
    age_days: int


class CleanResult(BaseModel):
    directory: str
    older_than_days: int
    applied: bool
    candidates: list[CleanCandidate] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)


def classify_report(payload: dict[str, Any]) -> str:
    """Return the browser report type for a parsed report payload."""
    declared = str(payload.get("report_type") or "")
    if declared in _TYPE_ALIASES:
        return _TYPE_ALIASES[declared]
    # Heuristics for reports that predate an explicit report_type field.
    if "verification_level" in payload and "models" in payload:
        return "multi_model"
    if "iterations" in payload and "latency_ms" in payload:
        return "fake_soak"
    return "unknown"


def _status_for(report_type: str, payload: dict[str, Any]) -> str | None:
    if report_type in {"acceptance", "go_live", "mac_activation"}:
        value = payload.get("final_status")
    else:
        value = payload.get("summary")
    return str(value) if value is not None else None


def _services_for(payload: dict[str, Any]) -> str | None:
    services = payload.get("services")
    if not isinstance(services, dict) or not services.get("requested"):
        return None
    startup = str(services.get("startup_status", "unknown"))
    shutdown = str(services.get("shutdown_status", "unknown"))
    return f"startup={startup}, shutdown={shutdown}"


def _next_actions_for(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("next_actions")
    if not isinstance(raw, list):
        return []
    # Redact defensively: next actions are commands/notes, never tokens or paths,
    # but an embedded absolute path is still reduced to its basename.
    return [redact_reason(str(item)) for item in raw if isinstance(item, str)]


def summarize_report(payload: dict[str, Any], path: Path) -> ReportSummary:
    report_type = classify_report(payload)
    generated_at = payload.get("generated_at") or payload.get("timestamp")
    level = payload.get("acceptance_level")
    backend = payload.get("runtime_backend")
    return ReportSummary(
        basename=path.name,
        report_type=report_type,
        generated_at=str(generated_at) if generated_at else None,
        status=_status_for(report_type, payload),
        acceptance_level=str(level) if level else None,
        runtime_backend=str(backend) if backend else None,
        services=_services_for(payload),
        next_actions=_next_actions_for(payload),
    )


def _read_report(path: Path) -> dict[str, Any] | None:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _report_files(directory: Path) -> list[Path]:
    root = directory.expanduser()
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.glob("*.json"):
        # Only real, non-symlink JSON files directly inside the directory.
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def _order_key(path: Path, payload: dict[str, Any]) -> tuple[float, float]:
    generated_at = payload.get("generated_at") or payload.get("timestamp")
    parsed = 0.0
    if isinstance(generated_at, str) and generated_at:
        try:
            parsed = parse_utc_iso(generated_at).timestamp()
        except ValueError:
            parsed = 0.0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (parsed, mtime)


def load_reports(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return ``(path, payload)`` for every readable report, newest first."""
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in _report_files(directory):
        payload = _read_report(path)
        if payload is not None:
            items.append((path, payload))
    items.sort(key=lambda item: _order_key(item[0], item[1]), reverse=True)
    return items


def list_report_summaries(directory: Path) -> ReportListing:
    summaries = [summarize_report(payload, path) for path, payload in load_reports(directory)]
    return ReportListing(directory="data/verification", count=len(summaries), reports=summaries)


def latest_report(directory: Path) -> ReportSummary | None:
    """Newest report of any *known* type (unknown-typed files are ignored)."""
    for path, payload in load_reports(directory):
        summary = summarize_report(payload, path)
        if summary.report_type in KNOWN_REPORT_TYPES:
            return summary
    return None


def latest_report_of_type(directory: Path, report_type: str) -> ReportSummary | None:
    for path, payload in load_reports(directory):
        summary = summarize_report(payload, path)
        if summary.report_type == report_type:
            return summary
    return None


def summarize_path(path: Path) -> ReportSummary | None:
    """Summarize an explicit report path (may be outside data/verification)."""
    resolved = path.expanduser()
    payload = _read_report(resolved)
    if payload is None:
        return None
    return summarize_report(payload, resolved)


def clean_reports(directory: Path, *, older_than_days: int, apply: bool) -> CleanResult:
    """Find (and with ``apply`` delete) report JSON files older than a cutoff.

    Dry-run by default. Only ``*.json`` files *directly inside* ``directory`` are
    ever considered, and a path is deleted only after its resolved parent is
    confirmed to be that directory — nothing outside ``data/verification`` can be
    touched.
    """
    root = directory.expanduser()
    result = CleanResult(
        directory="data/verification", older_than_days=older_than_days, applied=apply
    )
    if older_than_days < 0:
        return result
    cutoff = utc_now().timestamp() - older_than_days * 86_400
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        return result
    for path in _report_files(root):
        try:
            stat = path.stat()
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if resolved.parent != resolved_root:
            # Refuse anything that resolves outside the verification directory.
            continue
        if stat.st_mtime >= cutoff:
            continue
        age_days = int((utc_now().timestamp() - stat.st_mtime) // 86_400)
        result.candidates.append(CleanCandidate(basename=path.name, age_days=age_days))
        if apply:
            try:
                path.unlink()
                result.deleted.append(path.name)
            except OSError:
                continue
    return result


def known_report_types() -> Iterable[str]:
    return KNOWN_REPORT_TYPES
