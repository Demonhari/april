from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from april_common.config_fingerprint import config_fingerprint_digest
from april_common.report_freshness import freshness_from_payload
from april_common.settings import (
    AprilSettings,
)

_PATH_TEXT_RE = re.compile(r"~?(?:/[\w.\-]+){2,}/?")
_VERIFICATION_REPORT_TYPES = {
    "multi_model",
    "target_mac",
    "voice_live",
    "voice_conversation_live",
    "workflow",
    "soak",
}
_VERIFICATION_REPORT_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$")
_REAL_MODEL_REPORT_TYPES = {"multi_model", "target_mac"}
_BROWSER_REPORT_TYPES = {
    "acceptance",
    "go_live",
    "mac_activation",
    "voice_live",
    "wake_word_live",
    "voice_conversation_live",
    "multi_model",
    "workflow",
    "fake_soak",
}
_BROWSER_TYPE_ALIASES = {
    "acceptance": "acceptance",
    "go_live": "go_live",
    "mac_activation": "mac_activation",
    "voice_live": "voice_live",
    "wake_word_live": "wake_word_live",
    "voice_conversation_live": "voice_conversation_live",
    "multi_model": "multi_model",
    "workflow": "workflow",
    "soak": "fake_soak",
    "fake_soak": "fake_soak",
}


def _verification_root(settings: AprilSettings) -> Path:
    return (settings.home / "data" / "verification").resolve()


def _verification_report_files(settings: AprilSettings) -> list[Path]:
    root = _verification_root(settings)
    if not root.exists() or not root.is_dir():
        return []
    candidates: list[Path] = []
    for path in root.glob("*.json"):
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if (
            path.is_file()
            and not path.is_symlink()
            and _VERIFICATION_REPORT_BASENAME_RE.match(path.name)
            and _is_relative_to(resolved, root)
        ):
            candidates.append(path)
    return candidates


def _read_safe_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _classified_report_type(payload: dict[str, Any]) -> str:
    report_type = str(payload.get("report_type") or _infer_report_type(payload))
    return report_type if report_type in _VERIFICATION_REPORT_TYPES else "unknown"


def _report_matches_filter(report_type: str, filter_type: str) -> bool:
    if filter_type == "any":
        return True
    if filter_type == "real_model":
        return report_type in _REAL_MODEL_REPORT_TYPES
    return report_type == filter_type


def _latest_verification_report(
    settings: AprilSettings, *, report_type: str = "any"
) -> dict[str, Any]:
    # The latest report is selected *within the requested class* by the safe report
    # timestamp first, falling back to mtime only when the report timestamp is
    # absent/invalid. A newer voice-live report can never overwrite the latest
    # real-model report (or vice versa).
    filter_type = (
        report_type
        if report_type in {"any", "real_model", "voice_live", "voice_conversation_live", "workflow"}
        else "any"
    )
    candidates = _verification_report_files(settings)
    matching: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        payload = _read_safe_report(path)
        if payload is None:
            continue
        if _report_matches_filter(_classified_report_type(payload), filter_type):
            matching.append((path, payload))
    if not matching:
        if filter_type == "any" and candidates:
            # Files exist but none could be read as JSON objects.
            return {
                "status": "unreadable",
                "message": "latest verification report could not be read",
                "report": None,
            }
        return {
            "status": "not_verified",
            "message": "not verified yet",
            "report": None,
        }
    latest_path, latest_payload = max(matching, key=lambda item: _report_order_key(*item))
    return {
        "status": "ok",
        "message": "latest verification report",
        "report": _safe_report_payload(latest_payload, latest_path),
    }


def _reports_freshness(settings: AprilSettings) -> dict[str, Any]:
    """Per-type freshness for the latest report of each kind (redacted).

    Returns only basenames, statuses, ages, and stale booleans/reasons — never a
    token, prompt, transcript, patch, or absolute path. Staleness combines a
    per-type age TTL with the redacted config fingerprint embedded in each report.
    """
    current_fingerprint = config_fingerprint_digest(settings.home)
    latest: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is None:
            continue
        report_type = _browser_report_type(payload)
        if report_type == "unknown":
            continue
        key = _report_order_key(path, payload)
        if report_type not in latest or key > latest[report_type][0]:
            latest[report_type] = (key, path, payload)
    out: dict[str, Any] = {}
    for report_type, (_key, path, payload) in latest.items():
        fresh = freshness_from_payload(
            payload,
            report_type=report_type,
            current_fingerprint=current_fingerprint,
            basename=path.name,
        )
        status = payload.get("final_status") or payload.get("summary")
        out[report_type] = {
            "basename": path.name,
            "report_type": report_type,
            "status": str(status) if status is not None else None,
            "generated_at": fresh.generated_at,
            "age_seconds": fresh.age_seconds,
            "age_human": fresh.age_human,
            "stale": fresh.stale,
            "stale_reason": fresh.stale_reason,
            "config_fingerprint_matches": fresh.config_fingerprint_matches,
        }
    return out


def _latest_live_voice_flags(settings: AprilSettings) -> dict[str, bool]:
    """Read the latest live voice / wake-word verification flags from disk.

    Returns only two booleans (never a transcript, device, or path). Used to lift
    the offline voice milestone to its ``live_verified`` / ``wake_live_verified``
    rungs. Reading a report never opens the microphone.
    """
    voice_verified = False
    wake_verified = False
    conversation_verified = False
    voice_best: float | None = None
    wake_best: float | None = None
    conversation_best: float | None = None
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is None:
            continue
        declared = str(payload.get("report_type") or "")
        if declared == "voice_live":
            key = _report_order_key(path, payload)
            if voice_best is None or key > voice_best:
                voice_best = key
                voice_verified = bool(payload.get("voice_live_verified", False))
        elif declared == "wake_word_live":
            key = _report_order_key(path, payload)
            if wake_best is None or key > wake_best:
                wake_best = key
                wake_verified = bool(payload.get("wake_word_live_verified", False))
        elif declared == "voice_conversation_live":
            key = _report_order_key(path, payload)
            if conversation_best is None or key > conversation_best:
                conversation_best = key
                conversation_verified = bool(
                    payload.get("evidence_mode") == "real_hardware"
                    and payload.get("voice_conversation_live_verified", False)
                )
    return {
        "voice_live_verified": voice_verified,
        "wake_word_live_verified": wake_verified,
        "voice_conversation_live_verified": conversation_verified,
    }


def _verification_report_history(settings: AprilSettings) -> dict[str, Any]:
    matching: list[tuple[Path, dict[str, Any]]] = []
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is None:
            continue
        matching.append((path, payload))
    matching.sort(key=lambda item: _report_order_key(*item), reverse=True)
    reports: list[dict[str, Any]] = []
    for path, payload in matching:
        reports.append(_safe_report_payload(payload, path))
    if not reports:
        return {
            "status": "not_verified",
            "message": "not verified yet",
            "reports": [],
            "count": 0,
        }
    return {
        "status": "ok",
        "message": "verification report history",
        "reports": reports,
        "count": len(reports),
    }


def _verification_report_detail(settings: AprilSettings, report_basename: str) -> dict[str, Any]:
    path = _safe_report_path(settings, report_basename)
    payload = _read_safe_report(path)
    if payload is None:
        raise HTTPException(status_code=404, detail="verification report not found")
    return {
        "status": "ok",
        "message": "verification report",
        "report": _safe_report_payload(payload, path),
    }


def _browser_report_type(payload: dict[str, Any]) -> str:
    declared = str(payload.get("report_type") or "")
    if declared in _BROWSER_TYPE_ALIASES:
        return _BROWSER_TYPE_ALIASES[declared]
    if "verification_level" in payload and "models" in payload:
        return "multi_model"
    if "iterations" in payload and "latency_ms" in payload:
        return "fake_soak"
    return "unknown"


def _browser_report_summary(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    report_type = _browser_report_type(payload)
    status = (
        payload.get("final_status")
        if report_type in {"acceptance", "go_live", "mac_activation"}
        else payload.get("summary")
    )
    services = payload.get("services")
    services_summary: dict[str, Any] | None = None
    if isinstance(services, dict) and services.get("requested"):
        services_summary = {
            "mode": str(services.get("mode", "none")),
            "startup_status": str(services.get("startup_status", "unknown")),
            "shutdown_status": str(services.get("shutdown_status", "unknown")),
            "api_reachable": bool(services.get("api_reachable", False)),
            "runtime_reachable": bool(services.get("runtime_reachable", False)),
        }
    level = payload.get("acceptance_level")
    backend = payload.get("runtime_backend")
    summary: dict[str, Any] = {
        "basename": path.name,
        "report_type": report_type,
        "generated_at": str(payload.get("generated_at") or payload.get("timestamp") or ""),
        "status": str(status) if status is not None else None,
        "acceptance_level": str(level) if level else None,
        "runtime_backend": str(backend) if backend else None,
        "services": services_summary,
        "next_actions": _safe_string_list(payload.get("next_actions")),
    }
    if report_type == "go_live":
        # Surface the core-vs-hardened distinction so the browser can show a
        # working real-model core separately from the hardened go-live rung.
        # Every value here is a boolean, a small enum, or a redacted advisory.
        summary["core_real_model_ready"] = bool(payload.get("core_real_model_ready", False))
        summary["real_model_core_status"] = str(payload.get("real_model_core_status") or "not_run")
        summary["hardened_go_live_ready"] = bool(payload.get("hardened_go_live_ready", False))
        summary["hardening_warnings"] = _safe_string_list(payload.get("hardening_warnings"))
        summary["hardening_blockers"] = _safe_string_list(payload.get("hardening_blockers"))
    return summary


def _sorted_browser_items(settings: AprilSettings) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in _verification_report_files(settings):
        payload = _read_safe_report(path)
        if payload is not None:
            items.append((path, payload))
    items.sort(key=lambda item: _report_order_key(*item), reverse=True)
    return items


def _browser_reports(settings: AprilSettings) -> dict[str, Any]:
    reports = [
        _browser_report_summary(payload, path) for path, payload in _sorted_browser_items(settings)
    ]
    return {
        "status": "ok" if reports else "empty",
        "count": len(reports),
        "reports": reports,
    }


def _browser_latest(settings: AprilSettings, *, report_type: str | None = None) -> dict[str, Any]:
    for path, payload in _sorted_browser_items(settings):
        summary = _browser_report_summary(payload, path)
        if report_type is None:
            if summary["report_type"] in _BROWSER_REPORT_TYPES:
                return {"status": "ok", "report": summary}
        elif summary["report_type"] == report_type:
            return {"status": "ok", "report": summary}
    return {"status": "not_found", "report": None}


def _safe_report_path(settings: AprilSettings, report_basename: str) -> Path:
    if (
        report_basename != Path(report_basename).name
        or "/" in report_basename
        or "\\" in report_basename
        or Path(report_basename).is_absolute()
        or not _VERIFICATION_REPORT_BASENAME_RE.match(report_basename)
    ):
        raise HTTPException(status_code=400, detail="unsafe report basename")
    root = (settings.home / "data" / "verification").resolve()
    path = root / report_basename
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="verification report not found") from exc
    if path.is_symlink() or not path.is_file() or not _is_relative_to(resolved, root):
        raise HTTPException(status_code=400, detail="unsafe report path")
    return path


def _safe_report_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    report_type = str(payload.get("report_type") or _infer_report_type(payload))
    if report_type not in _VERIFICATION_REPORT_TYPES:
        report_type = "unknown"
    summary = str(payload.get("summary", "degraded"))
    safe: dict[str, Any] = {
        "file_basename": path.name,
        "basename": path.name,
        "generated_at": str(payload.get("generated_at") or payload.get("timestamp") or ""),
        "report_type": report_type,
        "summary": summary,
        "real_model_verified": _report_real_model_verified(payload, report_type),
        "verification_level": _safe_verification_level(payload),
        "real_models_exercised": _safe_int(payload.get("real_models_exercised")),
        "real_models_passed": _safe_int(payload.get("real_models_passed")),
        "any_real_model_exercised": bool(payload.get("any_real_model_exercised", False)),
        "any_real_model_passed": bool(payload.get("any_real_model_passed", False)),
        "core_model_set_verified": bool(payload.get("core_model_set_verified", False)),
        "all_available_models_verified": bool(payload.get("all_available_models_verified", False)),
        "all_configured_models_verified": bool(
            payload.get("all_configured_models_verified", False)
        ),
        "skipped": _safe_skipped(payload.get("skipped")),
        "threshold_failures": _safe_string_list(payload.get("threshold_failures")),
    }
    safe["skipped_count"] = len(safe["skipped"])
    safe["threshold_failure_count"] = len(safe["threshold_failures"])
    if isinstance(payload.get("models"), list):
        safe["models"] = [
            {
                "model_id": str(model.get("model_id", model.get("id", "unknown"))),
                "role": str(model.get("role", "unknown")),
                "backend": str(model.get("backend", "unknown")),
                "path_basename": _basename(model.get("path_basename") or model.get("path")),
                "available": bool(model.get("available", False)),
                "skipped_reason": _redact_path_text(str(model.get("skipped_reason")))
                if model.get("skipped_reason")
                else None,
            }
            for model in payload["models"]
            if isinstance(model, dict)
        ]
    if isinstance(payload.get("real_model"), dict):
        real_model = payload["real_model"]
        safe["models"] = [
            {
                "model_id": str(real_model.get("model_id", "unknown")),
                "role": str(real_model.get("role", "unknown")),
                "backend": str(payload.get("runtime_backend", "unknown")),
                "path_basename": _basename(real_model.get("path_basename")),
                "available": bool(real_model.get("attempted", False)),
                "skipped_reason": None,
            }
        ]
    if report_type == "voice_live":
        # Voice-live reports expose only safe booleans/counts: a live-verified flag
        # and per-stage successes. Never a transcript, an audio file path, or a
        # device name — VoiceLiveReport does not store those, and this allowlist
        # projection keeps it that way even if new raw fields are added later.
        safe["voice_live_verified"] = bool(payload.get("voice_live_verified", False))
        safe["recording_success"] = bool(payload.get("recording_success", False))
        safe["stt_success"] = bool(payload.get("stt_success", False))
        safe["tts_success"] = bool(payload.get("tts_success", False))
        safe["playback_user_confirmed"] = bool(payload.get("playback_user_confirmed", False))
    if report_type == "voice_conversation_live":
        safe["voice_conversation_live_verified"] = bool(
            payload.get("voice_conversation_live_verified", False)
        )
        safe["evidence_mode"] = str(payload.get("evidence_mode", "unknown"))
        safe["turn_count"] = _safe_int(payload.get("turn_count"))
        safe["same_conversation"] = bool(payload.get("same_conversation", False))
        safe["barge_in_detected"] = bool(payload.get("barge_in_detected", False))
        safe["two_turns_completed"] = bool(payload.get("two_turns_completed", False))
        safe["follow_up_opened"] = bool(payload.get("follow_up_opened", False))
    if report_type == "workflow":
        safe["real_model_exercised"] = bool(payload.get("real_model_exercised", False))
        safe["checks"] = _safe_workflow_checks(payload.get("checks"))
    if "checks_failed" in payload:
        safe["checks_failed"] = payload.get("checks_failed")
    if "check_failures" in payload:
        safe["check_failures"] = _safe_string_list(payload.get("check_failures"))
    if "failures" in payload:
        safe["failures"] = _safe_string_list(payload.get("failures"))
    return safe


def _safe_verification_level(payload: dict[str, Any]) -> str:
    value = str(payload.get("verification_level", "none"))
    return value if value in {"none", "partial", "core", "all"} else "none"


def _safe_int(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


def _infer_report_type(payload: dict[str, Any]) -> str:
    if "real_model" in payload:
        return "target_mac"
    if "recording_success" in payload or "playback_user_confirmed" in payload:
        return "voice_live"
    if "checks" in payload and str(payload.get("report_type")) == "workflow":
        return "workflow"
    if "iterations" in payload and "latency_ms" in payload:
        return "soak"
    return "unknown"


def _report_real_model_verified(payload: dict[str, Any], report_type: str) -> bool:
    if report_type == "voice_live":
        return False
    if report_type == "workflow":
        return bool(payload.get("real_model_verified", False))
    if report_type in _REAL_MODEL_REPORT_TYPES and isinstance(
        payload.get("real_model_verified"), bool
    ):
        return bool(payload["real_model_verified"])
    if report_type == "target_mac" and isinstance(payload.get("real_model"), dict):
        real_model = payload["real_model"]
        return (
            str(payload.get("runtime_backend")) != "fake"
            and bool(real_model.get("attempted"))
            and bool(real_model.get("load_success"))
            and bool(real_model.get("chat_success"))
            and bool(real_model.get("streaming_success"))
            and bool(real_model.get("unload_success"))
        )
    return False


def _report_order_key(path: Path, payload: dict[str, Any]) -> float:
    parsed = _safe_report_timestamp(payload)
    if parsed is not None:
        return parsed.timestamp()
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _safe_report_timestamp(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("generated_at") or payload.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_workflow_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    checks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "name": str(item.get("name", "unknown")),
                "status": str(item.get("status", "unknown")),
                "ok": bool(item.get("ok", False)),
                "detail": _safe_workflow_detail(str(item.get("detail", ""))),
            }
        )
    return checks


def _safe_workflow_detail(detail: str) -> str:
    lower = detail.lower()
    if "decision_summary" in lower:
        return "decision_summary redacted"
    sensitive_markers = (
        "prompt",
        "transcript",
        "token",
        "authorization",
        "bearer",
        "raw_tool_args",
        "tool args",
    )
    if any(marker in lower for marker in sensitive_markers):
        return "sensitive detail redacted"
    return _redact_path_text(detail)[:240]


def _safe_skipped(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    skipped: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        skipped.append(
            {
                "name": str(item.get("name", "unknown")),
                "reason": _redact_path_text(str(item.get("reason", ""))),
            }
        )
    return skipped


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_redact_path_text(str(item)) for item in value]


def _redact_path_text(text: str) -> str:
    def _basename(match: re.Match[str]) -> str:
        name = Path(match.group(0)).name
        return name or match.group(0)

    return _PATH_TEXT_RE.sub(_basename, text)


def _basename(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text or text == "[REDACTED]":
        return None
    return Path(text).name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
