from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from apps.runner.mac_report import environment_snapshot, redact_reason
from apps.runner.verification.types import (
    MissingChatResultError,
    VerifyCheck,
    WorkflowReportCheck,
    WorkflowVerificationReport,
)
from services.memory.database import connect_sqlite


def latest_brain_decision_marker(database: Path) -> int:
    if not database.exists():
        return 0
    try:
        with connect_sqlite(database) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(rowid), 0)
                FROM conversation_events
                WHERE event_type = 'brain_decision'
                """
            ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row is not None and row[0] is not None else 0


def brain_decision_after_marker(database: Path, marker: int) -> dict[str, Any]:
    if not database.exists():
        return {}
    try:
        with connect_sqlite(database) as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM conversation_events
                WHERE event_type = 'brain_decision' AND rowid > ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (marker,),
            ).fetchone()
    except sqlite3.Error:
        return {}
    if row is None:
        return {}
    try:
        payload = json.loads(str(row[0]))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_object_candidates(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates: list[dict[str, Any]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                raw = re.sub(r",(\s*[}\]])", r"\1", stripped[start : index + 1])
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                start = None
            elif depth < 0:
                depth = 0
                start = None
    return candidates


def chat_result_from_response(
    response: Any, *, context: str, snippet_chars: int = 240
) -> dict[str, Any]:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raw = " ".join(str(getattr(response, "text", "")).split())
        snippet = raw[: max(0, snippet_chars)] or "<empty body>"
        raise MissingChatResultError(f"{context} response missing result; body={snippet}")
    response.raise_for_status()
    return result


def build_workflow_report(
    checks: list[VerifyCheck],
    *,
    real_model_requested: bool,
    timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
    config_fingerprint: str | None = None,
) -> WorkflowVerificationReport:
    failed = [check for check in checks if not check.ok]
    real_model_exercised = real_model_requested and any(
        check.name == "real workflow planning route" and check.ok for check in checks
    )
    rendered = [
        WorkflowReportCheck(
            name=check.name,
            ok=check.ok,
            status=check.status or ("pass" if check.ok else "fail"),
            detail=safe_workflow_report_detail(check.detail),
        )
        for check in checks
    ]
    return WorkflowVerificationReport(
        generated_at=environment_snapshot().generated_at,
        config_fingerprint=config_fingerprint,
        summary="pass" if not failed else "fail",
        real_model_verified=real_model_exercised and not failed,
        real_model_exercised=real_model_exercised,
        checks=rendered,
        checks_failed=len(failed),
        check_failures=[check.name for check in failed],
        timeout_seconds=timeout_seconds if real_model_requested else None,
        max_output_tokens=max_output_tokens if real_model_requested else None,
    )


def write_workflow_report(report: WorkflowVerificationReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        report.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )
    return resolved


def safe_workflow_report_detail(detail: str) -> str:
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
    return redact_reason(detail)[:240]
