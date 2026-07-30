from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from services.evolution.rollout_models import (
    _IDENTIFIER_RE,
    _SAFE_OUTCOME_KEYS,
    _SHA256_RE,
    CanaryContext,
)


def _canary_eligible(context: CanaryContext) -> tuple[bool, str]:
    if context.source in {"voice", "wake", "background", "dreamer"} or context.live_voice:
        return False, "live_or_background_source_excluded"
    if context.mode != "standard" or context.high_risk_reasoning:
        return False, "high_risk_reasoning_excluded"
    if context.permission_level >= 3:
        return False, "approval_requiring_interaction_excluded"
    if context.risk_level not in {"none", "read_only"}:
        return False, "write_or_external_risk_excluded"
    if context.agent in {"coding_agent", "system_action_agent"}:
        return False, "write_capable_agent_excluded"
    if context.has_pending_approval:
        return False, "pending_approval_excluded"
    if (
        context.destructive
        or context.external_side_effect
        or context.security_sensitive
        or context.database_write
        or context.repository_write
        or context.background_evolution
    ):
        return False, "unsafe_interaction_excluded"
    read_only_tools = {
        "read_file",
        "search_files",
        "list_files",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "repo_indexer",
    }
    if any(tool not in read_only_tools for tool in context.tool_names):
        return False, "non_read_only_tool_excluded"
    return True, "eligible"


def _validate_safe_outcome(
    outcome: dict[str, bool | int | float],
) -> dict[str, bool | int | float]:
    if set(outcome) - _SAFE_OUTCOME_KEYS:
        raise ValueError("unsafe_rollout_outcome_key")
    safe: dict[str, bool | int | float] = {}
    for key, value in outcome.items():
        if key in {"latency_ms", "baseline_latency_ms"}:
            parsed = float(value)
            if parsed < 0 or parsed > 3_600_000:
                raise ValueError("rollout_latency_out_of_bounds")
            safe[key] = parsed
        elif isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int) and value in {0, 1}:
            safe[key] = bool(value)
        else:
            raise ValueError("rollout_outcome_must_be_boolean_or_bounded_latency")
    return safe


def _aggregate_outcome(
    aggregate: dict[str, Any],
    outcome: dict[str, bool | int | float],
) -> None:
    aggregate["sample_count"] = int(aggregate.get("sample_count", 0)) + 1
    aggregate["success_count"] = int(aggregate.get("success_count", 0)) + int(
        bool(outcome.get("success", True))
    )
    failure = bool(
        outcome.get("tool_failure")
        or outcome.get("coding_test_failed")
        or outcome.get("runtime_failure")
        or outcome.get("approval_denied")
        or outcome.get("user_correction")
        or outcome.get("negative_feedback")
        or outcome.get("regeneration")
        or not bool(outcome.get("success", True))
    )
    aggregate["failure_count"] = int(aggregate.get("failure_count", 0)) + int(failure)
    aggregate["structured_invalid_count"] = int(aggregate.get("structured_invalid_count", 0)) + int(
        outcome.get("structured_output_valid") is False
    )
    aggregate["repair_count"] = int(aggregate.get("repair_count", 0)) + int(
        bool(outcome.get("repair_attempted"))
    )
    aggregate["tool_success_count"] = int(aggregate.get("tool_success_count", 0)) + int(
        bool(outcome.get("tool_success"))
    )
    aggregate["tool_failure_count"] = int(aggregate.get("tool_failure_count", 0)) + int(
        bool(outcome.get("tool_failure"))
    )
    for key, field_name in (
        ("approval_denied", "approval_denial_count"),
        ("user_correction", "user_correction_count"),
        ("negative_feedback", "negative_feedback_count"),
        ("regeneration", "regeneration_count"),
        ("coding_test_passed", "coding_test_pass_count"),
        ("coding_test_failed", "coding_test_failure_count"),
        ("runtime_failure", "runtime_failure_count"),
        ("candidate_fallback", "fallback_count"),
        ("hard_failure", "hard_failure_count"),
    ):
        aggregate[field_name] = int(aggregate.get(field_name, 0)) + int(bool(outcome.get(key)))
    aggregate["latency_ms_total"] = float(aggregate.get("latency_ms_total", 0.0)) + float(
        outcome.get("latency_ms", 0.0)
    )
    aggregate["baseline_latency_ms_total"] = float(
        aggregate.get("baseline_latency_ms_total", 0.0)
    ) + float(outcome.get("baseline_latency_ms", 0.0))


def _outcome_event_summary(
    outcome: dict[str, bool | int | float],
) -> dict[str, Any]:
    return {
        "hard_failure": bool(outcome.get("hard_failure")),
        "runtime_failure": bool(outcome.get("runtime_failure")),
        "candidate_fallback": bool(outcome.get("candidate_fallback")),
        "success": bool(outcome.get("success", True)),
    }


def _encode_column_value(column: str, value: Any) -> Any:
    if column in {"metrics_json", "previous_active_artifact_json"} and not isinstance(value, str):
        return _canonical_json(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, field: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}_invalid")


def _validate_identifier(value: str, field: str) -> None:
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field}_invalid")


def _reason_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "_".join(str(value).casefold().split())
    safe = "".join(char for char in normalized if char.isalnum() or char == "_")
    return safe[:160] or "unknown"
