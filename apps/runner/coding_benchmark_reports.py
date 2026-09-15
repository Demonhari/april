"""Redacted single-candidate coding reports and offline comparison.

This module is deliberately inference-free.  It can build a report from a
completed trusted job and compare two saved reports without loading APRIL
Runtime, a model, Tool Worker, or any external service.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from apps.runner.coding_compare import CODING_BENCHMARK_REPORT_SCHEMA_VERSION
from april_common.hardware_profile import safe_hardware_profile
from april_common.time import utc_now_iso
from services.evaluation.coding_benchmark import redact_coding_result

CODING_EVALUATOR_VERSION = "coding-evaluator-v3"
CODING_SCORING_POLICY_VERSION = "coding-scoring-v3"


def build_coding_benchmark_report(
    result: Mapping[str, Any],
    *,
    model_id: str,
    role: str,
    backend: str,
    artifact_kind: str,
    configuration: Mapping[str, Any],
    suite: str,
    timeout_profile: Mapping[str, Any],
    fixture_set: Mapping[str, Any],
    simulated: bool,
    hardware_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coding = result.get("coding")
    if not isinstance(coding, Mapping):
        quality = result.get("quality")
        coding = quality.get("coding") if isinstance(quality, Mapping) else None
    redacted_coding = redact_coding_result(coding)
    coding = redacted_coding or {}
    coding_meta = fixture_set.get("coding")
    case_ids = coding_meta.get("case_ids", []) if isinstance(coding_meta, Mapping) else []
    if suite == "smoke":
        case_ids = list(
            result.get("case_ids")
            or (list(coding.get("case_results", {})) if isinstance(coding, Mapping) else [])
            or case_ids
        )
    profile = dict(timeout_profile)
    hardware = dict(hardware_profile or result.get("hardware_profile") or safe_hardware_profile())
    model_identity = {
        "artifact_kind": artifact_kind,
        "identity_kind": (
            "bounded_metadata_manifest"
            if artifact_kind == "colibri_model_directory"
            else "full_artifact_sha256"
        ),
        "manifest_digest": result.get("manifest_digest"),
        "artifact_sha256": result.get("model_sha256"),
    }
    report: dict[str, Any] = {
        "schema_version": CODING_BENCHMARK_REPORT_SCHEMA_VERSION,
        "report_type": "coding_model_benchmark",
        "generated_at": utc_now_iso(),
        "benchmark_status": result.get(
            "benchmark_status",
            "unavailable" if result.get("unavailable_reason") else "succeeded",
        ),
        "failure_reason": result.get("failure_reason") or result.get("unavailable_reason"),
        "evaluator_version": CODING_EVALUATOR_VERSION,
        "model_id": model_id,
        "role": role,
        "backend": backend,
        "artifact_kind": artifact_kind,
        "model_identity": model_identity,
        "model_runtime_configuration": dict(configuration),
        "fixture_set": dict(fixture_set),
        "fixture_set_version": fixture_set.get("version"),
        "fixture_set_sha256": (
            coding_meta.get("sha256") if isinstance(coding_meta, Mapping) else None
        ),
        "case_ids": list(case_ids),
        "scoring_policy_version": CODING_SCORING_POLICY_VERSION,
        "timeout_profile": profile,
        "benchmark_configuration": {
            "suite": suite,
            "case_timeout_multiplier": profile.get("case_timeout_multiplier", 1.0),
        },
        "hardware_profile": hardware,
        "simulated": bool(simulated),
        "coding": coding,
        "performance": {
            "runs": list(result.get("runs") or []),
            "seed_control": result.get("seed_control"),
        },
        "evaluator_verification_available": (
            coding.get("evaluator_verification_unavailable_count", 0) == 0
        ),
        "comparison_eligible": _comparison_eligible(
            suite=suite,
            simulated=simulated,
            case_ids=case_ids,
            coding=coding,
            fixture_set=fixture_set,
        ),
        "warnings": [
            "recommendation is advisory and never activates a model",
            *(
                [f"benchmark unavailable: {result['unavailable_reason']}"]
                if result.get("unavailable_reason")
                else []
            ),
            *(
                ["Colibri identity is a bounded metadata manifest, not a full weight digest"]
                if artifact_kind == "colibri_model_directory"
                else []
            ),
        ],
    }
    report["identity_digest"] = coding_report_identity_digest(report)
    return report


def _comparison_eligible(
    *,
    suite: str,
    simulated: bool,
    case_ids: list[Any],
    coding: Mapping[str, Any],
    fixture_set: Mapping[str, Any],
) -> bool:
    expected = fixture_set.get("coding")
    expected_ids = expected.get("case_ids") if isinstance(expected, Mapping) else None
    return bool(
        suite == "full"
        and not simulated
        and isinstance(expected_ids, list)
        and case_ids == expected_ids
        and coding.get("fixture_count") == len(expected_ids)
        and coding.get("evaluator_verification_unavailable_count", 1) == 0
        and isinstance(coding.get("case_results"), Mapping)
        and set(coding["case_results"]) == set(expected_ids)
    )


def coding_report_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "report_type": report.get("report_type"),
        "evaluator_version": report.get("evaluator_version"),
        "scoring_policy_version": report.get("scoring_policy_version"),
        "fixture_set_version": report.get("fixture_set_version"),
        "fixture_set_sha256": report.get("fixture_set_sha256"),
        "case_ids": report.get("case_ids"),
        "timeout_profile": report.get("timeout_profile"),
        "benchmark_configuration": report.get("benchmark_configuration"),
        "hardware_id": (
            report.get("hardware_profile", {}).get("id")
            if isinstance(report.get("hardware_profile"), Mapping)
            else None
        ),
        "simulated": report.get("simulated"),
        "model_id": report.get("model_id"),
        "model_identity": report.get("model_identity"),
    }


def coding_report_identity_digest(report: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        coding_report_identity(report), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_coding_report(report: Mapping[str, Any], path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and target.is_dir():
        raise ValueError(f"report path is an existing directory: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_coding_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("coding_benchmark_report_unreadable") from exc
    if not isinstance(payload, dict) or payload.get("report_type") != "coding_model_benchmark":
        raise ValueError("invalid_coding_benchmark_report")
    if payload.get("identity_digest") != coding_report_identity_digest(payload):
        raise ValueError("coding_benchmark_report_identity_mismatch")
    return payload


def compare_coding_benchmark_reports(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    reasons = _comparability_failures(first, second)
    model_ids = [str(first.get("model_id")), str(second.get("model_id"))]
    scores: list[Mapping[str, Any]] = [
        _comparison_score(first),
        _comparison_score(second),
    ]
    if reasons:
        recommendation = "insufficient_evidence"
        reason = ";".join(reasons)
    else:
        recommendation, reason = _recommend(scores)
    return {
        "schema_version": CODING_BENCHMARK_REPORT_SCHEMA_VERSION,
        "report_type": "coding_model_comparison",
        "generated_at": utc_now_iso(),
        "model_ids": model_ids,
        "comparison_identity": {
            "fixture_set_version": first.get("fixture_set_version"),
            "fixture_set_sha256": first.get("fixture_set_sha256"),
            "case_ids": first.get("case_ids"),
            "scoring_policy_version": first.get("scoring_policy_version"),
            "timeout_profile": first.get("timeout_profile"),
            "hardware_id": first.get("hardware_profile", {}).get("id"),
        },
        "scores": scores,
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "automatic_activation_performed": False,
        "warnings": ["offline comparison performed without model inference"],
    }


def _comparability_failures(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in (
        "fixture_set_version",
        "fixture_set_sha256",
        "case_ids",
        "scoring_policy_version",
        "evaluator_version",
        "timeout_profile",
        "benchmark_configuration",
    ):
        if first.get(field) != second.get(field):
            failures.append(f"mismatched_{field}")
    if first.get("simulated") or second.get("simulated"):
        failures.append("simulated_report")
    if not first.get("comparison_eligible") or not second.get("comparison_eligible"):
        failures.append("incomplete_or_ineligible_report")
    if not first.get("evaluator_verification_available") or not second.get(
        "evaluator_verification_available"
    ):
        failures.append("evaluator_verification_unavailable")
    first_hardware = first.get("hardware_profile")
    second_hardware = second.get("hardware_profile")
    if not isinstance(first_hardware, Mapping) or not isinstance(second_hardware, Mapping):
        failures.append("hardware_profile_missing")
    elif first_hardware.get("id") != second_hardware.get("id"):
        failures.append("material_hardware_mismatch")
    return list(dict.fromkeys(failures))


def _comparison_score(report: Mapping[str, Any]) -> dict[str, Any]:
    coding = report.get("coding")
    coding = coding if isinstance(coding, Mapping) else {}
    safety_failures = int(coding.get("safety_failures") or 0)
    return {
        "model_id": report.get("model_id"),
        "critical_safety_violation": safety_failures > 0,
        "safety_failures": safety_failures,
        "verified_agentic_success_rate": coding.get("agentic_verified_success_rate"),
        "verified_fixture_success_rate": coding.get("fixture_pass_rate"),
        "test_pass_rate": coding.get("test_pass_rate"),
        "repository_quality": 1.0 - float(coding.get("unnecessary_change_rate") or 0.0),
        "recovery_success_rate": coding.get("recovery_success_rate"),
        "structured_output_reliability": _structured_rate(coding),
        "performance": _performance_score(report),
    }


def _structured_rate(coding: Mapping[str, Any]) -> float | None:
    case_results = coding.get("case_results")
    rows = case_results.values() if isinstance(case_results, Mapping) else []
    values = [
        value
        for item in rows
        if isinstance(item, Mapping)
        and isinstance((value := item.get("structured_output_valid")), bool)
    ]
    return sum(values) / len(values) if values else None


def _performance_score(report: Mapping[str, Any]) -> float | None:
    performance = report.get("performance")
    runs = performance.get("runs") if isinstance(performance, Mapping) else None
    values = [run.get("tokens_per_second") for run in runs or [] if isinstance(run, Mapping)]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _recommend(scores: list[Mapping[str, Any]]) -> tuple[str, str]:
    if len(scores) != 2:
        return "insufficient_evidence", "exactly_two_candidates_required"
    if all(bool(score["critical_safety_violation"]) for score in scores):
        return "insufficient_evidence", "all_candidates_have_critical_safety_violations"
    safe = [score for score in scores if not bool(score["critical_safety_violation"])]
    if len(safe) == 1:
        return str(safe[0]["model_id"]), "safety_precedence"
    ordered = sorted(
        safe,
        key=lambda score: (
            float(score["verified_agentic_success_rate"] or -1.0),
            float(score["verified_fixture_success_rate"] or -1.0),
            float(score["test_pass_rate"] or -1.0),
            float(score["repository_quality"]),
            float(score["recovery_success_rate"] or -1.0),
            float(score["structured_output_reliability"] or -1.0),
            float(score["performance"] or -1.0),
        ),
        reverse=True,
    )
    primary_gap = abs(
        float(ordered[0]["verified_agentic_success_rate"] or 0.0)
        - float(ordered[1]["verified_agentic_success_rate"] or 0.0)
    )
    secondary_gap = max(
        abs(float(ordered[0][name] or 0.0) - float(ordered[1][name] or 0.0))
        for name in ("verified_fixture_success_rate", "test_pass_rate", "repository_quality")
    )
    if primary_gap < 0.05 and secondary_gap < 0.05:
        return "tied", "verified_correctness_and_quality_within_five_percent"
    return str(ordered[0]["model_id"]), "safety_then_verified_agentic_success_precedence"


def write_coding_comparison_report(report: Mapping[str, Any], path: Path) -> Path:
    return write_coding_report(report, path)
