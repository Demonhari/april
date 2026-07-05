"""Human review lifecycle for staged feedback eval cases.

``services.evolution.feedback_eval`` stages negative-feedback signals as
pending YAML cases under ``data/evolution/evals/pending/``. This module owns
the review lifecycle for those cases:

* promote — a human supplies the expected behaviour and the case becomes an
  active reviewed eval under ``data/evolution/evals/reviewed/``;
* reject — a human supplies a reason and the case moves to
  ``data/evolution/evals/rejected/``.

Invariants:

* Case ids are content digests assigned at staging; the strict id pattern
  (alphanumerics, ``-``, ``_`` only) makes path traversal unrepresentable.
* Every write goes through :class:`EvolutionWriteGuard`, so review artifacts
  can only ever land inside the fenced evolution data path.
* The expected behaviour of a promoted case is always human-supplied; the
  machine never invents an expectation from a negative signal.
* Only reviewed (promoted) cases are visible to the evaluator; pending and
  rejected cases never feed evals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SUMMARY_EXCERPT_CHARS = 200


class EvalReviewError(ValueError):
    """Invalid case id, unknown case, or missing human review input."""


def pending_eval_dir(settings: AprilSettings) -> Path:
    return settings.evolution_path / "evals" / "pending"


def reviewed_eval_dir(settings: AprilSettings) -> Path:
    return settings.evolution_path / "evals" / "reviewed"


def rejected_eval_dir(settings: AprilSettings) -> Path:
    return settings.evolution_path / "evals" / "rejected"


def validate_case_id(case_id: str) -> str:
    if not _CASE_ID_RE.match(case_id):
        raise EvalReviewError("invalid eval case id")
    return case_id


def _case_path(directory: Path, case_id: str) -> Path:
    validate_case_id(case_id)
    path = directory / f"{case_id}.yaml"
    # Belt and braces on top of the id pattern: the resolved file must stay
    # inside its directory and must not be a symlink out of the fence.
    resolved = path.resolve(strict=False)
    if resolved.parent != directory.resolve(strict=False) or path.is_symlink():
        raise EvalReviewError("invalid eval case id")
    return path


def _load_case(path: Path) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _case_summary(case_id: str, case: dict[str, Any]) -> dict[str, Any]:
    reason = str(case.get("reason") or "")
    return {
        "case_id": case_id,
        "case_type": str(case.get("case_type") or "unknown"),
        "signal": str(case.get("signal") or "unknown"),
        "status": str(case.get("status") or "unknown"),
        "created_at": case.get("created_at"),
        "reason_excerpt": reason[:_SUMMARY_EXCERPT_CHARS],
        "has_prompt": bool(case.get("prompt")),
        "has_bad_response_excerpt": bool(case.get("bad_response_excerpt")),
    }


def list_pending_cases(settings: AprilSettings) -> list[dict[str, Any]]:
    """Redacted-safe summaries of every staged, not-yet-reviewed eval case."""
    directory = pending_eval_dir(settings)
    if not directory.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        if not path.is_file() or path.is_symlink() or not _CASE_ID_RE.match(path.stem):
            continue
        case = _load_case(path)
        if case is None:
            continue
        summaries.append(_case_summary(path.stem, case))
    return summaries


def get_pending_case(settings: AprilSettings, case_id: str) -> dict[str, Any] | None:
    """The full pending case for local human review, or ``None`` if unknown."""
    path = _case_path(pending_eval_dir(settings), case_id)
    if not path.is_file():
        return None
    case = _load_case(path)
    if case is None:
        return None
    return {"case_id": case_id, **case}


def promote_pending_case(
    settings: AprilSettings,
    case_id: str,
    *,
    expected_behavior: str,
    guard: EvolutionWriteGuard | None = None,
    audit: AuditLogger | None = None,
    actor: str = "local-user",
) -> dict[str, Any]:
    """Promote one pending case into an active reviewed eval case.

    The reviewer-supplied ``expected_behavior`` is required; promotion never
    lets the model invent the expected answer. The pending file is removed so
    it is no longer counted as pending.
    """
    expected = expected_behavior.strip()
    if not expected:
        raise EvalReviewError("expected_behavior is required to promote an eval case")
    pending_path = _case_path(pending_eval_dir(settings), case_id)
    if not pending_path.is_file():
        raise EvalReviewError("unknown pending eval case")
    case = _load_case(pending_path)
    if case is None:
        raise EvalReviewError("pending eval case could not be read")
    active_guard = guard or EvolutionWriteGuard(settings, audit=audit)
    reviewed = {
        **case,
        "status": "reviewed",
        "expected_behavior": expected,
        "reviewed_at": utc_now_iso(),
    }
    written = active_guard.write_text(
        reviewed_eval_dir(settings) / f"{case_id}.yaml",
        yaml.safe_dump(reviewed, sort_keys=True),
    )
    # The pending file lives inside the fence by construction; validate anyway
    # so a tampered symlink can never delete something outside it.
    active_guard.validate_path(pending_path)
    pending_path.unlink(missing_ok=True)
    if audit is not None:
        audit.write(
            {
                "event_type": "feedback_eval_case_promoted",
                "actor": actor,
                "case_id": case_id,
                "path_basename": written.name,
                "expected_behavior_length": len(expected),
            }
        )
    return {"case_id": case_id, "status": "reviewed", "path_basename": written.name}


def reject_pending_case(
    settings: AprilSettings,
    case_id: str,
    *,
    reason: str,
    guard: EvolutionWriteGuard | None = None,
    audit: AuditLogger | None = None,
    actor: str = "local-user",
) -> dict[str, Any]:
    """Reject one pending case with a human-supplied reason."""
    rejection_reason = reason.strip()
    if not rejection_reason:
        raise EvalReviewError("a rejection reason is required")
    pending_path = _case_path(pending_eval_dir(settings), case_id)
    if not pending_path.is_file():
        raise EvalReviewError("unknown pending eval case")
    case = _load_case(pending_path)
    if case is None:
        raise EvalReviewError("pending eval case could not be read")
    active_guard = guard or EvolutionWriteGuard(settings, audit=audit)
    rejected = {
        **case,
        "status": "rejected",
        "rejection_reason": rejection_reason,
        "rejected_at": utc_now_iso(),
    }
    written = active_guard.write_text(
        rejected_eval_dir(settings) / f"{case_id}.yaml",
        yaml.safe_dump(rejected, sort_keys=True),
    )
    active_guard.validate_path(pending_path)
    pending_path.unlink(missing_ok=True)
    if audit is not None:
        audit.write(
            {
                "event_type": "feedback_eval_case_rejected",
                "actor": actor,
                "case_id": case_id,
                "path_basename": written.name,
                "reason_length": len(rejection_reason),
            }
        )
    return {"case_id": case_id, "status": "rejected", "path_basename": written.name}


def list_reviewed_eval_cases(settings: AprilSettings) -> list[dict[str, Any]]:
    """Active reviewed eval cases — the only review outcome evaluators may use.

    Pending cases (not yet reviewed) and rejected cases are never returned, so
    they can never feed candidate evaluation.
    """
    directory = reviewed_eval_dir(settings)
    if not directory.is_dir():
        return []
    cases: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        if not path.is_file() or path.is_symlink() or not _CASE_ID_RE.match(path.stem):
            continue
        case = _load_case(path)
        if case is None or case.get("status") != "reviewed":
            continue
        expected = str(case.get("expected_behavior") or "").strip()
        if not expected:
            # A reviewed case without human-supplied expectations is unusable.
            continue
        cases.append({"case_id": path.stem, **case})
    return cases
