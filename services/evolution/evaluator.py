from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from april_common.settings import AprilSettings

# Deterministic eval fixtures for overlay candidates. The baseline is fixed;
# a candidate must clear every check to score above it. No model call is
# involved, so the same candidate always evaluates identically.
BASELINE_SCORE = 0.5
PASSING_SCORE = 0.8
FAILING_SCORE = 0.2

_STRUCTURAL_RE = re.compile(
    r"(?im)^\s*(tools|permissions|allowed_tools|tool_registry|permission_level)\s*:"
)
_INJECTION_RE = re.compile(
    r"(?i)(ignore (all )?(previous|prior) instructions|disregard (your|the) (rules|policy)"
    r"|without approval|bypass (the )?permission)"
)
_MIN_CONTENT_CHARS = 20


@dataclass(frozen=True, slots=True)
class OverlayEvaluation:
    agent: str
    score: float
    baseline: float
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return self.score >= self.baseline

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "score": self.score,
            "baseline": self.baseline,
            "passed": self.passed,
            "checks": self.checks,
        }


def evaluate_overlay_candidate(
    *, agent: str, content: str, settings: AprilSettings
) -> OverlayEvaluation:
    """D5 fixture eval: deterministic safety and quality checks for one overlay."""
    max_chars = settings.evolution.prompt_overlay_max_chars
    checks = {
        "within_length_budget": max_chars <= 0 or len(content) <= max_chars,
        "no_structural_changes": not _STRUCTURAL_RE.search(content),
        "no_policy_injection": not _INJECTION_RE.search(content),
        "substantive_content": len(content.strip()) >= _MIN_CONTENT_CHARS,
    }
    score = PASSING_SCORE if all(checks.values()) else FAILING_SCORE
    return OverlayEvaluation(agent=agent, score=score, baseline=BASELINE_SCORE, checks=checks)
