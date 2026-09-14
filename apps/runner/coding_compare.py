"""Offline-safe coding-model comparison primitives.

The evaluator callback is intentionally injected.  A production evaluator must
use the existing Tool Worker/test harness; this module never executes generated
code or changes model configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel, Field

from april_common.hardware_profile import safe_hardware_profile
from services.april_runtime.model_registry import ModelDefinition

CODING_FIXTURES = (
    "python_bug_fix_v1",
    "python_multi_file_v1",
    "targeted_test_v1",
    "repository_navigation_v1",
    "failure_recovery_v1",
    "strict_structured_output_v1",
)


class CodingCaseMeasurement(BaseModel):
    completed: bool = False
    tests_passed: bool = False
    syntax_failure: bool = False
    forbidden_modifications: int = 0
    unnecessary_modifications: int = 0
    structured_json_valid: bool = False
    action_valid: bool = False
    recovered_after_failure: bool = False
    turns: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    timed_out: bool = False
    rss_bytes: int | None = Field(default=None, ge=0)


class CodingModelScore(BaseModel):
    model_id: str
    backend: str
    artifact_kind: str
    configuration: dict[str, object]
    cases: dict[str, CodingCaseMeasurement]
    success_rate: float
    test_pass_rate: float
    syntax_failure_count: int
    forbidden_modification_count: int
    unnecessary_modification_count: int
    structured_json_reliability: float
    action_validity: float
    recovery_rate: float
    average_turns: float
    average_latency_seconds: float
    timeout_count: int


class CodingComparisonReport(BaseModel):
    schema_version: int = 1
    report_type: str = "coding_model_comparison"
    model_ids: tuple[str, str]
    hardware_profile: dict[str, object]
    fixture_set: tuple[str, ...] = CODING_FIXTURES
    scores: list[CodingModelScore]
    recommendation: str
    warnings: list[str] = Field(default_factory=list)
    automatic_activation_performed: bool = False


def compare_coding_models(
    models: tuple[ModelDefinition, ModelDefinition],
    evaluate: Callable[[ModelDefinition, str], CodingCaseMeasurement],
    *,
    hardware: Mapping[str, object] | None = None,
) -> CodingComparisonReport:
    scores = [
        _score_model(model, {fixture: evaluate(model, fixture) for fixture in CODING_FIXTURES})
        for model in models
    ]
    ordered = sorted(
        scores,
        key=lambda score: (score.test_pass_rate, score.success_rate),
        reverse=True,
    )
    recommendation = (
        ordered[0].model_id
        if ordered and ordered[0].test_pass_rate > 0
        else "insufficient_evidence"
    )
    warnings = []
    if len({score.test_pass_rate for score in scores}) == 1:
        warnings.append("candidate test pass rates are tied")
    return CodingComparisonReport(
        model_ids=(models[0].id, models[1].id),
        hardware_profile=dict(hardware or safe_hardware_profile()),
        scores=scores,
        recommendation=recommendation,
        warnings=warnings,
    )


def _score_model(
    model: ModelDefinition, cases: dict[str, CodingCaseMeasurement]
) -> CodingModelScore:
    total = len(cases) or 1
    return CodingModelScore(
        model_id=model.id,
        backend=model.backend,
        artifact_kind=model.artifact_kind,
        configuration={
            "context_size": model.context_size,
            "max_output_tokens": model.max_output_tokens,
            "threads": model.threads,
            "chat_format": model.chat_format,
        },
        cases=cases,
        success_rate=sum(item.completed for item in cases.values()) / total,
        test_pass_rate=sum(item.tests_passed for item in cases.values()) / total,
        syntax_failure_count=sum(item.syntax_failure for item in cases.values()),
        forbidden_modification_count=sum(item.forbidden_modifications for item in cases.values()),
        unnecessary_modification_count=sum(
            item.unnecessary_modifications for item in cases.values()
        ),
        structured_json_reliability=sum(item.structured_json_valid for item in cases.values())
        / total,
        action_validity=sum(item.action_valid for item in cases.values()) / total,
        recovery_rate=sum(item.recovered_after_failure for item in cases.values()) / total,
        average_turns=sum(item.turns for item in cases.values()) / total,
        average_latency_seconds=sum(item.latency_seconds for item in cases.values()) / total,
        timeout_count=sum(item.timed_out for item in cases.values()),
    )
