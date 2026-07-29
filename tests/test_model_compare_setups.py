from __future__ import annotations

from copy import deepcopy

from apps.runner.commands import model_compare
from april_common.settings import BenchmarkSettings


def _summary(**overrides: float | bool | None) -> dict[str, object]:
    metrics: dict[str, float | None] = {
        "output_tokens_per_second": 10.0,
        "routing_accuracy": 0.95,
        "structured_json_reliability": 0.96,
        "coding_fixture_pass_rate": 0.94,
        "context_handling_reliability": 0.95,
        "load_unload_reliability": 1.0,
    }
    passed = bool(overrides.pop("passed", True))
    sustained = overrides.pop("sustained", 0.05)
    metrics.update(overrides)
    return {
        "passed": passed,
        "measurements": metrics,
        "sustained_performance": {
            "sustained_performance_degradation_fraction": sustained,
        },
    }


def test_comparison_fixture_set_is_identical_and_tokenizer_measured() -> None:
    assert model_compare._FIXTURE_SET_ID
    assert "long_context_tokenizer_measured_v1" in model_compare._FIXTURES
    assert len(set(model_compare._FIXTURES)) == len(model_compare._FIXTURES)


def test_routing_json_and_coding_each_affect_recommendation() -> None:
    thresholds = BenchmarkSettings()
    specialist = _summary()
    for metric, value in (
        ("routing_accuracy", 0.2),
        ("structured_json_reliability", 0.2),
        ("coding_fixture_pass_rate", 0.2),
    ):
        shared = _summary(**{metric: value})
        recommendation, failures = model_compare.recommend_setup(
            specialist,
            shared,
            thresholds,
            simulated=False,
        )
        assert recommendation == "candidate_failed_no_regression_gates"
        assert any(metric in failure for failure in failures)


def test_missing_measurement_is_insufficient_evidence() -> None:
    shared = _summary()
    metrics = shared["measurements"]
    assert isinstance(metrics, dict)
    metrics["context_handling_reliability"] = None
    recommendation, failures = model_compare.recommend_setup(
        _summary(),
        shared,
        BenchmarkSettings(),
        simulated=False,
    )
    assert recommendation == "insufficient_evidence"
    assert "missing:context_handling_reliability" in failures


def test_sustained_degradation_uses_first_and_final_rounds() -> None:
    report = model_compare.calculate_sustained_performance(
        [
            {
                "tokens_per_second": 10.0,
                "first_token_latency_seconds": 1.0,
            },
            {
                "tokens_per_second": 8.0,
                "first_token_latency_seconds": 1.25,
            },
        ]
    )
    assert report["throughput_degradation_fraction"] == 0.2
    assert report["latency_degradation_fraction"] == 0.25
    assert report["sustained_performance_degradation_fraction"] == 0.25


def test_simulated_result_never_produces_production_recommendation() -> None:
    recommendation, failures = model_compare.recommend_setup(
        _summary(),
        _summary(),
        BenchmarkSettings(),
        simulated=True,
    )
    assert recommendation == "insufficient_evidence"
    assert failures == ["simulated_results_are_not_production_evidence"]


def test_relative_quality_regression_retains_specialists() -> None:
    shared = _summary(context_handling_reliability=0.75)
    recommendation, failures = model_compare.recommend_setup(
        _summary(),
        shared,
        BenchmarkSettings(maximum_decline_fraction=0.1),
        simulated=False,
    )
    assert recommendation == "retain_specialist_configuration"
    assert "context_handling_reliability_regressed" in failures


def test_recommendation_is_pure_and_does_not_mutate_inputs() -> None:
    specialist = _summary()
    shared = _summary()
    before_specialist = deepcopy(specialist)
    before_shared = deepcopy(shared)
    recommendation, failures = model_compare.recommend_setup(
        specialist,
        shared,
        BenchmarkSettings(),
        simulated=False,
    )
    assert recommendation == "shared_model_recommended_for_manual_review"
    assert failures == []
    assert specialist == before_specialist
    assert shared == before_shared
