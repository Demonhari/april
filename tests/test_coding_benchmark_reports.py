from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from apps.runner.coding_benchmark_reports import (
    build_coding_benchmark_report,
    coding_report_identity_digest,
    compare_coding_benchmark_reports,
    load_coding_report,
    write_coding_report,
)
from services.evaluation.coding_benchmark import (
    CODING_SMOKE_CASE_IDS,
    coding_benchmark_timeout_profile,
)
from services.evaluation.model_quality import fixture_set_metadata
from services.jobs.registry import default_job_registry

ROOT = Path(__file__).resolve().parents[1]


def _report(
    tmp_path: Path,
    model_id: str,
    *,
    simulated: bool = False,
    suite: str = "full",
    timeout_profile: str = "full-local",
    safety_failures: int = 0,
    agentic_success: float = 0.75,
) -> dict[str, Any]:
    fixture_set = fixture_set_metadata(ROOT)
    case_ids = list(fixture_set["coding"]["case_ids"])
    coding = {
        "fixture_count": len(case_ids),
        "fixture_pass_rate": agentic_success,
        "test_pass_rate": agentic_success,
        "evaluator_verification_unavailable_count": 0,
        "agentic_verified_success_rate": agentic_success,
        "safety_failures": safety_failures,
        "unnecessary_change_rate": 0.0,
        "recovery_success_rate": 0.2,
        "case_results": {case_id: {"structured_output_valid": True} for case_id in case_ids},
    }
    if suite == "smoke":
        case_ids = list(CODING_SMOKE_CASE_IDS)
        coding["fixture_count"] = len(case_ids)
        coding["case_results"] = {
            case_id: {"structured_output_valid": True} for case_id in case_ids
        }
    return build_coding_benchmark_report(
        {
            "model_sha256": f"{model_id}-artifact",
            "model_basename": f"{model_id}.gguf",
            "artifact_kind": "gguf_file",
            "coding": coding,
            "runs": [{"tokens_per_second": 3.0}],
            "simulated": simulated,
        },
        model_id=model_id,
        role="coding",
        backend="llama_cpp",
        artifact_kind="gguf_file",
        configuration={"context_size": 4096, "max_output_tokens": 256},
        suite=suite,
        timeout_profile={
            "name": timeout_profile,
            "case_timeout_seconds": coding_benchmark_timeout_profile(
                timeout_profile
            ).case_timeout_seconds,
            "job_timeout_seconds": coding_benchmark_timeout_profile(
                timeout_profile
            ).job_timeout_seconds,
            "case_timeout_multiplier": 1.0,
        },
        fixture_set=fixture_set,
        simulated=simulated,
    )


def test_timeout_profiles_are_explicit_and_full_is_bounded() -> None:
    smoke = coding_benchmark_timeout_profile("smoke")
    full = coding_benchmark_timeout_profile("full-local")
    assert full.case_timeout_seconds > smoke.case_timeout_seconds
    assert full.job_timeout_seconds > full.case_timeout_seconds


def test_coding_benchmark_job_is_long_bounded_and_process_cancellable() -> None:
    definition = default_job_registry().require("model_coding_benchmark")
    assert definition.default_timeout_seconds == 21_600.0
    assert definition.cancellation_behavior == "process_group"
    assert definition.restart_safe is False


def test_single_candidate_report_is_atomic_and_self_identifying(tmp_path: Path) -> None:
    report = _report(tmp_path, "kat-candidate")
    target = tmp_path / "nested" / "kat.json"
    write_coding_report(report, target)
    loaded = load_coding_report(target)
    assert loaded["model_id"] == "kat-candidate"
    assert loaded["identity_digest"] == coding_report_identity_digest(loaded)
    assert not list(target.parent.glob("*.tmp"))


def test_smoke_report_is_not_comparison_eligible() -> None:
    report = _report(Path("."), "kat-candidate", suite="smoke", timeout_profile="smoke")
    assert report["case_ids"] == list(CODING_SMOKE_CASE_IDS)
    assert report["comparison_eligible"] is False


def test_full_report_contains_all_installed_case_ids_and_fixture_sha() -> None:
    report = _report(Path("."), "kat-candidate")
    coding = fixture_set_metadata(ROOT)["coding"]
    assert report["case_ids"] == coding["case_ids"]
    assert report["fixture_set_sha256"] == coding["sha256"]
    assert report["comparison_eligible"] is True


def test_offline_comparison_rejects_material_mismatches() -> None:
    first = _report(Path("."), "kat-candidate")
    second = _report(Path("."), "qwen-candidate")
    second["fixture_set_sha256"] = "different"
    second["identity_digest"] = coding_report_identity_digest(second)
    comparison = compare_coding_benchmark_reports(first, second)
    assert comparison["recommendation"] == "insufficient_evidence"
    assert "mismatched_fixture_set_sha256" in comparison["recommendation_reason"]

    second = _report(Path("."), "qwen-candidate")
    second["timeout_profile"] = {"name": "smoke"}
    second["identity_digest"] = coding_report_identity_digest(second)
    comparison = compare_coding_benchmark_reports(first, second)
    assert comparison["recommendation"] == "insufficient_evidence"
    assert "mismatched_timeout_profile" in comparison["recommendation_reason"]


def test_offline_comparison_is_safety_first_and_conservative() -> None:
    unsafe = _report(Path("."), "unsafe", safety_failures=1, agentic_success=1.0)
    safe = _report(Path("."), "safe", agentic_success=0.5)
    comparison = compare_coding_benchmark_reports(unsafe, safe)
    assert comparison["recommendation"] == "safe"

    tied = _report(Path("."), "a", agentic_success=0.5)
    other = _report(Path("."), "b", agentic_success=0.5)
    assert compare_coding_benchmark_reports(tied, other)["recommendation"] == "tied"

    unsafe_b = _report(Path("."), "unsafe-b", safety_failures=1, agentic_success=1.0)
    assert (
        compare_coding_benchmark_reports(unsafe, unsafe_b)["recommendation"]
        == "insufficient_evidence"
    )


def test_simulated_and_incomplete_reports_cannot_win() -> None:
    simulated = _report(Path("."), "fake", simulated=True)
    real = _report(Path("."), "real")
    assert compare_coding_benchmark_reports(simulated, real)["recommendation"] == (
        "insufficient_evidence"
    )
    incomplete = _report(Path("."), "incomplete")
    incomplete["comparison_eligible"] = False
    incomplete["identity_digest"] = coding_report_identity_digest(incomplete)
    assert compare_coding_benchmark_reports(incomplete, real)["recommendation"] == (
        "insufficient_evidence"
    )


def test_report_validation_and_comparison_edge_cases(tmp_path: Path) -> None:
    from apps.runner.coding_benchmark_reports import (
        _recommend,
        write_coding_comparison_report,
    )

    report = _report(tmp_path, "candidate")
    with pytest.raises(ValueError, match="existing directory"):
        write_coding_report(report, tmp_path)
    with pytest.raises(ValueError, match="unreadable"):
        load_coding_report(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_coding"):
        load_coding_report(invalid)
    report_path = tmp_path / "comparison.json"
    write_coding_comparison_report({"report_type": "coding_model_comparison"}, report_path)
    assert report_path.is_file()

    malformed = dict(report)
    malformed["identity_digest"] = "wrong"
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="identity_mismatch"):
        load_coding_report(malformed_path)

    missing_evidence = dict(report)
    missing_evidence["evaluator_verification_available"] = False
    no_hardware = dict(report)
    no_hardware["hardware_profile"] = None
    mismatch = compare_coding_benchmark_reports(missing_evidence, no_hardware)
    assert "evaluator_verification_unavailable" in mismatch["recommendation_reason"]
    assert "hardware_profile_missing" in mismatch["recommendation_reason"]
    assert _recommend([])[0] == "insufficient_evidence"
    scores = [
        {
            "model_id": "fast",
            "critical_safety_violation": False,
            "verified_agentic_success_rate": 0.9,
            "verified_fixture_success_rate": 0.9,
            "test_pass_rate": 0.9,
            "repository_quality": 0.9,
            "recovery_success_rate": 0.9,
            "structured_output_reliability": 0.9,
            "performance": 0.9,
        },
        {
            "model_id": "slow",
            "critical_safety_violation": False,
            "verified_agentic_success_rate": 0.5,
            "verified_fixture_success_rate": 0.5,
            "test_pass_rate": 0.5,
            "repository_quality": 0.5,
            "recovery_success_rate": 0.5,
            "structured_output_reliability": 0.5,
            "performance": 0.5,
        },
    ]
    assert _recommend(scores)[0] == "fast"


def test_report_builder_accepts_quality_coding_shape(tmp_path: Path) -> None:
    fixture_set = fixture_set_metadata(ROOT)
    report = build_coding_benchmark_report(
        {"quality": {"coding": {"fixture_count": 0}}, "case_ids": []},
        model_id="candidate",
        role="coding",
        backend="llama_cpp",
        artifact_kind="gguf_file",
        configuration={},
        suite="full",
        timeout_profile={"name": "full-local"},
        fixture_set=fixture_set,
        simulated=False,
    )
    assert report["coding"]["fixture_count"] == 0


def test_worker_coding_paths_are_fakeable_without_model_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.runner import model_job_worker

    settings = SimpleNamespace(
        runtime=SimpleNamespace(backend="fake"),
        environment="development",
        workers=SimpleNamespace(development_unsandboxed_override=False),
    )
    model = SimpleNamespace(
        artifact_kind="gguf_file",
        path=tmp_path / "candidate.gguf",
        role="coding",
        sha256="a" * 64,
        manifest_digest=None,
        basename="candidate.gguf",
    )
    quality = {
        "fixture_set": {},
        "coding": {"fixture_pass_rate": 1.0},
        "coding_fixture_pass_rate": 1.0,
        "timeout_profile": {"name": "smoke"},
        "suite": "smoke",
        "case_ids": list(CODING_SMOKE_CASE_IDS),
    }

    class FakeRun:
        run_index = 0
        ok = True
        first_token_latency_seconds = 0.1
        generation_time_seconds = 0.2
        output_tokens = 4
        tokens_per_second = 20.0
        process_rss_bytes = 1
        peak_process_rss_bytes = 2

    class FakeBenchmark:
        def __init__(self, **_: Any) -> None:
            pass

        def run_with_evaluation(self, callback: Any) -> tuple[list[FakeRun], dict[str, Any]]:
            return [FakeRun()], callback(self)

    async def fake_quality(*_: Any, **__: Any) -> dict[str, Any]:
        return quality

    monkeypatch.setattr(model_job_worker, "load_settings", lambda **_: settings)
    monkeypatch.setattr(model_job_worker, "validate_registered_model", lambda *_args, **_: model)
    monkeypatch.setattr(model_job_worker, "ModelBenchmark", FakeBenchmark)
    monkeypatch.setattr(model_job_worker, "_coding_quality_evaluation", fake_quality)
    payload = model_job_worker._coding_benchmark(
        tmp_path,
        "candidate",
        suite="smoke",
        timeout_profile="smoke",
        case_timeout_multiplier=1.0,
    )
    assert payload["coding_fixture_pass_rate"] == 1.0
    assert payload["runs"][0]["tokens_per_second"] == 20.0

    model.artifact_kind = "colibri_model_directory"

    async def fake_colibri(*_: Any, **__: Any) -> dict[str, Any]:
        return {"coding": {}, "suite": "smoke"}

    monkeypatch.setattr(model_job_worker, "_colibri_coding_benchmark", fake_colibri)
    payload = model_job_worker._coding_benchmark(
        tmp_path,
        "candidate",
        suite="smoke",
        timeout_profile="smoke",
        case_timeout_multiplier=1.0,
    )
    assert payload["artifact_kind"] == "colibri_model_directory"


@pytest.mark.asyncio
async def test_worker_quality_adapter_uses_fake_tool_worker_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.runner import model_job_worker

    settings = SimpleNamespace(
        home=tmp_path,
        environment="development",
        workers=SimpleNamespace(development_unsandboxed_override=False),
    )
    observed: dict[str, Any] = {}

    class FakeManager:
        def __init__(self, **kwargs: Any) -> None:
            observed["manager"] = kwargs

        async def start(self) -> str:
            return "fake-worker"

        async def stop(self) -> None:
            observed["stopped"] = True

    async def fake_evaluate(*_: Any, **kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"coding": {"fixture_pass_rate": 1.0}}

    monkeypatch.setattr(model_job_worker, "ToolWorkerProcessManager", FakeManager)
    monkeypatch.setattr(model_job_worker, "evaluate_coding_benchmark", fake_evaluate)
    result = await model_job_worker._coding_quality_evaluation(
        settings,
        session=SimpleNamespace(
            temp=tmp_path / "session",
            runtime_url="http://fake-runtime",
            runtime_token=None,
        ),
        model_id="candidate",
        suite="smoke",
        timeout_profile="smoke",
        case_timeout_multiplier=1.0,
    )
    assert result["coding"]["fixture_pass_rate"] == 1.0
    assert observed["tool_worker"] == "fake-worker"
    assert observed["suite"] == "smoke"
    assert observed["timeout_profile"] == "smoke"
    assert observed["stopped"] is True

    class UnavailableManager(FakeManager):
        async def start(self) -> str:
            raise model_job_worker.ToolWorkerUnavailable("offline")

    production = SimpleNamespace(
        home=tmp_path,
        environment="production",
        workers=SimpleNamespace(development_unsandboxed_override=False),
    )
    monkeypatch.setattr(model_job_worker, "ToolWorkerProcessManager", UnavailableManager)
    result = await model_job_worker._coding_quality_evaluation(
        production,
        session=SimpleNamespace(
            temp=tmp_path / "production-session",
            runtime_url="http://fake-runtime",
            runtime_token=None,
        ),
        model_id="candidate",
        suite="smoke",
        timeout_profile="smoke",
        case_timeout_multiplier=1.0,
    )
    assert result["coding"]["fixture_pass_rate"] == 1.0


@pytest.mark.asyncio
async def test_colibri_worker_adapters_are_inference_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.runner import model_job_worker

    settings = SimpleNamespace(
        home=tmp_path,
        environment="development",
        benchmark=SimpleNamespace(runs=1),
        workers=SimpleNamespace(development_unsandboxed_override=False),
    )

    class FakeBackend:
        unloaded = 0

        async def generate(self, *_: Any, **__: Any) -> Any:
            return SimpleNamespace(text="{}", output_tokens=2)

        async def tokenize(self, _text: str) -> list[int]:
            return [1, 2]

        async def unload(self) -> None:
            self.unloaded += 1

    class FakeManager:
        def __init__(self, **_: Any) -> None:
            pass

        async def start(self) -> str:
            return "worker"

        async def stop(self) -> None:
            pass

    backend = FakeBackend()
    original_colibri_backend = model_job_worker._colibri_backend
    original_colibri_quality = model_job_worker._colibri_quality
    monkeypatch.setattr(model_job_worker, "_colibri_backend", lambda *_: _async_value(backend))
    monkeypatch.setattr(
        model_job_worker,
        "_colibri_quality",
        lambda *_: _async_value(
            {
                "fixture_set": {},
                "routing": {},
                "strict_json": {},
                "coding": {},
                "context": {},
                "routing_accuracy": 1.0,
                "strict_json_first_pass_reliability": 1.0,
                "structured_json_reliability": 1.0,
                "coding_fixture_pass_rate": 1.0,
                "context_handling_reliability": 1.0,
            }
        ),
    )
    verification = await model_job_worker._colibri_verification(settings, "candidate")
    assert verification["passed"] is True
    benchmark = await model_job_worker._colibri_benchmark(settings, "candidate")
    assert benchmark["runs"][0]["prompt_token_count"] == 2

    monkeypatch.setattr(model_job_worker, "ToolWorkerProcessManager", FakeManager)
    monkeypatch.setattr(
        model_job_worker,
        "evaluate_coding_benchmark",
        lambda *_args, **_kwargs: _async_value(
            {
                "coding": {"fixture_pass_rate": 1.0},
                "coding_fixture_pass_rate": 1.0,
                "timeout_profile": {"name": "smoke"},
                "suite": "smoke",
                "case_ids": list(CODING_SMOKE_CASE_IDS),
            }
        ),
    )
    coding = await model_job_worker._colibri_coding_benchmark(
        settings,
        "candidate",
        suite="smoke",
        timeout_profile="smoke",
        case_timeout_multiplier=1.0,
    )
    assert coding["artifact_kind"] == "colibri_model_directory"

    monkeypatch.setattr(
        model_job_worker,
        "evaluate_model_quality",
        lambda *_args, **_kwargs: _async_value({"fixture_set": {}}),
    )
    monkeypatch.setattr(model_job_worker, "_colibri_quality", original_colibri_quality)
    quality = await model_job_worker._colibri_quality(settings, "candidate")
    assert quality["fixture_set"] == {}

    class FakeModel:
        colibri_tokenizer_path = Path("tokenizer.json")
        colibri_base_url = "http://127.0.0.1:8080"
        colibri_model_name = "candidate"
        name = "candidate"

        def resolved_path(self, _root: Path) -> Path:
            return tmp_path / "model"

        def model_copy(self, **_: Any) -> FakeModel:
            return self

    class FakeRegistry:
        root = tmp_path

        def get(self, _model_id: str) -> FakeModel:
            return FakeModel()

    class FakeColibri:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def load(self, _model: Any) -> None:
            pass

    monkeypatch.setattr(
        model_job_worker.ModelRegistry,
        "from_file",
        lambda *_args, **_kwargs: FakeRegistry(),
    )
    monkeypatch.setattr(model_job_worker, "ColibriBackend", FakeColibri)
    monkeypatch.setattr(model_job_worker, "_colibri_backend", original_colibri_backend)
    loaded = await model_job_worker._colibri_backend(settings, "candidate")
    assert loaded.kwargs["base_url"] == "http://127.0.0.1:8080"


async def _async_value(value: Any) -> Any:
    return value
