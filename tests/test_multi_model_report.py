from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from apps.runner.mac_report import (
    EnvironmentSnapshot,
    ReportThresholds,
    RoutingReport,
)
from apps.runner.multi_model_report import (
    PerModelResult,
    SpecialistSwitchReport,
    build_multi_model_report,
    per_model_threshold_failures,
    write_multi_model_report,
)
from apps.runner.verify import plan_multi_model_verification, skipped_result_for

ENV = EnvironmentSnapshot(
    generated_at="2026-06-26T00:00:00Z",
    os="Darwin 24.0.0",
    cpu_architecture="arm64",
    python_version="3.11.15",
)


def _brain_pass() -> PerModelResult:
    return PerModelResult(
        model_id="april-brain",
        role="brain",
        backend="llama_cpp",
        path_basename="granite-3.3-2b-q4_k_m.gguf",
        quantization="q4_k_m",
        available=True,
        context_size=8192,
        load_success=True,
        load_duration_seconds=1.2,
        chat_success=True,
        streaming_success=True,
        first_token_latency_seconds=0.3,
        tokens_per_second=20.0,
        output_token_count=16,
        unload_success=True,
        process_rss_bytes=400_000_000,
        structured_brain_json_success=True,
        routing=RoutingReport(total=38, passed=38, accuracy=1.0),
    )


def _coding_pass() -> PerModelResult:
    return PerModelResult(
        model_id="april-coding",
        role="coding",
        backend="llama_cpp",
        path_basename="qwen3-1.7b-q8_0.gguf",
        quantization="q8_0",
        available=True,
        context_size=8192,
        load_success=True,
        load_duration_seconds=0.8,
        chat_success=True,
        streaming_success=True,
        first_token_latency_seconds=0.2,
        tokens_per_second=30.0,
        output_token_count=8,
        unload_success=True,
        smoke_success=True,
        smoke_schema_valid=True,
        smoke_kind="coding_plan",
    )


def _reading_pass() -> PerModelResult:
    return PerModelResult(
        model_id="april-reading",
        role="reading",
        backend="llama_cpp",
        path_basename="qwen3-0.6b-q8_0.gguf",
        quantization="q8_0",
        available=True,
        context_size=4096,
        load_success=True,
        load_duration_seconds=0.7,
        chat_success=True,
        streaming_success=True,
        first_token_latency_seconds=0.2,
        tokens_per_second=35.0,
        output_token_count=8,
        unload_success=True,
        smoke_success=True,
        smoke_kind="reading_summary",
    )


def _all_switch_ok() -> SpecialistSwitchReport:
    return SpecialistSwitchReport(
        attempted=True,
        brain_loaded=True,
        coding_loaded=True,
        coding_unloaded=True,
        reading_loaded=True,
        reading_unloaded=True,
        brain_usable_after=True,
    )


# --- pure builder ----------------------------------------------------------


def test_all_models_pass_is_pass() -> None:
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), _coding_pass(), _reading_pass()],
        specialist_switch=_all_switch_ok(),
    )
    assert report.summary == "pass"
    assert report.real_model_verified is True
    assert report.real_models_exercised == 3
    assert report.real_models_passed == 3
    assert report.any_real_model_exercised is True
    assert report.any_real_model_passed is True
    assert report.core_model_set_verified is True
    assert report.all_available_models_verified is True
    assert report.all_configured_models_verified is True
    assert report.verification_level == "all"
    assert report.models_passed == 3
    assert report.skipped == []


def test_zero_real_models_is_verification_level_none() -> None:
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[],
        specialist_switch=None,
    )
    assert report.real_models_exercised == 0
    assert report.real_models_passed == 0
    assert report.any_real_model_exercised is False
    assert report.any_real_model_passed is False
    assert report.verification_level == "none"


def test_one_passing_model_only_is_partial_when_core_roles_are_missing() -> None:
    missing_coding = PerModelResult(
        model_id="april-coding",
        role="coding",
        backend="llama_cpp",
        path_basename="qwen-coding.gguf",
        available=False,
        skipped_reason="Missing model file.",
    )
    missing_reading = PerModelResult(
        model_id="april-reading",
        role="reading",
        backend="llama_cpp",
        path_basename="qwen-reading.gguf",
        available=False,
        skipped_reason="Missing model file.",
    )
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), missing_coding, missing_reading],
        specialist_switch=_all_switch_ok(),
    )
    assert report.real_model_verified is False
    assert report.core_model_set_verified is False
    assert report.all_configured_models_verified is False
    assert report.verification_level == "partial"
    assert report.summary == "degraded"


def test_brain_coding_reading_pass_is_core_when_optional_specialist_missing() -> None:
    missing_reasoning = PerModelResult(
        model_id="april-reasoning",
        role="reasoning",
        backend="llama_cpp",
        path_basename="reasoning.gguf",
        available=False,
        skipped_reason="Missing model file.",
    )
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), _coding_pass(), _reading_pass(), missing_reasoning],
        specialist_switch=_all_switch_ok(),
    )
    assert report.core_model_set_verified is True
    assert report.all_configured_models_verified is False
    assert report.verification_level == "core"
    assert report.summary == "degraded"


def test_fake_backend_can_never_be_real_verified() -> None:
    # A structurally-fine FAKE run is degraded at best and never real-verified.
    brain = _brain_pass()
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="fake",
        results=[brain],
        specialist_switch=_all_switch_ok(),
    )
    assert report.real_model_verified is False
    assert report.real_models_exercised == 0
    assert report.core_model_set_verified is False
    assert report.all_configured_models_verified is False
    assert report.verification_level == "none"
    assert report.summary != "pass"
    assert report.summary == "degraded"


def test_missing_model_is_skipped_not_passed() -> None:
    missing = PerModelResult(
        model_id="april-reading",
        role="reading",
        backend="llama_cpp",
        path_basename="qwen3-0.6b-q8_0.gguf",
        available=False,
        skipped_reason="Missing model file.",
    )
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), missing],
        specialist_switch=_all_switch_ok(),
    )
    # The missing model contributes a skip, never a pass, and the optional skip
    # degrades the otherwise-passing run.
    assert report.models_passed == 1
    assert report.summary == "degraded"
    assert report.all_configured_models_verified is False
    assert report.verification_level == "partial"
    assert any(item.name == "april-reading" for item in report.skipped)


def test_structural_failure_is_fail() -> None:
    broken = _coding_pass()
    broken.streaming_success = False
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), broken],
        specialist_switch=_all_switch_ok(),
    )
    assert report.summary == "fail"
    assert report.checks_failed == 1


def test_brain_structured_json_false_fails() -> None:
    brain = _brain_pass()
    brain.structured_brain_json_success = False
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[brain],
        specialist_switch=_all_switch_ok(),
    )
    assert report.summary == "fail"
    assert report.models_passed == 0
    assert any("structured Brain JSON" in failure for failure in report.check_failures)


def test_routing_below_threshold_fails_and_is_reported() -> None:
    brain = _brain_pass()
    brain.routing = RoutingReport(total=10, passed=8, accuracy=0.8)
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[brain],
        specialist_switch=_all_switch_ok(),
        thresholds=ReportThresholds(min_routing_accuracy=0.9),
    )
    assert report.summary == "degraded"
    assert report.models_passed == 1
    assert report.real_model_verified is True
    assert not any(
        "routing accuracy 0.80 below minimum 0.90" in item for item in report.check_failures
    )
    assert any(
        "routing accuracy 0.80 below minimum 0.90" in item for item in report.threshold_failures
    )


def test_brain_without_routing_evals_keeps_lifecycle_verified() -> None:
    brain = _brain_pass()
    brain.routing = RoutingReport(total=0, passed=0, accuracy=0.0)
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[brain],
        specialist_switch=_all_switch_ok(),
    )
    assert report.summary == "pass"
    assert report.real_model_verified is True
    assert not any("routing evals did not run" in failure for failure in report.check_failures)


def test_specialist_smoke_false_fails() -> None:
    coding = _coding_pass()
    coding.smoke_success = False
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), coding],
        specialist_switch=_all_switch_ok(),
    )
    assert report.summary == "fail"
    assert report.models_passed == 1
    assert any("specialist role smoke" in failure for failure in report.check_failures)


def test_specialist_smoke_schema_false_fails() -> None:
    coding = _coding_pass()
    coding.smoke_schema_valid = False
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), coding],
        specialist_switch=_all_switch_ok(),
    )
    assert report.summary == "fail"
    assert any("smoke schema" in failure for failure in report.check_failures)


def test_specialist_switch_failure_is_fail() -> None:
    switch = _all_switch_ok()
    switch.brain_usable_after = False
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass()],
        specialist_switch=switch,
    )
    assert report.summary == "fail"


def test_runtime_error_is_fail() -> None:
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass()],
        specialist_switch=None,
        runtime_error=True,
    )
    assert report.summary == "fail"


def test_require_real_model_without_any_available_fails() -> None:
    missing = PerModelResult(
        model_id="april-brain",
        role="brain",
        backend="llama_cpp",
        path_basename="granite.gguf",
        available=False,
        skipped_reason="Missing model file.",
    )
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[missing],
        specialist_switch=None,
        require_real_model=True,
    )
    assert report.summary == "fail"
    assert report.real_model_verified is False


def test_threshold_failure_degrades_but_does_not_fail() -> None:
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), _coding_pass()],
        specialist_switch=_all_switch_ok(),
        thresholds=ReportThresholds(min_tokens_per_second=1000.0),
    )
    assert report.summary == "degraded"
    assert report.threshold_failures


def test_rss_threshold_failure_degrades() -> None:
    failures = per_model_threshold_failures(_brain_pass(), ReportThresholds(max_rss_mb=1.0))
    assert any("process_rss_mb" in failure for failure in failures)


def test_report_is_redacted_by_construction() -> None:
    leaky = PerModelResult(
        model_id="april-reading",
        role="reading",
        backend="llama_cpp",
        path_basename="qwen3-0.6b-q8_0.gguf",
        available=False,
        skipped_reason="Missing model file: /Users/hari/april/models/qwen3-0.6b-q8_0.gguf",
    )
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass(), leaky],
        specialist_switch=_all_switch_ok(),
    )
    serialized = report.model_dump_json()
    # No directory structure, tokens, or secrets — only basenames survive.
    for banned in ("/Users/", "/april/", "Bearer ", "local-dev-token", "/models/"):
        assert banned not in serialized
    # The path-bearing skip reason is collapsed to a basename in both places.
    assert leaky.skipped_reason == "Missing model file: qwen3-0.6b-q8_0.gguf"
    skip = next(item for item in report.skipped if item.name == "april-reading")
    assert "/" not in skip.reason.replace("Missing model file: ", "")
    # Every model exposes only a basename, never a path.
    for model in report.models:
        assert model.path_basename is None or "/" not in model.path_basename


def test_report_has_discriminator_for_viewer() -> None:
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        results=[_brain_pass()],
        specialist_switch=_all_switch_ok(),
    )
    assert json.loads(report.model_dump_json())["report_type"] == "multi_model"


def test_write_multi_model_report_creates_file(tmp_path: Path) -> None:
    report = build_multi_model_report(
        environment=ENV,
        runtime_backend="fake",
        results=[],
        specialist_switch=None,
    )
    out = tmp_path / "data" / "verification" / "mac-readiness.json"
    written = write_multi_model_report(report, out)
    assert written.exists()
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["report_type"] == "multi_model"
    assert loaded["real_model_verified"] is False


# --- discovery / skip planning (no real runtime) ---------------------------


def _home_with_configs(tmp_path: Path) -> Path:
    shutil.copytree(Path.cwd() / "configs", tmp_path / "configs")
    return tmp_path


def test_plan_marks_missing_files_skipped(tmp_path: Path) -> None:
    home = _home_with_configs(tmp_path)
    models_path = home / "configs" / "models.yaml"
    data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    for model in data["models"].values():
        model["path"] = str(home / "models" / f"{model['id']}.gguf")
    models_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    plan = plan_multi_model_verification(home, llama_available=True)
    assert plan, "no configured models discovered"
    # The repo ships no GGUF files, so every configured model is skipped as
    # missing rather than silently treated as runnable.
    assert all(not entry.available for entry in plan)
    assert all("Missing model file" in (entry.reason or "") for entry in plan)
    # Skipped results never look like a pass.
    for entry in plan:
        result = skipped_result_for(entry)
        assert result.available is False
        assert result.structural_ok is False
        assert result.path_basename
        assert "/" not in result.path_basename


def test_plan_without_llama_is_skipped(tmp_path: Path) -> None:
    home = _home_with_configs(tmp_path)
    plan = plan_multi_model_verification(home, llama_available=False)
    assert all(not entry.available for entry in plan)
    assert all("llama-cpp-python is not installed" in (entry.reason or "") for entry in plan)
