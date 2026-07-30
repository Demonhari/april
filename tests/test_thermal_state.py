from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.runner.commands import model_compare
from apps.runner.verification.types import BenchmarkResult
from apps.runner.verify import ModelBenchmark
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    ResourceLimitReport,
    RestrictedProcessResult,
)
from april_common.thermal_state import (
    PMSET_THERMAL_COMMAND,
    ThermalStateResult,
    collect_thermal_state,
    parse_pmset_thermal_output,
    summarize_thermal_samples,
)


def _process_result(
    *,
    stdout: str = "",
    status: ProcessStatus = ProcessStatus.COMPLETED,
    returncode: int | None = 0,
) -> RestrictedProcessResult:
    return RestrictedProcessResult(
        status=status,
        returncode=returncode,
        stdout=stdout,
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=0.01,
        failure_code=None,
        resource_limits=ResourceLimitReport(ResourceLimitProfile.NONE, (), ()),
    )


def test_supported_intel_macos_pmset_output() -> None:
    captured: list[tuple[str, ...]] = []

    def runner(argv: object) -> RestrictedProcessResult:
        captured.append(tuple(argv))  # type: ignore[arg-type]
        return _process_result(
            stdout=(
                "CPU_Scheduler_Limit = 100\n"
                "CPU_Available_CPUs = 8\n"
                "CPU_Speed_Limit = 100\n"
                "Thermal_Level = 0\n"
            )
        )

    result = collect_thermal_state(
        platform_system=lambda: "Darwin",
        machine=lambda: "x86_64",
        runner=runner,
        sampled_at=lambda: "2026-07-30T00:00:00Z",
    )

    assert captured == [PMSET_THERMAL_COMMAND]
    assert result.available is True
    assert result.source == "pmset"
    assert result.cpu_speed_limit_percent == 100
    assert result.cpu_scheduler_limit_percent == 100
    assert result.thermal_level == 0
    assert result.throttling_observed is False


@pytest.mark.parametrize(
    "output",
    [
        "CPU Speed Limit: 99%\nCPU Scheduler Limit: 98%\nThermal Level: 2\n",
        " cpu_speed_limit=99 \n cpu_scheduler_limit = 98\n thermal_level = 2\n",
        "CPU_Speed_Limit : 99\nCPU_Scheduler_Limit:98\nThermal_Level:2\n",
    ],
)
def test_valid_pmset_format_variations(output: str) -> None:
    result = parse_pmset_thermal_output(output, sampled_at="2026-07-30T00:00:00Z")
    assert result.available is True
    assert result.cpu_speed_limit_percent == 99
    assert result.cpu_scheduler_limit_percent == 98
    assert result.thermal_level == 2


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("CPU_Speed_Limit = fast\n", "malformed_value"),
        ("CPU_Speed_Limit = 101\n", "malformed_value"),
        ("CPU_Speed_Limit = 100\nCPU_Speed_Limit = 90\n", "ambiguous_field"),
        ("CPU_Available_CPUs = 8\nNo thermal warning recorded\n", "no_recognized_fields"),
    ],
)
def test_malformed_or_missing_pmset_fields_are_unavailable(
    output: str,
    reason: str,
) -> None:
    result = parse_pmset_thermal_output(output)
    assert result.available is False
    assert result.failure_reason == reason
    assert result.throttling_observed is None


@pytest.mark.parametrize(
    ("status", "returncode", "reason"),
    [
        (ProcessStatus.COMPLETED, 1, "command_failed"),
        (ProcessStatus.START_FAILED, None, "command_failed"),
        (ProcessStatus.TIMED_OUT, None, "command_timeout"),
    ],
)
def test_pmset_command_failure_and_timeout(
    status: ProcessStatus,
    returncode: int | None,
    reason: str,
) -> None:
    result = collect_thermal_state(
        platform_system=lambda: "Darwin",
        machine=lambda: "x86_64",
        runner=lambda _argv: _process_result(status=status, returncode=returncode),
    )
    assert result.available is False
    assert result.failure_reason == reason
    assert result.throttling_observed is None


@pytest.mark.parametrize(
    ("system", "machine", "reason"),
    [
        ("Darwin", "arm64", "unsupported_architecture"),
        ("Linux", "x86_64", "unsupported_platform"),
    ],
)
def test_unsupported_platforms_do_not_run_pmset(
    system: str,
    machine: str,
    reason: str,
) -> None:
    called = False

    def runner(_argv: object) -> RestrictedProcessResult:
        nonlocal called
        called = True
        return _process_result()

    result = collect_thermal_state(
        platform_system=lambda: system,
        machine=lambda: machine,
        runner=runner,
    )
    assert result.available is False
    assert result.failure_reason == reason
    assert called is False


def test_speed_limit_reduction_is_direct_throttling() -> None:
    result = parse_pmset_thermal_output(
        "CPU_Speed_Limit = 82\nCPU_Scheduler_Limit = 100\nThermal_Level = 0\n"
    )
    assert result.available is True
    assert result.throttling_observed is True


def test_unavailable_measurement_is_distinct_from_no_throttling() -> None:
    unavailable = ThermalStateResult(
        available=False,
        sampled_at="2026-07-30T00:00:00Z",
        failure_reason="unsupported_platform",
    )
    summary = summarize_thermal_samples(
        [unavailable],
        performance_degradation_suggested=False,
    )
    assert summary.direct_measurement_unavailable is True
    assert summary.direct_throttling_observed is None
    assert summary.performance_degradation_suggested_throttling is False


def test_thermal_evidence_contains_no_raw_system_output() -> None:
    raw = (
        "HostName: private-mac\n"
        "UserName: private-user\n"
        "SerialNumber: SECRET-SERIAL\n"
        "CPU_Speed_Limit = 95\n"
    )
    sample = parse_pmset_thermal_output(raw, sampled_at="2026-07-30T00:00:00Z")
    summary = summarize_thermal_samples(
        [sample],
        performance_degradation_suggested=True,
    )
    encoded = summary.model_dump_json()
    for forbidden in ("private-mac", "private-user", "SECRET-SERIAL", raw):
        assert forbidden not in encoded


def test_model_benchmark_samples_before_between_and_after_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = 0

    def collect() -> ThermalStateResult:
        nonlocal samples
        samples += 1
        return ThermalStateResult(
            available=True,
            source="pmset",
            sampled_at=f"2026-07-30T00:00:0{samples}Z",
            cpu_speed_limit_percent=100,
            throttling_observed=False,
        )

    ports = iter([18101, 18102])
    monkeypatch.setattr("apps.runner.verify._free_port", lambda: next(ports))
    benchmark = ModelBenchmark(
        home=Path.cwd(),
        model_path=tmp_path / "model.gguf",
        prompt="hello",
        runs=3,
        max_output_tokens=4,
        keep_loaded=False,
        thermal_collector=collect,
    )
    monkeypatch.setattr(benchmark, "_prepare", lambda: None)
    monkeypatch.setattr(benchmark, "_env", lambda: {})
    monkeypatch.setattr(benchmark, "_start", lambda *_args: object())
    monkeypatch.setattr(benchmark, "_wait_json", lambda *_args, **_kwargs: {"status": "ok"})
    monkeypatch.setattr(
        benchmark,
        "_run_one",
        lambda index: BenchmarkResult(run_index=index, ok=True),
    )
    monkeypatch.setattr(benchmark, "_stop", lambda: None)

    results, _evaluation = benchmark.run_with_evaluation()

    assert len(results) == 3
    assert samples == 5
    assert len(benchmark.thermal_samples) == 5


def test_comparison_thermal_summary_keeps_direct_and_proxy_signals_separate() -> None:
    summaries = [
        summarize_thermal_samples(
            [
                parse_pmset_thermal_output(
                    "CPU_Speed_Limit = 80\nCPU_Scheduler_Limit = 95\nThermal_Level = 1\n"
                )
            ],
            performance_degradation_suggested=None,
        ).model_dump(mode="json")
    ]
    merged = model_compare.merge_thermal_evidence(
        summaries,
        performance_degradation_suggested=False,
    )
    payload = json.loads(merged.model_dump_json())
    assert payload["valid_sample_count"] == 1
    assert payload["minimum_cpu_speed_limit_percent"] == 80
    assert payload["minimum_cpu_scheduler_limit_percent"] == 95
    assert payload["maximum_thermal_level"] == 1
    assert payload["direct_throttling_observed"] is True
    assert payload["performance_degradation_suggested_throttling"] is False
