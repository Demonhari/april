"""Safe, redacted Intel macOS thermal-state sampling.

Only the explicitly recognized ``pmset -g therm`` numeric fields are retained.
Raw command output and unrelated host information never leave the collector.
"""

from __future__ import annotations

import platform
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    RestrictedProcessResult,
    run_restricted_process_sync,
)
from april_common.time import utc_now_iso

PMSET_THERMAL_COMMAND = ("/usr/bin/pmset", "-g", "therm")
THERMAL_SOURCE = "pmset"
THERMAL_TIMEOUT_SECONDS = 3.0

_FIELD_PATTERNS = {
    "cpu_speed_limit_percent": re.compile(
        r"^\s*CPU(?:_|[\t ]+)Speed(?:_|[\t ]+)Limit\s*(?:=|:)\s*(\d{1,3})\s*%?\s*$",
        re.IGNORECASE,
    ),
    "cpu_scheduler_limit_percent": re.compile(
        r"^\s*CPU(?:_|[\t ]+)Scheduler(?:_|[\t ]+)Limit\s*(?:=|:)\s*(\d{1,3})\s*%?\s*$",
        re.IGNORECASE,
    ),
    "thermal_level": re.compile(
        r"^\s*Thermal(?:_|[\t ]+)Level\s*(?:=|:)\s*(\d{1,3})\s*$",
        re.IGNORECASE,
    ),
}
_RECOGNIZED_PREFIX = re.compile(
    r"^\s*(?:CPU(?:_|[\t ]+)(?:Speed|Scheduler)(?:_|[\t ]+)Limit|"
    r"Thermal(?:_|[\t ]+)Level)\b",
    re.IGNORECASE,
)

ThermalCommandRunner = Callable[[Sequence[str]], RestrictedProcessResult]


class ThermalStateResult(BaseModel):
    available: bool
    source: str | None = None
    sampled_at: str
    cpu_speed_limit_percent: int | None = Field(default=None, ge=0, le=100)
    cpu_scheduler_limit_percent: int | None = Field(default=None, ge=0, le=100)
    thermal_level: int | None = Field(default=None, ge=0, le=100)
    throttling_observed: bool | None = None
    failure_reason: str | None = None


class ThermalEvidenceSummary(BaseModel):
    source: str | None = None
    valid_sample_count: int = Field(default=0, ge=0)
    minimum_cpu_speed_limit_percent: int | None = Field(default=None, ge=0, le=100)
    minimum_cpu_scheduler_limit_percent: int | None = Field(default=None, ge=0, le=100)
    maximum_thermal_level: int | None = Field(default=None, ge=0, le=100)
    direct_throttling_observed: bool | None = None
    performance_degradation_suggested_throttling: bool | None = None
    direct_measurement_unavailable: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


def collect_thermal_state(
    *,
    platform_system: Callable[[], str] = platform.system,
    machine: Callable[[], str] = platform.machine,
    runner: ThermalCommandRunner | None = None,
    sampled_at: Callable[[], str] = utc_now_iso,
) -> ThermalStateResult:
    """Collect one unprivileged Intel macOS sample, or a typed unavailable result."""
    timestamp = sampled_at()
    if platform_system() != "Darwin":
        return _unavailable(timestamp, "unsupported_platform")
    if machine() != "x86_64":
        return _unavailable(timestamp, "unsupported_architecture")
    completed = (runner or _run_pmset)(PMSET_THERMAL_COMMAND)
    if completed.status is ProcessStatus.TIMED_OUT:
        return _unavailable(timestamp, "command_timeout", source=THERMAL_SOURCE)
    if completed.status is not ProcessStatus.COMPLETED or completed.returncode != 0:
        return _unavailable(timestamp, "command_failed", source=THERMAL_SOURCE)
    return parse_pmset_thermal_output(completed.stdout, sampled_at=timestamp)


def parse_pmset_thermal_output(output: str, *, sampled_at: str | None = None) -> ThermalStateResult:
    """Parse only recognized bounded numeric fields; reject malformed ambiguity."""
    timestamp = sampled_at or utc_now_iso()
    values: dict[str, int] = {}
    for line in output.splitlines():
        matched = False
        for field, pattern in _FIELD_PATTERNS.items():
            match = pattern.fullmatch(line)
            if match is None:
                continue
            if field in values:
                return _unavailable(timestamp, "ambiguous_field", source=THERMAL_SOURCE)
            value = int(match.group(1))
            if not 0 <= value <= 100:
                return _unavailable(timestamp, "malformed_value", source=THERMAL_SOURCE)
            values[field] = value
            matched = True
            break
        if not matched and _RECOGNIZED_PREFIX.match(line):
            return _unavailable(timestamp, "malformed_value", source=THERMAL_SOURCE)
    if not values:
        return _unavailable(timestamp, "no_recognized_fields", source=THERMAL_SOURCE)
    speed = values.get("cpu_speed_limit_percent")
    scheduler = values.get("cpu_scheduler_limit_percent")
    level = values.get("thermal_level")
    throttling = (
        (speed is not None and speed < 100)
        or (scheduler is not None and scheduler < 100)
        or (level is not None and level > 0)
    )
    return ThermalStateResult(
        available=True,
        source=THERMAL_SOURCE,
        sampled_at=timestamp,
        cpu_speed_limit_percent=speed,
        cpu_scheduler_limit_percent=scheduler,
        thermal_level=level,
        throttling_observed=throttling,
    )


def summarize_thermal_samples(
    samples: Sequence[ThermalStateResult],
    *,
    performance_degradation_suggested: bool | None,
) -> ThermalEvidenceSummary:
    valid = [sample for sample in samples if sample.available]
    speed = [
        sample.cpu_speed_limit_percent
        for sample in valid
        if sample.cpu_speed_limit_percent is not None
    ]
    scheduler = [
        sample.cpu_scheduler_limit_percent
        for sample in valid
        if sample.cpu_scheduler_limit_percent is not None
    ]
    levels = [sample.thermal_level for sample in valid if sample.thermal_level is not None]
    failures = sorted(
        {
            sample.failure_reason
            for sample in samples
            if not sample.available and sample.failure_reason is not None
        }
    )
    return ThermalEvidenceSummary(
        source=(
            THERMAL_SOURCE if any(sample.source == THERMAL_SOURCE for sample in samples) else None
        ),
        valid_sample_count=len(valid),
        minimum_cpu_speed_limit_percent=min(speed) if speed else None,
        minimum_cpu_scheduler_limit_percent=min(scheduler) if scheduler else None,
        maximum_thermal_level=max(levels) if levels else None,
        direct_throttling_observed=(
            any(sample.throttling_observed is True for sample in valid) if valid else None
        ),
        performance_degradation_suggested_throttling=performance_degradation_suggested,
        direct_measurement_unavailable=not valid,
        failure_reasons=failures,
    )


def merge_thermal_evidence(
    summaries: Sequence[Mapping[str, object]],
    *,
    performance_degradation_suggested: bool | None,
) -> ThermalEvidenceSummary:
    """Validate and merge redacted child summaries without accepting raw output."""
    parsed: list[ThermalEvidenceSummary] = []
    for summary in summaries:
        try:
            parsed.append(ThermalEvidenceSummary.model_validate(summary))
        except ValueError:
            continue
    valid_count = sum(item.valid_sample_count for item in parsed)
    speed = [
        item.minimum_cpu_speed_limit_percent
        for item in parsed
        if item.minimum_cpu_speed_limit_percent is not None
    ]
    scheduler = [
        item.minimum_cpu_scheduler_limit_percent
        for item in parsed
        if item.minimum_cpu_scheduler_limit_percent is not None
    ]
    levels = [
        item.maximum_thermal_level for item in parsed if item.maximum_thermal_level is not None
    ]
    return ThermalEvidenceSummary(
        source=THERMAL_SOURCE if any(item.source == THERMAL_SOURCE for item in parsed) else None,
        valid_sample_count=valid_count,
        minimum_cpu_speed_limit_percent=min(speed) if speed else None,
        minimum_cpu_scheduler_limit_percent=min(scheduler) if scheduler else None,
        maximum_thermal_level=max(levels) if levels else None,
        direct_throttling_observed=(
            any(item.direct_throttling_observed is True for item in parsed) if valid_count else None
        ),
        performance_degradation_suggested_throttling=performance_degradation_suggested,
        direct_measurement_unavailable=valid_count == 0,
        failure_reasons=sorted(
            {reason for item in parsed for reason in item.failure_reasons}
            or ({"thermal_evidence_invalid"} if summaries and not parsed else set())
        ),
    )


def _run_pmset(argv: Sequence[str]) -> RestrictedProcessResult:
    return run_restricted_process_sync(
        list(argv),
        cwd=Path("/"),
        category=ProcessCategory.BENCHMARKING,
        timeout_seconds=THERMAL_TIMEOUT_SECONDS,
        max_stdout_bytes=4096,
        max_stderr_bytes=1024,
    )


def _unavailable(
    sampled_at: str,
    reason: str,
    *,
    source: str | None = None,
) -> ThermalStateResult:
    return ThermalStateResult(
        available=False,
        source=source,
        sampled_at=sampled_at,
        throttling_observed=None,
        failure_reason=reason,
    )
