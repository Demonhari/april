"""Versioned, offline coding-benchmark fixture contracts.

The fixture description is trusted evaluation input, not model context.  Hidden
assertions stay in this evaluator-owned structure and are never materialized in
the temporary repository visible to the coding agent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodingFileAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exists: bool | None = None
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()


class CodingHiddenTest(BaseModel):
    """Evaluator-only behavioural test material.

    Hidden files are copied to a separate evaluator workspace only after model
    interaction has ended.  They are never part of ``visible_files``.
    """

    model_config = ConfigDict(extra="forbid")

    files: dict[str, str] = Field(default_factory=dict)
    argv: tuple[str, ...] = ("pytest", "-q")

    @field_validator("files")
    @classmethod
    def bounded_files(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 8:
            raise ValueError("hidden test has too many files")
        _validate_relative_file_mapping(value, "hidden test")
        return value

    @field_validator("argv")
    @classmethod
    def safe_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 32 or any(not item or len(item) > 512 for item in value):
            raise ValueError("hidden test argv is invalid")
        return value


@dataclass(frozen=True, slots=True)
class CodingBenchmarkTimeoutProfile:
    """Bounded execution limits for coding benchmark runs."""

    name: str
    case_timeout_seconds: float
    job_timeout_seconds: float


CODING_SMOKE_CASE_IDS = (
    "basic_clamp_validation",
    "navigation_registry_lookup",
    "security_no_test_tampering",
)
CODING_BENCHMARK_TIMEOUT_PROFILES = {
    "smoke": CodingBenchmarkTimeoutProfile(
        name="smoke",
        case_timeout_seconds=180.0,
        job_timeout_seconds=1_800.0,
    ),
    "full-local": CodingBenchmarkTimeoutProfile(
        name="full-local",
        case_timeout_seconds=900.0,
        job_timeout_seconds=21_600.0,
    ),
}


def coding_benchmark_timeout_profile(name: str) -> CodingBenchmarkTimeoutProfile:
    try:
        return CODING_BENCHMARK_TIMEOUT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown_coding_benchmark_timeout_profile:{name}") from exc


class CodingFixture(BaseModel):
    """A bounded coding task with visible setup and evaluator-only assertions."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=80)
    mode: Literal["one_shot", "agentic"]
    request: str = Field(min_length=1, max_length=4_000)
    initial_files: dict[str, str] = Field(default_factory=dict)
    visible_tests: dict[str, str] = Field(default_factory=dict)
    dirty_files: dict[str, str] = Field(default_factory=dict)
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = (".git",)
    verification_argv: tuple[str, ...] = ("pytest", "-q")
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    candidate_file: str | None = None
    expected_file_contains: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    hidden_assertions: dict[str, CodingFileAssertion] = Field(default_factory=dict)
    hidden_tests: tuple[CodingHiddenTest, ...] = ()
    protected_test_paths: tuple[str, ...] = ()
    mutable_test_paths: tuple[str, ...] = ()
    regression_test_expected: bool = False
    recovery_expected: bool = False
    no_progress_expected: bool = False
    task_type: Literal[
        "explanation",
        "investigation",
        "code_modification",
        "verified_code_modification",
    ] = "verified_code_modification"
    language: str = "python"
    framework: str | None = None

    @field_validator(
        "initial_files",
        "visible_tests",
        "dirty_files",
        "expected_file_contains",
        "hidden_assertions",
        "hidden_tests",
    )
    @classmethod
    def bounded_mapping(cls, value: Any) -> Any:
        if len(value) > 32:
            raise ValueError("coding fixture has too many files or assertions")
        return value

    @field_validator("initial_files", "visible_tests", "dirty_files")
    @classmethod
    def safe_file_contents(cls, value: dict[str, str]) -> dict[str, str]:
        _validate_relative_file_mapping(value, "coding fixture")
        return value

    @field_validator(
        "allowed_paths",
        "forbidden_paths",
        "protected_test_paths",
        "mutable_test_paths",
    )
    @classmethod
    def safe_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            if normalize_relative_path(path) is None:
                raise ValueError("coding fixture path must remain relative")
        return value

    def visible_files(self) -> dict[str, str]:
        return {**self.initial_files, **self.visible_tests}

    def effective_protected_test_paths(self) -> tuple[str, ...]:
        if self.protected_test_paths:
            return self.protected_test_paths
        return tuple(path for path in self.visible_tests if _looks_like_test_path(path))


def _validate_relative_file_mapping(value: Mapping[str, str], label: str) -> None:
    for path, content in value.items():
        relative = Path(path) if isinstance(path, str) else Path(".")
        if (
            not isinstance(path, str)
            or not path
            or relative.is_absolute()
            or PureWindowsPath(path).is_absolute()
            or ".." in relative.parts
            or ".." in PurePosixPath(path.replace("\\", "/")).parts
            or "\x00" in path
        ):
            raise ValueError(f"{label} contains an invalid path")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 65_536:
            raise ValueError(f"{label} file is too large")


def _looks_like_test_path(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/")).name
    return name.startswith("test_") or name.endswith("_test.py") or ".test." in name


def normalize_relative_path(path: str) -> str | None:
    """Normalize a repository-relative path, rejecting escapes and absolutes."""

    if not isinstance(path, str) or not path or "\x00" in path:
        return None
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return None
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def path_is_within(path: str, parent: str) -> bool:
    """Return true only for a path equal to or below a relative parent."""

    normalized_path = normalize_relative_path(path)
    normalized_parent = normalize_relative_path(parent)
    if normalized_path is None or normalized_parent is None:
        return False
    path_parts = PurePosixPath(normalized_path).parts
    parent_parts = PurePosixPath(normalized_parent).parts
    return path_parts[: len(parent_parts)] == parent_parts


def coding_fixture_directory(home: Path, version: str = "v2") -> Path:
    return home / "data" / "evaluations" / "model_benchmark" / version


def load_coding_fixtures(home: Path, version: str = "v2") -> tuple[CodingFixture, ...]:
    path = coding_fixture_directory(home, version) / "coding.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != f"coding-{version}":
        raise ValueError("coding fixture version mismatch")
    raw = payload.get("fixtures")
    if not isinstance(raw, list):
        raise ValueError("coding fixture list is missing")
    fixtures = tuple(CodingFixture.model_validate(item) for item in raw)
    ids = [fixture.id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError("coding fixture IDs must be unique")
    return fixtures


def coding_fixture_digest(home: Path, version: str = "v2") -> str:
    path = coding_fixture_directory(home, version) / "coding.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_fixture_set(home: Path, version: str = "v2") -> dict[str, object]:
    fixtures = load_coding_fixtures(home, version)
    return {
        "version": f"model-quality-{version}",
        "coding_version": f"coding-{version}",
        "sha256": coding_fixture_digest(home, version),
        "case_ids": [fixture.id for fixture in fixtures],
        "case_count": len(fixtures),
        "categories": sorted({fixture.category for fixture in fixtures}),
    }


def evaluate_file_assertions(root: Path, assertions: dict[str, CodingFileAssertion]) -> bool:
    """Evaluate hidden assertions outside the model-visible fixture workspace."""

    for relative, assertion in assertions.items():
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False
        exists = path.is_file()
        if assertion.exists is not None and exists is not assertion.exists:
            return False
        if not exists:
            if assertion.contains or assertion.excludes:
                return False
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False
        if any(fragment not in content for fragment in assertion.contains):
            return False
        if any(fragment in content for fragment in assertion.excludes):
            return False
    return True


def redact_coding_result(value: object) -> dict[str, Any] | None:
    """Keep coding evidence useful without retaining requests or model output."""

    if not isinstance(value, Mapping):
        return None
    case_results = value.get("case_results")
    redacted_cases: dict[str, dict[str, Any]] = {}
    if isinstance(case_results, Mapping):
        for case_id, raw in case_results.items():
            if not isinstance(case_id, str) or not isinstance(raw, Mapping):
                continue
            redacted_cases[case_id] = {
                key: raw.get(key)
                for key in (
                    "id",
                    "category",
                    "mode",
                    "verified_case_success",
                    "agent_completed",
                    "completed",
                    "tests_passed",
                    "internal_verification_observed",
                    "internal_verification_passed",
                    "evaluator_verification_passed",
                    "evaluator_exit_status",
                    "evaluator_stdout_digest",
                    "evaluator_stderr_digest",
                    "evaluator_output_truncated",
                    "evaluator_repository_state_digest",
                    "final_repository_state_current",
                    "final_repository_state_correct",
                    "hidden_acceptance_passed",
                    "safety_passed",
                    "syntax_failure",
                    "forbidden_modifications",
                    "unnecessary_modifications",
                    "tests_modified_improperly",
                    "test_tampering",
                    "user_changes_preserved",
                    "structured_output_valid",
                    "action_valid",
                    "structured_output_failures",
                    "action_validation_failures",
                    "verification_failures_before_success",
                    "recovered_after_verification_failure",
                    "recovered_after_failure",
                    "no_progress_intervened",
                    "no_progress_warning_count",
                    "no_progress_replan_count",
                    "no_progress_stop_count",
                    "no_progress_event_count",
                    "turns",
                    "tool_calls",
                    "replan_count",
                    "latency_seconds",
                    "first_token_latency_seconds",
                    "output_tokens_per_second",
                    "peak_rss_bytes",
                    "timed_out",
                    "failure_reason",
                    "interventions",
                )
            }
    categories = value.get("category_scores")
    return {
        key: value.get(key)
        for key in (
            "fixture_count",
            "fixture_pass_rate",
            "test_pass_rate",
            "syntax_or_compilation_failures",
            "forbidden_file_modifications",
            "timeout_rate",
            "evaluator_verification_unavailable_count",
            "unnecessary_change_rate",
            "agentic_case_count",
            "agentic_verified_success_rate",
            "verified_success_rate",
            "safety_failures",
            "recovery_success_count",
            "recovery_success_rate",
            "no_progress_intervention_count",
        )
    } | {
        "category_scores": categories if isinstance(categories, Mapping) else {},
        "case_results": redacted_cases,
    }
