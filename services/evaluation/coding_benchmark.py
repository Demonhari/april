"""Versioned, offline coding-benchmark fixture contracts.

The fixture description is trusted evaluation input, not model context.  Hidden
assertions stay in this evaluator-owned structure and are never materialized in
the temporary repository visible to the coding agent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodingFileAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exists: bool | None = None
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()


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
    )
    @classmethod
    def bounded_mapping(cls, value: dict[str, object]) -> dict[str, object]:
        if len(value) > 32:
            raise ValueError("coding fixture has too many files or assertions")
        return value

    @field_validator("initial_files", "visible_tests", "dirty_files")
    @classmethod
    def safe_file_contents(cls, value: dict[str, str]) -> dict[str, str]:
        for path, content in value.items():
            relative = Path(path) if isinstance(path, str) else Path(".")
            if (
                not isinstance(path, str)
                or not path
                or relative.is_absolute()
                or ".." in relative.parts
                or "\x00" in path
            ):
                raise ValueError("coding fixture contains an invalid path")
            if not isinstance(content, str) or len(content.encode("utf-8")) > 65_536:
                raise ValueError("coding fixture file is too large")
        return value

    @field_validator("allowed_paths", "forbidden_paths")
    @classmethod
    def safe_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            relative = Path(path)
            if not path or relative.is_absolute() or ".." in relative.parts or "\x00" in path:
                raise ValueError("coding fixture path must remain relative")
        return value

    def visible_files(self) -> dict[str, str]:
        return {**self.initial_files, **self.visible_tests}


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
                    "completed",
                    "tests_passed",
                    "final_repository_state_correct",
                    "syntax_failure",
                    "forbidden_modifications",
                    "unnecessary_modifications",
                    "tests_modified_improperly",
                    "user_changes_preserved",
                    "structured_output_valid",
                    "action_valid",
                    "recovered_after_failure",
                    "no_progress_intervened",
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
            "unnecessary_change_rate",
            "agentic_case_count",
            "agentic_verified_success_rate",
            "safety_failures",
            "recovery_success_count",
            "no_progress_intervention_count",
        )
    } | {
        "category_scores": categories if isinstance(categories, Mapping) else {},
        "case_results": redacted_cases,
    }
