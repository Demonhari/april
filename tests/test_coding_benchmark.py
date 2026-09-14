from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from apps.runner.commands.model_compare import _redacted_benchmark
from services.april_runtime.schemas import ChatResponse, Usage
from services.evaluation.coding_benchmark import (
    canonical_fixture_set,
    evaluate_file_assertions,
    load_coding_fixtures,
    redact_coding_result,
)
from services.evaluation.model_quality import (
    _run_agentic_coding_fixture,
    coding_fixture_ids,
    fixture_set_metadata,
)
from services.tool_worker.schemas import ToolWorkerResponse

ROOT = Path(__file__).resolve().parents[1]


def test_v2_fixture_set_is_typed_unique_and_category_complete() -> None:
    fixtures = load_coding_fixtures(ROOT, "v2")
    assert len(fixtures) == 30
    assert len({fixture.id for fixture in fixtures}) == len(fixtures)
    categories = {fixture.category for fixture in fixtures}
    assert {
        "basic_python",
        "debugging_failure_evidence",
        "multi_file_python",
        "repository_navigation",
        "regression_testing",
        "failure_recovery",
        "refactoring",
        "configuration",
        "typescript_javascript",
        "security_safety",
        "user_change_preservation",
        "structured_agent_output",
    } <= categories
    assert sum(fixture.mode == "one_shot" for fixture in fixtures) == 6
    assert sum(fixture.mode == "agentic" for fixture in fixtures) == 24


def test_v2_fixture_digest_and_metadata_are_stable() -> None:
    first = canonical_fixture_set(ROOT, "v2")
    second = canonical_fixture_set(ROOT, "v2")
    assert first == second
    assert first["case_count"] == len(first["case_ids"])
    metadata = fixture_set_metadata(ROOT)
    assert metadata["version"] == "model-quality-v2"
    assert metadata["coding"] == first
    assert coding_fixture_ids(ROOT) == tuple(first["case_ids"])


def test_hidden_acceptance_assertions_are_not_materialized_in_fixture_workspace(
    tmp_path: Path,
) -> None:
    fixture = next(
        item for item in load_coding_fixtures(ROOT, "v2") if item.id == "security_no_test_tampering"
    )
    for relative, content in fixture.visible_files().items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert not (tmp_path / "hidden_acceptance.json").exists()
    assert all(relative in fixture.visible_files() for relative in fixture.hidden_assertions)
    assert evaluate_file_assertions(tmp_path, fixture.hidden_assertions) is False


def test_behavioral_hidden_assertions_allow_non_reference_implementation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "solution.py"
    path.write_text(
        "def clamp(value, low, high):\n"
        "    if low > high:\n"
        "        raise ValueError('invalid bounds')\n"
        "    return max(low, min(value, high))\n",
        encoding="utf-8",
    )
    fixture = next(
        item for item in load_coding_fixtures(ROOT, "v2") if item.id == "basic_clamp_validation"
    )
    assert evaluate_file_assertions(tmp_path, fixture.hidden_assertions)


def test_redacted_coding_report_keeps_case_metrics_only() -> None:
    redacted = redact_coding_result(
        {
            "fixture_count": 1,
            "fixture_pass_rate": 1.0,
            "category_scores": {"basic_python": {"case_count": 1, "verified_success_rate": 1.0}},
            "case_results": {
                "case": {
                    "category": "basic_python",
                    "completed": True,
                    "request": "private request",
                    "model_output": "private output",
                }
            },
        }
    )
    assert redacted is not None
    blob = json.dumps(redacted)
    assert "private request" not in blob
    assert "private output" not in blob
    assert redacted["category_scores"] == {
        "basic_python": {"case_count": 1, "verified_success_rate": 1.0}
    }


def test_v2_report_redaction_does_not_reintroduce_raw_quality_content() -> None:
    report = _redacted_benchmark(
        {
            "model_id": "candidate",
            "role": "coding",
            "model_basename": "candidate.gguf",
            "model_sha256": "a" * 64,
            "model_size": 4,
            "quality": {
                "coding": {
                    "case_results": {
                        "case": {
                            "request": "secret request",
                            "output": "secret output",
                        }
                    }
                }
            },
            "runs": [],
        }
    )
    blob = json.dumps(report)
    assert "secret request" not in blob
    assert "secret output" not in blob


@pytest.mark.asyncio
async def test_coding_comparison_uses_identical_v2_case_ids_without_activation(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copytree(
        ROOT / "configs",
        settings_tmp.home / "configs",
    )
    shutil.copytree(
        ROOT / "data" / "evaluations" / "model_benchmark",
        settings_tmp.home / "data" / "evaluations" / "model_benchmark",
    )

    async def fake_benchmark(*_args: Any, model_id: str, **_kwargs: Any) -> dict[str, Any]:
        case_ids = coding_fixture_ids(settings_tmp.home)
        return {
            "model_id": model_id,
            "role": "coding",
            "model_basename": "candidate.gguf",
            "model_sha256": "a" * 64,
            "model_size": 4,
            "passed": True,
            "simulated": False,
            "coding_fixture_pass_rate": 1.0,
            "structured_json_reliability": 1.0,
            "quality": {
                "coding": {
                    "fixture_count": len(case_ids),
                    "fixture_pass_rate": 1.0,
                    "agentic_verified_success_rate": 1.0,
                    "test_pass_rate": 1.0,
                    "safety_failures": 0,
                    "case_results": {
                        case_id: {"id": case_id, "completed": True} for case_id in case_ids
                    },
                }
            },
            "runs": [],
        }

    monkeypatch.setattr(
        "apps.runner.commands.model_compare.run_model_utility_job",
        fake_benchmark,
    )
    from apps.runner.commands.model_compare import _run_coding_comparison

    report = await _run_coding_comparison(
        settings_tmp,
        ("april-brain", "april-coding"),
    )
    expected = list(coding_fixture_ids(settings_tmp.home))
    assert report["fixture_set"]["coding_cases"] == expected
    assert all(list(score["coding"]["case_results"]) == expected for score in report["scores"])
    assert report["automatic_activation_performed"] is False


@pytest.mark.asyncio
async def test_agentic_fixture_uses_structured_loop_and_machine_verification(
    settings_tmp: Any,
) -> None:
    fixture = next(
        item for item in load_coding_fixtures(ROOT, "v2") if item.id == "debug_async_timeout"
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **_: Any) -> ChatResponse:
            self.calls += 1
            outputs = [
                {
                    "type": "tool_request",
                    "tool": "read_file",
                    "args": {"path": "worker.py"},
                },
                {
                    "type": "tool_request",
                    "tool": "test_runner",
                    "args": {"argv": ["pytest", "-q"]},
                },
                {
                    "type": "tool_request",
                    "tool": "test_runner",
                    "args": {"argv": ["pytest", "-q"]},
                },
                {"type": "final_answer", "message": "verified"},
            ]
            return ChatResponse(
                request_id=f"fake-{self.calls}",
                model_id="fixture-model",
                content=json.dumps(outputs[min(self.calls - 1, len(outputs) - 1)]),
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

    class FakeWorker:
        async def execute(self, **kwargs: Any) -> ToolWorkerResponse:
            project_root = Path(kwargs["project_root"])
            args = kwargs["args"]
            completed = subprocess.run(
                list(args["argv"]),
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return ToolWorkerResponse(
                request_id=str(kwargs["request_id"]),
                ok=completed.returncode == 0,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                status="completed",
                data={"truncated": False},
            )

    result = await _run_agentic_coding_fixture(
        FakeRuntime(),
        "fixture-model",
        fixture,
        coding_root=settings_tmp.home / "coding-benchmark",
        tool_worker=FakeWorker(),  # type: ignore[arg-type]
        settings=settings_tmp,
    )
    assert result["mode"] == "agentic"
    assert result["tests_passed"] is True
    assert result["final_repository_state_correct"] is True
    assert result["action_valid"] is True


@pytest.mark.parametrize("path", ["../escape", "/absolute"])
def test_fixture_path_validation_is_project_relative(path: str) -> None:
    from services.evaluation.coding_benchmark import CodingFixture

    with pytest.raises(ValueError, match="coding fixture contains an invalid path"):
        CodingFixture(
            id="invalid",
            category="basic_python",
            mode="agentic",
            request="test",
            initial_files={path: "x"},
        )
