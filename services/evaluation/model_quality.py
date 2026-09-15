from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from agents.coding.agent import coding_agent
from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.april_runtime.colibri_backend import ColibriBackend
from services.april_runtime.schemas import (
    ChatMessage,
    ChatResponse,
    GenerationOptions,
    ResponseFormat,
    Usage,
)
from services.brain.agent_loop import StructuredAgentLoop
from services.brain.deterministic_router import DeterministicRouter
from services.brain.model_routing import infer_model_route
from services.brain.repository_state import RepositoryState
from services.brain.route_contract import RouteCompiler
from services.brain.structured_output import grammar_safe_json_schema
from services.brain.task_contract import task_contract_for_request
from services.evaluation.coding_benchmark import (
    CodingFileAssertion,
    CodingFixture,
    canonical_fixture_set,
    coding_fixture_directory,
    evaluate_file_assertions,
    load_coding_fixtures,
    normalize_relative_path,
    path_is_within,
)
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
from services.tool_worker.client import ToolWorkerClient, ToolWorkerUnavailable
from skills.registry import default_registry

FIXTURE_SET_VERSION = "model-quality-v2"
_FILES = ("routing.json", "strict_json.json", "coding.json", "context.json")


def fixture_directory(home: Path, version: str | None = None) -> Path:
    selected = version or _selected_fixture_version(home)
    return coding_fixture_directory(home, selected)


def _selected_fixture_version(home: Path) -> str:
    v2 = coding_fixture_directory(home, "v2") / "coding.json"
    return "v2" if v2.is_file() else "v1"


def _fixture_path(home: Path, filename: str, version: str) -> Path:
    preferred = coding_fixture_directory(home, version) / filename
    if preferred.is_file():
        return preferred
    return coding_fixture_directory(home, "v1") / filename


def fixture_set_metadata(home: Path) -> dict[str, Any]:
    selected = _selected_fixture_version(home)
    digest = hashlib.sha256()
    installed = True
    versions: dict[str, str] = {}
    component_paths: dict[str, str] = {}
    for name in _FILES:
        path = _fixture_path(home, name, selected)
        try:
            payload = path.read_bytes()
            decoded = json.loads(payload)
        except (OSError, json.JSONDecodeError):
            installed = False
            continue
        digest.update(f"{selected}/{name}".encode())
        digest.update(b"\0")
        digest.update(payload)
        component_paths[name] = (
            f"{selected}/{name}" if path.parent.name == selected else "v1/" + name
        )
        if isinstance(decoded, dict) and isinstance(decoded.get("version"), str):
            versions[name] = decoded["version"]
    coding = None
    if installed and selected == "v2":
        try:
            coding = canonical_fixture_set(home, selected)
        except (OSError, ValueError, TypeError):
            installed = False
    elif installed:
        coding = {
            "version": "coding-v1",
            "sha256": hashlib.sha256(
                _fixture_path(home, "coding.json", selected).read_bytes()
            ).hexdigest(),
            "case_ids": list(coding_fixture_ids(home, selected)),
            "case_count": len(coding_fixture_ids(home, selected)),
            "categories": ["basic_python"],
        }
    return {
        "version": FIXTURE_SET_VERSION if selected == "v2" else "model-quality-v1",
        "sha256": digest.hexdigest() if installed else None,
        "installed": installed,
        "component_versions": versions,
        "component_paths": component_paths,
        "coding": coding,
    }


def coding_fixture_ids(home: Path, version: str | None = None) -> tuple[str, ...]:
    """Return the exact coding cases installed in the versioned fixture set."""

    selected = version or _selected_fixture_version(home)
    try:
        return tuple(fixture.id for fixture in load_coding_fixtures(home, selected))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    try:
        data = json.loads(_fixture_path(home, "coding.json", selected).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    fixtures = data.get("fixtures") if isinstance(data, dict) else None
    if not isinstance(fixtures, list):
        return ()
    return tuple(
        str(item["id"])
        for item in fixtures
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


def verified_case_success(measurement: Mapping[str, Any]) -> bool:
    """Canonical evaluator-owned definition of a passing coding case."""

    return (
        bool(measurement.get("agent_completed", measurement.get("completed")))
        and bool(measurement.get("evaluator_verification_passed", measurement.get("tests_passed")))
        and bool(measurement.get("final_repository_state_correct"))
        and bool(measurement.get("safety_passed"))
        and bool(measurement.get("user_changes_preserved", True))
        and not bool(
            measurement.get("test_tampering", measurement.get("tests_modified_improperly"))
        )
        and not bool(measurement.get("timed_out"))
    )


class ColibriEvaluationClient:
    """Runtime-shaped evaluator client backed directly by local Colibri."""

    supports_seed = False

    def __init__(self, *, backend: ColibriBackend, model_id: str) -> None:
        self.backend = backend
        self.model_id = model_id

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        response_format: ResponseFormat | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        del model_id
        selected = options or GenerationOptions()
        result = await self.backend.generate_messages(
            "",
            messages=messages,
            temperature=selected.temperature if selected.temperature is not None else 0.0,
            max_output_tokens=selected.max_output_tokens or 256,
            top_p=selected.top_p,
            stop=selected.stop,
            response_format=response_format,
            disable_thinking=(
                None if selected.enable_thinking is None else not selected.enable_thinking
            ),
        )
        return ChatResponse(
            request_id=request_id or f"colibri-eval-{self.model_id}",
            model_id=self.model_id,
            content=result.text,
            usage=Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
            ),
        )


def _evaluation_options(client: Any, **values: Any) -> GenerationOptions:
    if not getattr(client, "supports_seed", True):
        values.pop("seed", None)
    return GenerationOptions(**values)


async def evaluate_model_quality(
    settings: AprilSettings,
    *,
    runtime_url: str,
    runtime_token: str | None,
    model_id: str,
    coding_root: Path,
    tool_worker: ToolWorkerClient | None,
    client: Any | None = None,
    fixture_home: Path | None = None,
) -> dict[str, Any]:
    fixture_source = fixture_home or settings.home
    metadata = fixture_set_metadata(fixture_source)
    if not metadata["installed"]:
        raise RuntimeError("model_quality_fixtures_unavailable")
    client = client or RuntimeClient(runtime_url, token=runtime_token, timeout=180.0)
    selected = _selected_fixture_version(fixture_source)
    routing = await _routing(
        client,
        model_id,
        _load(_fixture_path(fixture_source, "routing.json", selected)),
        home=settings.home,
    )
    strict_json = await _strict_json(
        client, model_id, _load(_fixture_path(fixture_source, "strict_json.json", selected))
    )
    coding = await _coding(
        client,
        model_id,
        _load(_fixture_path(fixture_source, "coding.json", selected)),
        coding_root=coding_root,
        tool_worker=tool_worker,
        settings=settings,
        fixture_home=fixture_source,
    )
    context = await _context(
        client, model_id, _load(_fixture_path(fixture_source, "context.json", selected))
    )
    return {
        "fixture_set": metadata,
        "routing": routing,
        "strict_json": strict_json,
        "coding": coding,
        "context": context,
        "routing_accuracy": routing["aggregate_accuracy"],
        "strict_json_first_pass_reliability": strict_json["first_pass_schema_reliability"],
        "structured_json_reliability": strict_json["final_schema_reliability"],
        "coding_fixture_pass_rate": coding["fixture_pass_rate"],
        "context_handling_reliability": context["success_rate"],
    }


async def _routing(
    client: Any,
    model_id: str,
    data: Mapping[str, Any],
    *,
    home: Path,
) -> dict[str, Any]:
    fixtures = _fixtures(data)
    passed = 0
    invalid = 0
    repaired = 0
    wrong = 0
    semantic_passed = 0
    policy_passed = 0
    categories: dict[str, list[bool]] = {}
    policy_categories: dict[str, list[bool]] = {}
    deterministic_matches = 0
    deterministic_count = 0
    deterministic = DeterministicRouter()
    compiler = RouteCompiler.from_home(home)
    for fixture in fixtures:
        category = str(fixture["category"])
        first_pass = False
        repair_used = False
        try:
            outcome = await infer_model_route(
                client,
                model_id=model_id,
                message=str(fixture["request"]),
                history=None,
                request_id=f"benchmark-route-{fixture['id']}",
                compiler=compiler,
                max_output_tokens=192,
            )
            repair_used = outcome.repair_attempted
            first_pass = not repair_used and outcome.decision is not None
            semantic_ok = outcome.proposal is not None and _proposal_permitted(
                outcome.proposal.operation, outcome.proposal.tool_class, fixture
            )
            route_ok = outcome.decision is not None and _route_permitted(
                outcome.decision.model_dump(), fixture
            )
        except Exception:
            semantic_ok = False
            route_ok = False
            invalid += 1
        semantic_passed += int(semantic_ok)
        policy_passed += int(route_ok)
        if repair_used:
            repaired += 1
        if first_pass and not route_ok:
            wrong += 1
        passed += int(semantic_ok)
        categories.setdefault(category, []).append(semantic_ok)
        policy_categories.setdefault(category, []).append(route_ok)
        deterministic_result = deterministic.route(str(fixture["request"]))
        if deterministic_result is not None:
            deterministic_count += 1
            deterministic_matches += int(
                _route_permitted(deterministic_result.decision.model_dump(), fixture)
            )
    return {
        "fixture_count": len(fixtures),
        "aggregate_accuracy": passed / len(fixtures) if fixtures else 0.0,
        "passed": passed,
        "semantic_passed": semantic_passed,
        "policy_route_passed": policy_passed,
        "policy_route_accuracy": policy_passed / len(fixtures) if fixtures else 0.0,
        "invalid_output_count": invalid,
        "repair_counted_as_failure": repaired,
        "wrong_route_count": wrong,
        "per_category_accuracy": {
            category: sum(outcomes) / len(outcomes)
            for category, outcomes in sorted(categories.items())
        },
        "policy_route_per_category_accuracy": {
            category: sum(outcomes) / len(outcomes)
            for category, outcomes in sorted(policy_categories.items())
        },
        "model_router_only": True,
        "deterministic_router": {
            "matched_fixture_count": deterministic_count,
            "permitted_route_count": deterministic_matches,
            "included_in_model_accuracy": False,
        },
        "scoring_policy": str(data.get("scoring_policy", "")),
    }


async def _strict_json(
    client: Any,
    model_id: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    fixtures = _fixtures(data)
    counts: Counter[str] = Counter()
    first_schema = 0
    final_schema = 0
    for fixture in fixtures:
        schema = _mapping(fixture.get("schema"))
        response = await client.chat(
            model_id=model_id,
            messages=[
                ChatMessage(
                    role="system",
                    content="Return exactly one JSON object and no prose or markdown.",
                ),
                ChatMessage(role="user", content=str(fixture["prompt"])),
            ],
            options=_evaluation_options(client, temperature=0.0, max_output_tokens=256, seed=11),
            response_format=ResponseFormat(
                type="json_object", json_schema=grammar_safe_json_schema(dict(schema))
            ),
            request_id=f"benchmark-json-{fixture['id']}",
        )
        parsed, errors = _validate_json(response.content, schema)
        if parsed is not None:
            counts["valid_json_first_attempt"] += 1
        for error in errors:
            counts[error] += 1
        if parsed is not None and not errors:
            first_schema += 1
            final_schema += 1
            continue
        repair = await client.chat(
            model_id=model_id,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "Repair the value into exactly one object matching the supplied schema."
                    ),
                ),
                ChatMessage(role="user", content=response.content),
            ],
            options=_evaluation_options(client, temperature=0.0, max_output_tokens=256, seed=11),
            response_format=ResponseFormat(
                type="json_object", json_schema=grammar_safe_json_schema(dict(schema))
            ),
            request_id=f"benchmark-json-repair-{fixture['id']}",
        )
        repaired, repair_errors = _validate_json(repair.content, schema)
        if repaired is not None and not repair_errors:
            counts["repaired_output"] += 1
            final_schema += 1
        else:
            counts["invalid_after_repair"] += 1
    total = len(fixtures)
    return {
        "fixture_count": total,
        "valid_json_first_attempt": counts["valid_json_first_attempt"],
        "schema_valid_first_attempt": first_schema,
        "repaired_output": counts["repaired_output"],
        "invalid_after_repair": counts["invalid_after_repair"],
        "unsupported_fields": counts["unsupported_fields"],
        "missing_required_fields": counts["missing_required_fields"],
        "incorrect_enum_values": counts["incorrect_enum_values"],
        "first_pass_json_reliability": (
            counts["valid_json_first_attempt"] / total if total else 0.0
        ),
        "first_pass_schema_reliability": first_schema / total if total else 0.0,
        "final_schema_reliability": final_schema / total if total else 0.0,
    }


async def _coding(
    client: Any,
    model_id: str,
    data: Mapping[str, Any],
    *,
    coding_root: Path,
    tool_worker: ToolWorkerClient | None,
    settings: AprilSettings,
    fixture_home: Path,
) -> dict[str, Any]:
    fixtures = _fixtures(data)
    if tool_worker is None:
        return {
            "fixture_count": len(fixtures),
            "fixture_pass_rate": None,
            "unavailable_reason": "tool_worker_unavailable",
            "executed_only_through_tool_worker": True,
        }
    counts: Counter[str] = Counter()
    measurements: dict[str, dict[str, Any]] = {}
    selected_version = _selected_fixture_version(fixture_home)
    typed_fixtures = (
        {fixture.id: fixture for fixture in load_coding_fixtures(fixture_home, selected_version)}
        if selected_version == "v2"
        else {}
    )
    for fixture in fixtures:
        typed = typed_fixtures.get(str(fixture.get("id")))
        if typed is not None and typed.mode == "agentic":
            measurement = await _run_agentic_coding_fixture(
                client,
                model_id,
                typed,
                coding_root=coding_root,
                tool_worker=tool_worker,
                settings=settings,
            )
            measurement["id"] = typed.id
            measurements[typed.id] = measurement
            counts["passed"] += int(verified_case_success(measurement))
            counts["test_pass"] += int(measurement["evaluator_verification_passed"])
            counts["syntax_or_compilation_failures"] += int(measurement["syntax_failure"])
            counts["forbidden_file_modifications"] += int(measurement["forbidden_modifications"])
            counts["unnecessary_changes"] += int(measurement["unnecessary_modifications"])
            counts["safety_failures"] += int(not measurement["safety_passed"])
            counts["timeouts"] += int(measurement["timed_out"])
            counts["evaluator_unavailable"] += int(measurement["evaluator_exit_status"] is None)
            continue
        candidate_file = (
            typed.candidate_file if typed is not None else fixture.get("candidate_file")
        )
        request = typed.request if typed is not None else fixture.get("instruction")
        fixture_files = (
            typed.visible_files()
            if typed is not None
            else (fixture.get("fixture_files") or fixture.get("initial_files", {}))
        )
        test_argv = list(typed.verification_argv) if typed is not None else fixture.get("test_argv")
        if not isinstance(candidate_file, str) or not isinstance(request, str):
            raise ValueError("invalid_coding_fixture")
        response = await client.chat(
            model_id=model_id,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "Return exactly one JSON object with string fields filename and content. "
                        "Do not return a patch, markdown, or explanation."
                    ),
                ),
                ChatMessage(role="user", content=request),
            ],
            options=_evaluation_options(client, temperature=0.0, max_output_tokens=768, seed=13),
            response_format=ResponseFormat(
                type="json_object",
                json_schema=grammar_safe_json_schema(
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["filename", "content"],
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    }
                ),
            ),
            request_id=f"benchmark-code-{fixture['id']}",
        )
        try:
            candidate = json.loads(response.content)
            if (
                not isinstance(candidate, dict)
                or candidate.get("filename") != candidate_file
                or not isinstance(candidate.get("content"), str)
            ):
                raise ValueError("invalid_candidate")
            project = coding_root / str(fixture["id"])
            project.mkdir(parents=True, exist_ok=True)
            if not isinstance(fixture_files, dict):
                raise ValueError("invalid_fixture_files")
            if typed is None:
                visible_tests = fixture.get("visible_tests")
                if isinstance(visible_tests, dict):
                    fixture_files = {**fixture_files, **visible_tests}
            if not isinstance(test_argv, list):
                raise ValueError("invalid_test_argv")
            tool_result = await tool_worker.execute(
                request_id=f"benchmark-code-exec-{fixture['id']}",
                operation="benchmark_fixture",
                project_root=project,
                args={
                    "fixture_files": fixture_files,
                    "candidate_file": candidate_file,
                    "candidate_content": candidate["content"],
                    "expected_content": fixture.get("expected_content") if typed is None else None,
                    "test_argv": test_argv,
                },
                timeout_seconds=30.0,
                max_stdout_bytes=8_192,
                max_stderr_bytes=8_192,
            )
        except Exception:
            counts["syntax_or_compilation_failures"] += 1
            measurements[str(fixture["id"])] = {
                "id": str(fixture["id"]),
                "category": str(fixture.get("category", "basic_python")),
                "mode": "one_shot",
                "agent_completed": False,
                "completed": False,
                "tests_passed": False,
                "evaluator_verification_passed": False,
                "final_repository_state_correct": False,
                "safety_passed": False,
                "syntax_failure": True,
                "forbidden_modifications": 0,
                "unnecessary_modifications": 0,
                "structured_output_valid": False,
                "action_valid": False,
                "turns": 1,
                "tool_calls": 0,
                "replan_count": 0,
                "latency_seconds": 0.0,
                "timed_out": False,
                "failure_reason": "invalid_candidate_or_fixture",
                "verified_case_success": False,
            }
            continue
        counts["test_pass"] += int(tool_result.returncode == 0)
        failure = tool_result.failure_code or ""
        counts["timeouts"] += int(failure == "timeout")
        counts["evaluator_unavailable"] += int(tool_result.returncode is None)
        details = tool_result.data
        counts["forbidden_file_modifications"] += int(
            bool(details.get("forbidden_file_modification"))
        )
        counts["unnecessary_changes"] += int(bool(details.get("unnecessary_change")))
        counts["syntax_or_compilation_failures"] += int(
            bool(details.get("syntax_or_compilation_failure"))
        )
        hidden_ok = (
            evaluate_file_assertions(project, typed.hidden_assertions)
            if typed is not None
            else True
        )
        hidden_tests_ok = (
            await _run_hidden_tests(tool_worker, typed, project) if typed is not None else True
        )
        expected_ok = (
            evaluate_file_assertions(
                project,
                {
                    path: CodingFileAssertion(contains=fragments)
                    for path, fragments in typed.expected_file_contains.items()
                },
            )
            if typed is not None
            else True
        )
        hidden_ok = hidden_ok and hidden_tests_ok and expected_ok
        counts["safety_failures"] += int(bool(details.get("forbidden_file_modification")))
        measurement = {
            "id": str(fixture["id"]),
            "category": str(fixture.get("category", "basic_python")),
            "mode": "one_shot",
            "agent_completed": bool(tool_result.ok) and hidden_ok,
            "completed": bool(tool_result.ok) and hidden_ok,
            "tests_passed": tool_result.returncode == 0,
            "evaluator_verification_passed": tool_result.returncode == 0,
            "final_repository_state_correct": hidden_ok,
            "safety_passed": not bool(details.get("forbidden_file_modification")),
            "syntax_failure": bool(details.get("syntax_or_compilation_failure")),
            "forbidden_modifications": int(bool(details.get("forbidden_file_modification"))),
            "unnecessary_modifications": int(bool(details.get("unnecessary_change"))),
            "structured_output_valid": True,
            "action_valid": True,
            "turns": 1,
            "tool_calls": 1,
            "replan_count": 0,
            "latency_seconds": 0.0,
            "timed_out": failure == "timeout",
            "failure_reason": failure or None,
        }
        measurement["verified_case_success"] = verified_case_success(measurement)
        counts["passed"] += int(measurement["verified_case_success"])
        measurements[str(fixture["id"])] = measurement
    total = len(fixtures)
    agentic = [item for item in measurements.values() if item["mode"] == "agentic"]
    category_scores: dict[str, dict[str, float | int]] = {}
    for category in sorted({str(item.get("category", "")) for item in measurements.values()}):
        items = [item for item in measurements.values() if item.get("category") == category]
        category_scores[category] = {
            "case_count": len(items),
            "verified_success_rate": sum(bool(item.get("verified_case_success")) for item in items)
            / len(items),
        }
    return {
        "fixture_count": total,
        "fixture_pass_rate": counts["passed"] / total if total else 0.0,
        "test_pass_rate": counts["test_pass"] / total if total else 0.0,
        "syntax_or_compilation_failures": counts["syntax_or_compilation_failures"],
        "forbidden_file_modifications": counts["forbidden_file_modifications"],
        "timeout_rate": counts["timeouts"] / total if total else 0.0,
        "evaluator_verification_unavailable_count": counts["evaluator_unavailable"],
        "unnecessary_change_rate": counts["unnecessary_changes"] / total if total else 0.0,
        "executed_only_through_tool_worker": True,
        "agentic_case_count": len(agentic),
        "agentic_verified_success_rate": (
            sum(bool(item.get("verified_case_success")) for item in agentic) / len(agentic)
            if agentic
            else None
        ),
        "category_scores": category_scores,
        "case_results": measurements,
        "safety_failures": counts["safety_failures"],
        "recovery_success_count": sum(
            bool(item.get("recovered_after_failure")) for item in measurements.values()
        ),
        "recovery_success_rate": (
            sum(
                bool(item.get("recovered_after_verification_failure"))
                for item in measurements.values()
            )
            / total
            if total
            else 0.0
        ),
        "no_progress_intervention_count": sum(
            bool(item.get("no_progress_intervened")) for item in measurements.values()
        ),
    }


async def _run_agentic_coding_fixture(
    client: Any,
    model_id: str,
    fixture: CodingFixture,
    *,
    coding_root: Path,
    tool_worker: ToolWorkerClient,
    settings: AprilSettings,
) -> dict[str, Any]:
    """Run one fixture under its complete evaluator-owned timeout."""

    try:
        async with asyncio.timeout(fixture.timeout_seconds):
            return await _run_agentic_coding_fixture_inner(
                client,
                model_id,
                fixture,
                coding_root=coding_root,
                tool_worker=tool_worker,
                settings=settings,
            )
    except TimeoutError:
        return _timed_out_measurement(fixture)


def _timed_out_measurement(fixture: CodingFixture) -> dict[str, Any]:
    return {
        "category": fixture.category,
        "mode": fixture.mode,
        "agent_completed": False,
        "completed": False,
        "tests_passed": False,
        "evaluator_verification_passed": False,
        "final_repository_state_correct": False,
        "safety_passed": False,
        "forbidden_modifications": 0,
        "unnecessary_modifications": 0,
        "tests_modified_improperly": False,
        "test_tampering": False,
        "user_changes_preserved": False,
        "structured_output_valid": None,
        "action_valid": None,
        "structured_output_failures": None,
        "action_validation_failures": None,
        "recovered_after_verification_failure": False,
        "recovered_after_failure": False,
        "no_progress_intervened": False,
        "no_progress_warning_count": 0,
        "no_progress_replan_count": 0,
        "no_progress_stop_count": 0,
        "turns": 0,
        "tool_calls": 0,
        "replan_count": 0,
        "latency_seconds": None,
        "first_token_latency_seconds": None,
        "output_tokens_per_second": None,
        "peak_rss_bytes": None,
        "timed_out": True,
        "failure_reason": "fixture_timeout",
        "interventions": 0,
        "verified_case_success": False,
    }


async def _run_agentic_coding_fixture_inner(
    client: Any,
    model_id: str,
    fixture: CodingFixture,
    *,
    coding_root: Path,
    tool_worker: ToolWorkerClient,
    settings: AprilSettings,
) -> dict[str, Any]:
    """Run one case through StructuredAgentLoop and the trusted Tool Worker."""

    started = time.monotonic()
    project = coding_root / fixture.id
    project.mkdir(parents=True, exist_ok=True)
    _materialize_fixture_repository(project, fixture)
    database = Database(coding_root / f".{fixture.id}.db")
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    registry = default_registry()
    approvals = ApprovalStore(
        database,
        AuditLogger(coding_root / f".{fixture.id}.audit.jsonl"),
        expiry_seconds=300,
    )
    executor = ToolExecutionService(
        settings=settings,
        memory=memory,
        tool_registry=registry,
        permission_engine=PermissionEngine(registry),
        approvals=approvals,
        tool_worker=tool_worker,
    )
    previous_roots = os.environ.get("APRIL_ALLOWED_FILESYSTEM_ROOTS")
    os.environ["APRIL_ALLOWED_FILESYSTEM_ROOTS"] = str(project)
    from april_common.settings import reset_settings_cache

    reset_settings_cache()
    try:
        project_record = await memory.add_project(str(project))
        context = await executor.context(
            request_id=f"benchmark-{fixture.id}",
            conversation_id=await memory.create_conversation(project_id=project_record.id),
            actor="benchmark",
            agent_id="coding_agent",
            project_id=project_record.id,
            source="verify",
        )
        base_agent = coding_agent()
        agent = BaseAgent(base_agent.config.model_copy(update={"model_id": model_id}))
        contract = task_contract_for_request(
            run_id=f"benchmark-{fixture.id}",
            user_goal=fixture.request,
            agent_name=agent.name,
            agent_tools=set(agent.config.allowed_tools),
            project_id=project_record.id,
            project_root=str(project),
            intent=fixture.task_type,
            permission_level=3,
            risk_level="code_write",
            allowed_scope=fixture.allowed_paths,
        )
        loop = StructuredAgentLoop(runtime_client=client, tool_executor=executor, memory=memory)
        forbidden_baseline = _restricted_tree_snapshot(project, fixture.forbidden_paths)
        result = await loop.run(
            agent=agent,
            message=fixture.request,
            context=context,
            request_id=f"benchmark-{fixture.id}",
            task_contract=contract,
            run_metadata={"benchmark_fixture_id": fixture.id},
        )
        approvals_used = 0
        while result.status == "pending_approval" and approvals_used < 12:
            pending = result.pending_approval or {}
            approval_id = str(pending.get("approval_id", ""))
            suspended = await memory.get_suspended_agent_run_by_approval(approval_id)
            if suspended is None:
                break
            outcome = await executor.execute_approved(
                approval_id=approval_id,
                actor="benchmark",
                request_id=f"benchmark-approval-{fixture.id}-{approvals_used}",
            )
            if outcome.result is None or not outcome.result.ok:
                break
            resumed_context = await executor.context(
                request_id=f"benchmark-resume-{fixture.id}-{approvals_used}",
                conversation_id=suspended.conversation_id,
                actor="benchmark",
                agent_id=suspended.agent,
                project_id=suspended.project_id,
                source="approval",
                approval_id=approval_id,
            )
            result = await loop.resume(
                suspended=suspended,
                agent=agent,
                context=resumed_context,
                tool_result=outcome.result,
                request_id=f"benchmark-resume-{fixture.id}-{approvals_used}",
            )
            approvals_used += 1
        iterations = await database.fetchall(
            "SELECT state, tool_request_json, tool_result_json "
            "FROM agent_iterations WHERE run_id IN "
            "(SELECT id FROM agent_runs WHERE agent = 'coding_agent')"
        )
        tool_calls = await database.fetchall("SELECT id FROM tool_calls")
        run_rows = await database.fetchall(
            "SELECT metadata_json FROM agent_runs WHERE agent = 'coding_agent'"
        )
        replan_count = 0
        for row in run_rows:
            try:
                metadata = json.loads(str(row["metadata_json"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            control = metadata.get("run_control") if isinstance(metadata, dict) else None
            if isinstance(control, dict):
                replan_count = max(replan_count, int(control.get("replan_attempts", 0)))
        internal_verification_passed = any(_tool_iteration_passed(row) for row in iterations)
        final_state = RepositoryState.capture(project, project_id=fixture.id)
        evaluator = await _run_evaluator_verification(
            tool_worker,
            fixture,
            project,
        )
        after_evaluator_state = RepositoryState.capture(project, project_id=fixture.id)
        evaluator_state_current = final_state.digest == after_evaluator_state.digest
        evaluator_passed = bool(evaluator["passed"]) and evaluator_state_current
        hidden_ok = evaluate_file_assertions(project, fixture.hidden_assertions)
        expected_ok = evaluate_file_assertions(
            project,
            {
                path: CodingFileAssertion(contains=fragments)
                for path, fragments in fixture.expected_file_contains.items()
            },
        )
        hidden_tests_ok = await _run_hidden_tests(tool_worker, fixture, project)
        modified = _changed_relative_paths(project)
        dirty_preserved = all(
            (project / path).is_file() and (project / path).read_text(encoding="utf-8") == content
            for path, content in fixture.dirty_files.items()
        )
        allowed = {
            normalized
            for path in (fixture.allowed_paths or tuple(fixture.visible_files()))
            if (normalized := normalize_relative_path(path)) is not None
        }
        forbidden_count = sum(
            any(path_is_within(path, forbidden) for forbidden in fixture.forbidden_paths)
            for path in modified
        )
        forbidden_count += sum(
            before != after
            for path, before in forbidden_baseline.items()
            for after in [_restricted_tree_digest(project / path)]
        )
        unrelated_count = len(
            {
                path
                for path in modified
                if path not in fixture.dirty_files
                and not any(path_is_within(path, allowed_path) for allowed_path in allowed)
            }
        )
        protected = set(fixture.effective_protected_test_paths())
        mutable = set(fixture.mutable_test_paths)
        test_tampering = any(
            path_is_within(path, protected_parent)
            and not any(path_is_within(path, mutable_parent) for mutable_parent in mutable)
            for path in modified
            for protected_parent in protected
        )
        verification_failures, recovered = _verification_recovery_metrics(iterations)
        control = _run_control_metrics(run_rows)
        structured_failures, action_failures = _iteration_validation_metrics(iterations)
        safety_passed = forbidden_count == 0 and dirty_preserved and not test_tampering
        measurement = {
            "category": fixture.category,
            "mode": fixture.mode,
            "agent_completed": result.status == "ok",
            "completed": result.status == "ok",
            "tests_passed": evaluator_passed,
            "internal_verification_observed": bool(iterations),
            "internal_verification_passed": internal_verification_passed,
            "evaluator_verification_passed": evaluator_passed,
            "evaluator_verification_argv": list(fixture.verification_argv),
            "evaluator_exit_status": evaluator["exit_status"],
            "evaluator_stdout_digest": evaluator["stdout_digest"],
            "evaluator_stderr_digest": evaluator["stderr_digest"],
            "evaluator_output_truncated": evaluator["truncated"],
            "evaluator_repository_state_digest": final_state.digest,
            "final_repository_state_current": evaluator_state_current,
            "final_repository_state_correct": expected_ok and hidden_ok and hidden_tests_ok,
            "hidden_acceptance_passed": expected_ok and hidden_ok and hidden_tests_ok,
            "safety_passed": safety_passed,
            "syntax_failure": False,
            "forbidden_modifications": forbidden_count,
            "unnecessary_modifications": unrelated_count,
            "tests_modified_improperly": test_tampering,
            "test_tampering": test_tampering,
            "user_changes_preserved": dirty_preserved,
            "structured_output_valid": (None if not iterations else structured_failures == 0),
            "action_valid": None if not iterations else action_failures == 0,
            "structured_output_failures": structured_failures,
            "action_validation_failures": action_failures,
            "verification_failures_before_success": verification_failures,
            "recovered_after_verification_failure": recovered,
            "recovered_after_failure": recovered,
            "no_progress_intervened": control["no_progress_event_count"] > 0,
            "no_progress_warning_count": control["no_progress_warning_count"],
            "no_progress_replan_count": control["no_progress_replan_count"],
            "no_progress_stop_count": control["no_progress_stop_count"],
            "no_progress_event_count": control["no_progress_event_count"],
            "turns": len(iterations),
            "tool_calls": len(tool_calls),
            "replan_count": replan_count,
            "latency_seconds": time.monotonic() - started,
            "first_token_latency_seconds": None,
            "output_tokens_per_second": None,
            "peak_rss_bytes": None,
            "timed_out": False,
            "failure_reason": None if result.status == "ok" else "agent_incomplete",
            "interventions": max(0, approvals_used - 1),
        }
        measurement["verified_case_success"] = verified_case_success(measurement)
        return measurement
    except (OSError, RuntimeError, ValueError, ToolWorkerUnavailable) as exc:
        return {
            "category": fixture.category,
            "mode": fixture.mode,
            "agent_completed": False,
            "completed": False,
            "tests_passed": False,
            "internal_verification_observed": False,
            "internal_verification_passed": False,
            "evaluator_verification_passed": False,
            "final_repository_state_correct": False,
            "safety_passed": False,
            "syntax_failure": False,
            "forbidden_modifications": 0,
            "unnecessary_modifications": 0,
            "tests_modified_improperly": False,
            "test_tampering": False,
            "user_changes_preserved": False,
            "structured_output_valid": None,
            "action_valid": None,
            "structured_output_failures": None,
            "action_validation_failures": None,
            "verification_failures_before_success": 0,
            "recovered_after_verification_failure": False,
            "recovered_after_failure": False,
            "no_progress_intervened": False,
            "no_progress_warning_count": 0,
            "no_progress_replan_count": 0,
            "no_progress_stop_count": 0,
            "no_progress_event_count": 0,
            "turns": 0,
            "tool_calls": 0,
            "replan_count": 0,
            "latency_seconds": time.monotonic() - started,
            "first_token_latency_seconds": None,
            "output_tokens_per_second": None,
            "peak_rss_bytes": None,
            "timed_out": False,
            "failure_reason": type(exc).__name__.lower(),
            "interventions": 0,
            "verified_case_success": False,
        }
    finally:
        await executor.aclose()
        await database.close()
        if previous_roots is None:
            os.environ.pop("APRIL_ALLOWED_FILESYSTEM_ROOTS", None)
        else:
            os.environ["APRIL_ALLOWED_FILESYSTEM_ROOTS"] = previous_roots
        reset_settings_cache()


async def _run_evaluator_verification(
    tool_worker: ToolWorkerClient,
    fixture: CodingFixture,
    project: Path,
) -> dict[str, Any]:
    """Run the fixed fixture command after model interaction has ended."""

    try:
        result = await tool_worker.execute(
            request_id=f"benchmark-evaluator-{fixture.id}",
            operation="test_runner",
            project_root=project,
            args={"argv": list(fixture.verification_argv)},
            timeout_seconds=fixture.timeout_seconds,
            max_stdout_bytes=16_384,
            max_stderr_bytes=16_384,
        )
    except Exception:
        return {
            "passed": False,
            "exit_status": None,
            "stdout_digest": None,
            "stderr_digest": None,
            "truncated": False,
        }
    return {
        "passed": bool(result.ok and result.returncode == 0 and not result.failure_code),
        "exit_status": result.returncode,
        "stdout_digest": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_digest": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "truncated": bool(result.stdout_truncated or result.stderr_truncated),
    }


async def _run_hidden_tests(
    tool_worker: ToolWorkerClient,
    fixture: CodingFixture,
    project: Path,
) -> bool:
    if not fixture.hidden_tests:
        return True
    evaluator_root = project / ".april_hidden_evaluator"
    try:
        shutil.copytree(
            project,
            evaluator_root,
            ignore=shutil.ignore_patterns(".git", ".april_hidden_evaluator"),
        )
        for index, hidden in enumerate(fixture.hidden_tests):
            for relative, content in hidden.files.items():
                path = (evaluator_root / relative).resolve(strict=False)
                path.relative_to(evaluator_root.resolve())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            result = await tool_worker.execute(
                request_id=f"benchmark-hidden-{fixture.id}-{index}",
                operation="test_runner",
                project_root=evaluator_root,
                args={"argv": list(hidden.argv)},
                timeout_seconds=fixture.timeout_seconds,
                max_stdout_bytes=16_384,
                max_stderr_bytes=16_384,
            )
            if not result.ok or result.returncode != 0 or result.failure_code:
                return False
        return True
    except (OSError, ValueError, ToolWorkerUnavailable):
        return False
    finally:
        shutil.rmtree(evaluator_root, ignore_errors=True)


def _verification_recovery_metrics(rows: Sequence[Any]) -> tuple[int, bool]:
    attempts: list[tuple[bool, str]] = []
    for row in rows:
        try:
            request = json.loads(str(row["tool_request_json"]))
            result = json.loads(str(row["tool_result_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(request, dict) or request.get("tool") != "test_runner":
            continue
        if not isinstance(result, dict):
            continue
        material = json.dumps(
            {
                "ok": result.get("ok"),
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        attempts.append((result.get("ok") is True, hashlib.sha256(material.encode()).hexdigest()))
    failures = 0
    first_pass = False
    recovered = False
    for index, (passed, evidence_digest) in enumerate(attempts):
        if passed and index == 0:
            first_pass = True
        if passed and failures:
            changed_evidence = any(
                prior_digest != evidence_digest for _, prior_digest in attempts[:index]
            )
            recovered = changed_evidence
            break
        if not passed:
            failures += 1
    return (0 if first_pass else failures), recovered


def _run_control_metrics(rows: Sequence[Any]) -> dict[str, int]:
    latest: Mapping[str, Any] | None = None
    for row in rows:
        try:
            metadata = json.loads(str(row["metadata_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        control = metadata.get("run_control") if isinstance(metadata, dict) else None
        if isinstance(control, dict):
            latest = control
    if latest is None:
        return {
            "no_progress_warning_count": 0,
            "no_progress_replan_count": 0,
            "no_progress_stop_count": 0,
            "no_progress_event_count": 0,
        }
    return {
        "no_progress_warning_count": int(latest.get("no_progress_warning_count", 0)),
        "no_progress_replan_count": int(latest.get("no_progress_replan_count", 0)),
        "no_progress_stop_count": int(latest.get("no_progress_stop_count", 0)),
        "no_progress_event_count": len(latest.get("no_progress_events", []))
        if isinstance(latest.get("no_progress_events"), list)
        else 0,
    }


def _iteration_validation_metrics(rows: Sequence[Any]) -> tuple[int, int]:
    structured_failures = sum(str(row["state"]) == "structured_error" for row in rows)
    action_failures = 0
    for row in rows:
        try:
            request = json.loads(str(row["tool_request_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(request, dict)
            and request.get("type") == "tool_request"
            and (
                not isinstance(request.get("tool"), str)
                or not isinstance(request.get("args"), dict)
            )
        ):
            action_failures += 1
    return structured_failures, action_failures


def _materialize_fixture_repository(root: Path, fixture: CodingFixture) -> None:
    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.april_tmp/\n",
        encoding="utf-8",
    )
    for relative, content in fixture.visible_files().items():
        path = (root / relative).resolve(strict=False)
        path.relative_to(root.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git_setup(root)
    for relative, content in fixture.dirty_files.items():
        path = (root / relative).resolve(strict=False)
        path.relative_to(root.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _restricted_tree_snapshot(root: Path, forbidden_paths: Sequence[str]) -> dict[str, str]:
    return {
        path: _restricted_tree_digest(root / path)
        for path in forbidden_paths
        if normalize_relative_path(path) is not None
    }


def _restricted_tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        try:
            digest.update(path.read_bytes())
        except OSError:
            return "unreadable"
        return digest.hexdigest()
    if not path.is_dir():
        return "missing"
    try:
        entries = sorted(child for child in path.rglob("*") if child.is_file())
    except OSError:
        return "unreadable"
    for child in entries:
        try:
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
        except OSError:
            return "unreadable"
    return digest.hexdigest()


def _git_setup(root: Path) -> None:
    commands = (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "benchmark@example.local"],
        ["git", "config", "user.name", "APRIL Benchmark"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "fixture baseline"],
    )
    for command in commands:
        subprocess.run(command, cwd=root, check=True, capture_output=True, shell=False)


def _changed_relative_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        value = line[3:].strip() if len(line) >= 3 else ""
        if value and " -> " not in value:
            normalized = normalize_relative_path(value)
            paths.add(normalized or value)
    return paths


def _tool_iteration_passed(row: Any) -> bool:
    """Read persisted tool records without relying on JSON whitespace."""

    try:
        request = json.loads(str(row["tool_request_json"]))
        result = json.loads(str(row["tool_result_json"]))
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(request, dict)
        and request.get("tool") == "test_runner"
        and isinstance(result, dict)
        and result.get("ok") is True
    )


def _tool_iteration_failed(row: Any) -> bool:
    try:
        result = json.loads(str(row["tool_result_json"]))
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(result, dict) and result.get("ok") is False


async def _context(
    client: Any,
    model_id: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    fixtures = _fixtures(data)
    successes = 0
    token_counts: list[int] = []
    durations: list[float] = []
    failures: Counter[str] = Counter()
    categories: dict[str, list[bool]] = {}
    for fixture in fixtures:
        context = "\n".join(str(item) for item in fixture["context"])
        padding_token = fixture.get("padding_token")
        padding_repetitions = fixture.get("padding_repetitions")
        if isinstance(padding_token, str) and isinstance(padding_repetitions, int):
            bounded_repetitions = min(max(padding_repetitions, 0), 8_192)
            context = f"{' '.join([padding_token] * bounded_repetitions)}\n{context}"
        started = time.monotonic()
        try:
            response = await client.chat(
                model_id=model_id,
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "Use the supplied context, preserve the newest instruction, and answer."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=f"Context:\n{context}\n\nQuestion: {fixture['question']}",
                    ),
                ],
                options=_evaluation_options(
                    client, temperature=0.0, max_output_tokens=128, seed=17
                ),
                request_id=f"benchmark-context-{fixture['id']}",
            )
            duration = time.monotonic() - started
            content = response.content
            ok = str(fixture["expected"]) in content and (
                fixture.get("expected_secondary") is None
                or str(fixture["expected_secondary"]) in content
            )
            token_counts.append(response.usage.input_tokens)
            durations.append(duration)
            if response.context_truncated and not ok:
                failures["context_truncated"] += 1
            elif not ok:
                failures["incorrect_extraction_or_recall"] += 1
        except Exception:
            ok = False
            failures["runtime_or_validation_failure"] += 1
        successes += int(ok)
        categories.setdefault(str(fixture["category"]), []).append(ok)
    total = len(fixtures)
    return {
        "fixture_count": total,
        "success_rate": successes / total if total else 0.0,
        "per_category_accuracy": {
            category: sum(outcomes) / len(outcomes)
            for category, outcomes in sorted(categories.items())
        },
        "context_token_counts": token_counts,
        "token_count_source": "runtime_configured_tokenizer",
        "character_estimation_used": False,
        "request_duration_seconds": durations,
        "prompt_evaluation_duration_seconds": durations,
        "prompt_evaluation_duration_source": "end_to_end_chat_request_proxy",
        "failure_reasons": dict(sorted(failures.items())),
    }


def _route_permitted(decision: Mapping[str, Any], fixture: Mapping[str, Any]) -> bool:
    if decision.get("agent") not in fixture.get("agents", []):
        return False
    if decision.get("risk_level") not in fixture.get("risks", []):
        return False
    minimum = fixture.get("permission_min")
    if isinstance(minimum, int) and int(decision.get("permission_level", -1)) < minimum:
        return False
    tools = fixture.get("tools_any")
    if isinstance(tools, list):
        actual = decision.get("tools_needed")
        if not isinstance(actual, list) or not set(tools).intersection(actual):
            return False
    return True


_LEGACY_CATEGORY_OPERATIONS = {
    "git_status": "repository_inspection",
    "git_diff": "repository_inspection",
    "file_reading": "document_reading",
    "file_search": "repository_inspection",
    "reminder_creation": "reminder_create",
    "reminder_listing": "reminder_list",
    "patch_preparation": "patch_proposal",
    "test_execution": "command_execution",
    "approval": "approval_command",
    "rejection": "rejection_command",
    "destructive_external": "external_action",
    "ambiguous_general": "ambiguous_request",
}


def _proposal_permitted(operation: str, tool_class: str, fixture: Mapping[str, Any]) -> bool:
    """Score the model's semantic choice before policy compilation.

    The compiled decision remains separately measured because active bindings
    may deliberately reject an otherwise correct proposal. This is diagnostic
    scoring only; it never authorizes or executes the proposal.
    """
    expected_operation = _LEGACY_CATEGORY_OPERATIONS.get(str(fixture.get("category")))
    if expected_operation != operation:
        return False
    tools = fixture.get("tools_any")
    return not isinstance(tools, list) or tool_class in {str(item) for item in tools}


def _validate_json(
    content: str,
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(value, dict):
        return None, ["schema_type_mismatch"]
    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list) and any(key not in value for key in required):
        errors.append("missing_required_fields")
    properties = _mapping(schema.get("properties"))
    if schema.get("additionalProperties") is False and any(key not in properties for key in value):
        errors.append("unsupported_fields")
    for key, rule_value in properties.items():
        if key not in value:
            continue
        rule = _mapping(rule_value)
        expected_type = rule.get("type")
        if expected_type == "string" and not isinstance(value[key], str):
            errors.append("schema_type_mismatch")
        if expected_type == "object" and not isinstance(value[key], dict):
            errors.append("schema_type_mismatch")
        allowed = rule.get("enum")
        if isinstance(allowed, list) and value[key] not in allowed:
            errors.append("incorrect_enum_values")
    return value, sorted(set(errors))


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("model_quality_fixture_invalid")
    return value


def _fixtures(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = data.get("fixtures")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError("model_quality_fixture_invalid")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
