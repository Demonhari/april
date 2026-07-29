from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat
from services.brain.deterministic_router import DeterministicRouter
from services.brain.parser import parse_brain_decision
from services.brain.router import ROUTER_SYSTEM_PROMPT
from services.brain.structured_output import BRAIN_DECISION_RESPONSE_FORMAT
from services.tool_worker.client import ToolWorkerClient

FIXTURE_SET_VERSION = "model-quality-v1"
_FILES = ("routing.json", "strict_json.json", "coding.json", "context.json")


def fixture_directory(home: Path) -> Path:
    return home / "data" / "evaluations" / "model_benchmark" / "v1"


def fixture_set_metadata(home: Path) -> dict[str, Any]:
    root = fixture_directory(home)
    digest = hashlib.sha256()
    installed = True
    versions: dict[str, str] = {}
    for name in _FILES:
        path = root / name
        try:
            payload = path.read_bytes()
            decoded = json.loads(payload)
        except (OSError, json.JSONDecodeError):
            installed = False
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        if isinstance(decoded, dict) and isinstance(decoded.get("version"), str):
            versions[name] = decoded["version"]
    return {
        "version": FIXTURE_SET_VERSION,
        "sha256": digest.hexdigest() if installed else None,
        "installed": installed,
        "component_versions": versions,
    }


async def evaluate_model_quality(
    settings: AprilSettings,
    *,
    runtime_url: str,
    runtime_token: str | None,
    model_id: str,
    coding_root: Path,
    tool_worker: ToolWorkerClient | None,
) -> dict[str, Any]:
    metadata = fixture_set_metadata(settings.home)
    if not metadata["installed"]:
        raise RuntimeError("model_quality_fixtures_unavailable")
    client = RuntimeClient(runtime_url, token=runtime_token, timeout=180.0)
    root = fixture_directory(settings.home)
    routing = await _routing(client, model_id, _load(root / "routing.json"))
    strict_json = await _strict_json(client, model_id, _load(root / "strict_json.json"))
    coding = await _coding(
        client,
        model_id,
        _load(root / "coding.json"),
        coding_root=coding_root,
        tool_worker=tool_worker,
    )
    context = await _context(client, model_id, _load(root / "context.json"))
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
    client: RuntimeClient,
    model_id: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    fixtures = _fixtures(data)
    passed = 0
    invalid = 0
    repaired = 0
    wrong = 0
    categories: dict[str, list[bool]] = {}
    deterministic_matches = 0
    deterministic_count = 0
    deterministic = DeterministicRouter()
    for fixture in fixtures:
        category = str(fixture["category"])
        first_pass = False
        repair_used = False
        try:
            response = await client.chat(
                model_id=model_id,
                messages=[
                    ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=str(fixture["request"])),
                ],
                options=GenerationOptions(temperature=0.0, max_output_tokens=512, seed=7),
                response_format=BRAIN_DECISION_RESPONSE_FORMAT,
                request_id=f"benchmark-route-{fixture['id']}",
            )
            try:
                decision = parse_brain_decision(response.content)
                first_pass = True
            except Exception:
                repair_used = True
                repair = await client.chat(
                    model_id=model_id,
                    messages=[
                        ChatMessage(
                            role="system",
                            content="Repair into exactly one schema-valid routing JSON object.",
                        ),
                        ChatMessage(role="user", content=response.content),
                    ],
                    options=GenerationOptions(temperature=0.0, max_output_tokens=512, seed=7),
                    response_format=BRAIN_DECISION_RESPONSE_FORMAT,
                    request_id=f"benchmark-route-repair-{fixture['id']}",
                )
                decision = parse_brain_decision(repair.content, method="model_repair")
            route_ok = first_pass and _route_permitted(decision.model_dump(), fixture)
        except Exception:
            decision = None
            route_ok = False
            invalid += 1
        if repair_used:
            repaired += 1
        elif first_pass and not route_ok:
            wrong += 1
        passed += int(route_ok)
        categories.setdefault(category, []).append(route_ok)
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
        "invalid_output_count": invalid,
        "repair_counted_as_failure": repaired,
        "wrong_route_count": wrong,
        "per_category_accuracy": {
            category: sum(outcomes) / len(outcomes)
            for category, outcomes in sorted(categories.items())
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
    client: RuntimeClient,
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
            options=GenerationOptions(temperature=0.0, max_output_tokens=256, seed=11),
            response_format=ResponseFormat(type="json_object", json_schema=dict(schema)),
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
            options=GenerationOptions(temperature=0.0, max_output_tokens=256, seed=11),
            response_format=ResponseFormat(type="json_object", json_schema=dict(schema)),
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
    client: RuntimeClient,
    model_id: str,
    data: Mapping[str, Any],
    *,
    coding_root: Path,
    tool_worker: ToolWorkerClient | None,
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
    for fixture in fixtures:
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
                ChatMessage(role="user", content=str(fixture["instruction"])),
            ],
            options=GenerationOptions(temperature=0.0, max_output_tokens=768, seed=13),
            response_format=ResponseFormat(
                type="json_object",
                json_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["filename", "content"],
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            ),
            request_id=f"benchmark-code-{fixture['id']}",
        )
        try:
            candidate = json.loads(response.content)
            if (
                not isinstance(candidate, dict)
                or candidate.get("filename") != fixture["candidate_file"]
                or not isinstance(candidate.get("content"), str)
            ):
                raise ValueError("invalid_candidate")
            project = coding_root / str(fixture["id"])
            project.mkdir(parents=True, exist_ok=True)
            tool_result = await tool_worker.execute(
                request_id=f"benchmark-code-exec-{fixture['id']}",
                operation="benchmark_fixture",
                project_root=project,
                args={
                    "fixture_files": fixture["fixture_files"],
                    "candidate_file": fixture["candidate_file"],
                    "candidate_content": candidate["content"],
                    "expected_content": fixture.get("expected_content"),
                    "test_argv": fixture["test_argv"],
                },
                timeout_seconds=30.0,
                max_stdout_bytes=8_192,
                max_stderr_bytes=8_192,
            )
        except Exception:
            counts["syntax_or_compilation_failures"] += 1
            continue
        counts["passed"] += int(tool_result.ok)
        counts["test_pass"] += int(tool_result.returncode == 0)
        failure = tool_result.failure_code or ""
        counts["timeouts"] += int(failure == "timeout")
        details = tool_result.data
        counts["forbidden_file_modifications"] += int(
            bool(details.get("forbidden_file_modification"))
        )
        counts["unnecessary_changes"] += int(bool(details.get("unnecessary_change")))
        counts["syntax_or_compilation_failures"] += int(
            bool(details.get("syntax_or_compilation_failure"))
        )
    total = len(fixtures)
    return {
        "fixture_count": total,
        "fixture_pass_rate": counts["passed"] / total if total else 0.0,
        "test_pass_rate": counts["test_pass"] / total if total else 0.0,
        "syntax_or_compilation_failures": counts["syntax_or_compilation_failures"],
        "forbidden_file_modifications": counts["forbidden_file_modifications"],
        "timeout_rate": counts["timeouts"] / total if total else 0.0,
        "unnecessary_change_rate": counts["unnecessary_changes"] / total if total else 0.0,
        "executed_only_through_tool_worker": True,
    }


async def _context(
    client: RuntimeClient,
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
                options=GenerationOptions(temperature=0.0, max_output_tokens=128, seed=17),
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
