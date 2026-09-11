from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import typer

from agents.registry import default_agent_registry
from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.perf_tune import (
    run_real_tune as _run_real_tune_impl,
)
from april_common.settings import load_settings
from services.april_runtime.client import RuntimeClient
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.perf_profile import (
    TUNABLE_FIELDS,
    profile_fingerprint,
    profile_inputs,
)
from services.april_runtime.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerationOptions,
    ResponseFormat,
)
from services.brain.capabilities import trusted_capability_summary
from services.brain.memory_policy import AgentMemoryContext
from services.brain.orchestration.finalization_flow import conversation_chat_messages
from services.brain.request_context import RequestContext
from services.brain.router import BrainRouter
from services.memory.schemas import Message
from skills.registry import default_registry


class _LocalRuntimeClient:
    def __init__(self, lifecycle: ModelLifecycle) -> None:
        self.lifecycle = lifecycle

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions,
        response_format: ResponseFormat,
        request_id: str | None = None,
    ) -> ChatResponse:
        return await self.lifecycle.generate(
            ChatRequest(
                model_id=model_id,
                messages=messages,
                options=options,
                response_format=response_format,
                request_id=request_id,
            )
        )


@_registry.perf_app.command("bench")
def perf_bench(
    fake: bool = typer.Option(False, "--fake"),
    role: str = typer.Option("all", "--role"),
    repeat: int = typer.Option(1, "--repeat", min=1, max=20),
    report: Path | None = typer.Option(None, "--report"),
    compare_prefix_cache: bool = typer.Option(False, "--compare-prefix-cache"),
) -> None:
    if role not in {"brain", "coding", "reading", "all"}:
        raise typer.BadParameter("role must be brain, coding, reading, or all")
    settings = load_settings()
    if not fake:
        _require_real_runtime(settings.home, settings.runtime.backend, command="bench")
        output = _run_real_bench(
            settings.home,
            role=role,
            repeat=repeat,
            compare_prefix_cache=compare_prefix_cache,
        )
    else:
        output = asyncio.run(
            _run_fake_bench(
                settings.home,
                role=role,
                repeat=repeat,
                compare_prefix_cache=compare_prefix_cache,
            )
        )
    target = report or (
        settings.home
        / "data"
        / "verification"
        / f"perf-bench-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(str(target))


async def _run_fake_bench(
    home: Path, *, role: str, repeat: int, compare_prefix_cache: bool = False
) -> dict[str, Any]:
    if compare_prefix_cache:
        off = await _run_fake_bench(home, role=role, repeat=repeat)
        on = await _run_fake_bench(home, role=role, repeat=repeat)
        return {
            "schema": "april.perf.bench.v2",
            "created_at": datetime.now(UTC).isoformat(),
            "simulated": True,
            "role": role,
            "repeat": repeat,
            "modes": {"off": off, "on": on},
            "comparison": {
                "routing_decisions_identical_across_modes": _routing_modes_identical(off, on),
                "fallback_or_failure_count": {
                    "off": off.get("fallback_count", 0),
                    "on": on.get("fallback_count", 0),
                },
            },
        }
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    lifecycle = ModelLifecycle(registry, root_backend="fake")
    local = _LocalRuntimeClient(lifecycle)
    router = BrainRouter(cast(Any, local), brain_model_id="april-brain")
    cases: list[dict[str, Any]] = []
    brain_cases = []
    if role in {"brain", "all"}:
        fixture = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
        import yaml

        brain_cases = yaml.safe_load(fixture.read_text(encoding="utf-8")).get("cases", [])
        for pass_number in range(1, max(2, repeat) + 1):
            for case_number, case in enumerate(brain_cases, start=1):
                started = time.monotonic()
                result = await router.route_result(str(case["message"]))
                cases.append(
                    {
                        "pass": pass_number,
                        "case_id": f"case-{case_number:03d}",
                        "workload": "brain",
                        "phase": "routing",
                        "latency_ms": (time.monotonic() - started) * 1000,
                        "timing": result.runtime_timing,
                        "route_source": result.route_source,
                        "operation": result.decision.intent,
                        "context": result.decision.agent,
                        "tool_class": result.decision.tools_needed[0]
                        if result.decision.tools_needed
                        else "none",
                        "prefix_cache": {},
                    }
                )
                agent_request = ChatRequest(
                    model_id="april-brain",
                    messages=[
                        ChatMessage(role="system", content="Synthetic agent system."),
                        ChatMessage(role="user", content="Synthetic history."),
                        ChatMessage(role="user", content="Synthetic question."),
                    ],
                    options=GenerationOptions(temperature=0.0, max_output_tokens=48, seed=17),
                )
                stream_events = [event async for event in lifecycle.stream(agent_request)]
                usage = next((payload for name, payload in stream_events if name == "usage"), {})
                cases.append(
                    {
                        "pass": pass_number,
                        "case_id": f"case-{case_number:03d}",
                        "workload": "brain",
                        "phase": "agent",
                        "timing": usage.get("timing", {}),
                        "prefix_cache": usage.get("prefix_cache", {}),
                        "finish_reason": "stop",
                    }
                )
    specialist_roles = (
        [] if role == "brain" else ([role] if role != "all" else ["coding", "reading"])
    )
    for specialist_role in specialist_roles:
        model = next((item for item in registry.list() if item.role == specialist_role), None)
        if model is None:
            continue
        prompt = "Performance workload: " + "local deterministic token " * 250
        for _ in range(repeat):
            response = await lifecycle.generate(
                ChatRequest(
                    model_id=model.id,
                    messages=[ChatMessage(role="user", content=prompt)],
                    options=GenerationOptions(temperature=0.0, max_output_tokens=64, seed=17),
                )
            )
            cases.append(
                {
                    "role": specialist_role,
                    "workload": specialist_role,
                    "phase": "agent",
                    "case_id": f"{specialist_role}-001",
                    "latency_ms": response.diagnostics["timing"]["total_ms"],
                    "timing": response.diagnostics["timing"],
                    "prefix_cache": response.diagnostics.get("prefix_cache", {}),
                    "finish_reason": response.finish_reason,
                }
            )
    decisions = {}
    if brain_cases:
        first = [
            item["operation"]
            for item in cases
            if item.get("pass") == 1 and item.get("phase") == "routing"
        ]
        second = [
            item["operation"]
            for item in cases
            if item.get("pass") == 2 and item.get("phase") == "routing"
        ]
        decisions = {
            "fallback_count": sum(1 for item in cases if item.get("route_source") == "fallback"),
            "routing_decisions_identical_across_passes": first == second,
        }
    return {
        "schema": "april.perf.bench.v2",
        "created_at": datetime.now(UTC).isoformat(),
        "simulated": True,
        "role": role,
        "repeat": repeat,
        "cases": cases,
        **decisions,
        "summary": _bench_summary(cases),
        "comparison": None,
    }


def _bench_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case.get("workload", "unknown")), []).append(case)
    result: dict[str, Any] = {}
    for workload, items in grouped.items():
        routing_timings = [
            item.get("timing") or {} for item in items if item.get("phase") == "routing"
        ]
        agent_timings = [item.get("timing") or {} for item in items if item.get("phase") == "agent"]
        totals = [
            float(timing["total_ms"])
            for timing in routing_timings
            if isinstance(timing.get("total_ms"), (int, float))
        ]
        agent_ttft = [
            float(timing["ttft_ms"])
            for timing in agent_timings
            if isinstance(timing.get("ttft_ms"), (int, float))
        ]
        agent_totals = [
            float(timing["total_ms"])
            for timing in agent_timings
            if isinstance(timing.get("total_ms"), (int, float))
        ]
        rates = [
            float(timing["prompt_eval_tokens_per_second"])
            for timing in [item.get("timing") or {} for item in items]
            if isinstance(timing.get("prompt_eval_tokens_per_second"), (int, float))
            and timing["prompt_eval_tokens_per_second"] > 0
        ]
        reuse = [
            float(timing["prompt_reuse_ratio"])
            for timing in [item.get("timing") or {} for item in items]
            if isinstance(timing.get("prompt_reuse_ratio"), (int, float))
        ]
        restore = [
            float((item.get("prefix_cache") or {})["restore_ms"])
            for item in items
            if isinstance((item.get("prefix_cache") or {}).get("restore_ms"), (int, float))
        ]
        save = [
            float((item.get("prefix_cache") or {})["save_ms"])
            for item in items
            if isinstance((item.get("prefix_cache") or {}).get("save_ms"), (int, float))
        ]
        state_sizes = [
            int((item.get("prefix_cache") or {})["state_bytes"])
            for item in items
            if isinstance((item.get("prefix_cache") or {}).get("state_bytes"), int)
        ]
        result[workload] = {
            "calls": len(items),
            "routing_calls": len(routing_timings),
            "agent_calls": len(agent_timings),
            "median_total_ms": statistics.median(
                [
                    float((item.get("timing") or {})["total_ms"])
                    for item in items
                    if isinstance((item.get("timing") or {}).get("total_ms"), (int, float))
                ]
            )
            if any(
                isinstance((item.get("timing") or {}).get("total_ms"), (int, float))
                for item in items
            )
            else None,
            "median_routing_total_ms": statistics.median(totals) if totals else None,
            "p90_total_ms": _percentile(totals, 0.9),
            "median_agent_ttft_ms": statistics.median(agent_ttft) if agent_ttft else None,
            "p90_agent_ttft_ms": _percentile(agent_ttft, 0.9),
            "median_agent_total_ms": statistics.median(agent_totals) if agent_totals else None,
            "p90_agent_total_ms": _percentile(agent_totals, 0.9),
            "median_prompt_eval_tokens_per_second": statistics.median(rates) if rates else None,
            "median_prompt_reuse_ratio": statistics.median(reuse) if reuse else None,
            "median_restore_ms": statistics.median(restore) if restore else None,
            "median_save_ms": statistics.median(save) if save else None,
            "max_state_bytes": max(state_sizes) if state_sizes else None,
        }
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _run_real_bench(
    home: Path, *, role: str, repeat: int, compare_prefix_cache: bool = False
) -> dict[str, Any]:
    """Measure a fresh, temporary runtime through the typed RuntimeClient."""
    if compare_prefix_cache:
        off = _run_real_bench_mode(home, role=role, repeat=repeat, cache_enabled=False)
        on = _run_real_bench_mode(home, role=role, repeat=repeat, cache_enabled=True)
        return _compare_bench_modes(role=role, repeat=repeat, off=off, on=on)
    return _redact_bench_report(
        _run_real_bench_mode(home, role=role, repeat=repeat, cache_enabled=True)
    )


def _run_real_bench_mode(
    home: Path, *, role: str, repeat: int, cache_enabled: bool
) -> dict[str, Any]:
    from apps.runner.verification.multi_model import AllConfiguredModelsVerifier

    session = AllConfiguredModelsVerifier(
        home=home,
        require_real_model=True,
        max_output_tokens=48,
        routing_evaluation=False,
    )
    cases: list[dict[str, Any]] = []
    try:
        session._prepare()
        env = session._env()
        env["APRIL_RUNTIME_PREFIX_CACHE"] = "on" if cache_enabled else "off"
        session.runtime = session._start("services.april_runtime.server", env, session.runtime_log)
        session.api = session._start("services.api.server", env, session.api_log)
        session._wait_json(session.runtime_url + "/runtime/health", auth_runtime=True)
        client = RuntimeClient(session.runtime_url, token=session.runtime_token, timeout=180.0)
        registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
        selected_roles = {"brain", "coding", "reading"} if role == "all" else {role}
        selected = [item for item in registry.list() if item.role in selected_roles]
        before_health = asyncio.run(client.health())
        for model in selected:
            asyncio.run(client.load(model.id))
            asyncio.run(
                client.chat(
                    model_id=model.id,
                    messages=[ChatMessage(role="user", content="Synthetic performance warmup.")],
                    options=GenerationOptions(temperature=0.0, max_output_tokens=1, seed=17),
                    request_id=f"bench-warmup-{model.id}",
                )
            )
        bench_settings = load_settings(root=home)
        agent_registry = default_agent_registry()
        capability_summary = trusted_capability_summary(
            settings=bench_settings,
            agent_registry=agent_registry,
            tool_registry=default_registry(),
            model_registry=registry,
            request_context=RequestContext.unknown(),
        )
        router = BrainRouter(client)
        if role in {"brain", "all"}:
            import yaml

            fixture = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
            raw_cases = yaml.safe_load(fixture.read_text(encoding="utf-8")).get("cases", [])
            for pass_number in range(1, max(2, repeat) + 1):
                for case_number, case in enumerate(raw_cases, start=1):
                    result = asyncio.run(
                        router.route_result(
                            str(case["message"]), request_id=f"bench-{pass_number}-{case_number}"
                        )
                    )
                    cases.append(
                        {
                            "pass": pass_number,
                            "case_id": f"case-{case_number:03d}",
                            "workload": "brain",
                            "phase": "routing",
                            "latency_ms": result.routing_latency_ms,
                            "timing": result.runtime_timing or {},
                            "route_source": result.route_source.value,
                            "operation": result.decision.intent,
                            "context": result.decision.agent,
                            "tool_class": result.decision.tools_needed[0]
                            if result.decision.tools_needed
                            else "none",
                            "prefix_cache": result.runtime_prefix_cache,
                        }
                    )
                    agent = agent_registry.get("general_agent")
                    if agent is None:
                        raise RuntimeError("general_agent is not configured")
                    messages = conversation_chat_messages(
                        system_prompt=agent.system_prompt,
                        memory_context=AgentMemoryContext(history=_synthetic_history()),
                        current_prompt=(
                            f"{capability_summary}\n\n"
                            f"Synthetic routing case request:\n{case['message']}"
                        ),
                    )
                    usage, generated_text, finish_reason = asyncio.run(
                        _stream_bench_call(
                            client,
                            model_id="april-brain",
                            messages=messages,
                            max_output_tokens=48,
                            request_id=f"bench-agent-{pass_number}-{case_number}",
                        )
                    )
                    cases.append(
                        {
                            "pass": pass_number,
                            "case_id": f"case-{case_number:03d}",
                            "workload": "brain",
                            "phase": "agent",
                            "latency_ms": _timing_value(usage, "total_ms"),
                            "timing": usage.get("timing", {}),
                            "prefix_cache": usage.get("prefix_cache", {}),
                            "finish_reason": finish_reason,
                            "_generated_text": generated_text,
                        }
                    )
        if role in {"coding", "reading", "all"}:
            specialist_roles = ("coding", "reading") if role == "all" else (role,)
            for specialist_role in specialist_roles:
                specialist_model = next(
                    (item for item in selected if item.role == specialist_role), None
                )
                agent = agent_registry.get(f"{specialist_role}_agent")
                if specialist_model is None or agent is None:
                    continue
                questions = (
                    "Summarize the deterministic local performance context.",
                    "List two relevant observations from the same synthetic context.",
                )
                synthetic_context = "synthetic context token " * 750
                for pass_number in range(1, max(1, repeat) + 1):
                    for question_number, question in enumerate(questions, start=1):
                        if question_number == 2:
                            routed = asyncio.run(
                                router.route_result(
                                    "Synthetic performance routing handoff.",
                                    request_id=(f"bench-{specialist_role}-route-{pass_number}"),
                                )
                            )
                            cases.append(
                                {
                                    "pass": pass_number,
                                    "case_id": f"{specialist_role}-route-{pass_number:03d}",
                                    "workload": specialist_role,
                                    "phase": "routing",
                                    "latency_ms": routed.routing_latency_ms,
                                    "timing": routed.runtime_timing,
                                    "prefix_cache": routed.runtime_prefix_cache,
                                    "route_source": routed.route_source.value,
                                    "operation": routed.decision.intent,
                                    "context": routed.decision.agent,
                                    "tool_class": (
                                        routed.decision.tools_needed[0]
                                        if routed.decision.tools_needed
                                        else "none"
                                    ),
                                }
                            )
                        messages = conversation_chat_messages(
                            system_prompt=agent.system_prompt,
                            memory_context=AgentMemoryContext(history=_synthetic_history()),
                            current_prompt=(
                                f"{capability_summary}\n\n{synthetic_context}\n\n{question}"
                            ),
                        )
                        usage, generated_text, finish_reason = asyncio.run(
                            _stream_bench_call(
                                client,
                                model_id=specialist_model.id,
                                messages=messages,
                                max_output_tokens=48,
                                request_id=(
                                    f"bench-{specialist_role}-agent-{pass_number}-{question_number}"
                                ),
                            )
                        )
                        cases.append(
                            {
                                "pass": pass_number,
                                "case_id": f"{specialist_role}-{question_number:03d}",
                                "workload": specialist_role,
                                "phase": "agent",
                                "latency_ms": _timing_value(usage, "total_ms"),
                                "timing": usage.get("timing", {}),
                                "prefix_cache": usage.get("prefix_cache", {}),
                                "finish_reason": finish_reason,
                                "_generated_text": generated_text,
                            }
                        )
        after_health = asyncio.run(client.health())
        return {
            "schema": "april.perf.bench.v2",
            "created_at": datetime.now(UTC).isoformat(),
            "simulated": False,
            "role": role,
            "repeat": repeat,
            "cache_enabled": cache_enabled,
            "cases": cases,
            "summary": _bench_summary(cases),
            "fallback_count": sum(1 for item in cases if item.get("route_source") == "fallback"),
            "routing_decisions_identical_across_passes": _routing_passes_identical(cases),
            "runtime_rss": {
                "before": _health_memory(before_health),
                "after": _health_memory(after_health),
            },
            "isolation": "temporary_verify_services",
        }
    finally:
        session._stop()
        import shutil

        shutil.rmtree(session.temp, ignore_errors=True)


def _routing_passes_identical(cases: list[dict[str, Any]]) -> bool | None:
    grouped: dict[int, list[tuple[object, object, object]]] = {}
    for item in cases:
        if item.get("phase") != "routing":
            continue
        grouped.setdefault(int(item["pass"]), []).append(
            (item.get("operation"), item.get("context"), item.get("tool_class"))
        )
    values = list(grouped.values())
    return values[0] == values[1] if len(values) >= 2 else None


def _compare_bench_modes(
    *, role: str, repeat: int, off: dict[str, Any], on: dict[str, Any]
) -> dict[str, Any]:
    divergence = _agent_text_divergence(off, on)
    clean_off = _redact_bench_report(off)
    clean_on = _redact_bench_report(on)
    return {
        "schema": "april.perf.bench.v2",
        "created_at": datetime.now(UTC).isoformat(),
        "simulated": False,
        "role": role,
        "repeat": repeat,
        "modes": {"off": clean_off, "on": clean_on},
        "comparison": {
            "routing_decisions_identical_across_modes": _routing_modes_identical(off, on),
            "fallback_or_failure_count": {
                "off": off.get("fallback_count", 0),
                "on": on.get("fallback_count", 0),
            },
            "text_divergence_count": divergence,
            "per_workload_deltas": _summary_deltas(off, on),
        },
    }


def _routing_modes_identical(off: dict[str, Any], on: dict[str, Any]) -> bool:
    def values(report: dict[str, Any]) -> list[tuple[object, object, object]]:
        return [
            (item.get("operation"), item.get("context"), item.get("tool_class"))
            for item in report.get("cases", [])
            if item.get("phase") == "routing"
        ]

    return values(off) == values(on)


def _synthetic_history() -> list[Message]:
    return [
        Message(
            id="perf-history-1",
            conversation_id="perf-bench",
            role="user",
            content="A fixed synthetic history question.",
            created_at="2026-01-01T00:00:00Z",
        ),
        Message(
            id="perf-history-2",
            conversation_id="perf-bench",
            role="assistant",
            content="A fixed synthetic history answer.",
            created_at="2026-01-01T00:00:01Z",
        ),
    ]


async def _stream_bench_call(
    client: RuntimeClient,
    *,
    model_id: str,
    messages: list[ChatMessage],
    max_output_tokens: int,
    request_id: str,
) -> tuple[dict[str, Any], str, str]:
    generated: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = "stop"
    async for raw in client.stream(
        model_id=model_id,
        messages=messages,
        options=GenerationOptions(temperature=0.0, max_output_tokens=max_output_tokens, seed=17),
        request_id=request_id,
    ):
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            continue
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            continue
        if event.get("event") == "token":
            value = payload.get("text")
            if isinstance(value, str):
                generated.append(value)
        elif event.get("event") == "usage":
            usage = payload
        elif event.get("event") == "done":
            value = payload.get("finish_reason")
            if isinstance(value, str):
                finish_reason = value
    return usage, "".join(generated), finish_reason


def _timing_value(payload: dict[str, Any], key: str) -> float | None:
    value = (payload.get("timing") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _health_memory(payload: dict[str, Any]) -> dict[str, int | None]:
    return {
        "rss_bytes": payload.get("process_rss_bytes")
        if isinstance(payload.get("process_rss_bytes"), int)
        else None,
        "peak_rss_bytes": payload.get("process_peak_rss_bytes")
        if isinstance(payload.get("process_peak_rss_bytes"), int)
        else None,
    }


def _agent_text_divergence(off: dict[str, Any], on: dict[str, Any]) -> int:
    def texts(report: dict[str, Any]) -> dict[str, str]:
        return {
            str(item.get("case_id")): str(item.get("_generated_text", ""))
            for item in report.get("cases", [])
            if item.get("phase") == "agent"
        }

    left, right = texts(off), texts(on)
    return sum(left.get(case_id) != value for case_id, value in right.items())


def _redact_bench_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove in-memory-only generated text before a report is serialized."""
    result = dict(report)
    result["cases"] = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in report.get("cases", [])
    ]
    return result


def _summary_deltas(off: dict[str, Any], on: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    off_summary = off.get("summary", {})
    on_summary = on.get("summary", {})
    if not isinstance(off_summary, dict) or not isinstance(on_summary, dict):
        return deltas
    for workload in sorted(set(off_summary) | set(on_summary)):
        left = off_summary.get(workload, {})
        right = on_summary.get(workload, {})
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        fields: dict[str, float | None] = {}
        for key, value in right.items():
            baseline = left.get(key)
            if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                fields[key] = float(value) - float(baseline)
        deltas[str(workload)] = fields
    return deltas


@_registry.perf_app.command("tune")
def perf_tune(
    role: str = typer.Option("all", "--role"),
    max_minutes: float = typer.Option(30.0, "--max-minutes", min=0.01),
    cooldown_seconds: float = typer.Option(20.0, "--cooldown-seconds", min=0.0),
    dry_run: bool = typer.Option(False, "--dry-run"),
    fake: bool = typer.Option(False, "--fake"),
) -> None:
    if role not in {"brain", "coding", "reading", "all"}:
        raise typer.BadParameter("role must be brain, coding, reading, or all")
    settings = load_settings()
    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml", root=settings.home
    )
    selected_models = [model for model in registry.list() if role == "all" or model.role == role]
    candidates = {model.id: dict(_candidate_values(model)) for model in selected_models}
    plan = {
        "role": role,
        "max_minutes": max_minutes,
        "knobs": sorted(TUNABLE_FIELDS),
        "physical_cores": _physical_cores_for_plan(),
        "logical_cores": os.cpu_count(),
        "candidates": candidates,
        "thread_candidates_skipped": _physical_cores_for_plan() is None,
        "status": "plan_only" if dry_run else "simulated" if fake else "requires_real_runtime",
    }
    if dry_run:
        console.print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if not fake:
        _require_real_runtime(settings.home, settings.runtime.backend, command="tune")
        # The real sweep is deliberately isolated to this operator command.
        # It is implemented through fresh lifecycle instances below; no profile
        # is written until a candidate satisfies every acceptance gate.
        output = asyncio.run(
            _run_real_tune(
                settings.home,
                role=role,
                max_minutes=max_minutes,
                cooldown_seconds=cooldown_seconds,
            )
        )
        target = (
            settings.home
            / "data"
            / "verification"
            / (f"perf-tune-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        console.print(str(target))
        return
    target = (
        settings.home
        / "data"
        / "verification"
        / (f"perf-tune-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"schema": "april.perf.tune.v2", **plan, "simulated": fake}, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(str(target))


def _require_real_runtime(home: Path, backend: str, *, command: str) -> None:
    if backend != "llama_cpp":
        console.print(f"Next: APRIL_RUNTIME_BACKEND=llama_cpp run april perf {command}")
        raise typer.Exit(1)
    if importlib.util.find_spec("llama_cpp") is None:
        console.print("Next: pip install -e '.[runtime]'")
        raise typer.Exit(1) from None
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    if any(not model.resolved_path(home).is_file() for model in registry.list()):
        console.print("Next: run april verify --all-configured-models --require-real-model")
        raise typer.Exit(1)


def _physical_cores_for_plan() -> int | None:
    from april_common.hardware_profile import physical_cpu_count

    return physical_cpu_count()


async def _run_real_tune(
    home: Path, *, role: str, max_minutes: float, cooldown_seconds: float
) -> dict[str, Any]:
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    return await _run_real_tune_impl(
        home,
        role=role,
        max_minutes=max_minutes,
        cooldown_seconds=cooldown_seconds,
        registry=registry,
        candidate_values=_candidate_values,
        accept_candidate=_accept_candidate,
        profile_inputs=profile_inputs,
        profile_fingerprint=profile_fingerprint,
        tunable_fields=TUNABLE_FIELDS,
    )


def _candidate_values(model: Any) -> list[tuple[str, list[Any]]]:
    physical = _physical_cores_for_plan()
    logical = os.cpu_count()
    values: list[tuple[str, list[Any]]] = []
    if physical is not None:
        values.extend(
            [
                (
                    "threads_batch",
                    sorted({max(1, physical - 2), physical, logical or physical}),
                ),
            ]
        )
    else:
        values = []
    values.extend(
        [
            (
                "n_batch=n_ubatch",
                [min(value, model.context_size) for value in (128, 256, 512)],
            ),
            ("flash_attn", [False, True]),
        ]
    )
    if physical is not None:
        values.append(("threads", sorted({4, 6, physical})))
    return values


def _accept_candidate(
    baseline: dict[str, Any], candidate: dict[str, Any], knob: str
) -> tuple[bool, str | None]:
    if candidate.get("semantic_check", candidate.get("outputs")) != baseline.get(
        "semantic_check", baseline.get("outputs")
    ):
        return False, "semantic_drift"
    if baseline.get("routing_decisions") != candidate.get("routing_decisions"):
        return False, "semantic_drift"
    if not candidate.get("valid_prefill", True) or not baseline.get("valid_prefill", True):
        return False, "unmeasured"
    if candidate["peak_rss"] > baseline["peak_rss"] * 1.15:
        return False, "rss"
    metric_key = "metric" if knob == "threads" else "prompt_metric"
    if candidate.get(metric_key, 0) <= 0 or baseline.get(metric_key, 0) <= 0:
        return False, "unmeasured"
    if candidate[metric_key] < baseline[metric_key] * 1.05:
        return False, "slower"
    return True, None
