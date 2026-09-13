"""Fake performance bench execution kept out of the CLI composition module."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from agents.registry import default_agent_registry
from apps.runner.perf_workload import bench_capability_context, fit_agent_workload
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatRequest, GenerationOptions
from services.brain.memory_policy import AgentMemoryContext
from services.brain.router import BrainRouter
from skills.registry import default_registry


async def run_fake_bench(
    home: Path,
    *,
    role: str,
    repeat: int,
    compare_prefix_cache: bool = False,
    compare_layout: bool = False,
    layout_enabled: bool = False,
) -> dict[str, Any]:
    from apps.runner.commands import runner_perf

    if compare_prefix_cache or compare_layout:
        off = await run_fake_bench(home, role=role, repeat=repeat)
        on = await run_fake_bench(home, role=role, repeat=repeat, layout_enabled=compare_layout)
        kind = "layout" if compare_layout and not compare_prefix_cache else "prefix_cache"
        return {
            "schema": "april.perf.bench.v2",
            "created_at": datetime.now(UTC).isoformat(),
            "simulated": True,
            "role": role,
            "repeat": repeat,
            "modes": {"off": off, "on": on},
            "partial": bool(off.get("partial") or on.get("partial")),
            "comparison_kind": kind,
            "comparison": {
                "kind": kind,
                "routing_decisions_identical_across_modes": runner_perf._routing_modes_identical(
                    off, on
                ),
                "fallback_or_failure_count": {
                    "off": off.get("fallback_count", 0),
                    "on": on.get("fallback_count", 0),
                },
                "text_divergence_count": runner_perf._agent_text_divergence(off, on),
                "per_workload_deltas": runner_perf._summary_deltas(off, on),
            },
        }
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    lifecycle = runner_perf.ModelLifecycle(registry, root_backend="fake")
    local = runner_perf._LocalRuntimeClient(lifecycle)
    router = BrainRouter(cast(Any, local), brain_model_id="april-brain")
    settings = load_settings(root=home)
    agent_registry = default_agent_registry()
    capability_summary, stable_prefix = bench_capability_context(
        settings=settings,
        agent_registry=agent_registry,
        tool_registry=default_registry(),
        model_registry=registry,
        layout_enabled=layout_enabled,
    )
    cases: list[dict[str, Any]] = []
    brain_cases: list[dict[str, Any]] = []
    if role in {"brain", "all"}:
        import yaml

        fixture = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
        brain_cases = yaml.safe_load(fixture.read_text(encoding="utf-8")).get("cases", [])
        for pass_number in range(1, max(2, repeat) + 1):
            for case_number, case in enumerate(brain_cases, start=1):
                started = time.monotonic()
                routing_case: dict[str, Any] = {
                    "pass": pass_number,
                    "case_id": f"case-{case_number:03d}",
                    "workload": "brain",
                    "phase": "routing",
                }
                try:
                    result = await router.route_result(str(case["message"]))
                    routing_case.update(
                        {
                            "latency_ms": (time.monotonic() - started) * 1000,
                            "timing": result.runtime_timing,
                            "route_source": result.route_source,
                            "intent": result.decision.intent,
                            "agent": result.decision.agent,
                            "tool": result.decision.tools_needed[0]
                            if result.decision.tools_needed
                            else "none",
                            "prefix_cache": {},
                        }
                    )
                except Exception as exc:
                    routing_case["error"] = runner_perf._bench_error(exc)
                cases.append(routing_case)
                agent_case: dict[str, Any] = {
                    "pass": pass_number,
                    "case_id": f"case-{case_number:03d}",
                    "workload": "brain",
                    "phase": "agent",
                }
                try:
                    agent = agent_registry.get("general_agent")
                    brain_model = registry.get("april-brain")
                    if agent is None:
                        raise RuntimeError("configured general agent unavailable")
                    plan = await fit_agent_workload(
                        client=local,
                        model=brain_model,
                        agent=agent,
                        capability_summary=capability_summary,
                        memory_context=AgentMemoryContext(history=runner_perf._synthetic_history()),
                        question="Synthetic question.",
                        stable_prefix=stable_prefix,
                        max_output_tokens=48,
                        allow_filler=False,
                    )
                    if plan["skipped"]:
                        cases.append(
                            {**agent_case, "phase": "workload_skipped", "reason": plan["reason"]}
                        )
                        continue
                    events = [
                        event
                        async for event in lifecycle.stream(
                            ChatRequest(
                                model_id=brain_model.id,
                                messages=plan["messages"],
                                options=GenerationOptions(
                                    temperature=0.0, max_output_tokens=48, seed=17
                                ),
                            )
                        )
                    ]
                    error_event = next((p for n, p in events if n == "error"), None)
                    usage = next((p for n, p in events if n == "usage"), {})
                    if error_event is not None:
                        agent_case["error"] = {
                            "error_code": str(error_event.get("code", "BENCH_CALL_FAILED")),
                            "error_type": "RuntimeStreamError",
                        }
                    else:
                        agent_case.update(
                            {
                                "timing": usage.get("timing", {}),
                                "prefix_cache": usage.get("prefix_cache", {}),
                                "finish_reason": "stop",
                            }
                        )
                except Exception as exc:
                    agent_case["error"] = runner_perf._bench_error(exc)
                cases.append(agent_case)
    specialist_roles = (
        [] if role == "brain" else ([role] if role != "all" else ["coding", "reading"])
    )
    for specialist_role in specialist_roles:
        model = next((item for item in registry.list() if item.role == specialist_role), None)
        if model is None:
            continue
        for _ in range(repeat):
            specialist_case: dict[str, Any] = {
                "role": specialist_role,
                "workload": specialist_role,
                "phase": "agent",
                "case_id": f"{specialist_role}-001",
            }
            try:
                agent = agent_registry.get(f"{specialist_role}_agent")
                if agent is None:
                    raise RuntimeError("configured specialist agent unavailable")
                plan = await fit_agent_workload(
                    client=local,
                    model=model,
                    agent=agent,
                    max_output_tokens=64,
                    capability_summary=capability_summary,
                    memory_context=AgentMemoryContext(history=runner_perf._synthetic_history()),
                    question="Synthetic performance question.",
                    stable_prefix=stable_prefix,
                )
                if plan["skipped"]:
                    cases.append(
                        {
                            **specialist_case,
                            "phase": "workload_skipped",
                            "reason": plan["reason"],
                        }
                    )
                    continue
                response = await lifecycle.generate(
                    ChatRequest(
                        model_id=model.id,
                        messages=plan["messages"],
                        options=GenerationOptions(temperature=0.0, max_output_tokens=64, seed=17),
                    )
                )
                specialist_case.update(
                    {
                        "latency_ms": response.diagnostics["timing"]["total_ms"],
                        "timing": response.diagnostics["timing"],
                        "prefix_cache": response.diagnostics.get("prefix_cache", {}),
                        "finish_reason": response.finish_reason,
                    }
                )
            except Exception as exc:
                specialist_case["error"] = runner_perf._bench_error(exc)
            cases.append(specialist_case)
    decisions: dict[str, Any] = {}
    if brain_cases:
        first = [
            item.get("intent")
            for item in cases
            if item.get("pass") == 1 and item.get("phase") == "routing" and "intent" in item
        ]
        second = [
            item.get("intent")
            for item in cases
            if item.get("pass") == 2 and item.get("phase") == "routing" and "intent" in item
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
        "partial": any("error" in item for item in cases),
        **decisions,
        "summary": runner_perf._bench_summary(cases),
        "comparison": None,
    }
