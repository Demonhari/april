from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from agents.registry import default_agent_registry
from apps.cli.render import console
from apps.runner import bench_reporting as _bench_reporting
from apps.runner.commands import registry as _registry
from apps.runner.perf_tune import (
    run_real_tune as _run_real_tune_impl,
)
from apps.runner.perf_workload import (
    TUNE_ABAB_PAIRS,
    TUNE_FINAL_RECHECK_WORKERS,
    TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
    TUNE_ROUTING_WORKERS,
    TUNE_RUNS_PER_WORKER,
    TUNE_SIDES_PER_PAIR,
    bench_capability_context,
    fit_agent_workload,
    fit_bench_message_set,
    tune_rate_assumptions,
    tune_target_prompt_tokens,
    tune_warmup_prompt_tokens,
)
from apps.runner.report_io import ReportPathError, preflight_report_path, write_json_report
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
from services.brain.request_context import RequestContext
from services.brain.router import BrainRouter
from services.memory.schemas import Message
from skills.registry import default_registry

_agent_text_divergence = _bench_reporting.agent_text_divergence
_bench_summary = _bench_reporting.bench_summary
_health_memory = _bench_reporting.health_memory
_redact_bench_report = _bench_reporting.redact_bench_report
_routing_modes_identical = _bench_reporting.routing_modes_identical
_routing_passes_identical = _bench_reporting.routing_passes_identical
_summary_deltas = _bench_reporting.summary_deltas
_timing_value = _bench_reporting.timing_value


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

    async def count_message_tokens(self, *, model_id: str, messages: list[ChatMessage]) -> int:
        return await self.lifecycle.count_message_tokens(model_id, messages)


_fit_bench_message_set = fit_bench_message_set


@_registry.perf_app.command("bench")
def perf_bench(
    fake: bool = typer.Option(False, "--fake"),
    role: str = typer.Option("all", "--role"),
    repeat: int = typer.Option(1, "--repeat", min=1, max=20),
    report: Path | None = typer.Option(None, "--report"),
    compare_prefix_cache: bool = typer.Option(False, "--compare-prefix-cache"),
    compare_layout: bool = typer.Option(False, "--compare-layout"),
    print_commands: bool = typer.Option(False, "--print-commands"),
) -> None:
    if role not in {"brain", "coding", "reading", "all"}:
        raise typer.BadParameter("role must be brain, coding, reading, or all")
    if print_commands is True:
        for command in _recommended_perf_commands():
            typer.echo(command)
        return
    settings = load_settings()
    default_target = _default_report_path(settings.home, "perf-bench")
    report_path = report if isinstance(report, Path) else None
    target = report_path.expanduser() if report_path is not None else default_target
    try:
        target = preflight_report_path(target)
    except ReportPathError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    typer.echo(f"Report destination: {target}")
    try:
        if not fake:
            _require_real_runtime(settings.home, settings.runtime.backend, command="bench")
            output = _run_real_bench(
                settings.home,
                role=role,
                repeat=repeat,
                compare_prefix_cache=compare_prefix_cache,
                compare_layout=compare_layout,
            )
        else:
            output = asyncio.run(
                _run_fake_bench(
                    settings.home,
                    role=role,
                    repeat=repeat,
                    compare_prefix_cache=compare_prefix_cache,
                    compare_layout=compare_layout,
                )
            )
    except typer.Exit:
        raise
    except Exception as exc:
        output = _bench_failure_report(role=role, repeat=repeat, simulated=fake, exc=exc)
    written = _write_report_with_fallback(target, default_target, output)
    typer.echo(str(written))


def _default_report_path(home: Path, prefix: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return home / "data" / "verification" / f"{prefix}-{stamp}.json"


def _write_report_with_fallback(target: Path, fallback: Path, output: dict[str, Any]) -> Path:
    try:
        return write_json_report(target, output)
    except (OSError, ReportPathError) as primary_error:
        if target == fallback:
            raise
        try:
            written = write_json_report(fallback, output)
        except (OSError, ReportPathError):
            raise primary_error from None
        typer.echo(f"Primary report write failed; wrote fallback report to {written}")
        return written


def _recommended_perf_commands() -> tuple[str, ...]:
    return (
        "run april perf bench --role brain --compare-prefix-cache "
        "--report data/verification/perf-bench-cache-brain.json",
        "run april perf bench --role all --compare-prefix-cache "
        "--report data/verification/perf-bench-cache-all.json",
        "run april perf bench --role brain --compare-layout "
        "--report data/verification/perf-bench-layout.json",
    )


def _bench_failure_report(
    *, role: str, repeat: int, simulated: bool, exc: Exception
) -> dict[str, Any]:
    return {
        "schema": "april.perf.bench.v2",
        "created_at": datetime.now(UTC).isoformat(),
        "simulated": simulated,
        "role": role,
        "repeat": repeat,
        "partial": True,
        "error": {"error_code": "BENCH_FAILED", "error_type": type(exc).__name__},
        "cases": [],
        "summary": {},
    }


def _bench_error(exc: Exception) -> dict[str, str]:
    return {"error_code": "BENCH_CALL_FAILED", "error_type": type(exc).__name__}


async def _run_fake_bench(
    home: Path,
    *,
    role: str,
    repeat: int,
    compare_prefix_cache: bool = False,
    compare_layout: bool = False,
    layout_enabled: bool = False,
) -> dict[str, Any]:
    from apps.runner.perf_fake_bench import run_fake_bench

    return await run_fake_bench(
        home,
        role=role,
        repeat=repeat,
        compare_prefix_cache=compare_prefix_cache,
        compare_layout=compare_layout,
        layout_enabled=layout_enabled,
    )


def _run_real_bench(
    home: Path,
    *,
    role: str,
    repeat: int,
    compare_prefix_cache: bool = False,
    compare_layout: bool = False,
) -> dict[str, Any]:
    def run_mode(*, cache_enabled: bool, layout_enabled: bool = False) -> dict[str, Any]:
        try:
            return _run_real_bench_mode(
                home,
                role=role,
                repeat=repeat,
                cache_enabled=cache_enabled,
                layout_enabled=layout_enabled,
            )
        except Exception as exc:
            return _bench_failure_report(role=role, repeat=repeat, simulated=False, exc=exc)

    if compare_prefix_cache:
        off = run_mode(cache_enabled=False)
        on = run_mode(cache_enabled=True)
        return _compare_bench_modes(
            role=role, repeat=repeat, off=off, on=on, comparison_kind="prefix_cache"
        )
    if compare_layout:
        off = run_mode(cache_enabled=True, layout_enabled=False)
        on = run_mode(cache_enabled=True, layout_enabled=True)
        return _compare_bench_modes(
            role=role, repeat=repeat, off=off, on=on, comparison_kind="layout"
        )
    return _redact_bench_report(run_mode(cache_enabled=True))


def _run_real_bench_mode(
    home: Path,
    *,
    role: str,
    repeat: int,
    cache_enabled: bool,
    layout_enabled: bool = False,
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
        env["APRIL_RUNTIME_PREFIX_PREWARM"] = "off"
        env["APRIL_ORCHESTRATION_STABLE_PREFIX_LAYOUT"] = "on" if layout_enabled else "off"
        session.runtime = session._start("services.april_runtime.server", env, session.runtime_log)
        session._wait_json(session.runtime_url + "/runtime/health", auth_runtime=True)
        client = RuntimeClient(session.runtime_url, token=session.runtime_token, timeout=180.0)
        registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
        selected_roles = {"brain", "coding", "reading"} if role == "all" else {role}
        selected = [item for item in registry.list() if item.role in selected_roles]
        before_health = asyncio.run(client.health())
        unavailable_models: set[str] = set()
        for model in selected:
            try:
                asyncio.run(client.load(model.id))
                asyncio.run(
                    client.chat(
                        model_id=model.id,
                        messages=[
                            ChatMessage(role="user", content="Synthetic performance warmup.")
                        ],
                        options=GenerationOptions(temperature=0.0, max_output_tokens=1, seed=17),
                        request_id=f"bench-warmup-{model.id}",
                    )
                )
            except Exception as exc:
                unavailable_models.add(model.id)
                cases.append(
                    {
                        "workload": str(model.role),
                        "phase": "workload_error",
                        "case_id": f"{model.role}-startup",
                        "error": _bench_error(exc),
                    }
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
        capability_summary, stable_prefix = bench_capability_context(
            settings=bench_settings,
            agent_registry=agent_registry,
            tool_registry=default_registry(),
            model_registry=registry,
            layout_enabled=layout_enabled,
        )
        router = BrainRouter(client)
        if role in {"brain", "all"}:
            import yaml

            fixture = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
            raw_cases = yaml.safe_load(fixture.read_text(encoding="utf-8")).get("cases", [])
            for pass_number in range(1, max(2, repeat) + 1):
                for case_number, case in enumerate(raw_cases, start=1):
                    routing_case: dict[str, Any] = {
                        "pass": pass_number,
                        "case_id": f"case-{case_number:03d}",
                        "workload": "brain",
                        "phase": "routing",
                    }
                    try:
                        result = asyncio.run(
                            router.route_result(
                                str(case["message"]),
                                request_id=f"bench-{pass_number}-{case_number}",
                            )
                        )
                        routing_case.update(
                            {
                                "latency_ms": result.routing_latency_ms,
                                "timing": result.runtime_timing or {},
                                "route_source": result.route_source.value,
                                "intent": result.decision.intent,
                                "agent": result.decision.agent,
                                "tool": result.decision.tools_needed[0]
                                if result.decision.tools_needed
                                else "none",
                                "prefix_cache": result.runtime_prefix_cache,
                            }
                        )
                    except Exception as exc:
                        routing_case["error"] = _bench_error(exc)
                    cases.append(routing_case)
                    agent = agent_registry.get("general_agent")
                    if agent is None:
                        cases.append(
                            {
                                "pass": pass_number,
                                "case_id": f"case-{case_number:03d}",
                                "workload": "brain",
                                "phase": "agent",
                                "error": {
                                    "error_code": "WORKLOAD_UNAVAILABLE",
                                    "error_type": "ConfigurationError",
                                },
                            }
                        )
                        continue
                    agent_case: dict[str, Any] = {
                        "pass": pass_number,
                        "case_id": f"case-{case_number:03d}",
                        "workload": "brain",
                        "phase": "agent",
                    }
                    try:
                        plan = asyncio.run(
                            fit_agent_workload(
                                client=client,
                                model=registry.get("april-brain"),
                                agent=agent,
                                max_output_tokens=48,
                                capability_summary=capability_summary,
                                memory_context=AgentMemoryContext(history=_synthetic_history()),
                                question=f"Synthetic routing case request: {case['message']}",
                                stable_prefix=stable_prefix,
                                allow_filler=False,
                            )
                        )
                        if plan["skipped"]:
                            cases.append(
                                {
                                    **agent_case,
                                    "phase": "workload_skipped",
                                    "reason": plan["reason"],
                                }
                            )
                            continue
                        usage, generated_text, finish_reason = asyncio.run(
                            _stream_bench_call(
                                client,
                                model_id="april-brain",
                                messages=plan["messages"],
                                max_output_tokens=48,
                                request_id=f"bench-agent-{pass_number}-{case_number}",
                            )
                        )
                        agent_case.update(
                            {
                                "latency_ms": _timing_value(usage, "total_ms"),
                                "timing": usage.get("timing", {}),
                                "prefix_cache": usage.get("prefix_cache", {}),
                                "finish_reason": finish_reason,
                                "_generated_text": generated_text,
                            }
                        )
                    except Exception as exc:
                        agent_case["error"] = _bench_error(exc)
                    cases.append(agent_case)
        if role in {"coding", "reading", "all"}:
            specialist_roles = ("coding", "reading") if role == "all" else (role,)
            for specialist_role in specialist_roles:
                specialist_model = next(
                    (item for item in selected if item.role == specialist_role), None
                )
                agent = agent_registry.get(f"{specialist_role}_agent")
                if (
                    specialist_model is None
                    or specialist_model.id in unavailable_models
                    or agent is None
                ):
                    continue
                questions = (
                    "Summarize the deterministic local performance context.",
                    "List two relevant observations from the same synthetic context.",
                )
                for pass_number in range(1, max(1, repeat) + 1):
                    for question_number, question in enumerate(questions, start=1):
                        if question_number == 2:
                            route_case = {
                                "pass": pass_number,
                                "case_id": f"{specialist_role}-route-{pass_number:03d}",
                                "workload": specialist_role,
                                "phase": "routing",
                            }
                            try:
                                routed = asyncio.run(
                                    router.route_result(
                                        "Synthetic performance routing handoff.",
                                        request_id=(f"bench-{specialist_role}-route-{pass_number}"),
                                    )
                                )
                                route_case.update(
                                    {
                                        "latency_ms": routed.routing_latency_ms,
                                        "timing": routed.runtime_timing,
                                        "prefix_cache": routed.runtime_prefix_cache,
                                        "route_source": routed.route_source.value,
                                        "intent": routed.decision.intent,
                                        "agent": routed.decision.agent,
                                        "tool": (
                                            routed.decision.tools_needed[0]
                                            if routed.decision.tools_needed
                                            else "none"
                                        ),
                                    }
                                )
                            except Exception as exc:
                                route_case["error"] = _bench_error(exc)
                            cases.append(route_case)
                        agent_case = {
                            "pass": pass_number,
                            "case_id": f"{specialist_role}-{question_number:03d}",
                            "workload": specialist_role,
                            "phase": "agent",
                        }
                        try:
                            plan = asyncio.run(
                                fit_agent_workload(
                                    client=client,
                                    model=specialist_model,
                                    agent=agent,
                                    max_output_tokens=48,
                                    capability_summary=capability_summary,
                                    memory_context=AgentMemoryContext(history=_synthetic_history()),
                                    question=question,
                                    stable_prefix=stable_prefix,
                                )
                            )
                            if plan["skipped"]:
                                cases.append(
                                    {
                                        **agent_case,
                                        "phase": "workload_skipped",
                                        "reason": plan["reason"],
                                    }
                                )
                                continue
                            usage, generated_text, finish_reason = asyncio.run(
                                _stream_bench_call(
                                    client,
                                    model_id=specialist_model.id,
                                    messages=plan["messages"],
                                    max_output_tokens=48,
                                    request_id=(
                                        f"bench-{specialist_role}-agent-{pass_number}-{question_number}"
                                    ),
                                )
                            )
                            agent_case.update(
                                {
                                    "latency_ms": _timing_value(usage, "total_ms"),
                                    "timing": usage.get("timing", {}),
                                    "prefix_cache": usage.get("prefix_cache", {}),
                                    "finish_reason": finish_reason,
                                    "_generated_text": generated_text,
                                }
                            )
                        except Exception as exc:
                            agent_case["error"] = _bench_error(exc)
                        cases.append(agent_case)
        after_health = asyncio.run(client.health())
        return {
            "schema": "april.perf.bench.v2",
            "created_at": datetime.now(UTC).isoformat(),
            "simulated": False,
            "role": role,
            "repeat": repeat,
            "cache_enabled": cache_enabled,
            "cases": cases,
            "partial": any("error" in item for item in cases),
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


def _compare_bench_modes(
    *,
    role: str,
    repeat: int,
    off: dict[str, Any],
    on: dict[str, Any],
    comparison_kind: str = "prefix_cache",
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
        "comparison_kind": comparison_kind,
        "partial": bool(off.get("partial") or on.get("partial")),
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


@_registry.perf_app.command("tune")
def perf_tune(
    role: str = typer.Option("all", "--role"),
    max_minutes: float = typer.Option(120.0, "--max-minutes", min=0.01),
    cooldown_seconds: float = typer.Option(20.0, "--cooldown-seconds", min=0.0),
    dry_run: bool = typer.Option(False, "--dry-run"),
    fake: bool = typer.Option(False, "--fake"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    if role not in {"brain", "coding", "reading", "all"}:
        raise typer.BadParameter("role must be brain, coding, reading, or all")
    settings = load_settings()
    default_target = _default_report_path(settings.home, "perf-tune")
    report_path = report if isinstance(report, Path) else None
    target = report_path.expanduser() if report_path is not None else default_target
    try:
        target = preflight_report_path(target)
    except ReportPathError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml", root=settings.home
    )
    selected_models = [model for model in registry.list() if role == "all" or model.role == role]
    candidates = {model.id: dict(_candidate_values(model)) for model in selected_models}
    launches: dict[str, int] = {}
    estimates: dict[str, dict[str, Any]] = {}
    workload_token_sizes = {
        model.id: {
            "measurement_max_output_tokens": TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
            "target_prompt_tokens": tune_target_prompt_tokens(getattr(model, "context_size", 1024)),
            "reserved_output_tokens": TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
            "warmup_prompt_tokens": tune_warmup_prompt_tokens(),
        }
        for model in selected_models
    }
    for model in selected_models:
        candidate_count = sum(len(values) for values in candidates[model.id].values())
        final_recheck_workers = TUNE_FINAL_RECHECK_WORKERS if candidate_count else 0
        routing_workers = TUNE_ROUTING_WORKERS if model.role == "brain" else 0
        launches[model.id] = (
            candidate_count * TUNE_ABAB_PAIRS * TUNE_SIDES_PER_PAIR
            + final_recheck_workers
            + routing_workers
        )
        rates = tune_rate_assumptions(settings.home, model.id)
        target_tokens = workload_token_sizes[model.id]["target_prompt_tokens"]
        worker_seconds = rates["load_seconds"] + TUNE_RUNS_PER_WORKER * (
            target_tokens / rates["prefill_tokens_per_second"]
            + TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS / rates["decode_tokens_per_second"]
        )
        cooldowns = max(0, launches[model.id] - 1) * cooldown_seconds
        estimates[model.id] = {
            "estimated_minutes": (launches[model.id] * worker_seconds + cooldowns) / 60.0,
            "worker_seconds": worker_seconds,
            "cooldown_seconds": cooldowns,
            "load_seconds": rates["load_seconds"],
            "prefill_tokens_per_second": rates["prefill_tokens_per_second"],
            "decode_tokens_per_second": rates["decode_tokens_per_second"],
            "runs_per_worker": 2,
            "target_prompt_tokens": target_tokens,
        }
    over_budget = {
        model_id: item["estimated_minutes"]
        for model_id, item in estimates.items()
        if item["estimated_minutes"] > max_minutes
    }
    plan = {
        "role": role,
        "max_minutes": max_minutes,
        "knobs": sorted(TUNABLE_FIELDS),
        "physical_cores": _physical_cores_for_plan(),
        "logical_cores": os.cpu_count(),
        "candidates": candidates,
        "worker_launches": launches,
        "worker_launch_formula": {
            "pairs": TUNE_ABAB_PAIRS,
            "sides_per_pair": TUNE_SIDES_PER_PAIR,
            "runs_per_worker": TUNE_RUNS_PER_WORKER,
            "final_recheck_workers": TUNE_FINAL_RECHECK_WORKERS,
            "routing_workers_for_brain": TUNE_ROUTING_WORKERS,
        },
        "workload_token_sizes": workload_token_sizes,
        "estimated_minutes": {
            model_id: item["estimated_minutes"] for model_id, item in estimates.items()
        },
        "estimate_assumptions": estimates,
        "over_budget_models": over_budget,
        "profile_written": False,
        "winner_settings": {},
        "thread_candidates_skipped": _physical_cores_for_plan() is None,
        "status": "plan_only" if dry_run else "simulated" if fake else "requires_real_runtime",
    }
    typer.echo(
        f"Report destination: {target}; estimated runtime: "
        f"{sum(item['estimated_minutes'] for item in estimates.values()):.1f} minutes; "
        f"worker launches: {sum(launches.values())}"
    )
    for model_id, minutes in over_budget.items():
        typer.echo(
            f"Warning: estimated runtime for {model_id} ({minutes:.1f} minutes) "
            f"exceeds max_minutes={max_minutes:.1f}"
        )
    if dry_run:
        typer.echo(json.dumps(plan, indent=2, sort_keys=True))
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
                development_unsandboxed_override=getattr(
                    getattr(settings, "workers", None),
                    "development_unsandboxed_override",
                    False,
                ),
            )
        )
        written = _write_report_with_fallback(target, default_target, output)
        typer.echo(str(written))
        return
    written = _write_report_with_fallback(
        target,
        default_target,
        {"schema": "april.perf.tune.v2", **plan, "simulated": fake},
    )
    typer.echo(str(written))


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
    home: Path,
    *,
    role: str,
    max_minutes: float,
    cooldown_seconds: float,
    development_unsandboxed_override: bool = False,
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
        development_unsandboxed_override=development_unsandboxed_override,
    )


def _candidate_values(model: Any) -> list[tuple[str, list[Any]]]:
    physical = _physical_cores_for_plan()
    logical = os.cpu_count()
    values: list[tuple[str, list[Any]]] = []
    current_batch = int(getattr(model, "threads_batch", None) or model.threads)
    current_n_batch = int(getattr(model, "n_batch", None) or model.context_size)
    current_n_ubatch = min(int(getattr(model, "n_ubatch", None) or 512), current_n_batch)
    current_flash = bool(getattr(model, "flash_attn", None) or False)

    def changed(knob: str, value: Any) -> bool:
        if knob == "threads_batch":
            return int(value) != current_batch
        if knob == "n_batch=n_ubatch":
            return int(value) != current_n_batch or int(value) != current_n_ubatch
        if knob == "flash_attn":
            return bool(value) != current_flash
        return int(value) != int(model.threads)

    if physical is not None:
        batch_values = sorted({max(1, physical - 2), physical, logical or physical})
        values.append(
            ("threads_batch", [value for value in batch_values if changed("threads_batch", value)])
        )
    batch_values = [min(value, model.context_size) for value in (128, 256, 512)]
    values.append(
        (
            "n_batch=n_ubatch",
            [value for value in batch_values if changed("n_batch=n_ubatch", value)],
        )
    )
    values.append(
        ("flash_attn", [value for value in (False, True) if changed("flash_attn", value)])
    )
    if physical is not None:
        values.append(
            ("threads", [value for value in sorted({4, 6, physical}) if changed("threads", value)])
        )
    return values


def _accept_candidate(
    baseline: dict[str, Any], candidate: dict[str, Any], knob: str
) -> tuple[bool, str | None]:
    if candidate.get("semantic_check", candidate.get("outputs")) != baseline.get(
        "semantic_check", baseline.get("outputs")
    ):
        return False, "semantic_drift"
    baseline_routing = baseline.get("routing_decisions")
    candidate_routing = candidate.get("routing_decisions")
    if "routing_decisions" in baseline or "routing_decisions" in candidate:
        if not baseline_routing or not candidate_routing:
            return False, "unmeasured"
        if baseline_routing != candidate_routing:
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
