from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import typer

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from april_common.settings import load_settings
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
from services.brain.router import BrainRouter


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
) -> None:
    if role not in {"brain", "coding", "reading", "all"}:
        raise typer.BadParameter("role must be brain, coding, reading, or all")
    settings = load_settings()
    if not fake:  # pragma: no cover - requires an installed runtime and GGUFs
        _require_real_runtime(settings.home, settings.runtime.backend, command="bench")
        output = _run_real_bench(settings.home, role=role, repeat=repeat)
    else:
        output = asyncio.run(_run_fake_bench(settings.home, role=role, repeat=repeat))
    target = report or (
        settings.home
        / "data"
        / "verification"
        / f"perf-bench-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(str(target))


async def _run_fake_bench(home: Path, *, role: str, repeat: int) -> dict[str, Any]:
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
    specialist_roles = (
        [] if role == "brain" else ([role] if role != "all" else ["coding", "reading"])
    )
    for specialist_role in specialist_roles:
        model = next((item for item in registry.list() if item.role == specialist_role), None)
        if model is None:  # pragma: no cover - configured roles have registry entries
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
                    "latency_ms": response.diagnostics["timing"]["total_ms"],
                    "timing": response.diagnostics["timing"],
                }
            )
    decisions = {}
    if brain_cases:
        first = [item["operation"] for item in cases if item.get("pass") == 1]
        second = [item["operation"] for item in cases if item.get("pass") == 2]
        decisions = {
            "fallback_count": sum(1 for item in cases if item.get("route_source") == "fallback"),
            "routing_decisions_identical_across_passes": first == second,
        }
    return {
        "schema": "april.perf.bench.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "simulated": True,
        "role": role,
        "repeat": repeat,
        "cases": cases,
        **decisions,
    }


def _run_real_bench(home: Path, *, role: str, repeat: int) -> dict[str, Any]:  # pragma: no cover
    """Use the verifier's isolated-service path for real routing evidence."""
    from apps.runner.verify import run_routing_only_verification

    report = run_routing_only_verification(home, max_output_tokens=32)
    cases: list[dict[str, Any]] = []
    routing = getattr(report, "routing", None)
    for case_number, case in enumerate(
        getattr(routing, "cases", []) if routing is not None else [], start=1
    ):
        cases.append(
            {
                "case_id": f"case-{case_number:03d}",
                "latency_ms": None,
                "timing": {},
                "route_source": case.route_source,
                "operation": case.actual_intent,
                "context": case.actual_agent,
                "tool_class": None,
                "prefix_cache": {},
            }
        )
    return {
        "schema": "april.perf.bench.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "simulated": False,
        "role": role,
        "repeat": repeat,
        "cases": cases,
        "fallback_count": getattr(routing, "fallback_count", 0) if routing else 0,
        "routing_decisions_identical_across_passes": None,
        "isolation": "temporary_verify_services",
    }


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
    if not fake:  # pragma: no cover - requires an installed runtime and GGUFs
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
        json.dumps({"schema": "april.perf.tune.v1", **plan, "simulated": fake}, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(str(target))


def _require_real_runtime(home: Path, backend: str, *, command: str) -> None:  # pragma: no cover
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
) -> dict[str, Any]:  # pragma: no cover
    """Run a bounded local sweep and save profiles only for accepted winners."""
    started = time.monotonic()
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    models = [model for model in registry.list() if role == "all" or model.role == role]
    results: list[dict[str, Any]] = []
    pending_profiles: list[dict[str, Any]] = []
    exhausted = False
    for model in models:
        if time.monotonic() - started >= max_minutes * 60:
            exhausted = True
            break
        baseline = await _measure_real_candidate(home, model, model)
        winner = model
        winner_measure = baseline
        candidate_plan = await asyncio.to_thread(_candidate_values, model)
        for knob, values in candidate_plan:
            for value in values:
                if time.monotonic() - started >= max_minutes * 60:
                    exhausted = True
                    break
                candidate = model.model_copy(
                    update=(
                        {"n_batch": value, "n_ubatch": value}
                        if knob == "n_batch=n_ubatch"
                        else {knob: value}
                    )
                )
                measured = await _measure_real_candidate(home, model, candidate)
                accepted, reason = _accept_candidate(baseline, measured, knob)
                results.append(
                    {
                        "model_id": model.id,
                        "knob": knob,
                        "value": value,
                        "accepted": accepted,
                        "reason": reason,
                    }
                )
                if accepted and measured["metric"] > winner_measure["metric"]:
                    winner, winner_measure = candidate, measured
                if cooldown_seconds:
                    await asyncio.sleep(cooldown_seconds)
        if winner != model:
            inputs = await asyncio.to_thread(profile_inputs, winner, home)
            profile = {
                "schema": "april.perf.profile.v1",
                "model_id": model.id,
                "role": model.role,
                "fingerprint": profile_fingerprint(inputs),
                "fingerprint_inputs": inputs,
                "settings": {
                    key: getattr(winner, key)
                    for key in TUNABLE_FIELDS
                    if getattr(winner, key) != getattr(model, key)
                },
            }
            pending_profiles.append(profile)
    profiles: list[dict[str, Any]] = []
    if not exhausted:
        for profile in pending_profiles:
            profile_path = home / "data" / "perf" / "profiles" / f"{profile['fingerprint']}.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
            profiles.append(
                {"model_id": profile["model_id"], "fingerprint": profile["fingerprint"]}
            )
    return {
        "schema": "april.perf.tune.v1",
        "simulated": False,
        "role": role,
        "results": results,
        "profiles": profiles,
        "stopped_due_to_budget": exhausted or time.monotonic() - started >= max_minutes * 60,
    }


async def _measure_real_candidate(  # pragma: no cover
    home: Path, _baseline: Any, candidate: Any
) -> dict[str, Any]:
    candidate_registry = ModelRegistry.from_dict(
        {"models": {candidate.id: candidate.model_dump(mode="json")}}, root=home
    )
    lifecycle = ModelLifecycle(candidate_registry, root_backend="llama_cpp")
    prompt = "Performance workload: " + "local deterministic token " * 250
    warmup = ChatRequest(
        model_id=candidate.id,
        messages=[ChatMessage(role="user", content=prompt)],
        options=GenerationOptions(temperature=0.0, max_output_tokens=32, seed=17),
    )
    await lifecycle.generate(warmup)
    measured: list[float] = []
    prompt_measured: list[float] = []
    outputs: list[str] = []
    peak_rss = 0
    for _ in range(3):
        before = time.monotonic()
        response = await lifecycle.generate(warmup)
        elapsed = max(time.monotonic() - before, 0.000001)
        measured.append(float(response.usage.output_tokens) / elapsed)
        prompt_timing = response.usage.timing or {}
        prompt_tokens = float(prompt_timing.get("prompt_eval_tokens", 0) or 0)
        prompt_ms = float(prompt_timing.get("prompt_eval_ms", 0) or 0)
        prompt_measured.append(prompt_tokens / (prompt_ms / 1000.0) if prompt_ms else 0.0)
        outputs.append(response.content)
        peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    await lifecycle.cleanup()
    return {
        "metric": sorted(measured)[len(measured) // 2],
        "prompt_metric": sorted(prompt_measured)[len(prompt_measured) // 2],
        "outputs": outputs,
        "peak_rss": peak_rss,
    }


def _candidate_values(model: Any) -> list[tuple[str, list[Any]]]:  # pragma: no cover
    physical = _physical_cores_for_plan()
    logical = os.cpu_count()
    values: list[tuple[str, list[Any]]] = []
    if physical is not None:
        values.extend(
            [
                ("threads", sorted({4, 6, physical})),
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
    return values


def _accept_candidate(
    baseline: dict[str, Any], candidate: dict[str, Any], knob: str
) -> tuple[bool, str | None]:
    if candidate["outputs"] != baseline["outputs"]:
        return False, "semantic_drift"
    if candidate["peak_rss"] > baseline["peak_rss"] * 1.15:
        return False, "rss"
    metric_key = "metric" if knob == "threads" else "prompt_metric"
    if candidate[metric_key] < baseline[metric_key] * 1.05:
        return False, "slower"
    return True, None
