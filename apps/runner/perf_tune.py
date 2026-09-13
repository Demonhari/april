"""Isolated measurement orchestration for the operator performance tuner."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from april_common.process_environment import ProcessCategory
from april_common.process_runner import ResourceLimitProfile, run_restricted_process_sync


class TuneBudgetExceeded(RuntimeError):
    """Internal signal used to prevent profile writes after budget exhaustion."""


class TuneWorkerFailed(RuntimeError):
    """A candidate subprocess failed without invalidating the whole sweep."""

    def __init__(self, side: str) -> None:
        super().__init__(f"{side} worker failed")
        self.side = side


async def run_real_tune(
    home: Path,
    *,
    role: str,
    max_minutes: float,
    cooldown_seconds: float,
    registry: Any,
    candidate_values: Any,
    accept_candidate: Any,
    profile_inputs: Any,
    profile_fingerprint: Any,
    tunable_fields: Any,
    measure_candidate: Any = None,
    development_unsandboxed_override: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    models = [model for model in registry.list() if role == "all" or model.role == role]
    results: list[dict[str, Any]] = []
    pending_profiles: list[dict[str, Any]] = []
    exhausted = False
    if measure_candidate is None:

        async def measure_fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return await measure_real_candidate(
                *args,
                development_unsandboxed_override=development_unsandboxed_override,
                **kwargs,
            )
    else:
        measure_fn = measure_candidate
    for model in models:
        if time.monotonic() - started >= max_minutes * 60:
            exhausted = True
            break
        winner = model
        winner_measure: dict[str, Any] | None = None
        model_failed = False
        routing_check_failed = False
        if model.role == "brain" and measure_candidate is None:
            try:
                await measure_fn(
                    home,
                    model,
                    model,
                    runs=1,
                    nonce_prefix="routing-start",
                    routing_check=True,
                )
            except TuneBudgetExceeded:
                exhausted = True
                break
            except Exception:
                routing_check_failed = True
        for knob, values in await asyncio.to_thread(candidate_values, winner):
            for value in values:
                if time.monotonic() - started >= max_minutes * 60:
                    exhausted = True
                    break
                candidate = winner.model_copy(
                    update=(
                        {"n_batch": value, "n_ubatch": value}
                        if knob == "n_batch=n_ubatch"
                        else {knob: value}
                    )
                )
                try:
                    baseline_measure, measured = await measure_abab_pair(
                        home,
                        winner,
                        candidate,
                        cooldown_seconds=cooldown_seconds,
                        deadline=started + max_minutes * 60,
                        measure_candidate=measure_fn,
                    )
                except TuneBudgetExceeded:
                    exhausted = True
                    break
                except TuneWorkerFailed as exc:
                    results.append(
                        {
                            "model_id": model.id,
                            "knob": knob,
                            "value": value,
                            "accepted": False,
                            "reason": (
                                "baseline_failed" if exc.side == "baseline" else "worker_failed"
                            ),
                        }
                    )
                    if exc.side == "baseline":
                        model_failed = True
                        break
                    continue
                accepted, reason = accept_candidate(baseline_measure, measured, knob)
                results.append(
                    {
                        "model_id": model.id,
                        "knob": knob,
                        "value": value,
                        "accepted": accepted,
                        "reason": reason,
                        "baseline_metric": baseline_measure.get("metric"),
                        "candidate_metric": measured.get("metric"),
                        "baseline_prompt_metric": baseline_measure.get("prompt_metric"),
                        "candidate_prompt_metric": measured.get("prompt_metric"),
                    }
                )
                if accepted:
                    winner = candidate
                    winner_measure = measured
                elif winner_measure is None:
                    winner_measure = baseline_measure
            if exhausted:
                break
            if model_failed:
                break
        if model_failed:
            continue
        if winner != model:
            try:
                final_profile, final_baseline = await measure_abab_pair(
                    home,
                    winner,
                    model,
                    cooldown_seconds=cooldown_seconds,
                    deadline=started + max_minutes * 60,
                    measure_candidate=measure_fn,
                    first=winner,
                    second=model,
                )
            except TuneBudgetExceeded:
                exhausted = True
                break
            except TuneWorkerFailed:
                results.append(
                    {
                        "model_id": model.id,
                        "knob": "final_recheck",
                        "value": None,
                        "accepted": False,
                        "reason": "worker_failed",
                    }
                )
                continue
            final_knob = changed_tunable_knob(model, winner)
            final_ok, final_reason = accept_candidate(final_baseline, final_profile, final_knob)
            if not final_ok:
                results.append(
                    {
                        "model_id": model.id,
                        "knob": "final_recheck",
                        "value": None,
                        "accepted": False,
                        "reason": final_reason or "rejected",
                    }
                )
                continue
            inputs = await asyncio.to_thread(profile_inputs, winner, home)
            if model.role == "brain" and measure_candidate is None:
                try:
                    await measure_fn(
                        home,
                        model,
                        winner,
                        runs=1,
                        nonce_prefix="routing-final",
                        routing_check=True,
                    )
                except TuneBudgetExceeded:
                    exhausted = True
                    break
                except Exception:
                    routing_check_failed = True
            if routing_check_failed:
                results.append(
                    {
                        "model_id": model.id,
                        "knob": "routing_check",
                        "value": None,
                        "accepted": False,
                        "reason": "worker_failed",
                    }
                )
                continue
            pending_profiles.append(
                {
                    "schema": "april.perf.profile.v2",
                    "model_id": model.id,
                    "role": model.role,
                    "fingerprint": profile_fingerprint(inputs),
                    "fingerprint_inputs": inputs,
                    "settings": {
                        key: getattr(winner, key)
                        for key in tunable_fields
                        if getattr(winner, key) != getattr(model, key)
                    },
                }
            )
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
        "schema": "april.perf.tune.v2",
        "simulated": False,
        "role": role,
        "results": results,
        "profiles": profiles,
        "stopped_due_to_budget": exhausted or time.monotonic() - started >= max_minutes * 60,
    }


async def measure_abab_pair(
    home: Path,
    baseline: Any,
    candidate: Any,
    *,
    cooldown_seconds: float,
    deadline: float,
    measure_candidate: Any,
    pairs: int = 2,
    runs: int = 2,
    first: Any | None = None,
    second: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = first or baseline
    right = second or candidate
    left_samples: list[dict[str, Any]] = []
    right_samples: list[dict[str, Any]] = []
    for pair in range(pairs):
        if time.monotonic() >= deadline:
            raise TuneBudgetExceeded
        try:
            left_samples.append(
                await measure_candidate(
                    home, baseline, left, runs=runs, nonce_prefix=f"pair-{pair}-baseline"
                )
            )
        except TuneWorkerFailed as exc:
            raise TuneWorkerFailed("baseline") from exc
        except TuneBudgetExceeded:
            raise
        except Exception as exc:
            raise TuneWorkerFailed("baseline") from exc
        if cooldown_seconds:
            await asyncio.sleep(cooldown_seconds)
        if time.monotonic() >= deadline:
            raise TuneBudgetExceeded
        try:
            right_samples.append(
                await measure_candidate(
                    home, baseline, right, runs=runs, nonce_prefix=f"pair-{pair}-candidate"
                )
            )
        except TuneWorkerFailed:
            raise
        except TuneBudgetExceeded:
            raise
        except Exception as exc:
            raise TuneWorkerFailed("candidate") from exc
        if cooldown_seconds:
            await asyncio.sleep(cooldown_seconds)
    return _aggregate_measurements(left_samples), _aggregate_measurements(right_samples)


def _aggregate_measurements(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"metric": 0.0, "prompt_metric": 0.0, "outputs": [], "peak_rss": 0}
    return {
        "metric": statistics.median(float(item.get("metric", 0.0)) for item in samples),
        "prompt_metric": statistics.median(
            float(item.get("prompt_metric", 0.0)) for item in samples
        ),
        "outputs": [output for item in samples for output in item.get("outputs", [])],
        "semantic_check": [item.get("semantic_check") for item in samples],
        "routing_decisions": [item.get("routing_decisions", []) for item in samples],
        "peak_rss": max(int(item.get("peak_rss", 0)) for item in samples),
        "valid_prefill": all(bool(item.get("valid_prefill", False)) for item in samples),
    }


def changed_tunable_knob(baseline: Any, candidate: Any) -> str:
    if baseline.threads_batch != candidate.threads_batch:
        return "threads_batch"
    if baseline.n_batch != candidate.n_batch or baseline.n_ubatch != candidate.n_ubatch:
        return "n_batch=n_ubatch"
    if baseline.flash_attn != candidate.flash_attn:
        return "flash_attn"
    return "threads"


async def measure_real_candidate(
    home: Path,
    _baseline: Any,
    candidate: Any,
    *,
    runs: int = 2,
    nonce_prefix: str = "run",
    development_unsandboxed_override: bool = False,
    routing_check: bool = False,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "home": str(home),
            "model": candidate.model_dump(mode="json"),
            "runs": runs,
            "nonce_prefix": nonce_prefix,
            "routing_check": routing_check,
        },
        separators=(",", ":"),
    )
    result = await asyncio.to_thread(
        run_restricted_process_sync,
        [sys.executable, "-m", "apps.runner.perf_worker", "--payload", payload],
        cwd=home,
        category=ProcessCategory.BENCHMARKING,
        timeout_seconds=3600.0,
        max_stdout_bytes=1_000_000,
        max_stderr_bytes=64_000,
        resource_limit_profile=ResourceLimitProfile.MODEL_UTILITY,
        april_home=home,
        development_unsandboxed_override=development_unsandboxed_override,
    )
    if result.returncode != 0:
        raise TuneWorkerFailed("candidate")
    try:
        parsed = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise TuneWorkerFailed("candidate") from exc
    metrics = parsed.get("metrics", [])
    metric_values = [float(item.get("metric", 0.0)) for item in metrics]
    prompt_values = [float(item.get("prompt_metric", 0.0)) for item in metrics]
    return {
        "metric": statistics.median(metric_values) if metric_values else 0.0,
        "prompt_metric": statistics.median(prompt_values) if prompt_values else 0.0,
        "outputs": [item.get("semantic_digest") for item in metrics],
        "semantic_check": parsed.get("semantic_check_digest"),
        "routing_decisions": parsed.get("routing_decisions", []),
        "peak_rss": int(parsed.get("peak_rss_bytes", 0) or 0),
        "valid_prefill": bool(parsed.get("valid_prefill", False)),
    }
