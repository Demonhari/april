"""Redacted report calculations for the operator performance bench."""

from __future__ import annotations

import statistics
from typing import Any


def bench_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        pass_number = int(case.get("pass", 1) or 1)
        group = "first_pass" if pass_number == 1 else "repeat_passes"
        grouped.setdefault((str(case.get("workload", "unknown")), group), []).append(case)
    result: dict[str, Any] = {}
    for (workload, group), items in grouped.items():
        result.setdefault(workload, {})[group] = _group_summary(items)
    for workload in result:
        items = [item for item in cases if str(item.get("workload", "unknown")) == workload]
        result[workload].update(
            {
                key: value
                for key, value in _group_summary(items).items()
                if key in {"median_total_ms", "calls", "routing_calls", "agent_calls"}
            }
        )
    return result


def _group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    routing = [
        item.get("timing") or {}
        for item in items
        if item.get("phase") == "routing" and item.get("route_source") != "deterministic"
    ]
    agents = [item.get("timing") or {} for item in items if item.get("phase") == "agent"]
    totals = _numbers(routing, "total_ms")
    agent_ttft = _numbers(agents, "ttft_ms")
    agent_totals = _numbers(agents, "total_ms")
    prefill = [item.get("timing") or {} for item in items]
    rates = [
        float(timing["prompt_eval_tokens_per_second"])
        for timing in prefill
        if isinstance(timing.get("prompt_eval_tokens_per_second"), (int, float))
        and timing["prompt_eval_tokens_per_second"] > 0
        and isinstance(timing.get("prompt_eval_tokens"), (int, float))
        and timing["prompt_eval_tokens"] >= 64
    ]
    prompt_counts = [
        int(timing["prompt_eval_tokens"])
        for timing in prefill
        if isinstance(timing.get("prompt_eval_tokens"), (int, float))
        and timing["prompt_eval_tokens"] >= 64
    ]
    reuse = _numbers(prefill, "prompt_reuse_ratio")
    restore = _prefix_numbers(items, "restore_ms")
    save = _prefix_numbers(items, "save_ms")
    state_sizes = [
        int((item.get("prefix_cache") or {})["state_bytes"])
        for item in items
        if isinstance((item.get("prefix_cache") or {}).get("state_bytes"), int)
    ]
    all_totals = _numbers([item.get("timing") or {} for item in items], "total_ms")
    return {
        "calls": len(items),
        "routing_calls": len(routing),
        "agent_calls": len(agents),
        "median_total_ms": statistics.median(all_totals) if all_totals else None,
        "median_routing_total_ms": statistics.median(totals) if totals else None,
        "p90_routing_total_ms": percentile(totals, 0.9),
        "median_agent_ttft_ms": statistics.median(agent_ttft) if agent_ttft else None,
        "p90_agent_ttft_ms": percentile(agent_ttft, 0.9),
        "median_agent_total_ms": statistics.median(agent_totals) if agent_totals else None,
        "p90_agent_total_ms": percentile(agent_totals, 0.9),
        "median_prompt_eval_tokens_per_second": statistics.median(rates) if rates else None,
        "median_prompt_eval_tokens": statistics.median(prompt_counts) if prompt_counts else None,
        "median_prompt_reuse_ratio": statistics.median(reuse) if reuse else None,
        "median_restore_ms": statistics.median(restore) if restore else None,
        "median_save_ms": statistics.median(save) if save else None,
        "max_state_bytes": max(state_sizes) if state_sizes else None,
    }


def _numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    return [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]


def _prefix_numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float((item.get("prefix_cache") or {})[key])
        for item in items
        if isinstance((item.get("prefix_cache") or {}).get(key), (int, float))
    ]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))]


def routing_passes_identical(cases: list[dict[str, Any]]) -> bool | None:
    grouped: dict[int, list[tuple[object, object, object]]] = {}
    for item in cases:
        if item.get("phase") != "routing":
            continue
        grouped.setdefault(int(item["pass"]), []).append(
            (item.get("intent"), item.get("agent"), item.get("tool"))
        )
    values = list(grouped.values())
    return values[0] == values[1] if len(values) >= 2 else None


def routing_modes_identical(off: dict[str, Any], on: dict[str, Any]) -> bool:
    def values(report: dict[str, Any]) -> list[tuple[object, object, object]]:
        return [
            (item.get("intent"), item.get("agent"), item.get("tool"))
            for item in report.get("cases", [])
            if item.get("phase") == "routing" and item.get("route_source") != "deterministic"
        ]

    return values(off) == values(on)


def agent_text_divergence(off: dict[str, Any], on: dict[str, Any]) -> int:
    def texts(report: dict[str, Any]) -> dict[tuple[object, object, object], str]:
        return {
            (item.get("pass"), item.get("workload"), item.get("case_id")): str(
                item.get("_generated_text", "")
            )
            for item in report.get("cases", [])
            if item.get("phase") == "agent"
        }

    left, right = texts(off), texts(on)
    return sum(left.get(case_id) != value for case_id, value in right.items())


def redact_bench_report(report: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    result["cases"] = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in report.get("cases", [])
    ]
    return result


def summary_deltas(off: dict[str, Any], on: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    off_summary = off.get("summary", {})
    on_summary = on.get("summary", {})
    if not isinstance(off_summary, dict) or not isinstance(on_summary, dict):
        return deltas
    for workload in sorted(set(off_summary) | set(on_summary)):
        left_all = off_summary.get(workload, {})
        right_all = on_summary.get(workload, {})
        if isinstance(left_all, dict) and "first_pass" in left_all:
            left = left_all.get("first_pass", {})
            right = right_all.get("first_pass", {}) if isinstance(right_all, dict) else {}
        else:
            left, right = left_all, right_all
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        deltas[str(workload)] = {
            key: float(value) - float(left[key])
            for key, value in right.items()
            if isinstance(value, (int, float)) and isinstance(left.get(key), (int, float))
        }
    return deltas


def health_memory(payload: dict[str, Any]) -> dict[str, int | None]:
    return {
        "rss_bytes": payload.get("process_rss_bytes")
        if isinstance(payload.get("process_rss_bytes"), int)
        else None,
        "peak_rss_bytes": payload.get("process_peak_rss_bytes")
        if isinstance(payload.get("process_peak_rss_bytes"), int)
        else None,
    }


def timing_value(payload: dict[str, Any], key: str) -> float | None:
    value = (payload.get("timing") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None
