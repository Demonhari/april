"""Deterministic, tokenizer-driven performance workload construction."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agents.registry import AgentRegistry
from april_common.settings import AprilSettings
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatMessage
from services.brain.capabilities import (
    trusted_capability_summary,
    trusted_capability_summary_parts,
)
from services.brain.memory_policy import AgentMemoryContext
from services.brain.orchestration.finalization_flow import conversation_chat_messages
from services.brain.request_context import RequestContext
from skills.registry import ToolRegistry

FILLER_UNIT = "synthetic context token "
MIN_WORKLOAD_TOKENS = 128
TUNE_WORKLOAD_FRACTION = 0.60
TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS = 32
TUNE_WARMUP_PROMPT = "Synthetic tune warmup."
TUNE_ABAB_PAIRS = 2
TUNE_RUNS_PER_WORKER = 2
TUNE_SIDES_PER_PAIR = 2
TUNE_FINAL_RECHECK_WORKERS = 4
TUNE_ROUTING_WORKERS = 2
# Conservative estimates used only for operator-facing dry-run planning.
TUNE_DEFAULT_LOAD_SECONDS = 60.0
TUNE_DEFAULT_PREFILL_TOKENS_PER_SECOND = 10.0
TUNE_DEFAULT_DECODE_TOKENS_PER_SECOND = 2.0


def tune_target_prompt_tokens(
    context_size: int, reserved_output_tokens: int = TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS
) -> int:
    return int(TUNE_WORKLOAD_FRACTION * context_size) - reserved_output_tokens


def tune_warmup_prompt_tokens() -> int:
    return len(TUNE_WARMUP_PROMPT.split())


def tune_rate_assumptions(home: Path, model_id: str) -> dict[str, float]:
    rates = {
        "load_seconds": TUNE_DEFAULT_LOAD_SECONDS,
        "prefill_tokens_per_second": TUNE_DEFAULT_PREFILL_TOKENS_PER_SECOND,
        "decode_tokens_per_second": TUNE_DEFAULT_DECODE_TOKENS_PER_SECOND,
    }
    directory = home / "data" / "perf" / "profiles"
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("model_id") != model_id:
            continue
        measured = payload.get("estimate") or payload.get("estimates")
        if not isinstance(measured, dict):
            continue
        for key in rates:
            value = measured.get(key)
            if isinstance(value, (int, float)) and value > 0:
                rates[key] = float(value)
        break
    return rates


def build_filler(target_tokens: int, count_tokens: Callable[[str], int]) -> str:
    """Return whole filler repetitions whose measured size fits the target."""

    if target_tokens <= 0:
        return ""
    high = 1
    while count_tokens(FILLER_UNIT * high) < target_tokens:
        high *= 2
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(FILLER_UNIT * middle) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    return FILLER_UNIT * low


async def build_filler_async(
    target_tokens: int, count_tokens: Callable[[str], Awaitable[int]]
) -> str:
    """Async counterpart for RuntimeClient/backend tokenizer calls."""

    if target_tokens <= 0:
        return ""
    high = 1
    while await count_tokens(FILLER_UNIT * high) < target_tokens:
        high *= 2
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        if await count_tokens(FILLER_UNIT * middle) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    return FILLER_UNIT * low


def workload_budget(context_size: int, reserved_output_tokens: int, overhead_tokens: int) -> int:
    """Return the 70%-of-context input budget, floored at 128 tokens."""

    raw = math.floor(0.70 * context_size) - reserved_output_tokens - overhead_tokens
    return max(MIN_WORKLOAD_TOKENS, raw)


def workload_budget_for_fraction(
    context_size: int,
    reserved_output_tokens: int,
    overhead_tokens: int,
    fraction: float,
) -> int:
    """Return a bounded input budget for a workload-specific context fraction."""

    raw = math.floor(fraction * context_size) - reserved_output_tokens - overhead_tokens
    return max(MIN_WORKLOAD_TOKENS, raw)


def raw_workload_budget(
    context_size: int,
    reserved_output_tokens: int,
    overhead_tokens: int,
    fraction: float = 0.70,
) -> int:
    """Return the un-clamped budget used to decide whether to skip a workload."""

    return math.floor(fraction * context_size) - reserved_output_tokens - overhead_tokens


def build_bench_message_set(
    *,
    system_prompt: str,
    capability_summary: str,
    memory_context: AgentMemoryContext,
    question: str,
    synthetic_context: str = "",
    stable_prefix: str | None = None,
) -> list[ChatMessage]:
    """Build the production-shaped, deterministic benchmark message set."""

    current_prompt = f"{capability_summary}\n\n"
    if synthetic_context:
        current_prompt += f"{synthetic_context}\n\n"
    current_prompt += question
    return conversation_chat_messages(
        system_prompt=system_prompt,
        memory_context=memory_context,
        current_prompt=current_prompt,
        stable_prefix=stable_prefix,
    )


async def fit_bench_message_set(
    *,
    client: Any,
    model_id: str,
    context_size: int,
    max_output_tokens: int,
    system_prompt: str,
    capability_summary: str,
    memory_context: AgentMemoryContext,
    question: str,
    stable_prefix: str | None = None,
    allow_filler: bool = True,
) -> dict[str, Any]:
    """Fit one complete benchmark payload using the model's tokenizer."""

    async def count(messages: list[ChatMessage]) -> int:
        return int(await client.count_message_tokens(model_id=model_id, messages=messages))

    base_messages = build_bench_message_set(
        system_prompt=system_prompt,
        capability_summary=capability_summary,
        memory_context=memory_context,
        question=question,
        stable_prefix=stable_prefix,
    )
    fixed_tokens = await count(base_messages)
    target_tokens = int(0.70 * context_size) - max_output_tokens
    raw_budget = raw_workload_budget(context_size, max_output_tokens, fixed_tokens, fraction=0.70)
    if fixed_tokens > target_tokens:
        return {
            "skipped": True,
            "reason": "fixed_prompt_exceeds_budget",
            "token_count": fixed_tokens,
            "target_tokens": target_tokens,
        }
    if raw_budget < MIN_WORKLOAD_TOKENS:
        return {
            "skipped": True,
            "reason": "workload_budget_below_minimum",
            "token_count": fixed_tokens,
            "target_tokens": target_tokens,
        }
    if not allow_filler:
        return {
            "skipped": False,
            "messages": base_messages,
            "token_count": fixed_tokens,
            "target_tokens": target_tokens,
            "workload_tokens": 0,
        }

    async def count_text(text: str) -> int:
        return int(
            await client.count_message_tokens(
                model_id=model_id,
                messages=[ChatMessage(role="user", content=text)],
            )
        )

    filler = await build_filler_async(raw_budget, count_text)
    messages = build_bench_message_set(
        system_prompt=system_prompt,
        capability_summary=capability_summary,
        memory_context=memory_context,
        question=question,
        synthetic_context=filler,
        stable_prefix=stable_prefix,
    )
    measured = await count(messages)
    for _ in range(4):
        if measured <= target_tokens:
            break
        excess = measured - target_tokens
        filler = await build_filler_async(max(0, raw_budget - excess), count_text)
        messages = build_bench_message_set(
            system_prompt=system_prompt,
            capability_summary=capability_summary,
            memory_context=memory_context,
            question=question,
            synthetic_context=filler,
            stable_prefix=stable_prefix,
        )
        measured = await count(messages)
    if measured > target_tokens:
        return {
            "skipped": True,
            "reason": "sized_prompt_exceeds_budget",
            "token_count": measured,
            "target_tokens": target_tokens,
        }
    return {
        "skipped": False,
        "messages": messages,
        "token_count": measured,
        "target_tokens": target_tokens,
        "workload_tokens": int(await count_text(filler)) if filler else 0,
    }


async def fit_agent_workload(
    *,
    client: Any,
    model: Any,
    agent: Any,
    capability_summary: str,
    memory_context: AgentMemoryContext,
    question: str,
    max_output_tokens: int,
    stable_prefix: str | None = None,
    allow_filler: bool = True,
) -> dict[str, Any]:
    return await fit_bench_message_set(
        client=client,
        model_id=model.id,
        context_size=model.context_size,
        max_output_tokens=max_output_tokens,
        system_prompt=agent.system_prompt,
        capability_summary=capability_summary,
        memory_context=memory_context,
        question=question,
        stable_prefix=stable_prefix,
        allow_filler=allow_filler,
    )


def bench_capability_context(
    *,
    settings: AprilSettings,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelRegistry,
    layout_enabled: bool,
) -> tuple[str, str | None]:
    summary = trusted_capability_summary(
        settings=settings,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
        request_context=RequestContext.unknown(),
    )
    if not layout_enabled:
        return summary, None
    stable, volatile = trusted_capability_summary_parts(
        settings=settings,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
        request_context=RequestContext.unknown(),
    )
    return volatile, stable
