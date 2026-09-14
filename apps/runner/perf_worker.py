"""One isolated, prefix-cache-free performance measurement for perf tune."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, cast

from apps.runner.perf_workload import (
    MIN_WORKLOAD_TOKENS,
    TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
    TUNE_WARMUP_PROMPT,
    TUNE_WORKLOAD_FRACTION,
    build_filler,
    build_filler_async,
    raw_workload_budget,
    tune_target_prompt_tokens,
)
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry
from services.april_runtime.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerationOptions,
    ResponseFormat,
)
from services.brain.router import BrainRouter


class _LifecycleClient:
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


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if __import__("sys").platform == "darwin" else value * 1024


def _workload_prompt(
    nonce: str,
    context_size: int,
    count_tokens: Any | None = None,
    max_output_tokens: int = TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
) -> str:
    """Build a deterministic tune prompt; sizing uses an injected tokenizer."""

    fixed = f"Measurement nonce: {nonce}\n"
    if count_tokens is None:
        token_count = max(256, min(768, context_size // 3))
        return fixed + "synthetic context token " * token_count
    fixed_tokens = int(count_tokens(fixed))
    filler_budget = raw_workload_budget(
        context_size,
        max_output_tokens,
        fixed_tokens,
        fraction=TUNE_WORKLOAD_FRACTION,
    )
    if filler_budget < MIN_WORKLOAD_TOKENS:
        return fixed
    filler = build_filler(filler_budget, count_tokens)
    return fixed + filler


async def _sized_workload_prompt(
    lifecycle: ModelLifecycle,
    model_id: str,
    nonce: str,
    context_size: int,
    max_output_tokens: int,
) -> tuple[str | None, int, str | None]:
    async def count(text: str) -> int:
        counter = getattr(lifecycle, "count_message_tokens", None)
        if callable(counter):
            return int(
                await counter(
                    model_id,
                    [ChatMessage(role="user", content=text)],
                )
            )
        return len(text.split())

    fixed = f"Measurement nonce: {nonce}\n"
    fixed_tokens = await count(fixed)
    raw_budget = raw_workload_budget(
        context_size,
        max_output_tokens,
        fixed_tokens,
        fraction=TUNE_WORKLOAD_FRACTION,
    )
    if raw_budget < MIN_WORKLOAD_TOKENS:
        return None, fixed_tokens, "workload_budget_below_minimum"
    filler = await build_filler_async(raw_budget, count)
    prompt = fixed + filler
    measured = await count(prompt)
    target = tune_target_prompt_tokens(context_size, max_output_tokens)
    for _ in range(4):
        if measured <= target:
            break
        filler = await build_filler_async(max(0, raw_budget - (measured - target)), count)
        prompt = fixed + filler
        measured = await count(prompt)
    if measured > target:
        return None, measured, "sized_prompt_exceeds_budget"
    return prompt, measured, None


async def measure(payload: dict[str, Any]) -> dict[str, Any]:
    model = ModelDefinition.model_validate(payload["model"])
    home = Path(str(payload["home"])).resolve()
    registry = ModelRegistry.from_dict(
        {"models": {model.id: model.model_dump(mode="json")}}, root=home
    )
    lifecycle = ModelLifecycle(registry, root_backend="llama_cpp", prefix_cache_enabled=False)
    await lifecycle.generate(
        ChatRequest(
            model_id=model.id,
            messages=[ChatMessage(role="user", content=TUNE_WARMUP_PROMPT)],
            options=GenerationOptions(temperature=0.0, max_output_tokens=1, seed=17),
        )
    )
    metrics: list[dict[str, Any]] = []
    for index in range(int(payload.get("runs", 3))):
        started = time.monotonic()
        prompt, workload_tokens, workload_skip_reason = await _sized_workload_prompt(
            lifecycle,
            model.id,
            f"{payload.get('nonce_prefix', 'run')}-{index}",
            model.context_size,
            TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
        )
        if prompt is None:
            await lifecycle.cleanup()
            return {
                "metrics": [],
                "routing_decisions": [],
                "workload_skipped": {
                    "reason": workload_skip_reason or "workload_unavailable",
                    "token_count": workload_tokens,
                },
                "peak_rss_bytes": _rss_bytes(),
                "valid_prefill": False,
            }
        response = await lifecycle.generate(
            ChatRequest(
                model_id=model.id,
                messages=[
                    ChatMessage(
                        role="user",
                        content=prompt,
                    )
                ],
                options=GenerationOptions(
                    temperature=0.0,
                    max_output_tokens=TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
                    seed=17,
                ),
            )
        )
        timing = response.usage.timing or {}
        prompt_tokens = int(timing.get("prompt_tokens", response.usage.input_tokens))
        prompt_eval_tokens = int(timing.get("prompt_eval_tokens", 0) or 0)
        metrics.append(
            {
                "metric": float(timing.get("eval_tokens_per_second", 0.0) or 0.0),
                "prompt_metric": float(timing.get("prompt_eval_tokens_per_second", 0.0) or 0.0),
                "prompt_tokens": prompt_tokens,
                "prompt_eval_tokens": prompt_eval_tokens,
                "semantic_digest": hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
                "elapsed_ms": (time.monotonic() - started) * 1000.0,
            }
        )
    check = await lifecycle.generate(
        ChatRequest(
            model_id=model.id,
            messages=[
                ChatMessage(role="user", content="A fixed nonce-free semantic check workload.")
            ],
            options=GenerationOptions(
                temperature=0.0,
                max_output_tokens=TUNE_MEASUREMENT_MAX_OUTPUT_TOKENS,
                seed=17,
            ),
        )
    )
    routing_decisions: list[tuple[str, str, str]] = []
    routing_ran = model.role == "brain" and bool(payload.get("routing_check", True))
    if routing_ran:
        import yaml

        fixture = home / "tests" / "fixtures" / "evals" / "brain_routes.yaml"
        raw_cases = yaml.safe_load(fixture.read_text(encoding="utf-8")).get("cases", [])
        router = BrainRouter(cast(Any, _LifecycleClient(lifecycle)), brain_model_id=model.id)
        for case in raw_cases[:12]:
            route = await router.route_result(str(case["message"]))
            routing_decisions.append(
                (
                    route.decision.intent,
                    route.decision.agent,
                    route.decision.tools_needed[0] if route.decision.tools_needed else "none",
                )
            )
    await lifecycle.cleanup()
    return {
        "metrics": metrics,
        "semantic_check_digest": hashlib.sha256(check.content.encode("utf-8")).hexdigest(),
        "routing_decisions": routing_decisions,
        "routing_ran": routing_ran,
        "peak_rss_bytes": _rss_bytes(),
        "valid_prefill": all(
            item["prompt_eval_tokens"] >= 0.9 * item["prompt_tokens"] for item in metrics
        ),
        "workload_token_count": max(
            (int(item.get("prompt_tokens", 0)) for item in metrics), default=0
        ),
    }


def main() -> None:  # pragma: no cover - real GGUF worker entrypoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload)
    print(json.dumps(asyncio.run(measure(payload)), separators=(",", ":")))


if __name__ == "__main__":
    main()
