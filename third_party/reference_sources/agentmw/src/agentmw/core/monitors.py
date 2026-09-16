"""Monitor orchestrator.

LLM monitor is primary. Heuristics are fallback (and optional prefilter).
Public API (`MonitorResult`, `MonitorReport`, `run_monitors`) is stable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agentmw.core.config import AgentmwConfig, MonitorConfig, default_config
from agentmw.core.providers import LLMProvider, ProviderError, select_provider

logger = logging.getLogger("agentmw.monitors")


@dataclass
class MonitorResult:
    name: str
    triggered: bool
    reason: str = ""
    correction: str = ""

    def __bool__(self) -> bool:
        return self.triggered


Message = dict[str, Any]
Monitor = Callable[[list[Message]], MonitorResult]


@dataclass
class MonitorReport:
    results: list[MonitorResult] = field(default_factory=list)
    used_llm: bool = False
    used_heuristics: bool = False
    provider_name: str = ""

    @property
    def triggered(self) -> list[MonitorResult]:
        return [r for r in self.results if r.triggered]

    def correction_text(self) -> str:
        parts = [f"[{r.name}] {r.correction}" for r in self.triggered if r.correction]
        return "\n".join(parts)


def _dedup(results: list[MonitorResult]) -> list[MonitorResult]:
    """LLM and heuristics may both flag the same class — keep the LLM verdict."""
    seen: set[str] = set()
    out: list[MonitorResult] = []
    # LLM results have name prefixed "llm:"; prefer them.
    for r in sorted(results, key=lambda x: 0 if x.name.startswith("llm:") else 1):
        canonical = r.name.split(":", 1)[-1] if r.name.startswith("llm:") else r.name
        # normalize llm category names to canonical names already used by heuristics
        canonical_map = {
            "loop": "loop",
            "redundant_tool_call": "redundant_tool_call",
            "contradiction": "contradiction",
            "abandonment": "abandonment",
            "hallucination": "hallucination",
            "second_guessing": "second_guessing",
        }
        canonical = canonical_map.get(canonical, canonical)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(r)
    return out


def run_monitors(
    messages: list[Message],
    monitors: list[Monitor] | None = None,
    *,
    config: AgentmwConfig | None = None,
    provider: LLMProvider | None = None,
) -> MonitorReport:
    """Run the monitor pipeline.

    - If `monitors` is provided explicitly, run only those (legacy path).
    - Otherwise, orchestrate: LLM primary + heuristics fallback per `config.pipeline`.
    """
    cfg = config or default_config()

    # Legacy path: caller provides explicit monitors.
    if monitors is not None:
        return MonitorReport(
            results=[m(messages) for m in monitors],
            used_heuristics=True,
        )

    pipeline = cfg.pipeline
    results: list[MonitorResult] = []
    used_llm = False
    used_heuristics = False
    prov = provider if provider is not None else (select_provider(cfg.provider) if pipeline.use_llm else None)

    # 1. Optional heuristic prefilter (cheap, deterministic).
    if pipeline.heuristics_prefilter:
        from agentmw.core.heuristics import run_heuristics
        results.extend(run_heuristics(messages, cfg.monitors))
        used_heuristics = True

    # 2. Primary LLM review.
    if pipeline.use_llm and prov is not None and getattr(prov, "available", False):
        from agentmw.core.llm_monitor import llm_review
        try:
            results.extend(llm_review(messages, prov))
            used_llm = True
        except ProviderError as e:
            logger.warning("LLM monitor failed (%s); falling back to heuristics.", e)
            if pipeline.use_heuristics_fallback and not used_heuristics:
                from agentmw.core.heuristics import run_heuristics
                results.extend(run_heuristics(messages, cfg.monitors))
                used_heuristics = True

    # 3. Heuristics fallback if nothing else ran.
    if not results and pipeline.use_heuristics_fallback and not used_heuristics:
        from agentmw.core.heuristics import run_heuristics
        results.extend(run_heuristics(messages, cfg.monitors))
        used_heuristics = True

    return MonitorReport(
        results=_dedup(results),
        used_llm=used_llm,
        used_heuristics=used_heuristics,
        provider_name=prov.name if prov is not None else "",
    )


# --- Backward-compat re-exports (the regex monitors used to live here) ---

from agentmw.core.heuristics import (  # noqa: E402
    _SECOND_GUESS_RE,
    _iter_tool_uses,
    _text_of,
    _tool_signature,
    loop_monitor as _h_loop_monitor,
    redundant_tool_call_monitor as _h_redundant_monitor,
    second_guessing_monitor as _h_second_guess_monitor,
)


def loop_monitor(messages: list[Message]) -> MonitorResult:
    """Backward-compatible heuristic loop monitor with default thresholds."""
    return _h_loop_monitor(messages, MonitorConfig())


def redundant_tool_call_monitor(messages: list[Message]) -> MonitorResult:
    return _h_redundant_monitor(messages, MonitorConfig())


def second_guessing_monitor(messages: list[Message]) -> MonitorResult:
    return _h_second_guess_monitor(messages, MonitorConfig())


DEFAULT_MONITORS: list[Monitor] = [
    loop_monitor,
    redundant_tool_call_monitor,
    second_guessing_monitor,
]


__all__ = [
    "MonitorResult",
    "MonitorReport",
    "Monitor",
    "run_monitors",
    "loop_monitor",
    "redundant_tool_call_monitor",
    "second_guessing_monitor",
    "DEFAULT_MONITORS",
    "_iter_tool_uses",
    "_text_of",
    "_tool_signature",
]
