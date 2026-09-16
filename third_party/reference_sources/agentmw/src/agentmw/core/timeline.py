"""Time-travel: walk a saved trace step-by-step, find the exact turn at which
each monitor would have fired for the first time, and quantify the waste
that followed.

A trace is a JSON list of Anthropic-style messages. We rebuild monitor state
incrementally so we can name the *point of no return* precisely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentmw.core.config import AgentmwConfig, default_config
from agentmw.core.memory import ReasoningLibrary, Pattern
from agentmw.core.monitors import MonitorResult, _text_of, run_monitors
from agentmw.core.providers import LLMProvider, select_provider

Message = dict[str, Any]


@dataclass
class TimelineStep:
    idx: int
    role: str
    summary: str
    bytes_at_step: int
    new_fires: list[MonitorResult] = field(default_factory=list)
    all_fires: list[MonitorResult] = field(default_factory=list)
    judge_result: MonitorResult | None = None


@dataclass
class TimelineReport:
    steps: list[TimelineStep]
    total_bytes: int
    first_fire_step: int | None
    wasted_bytes_after_first_fire: int
    wasted_messages_after_first_fire: int
    recalled_patterns: list[Pattern]

    def first_fire(self) -> tuple[int, list[MonitorResult]] | None:
        for s in self.steps:
            if s.new_fires:
                return s.idx, s.new_fires
        return None


def _summarize(msg: Message) -> str:
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, str):
        return content[:80].replace("\n", " ")
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", "")[:60].replace("\n", " "))
            elif t == "tool_use":
                inp = b.get("input", {})
                inp_str = json.dumps(inp, default=str)[:50] if isinstance(inp, dict) else str(inp)[:50]
                parts.append(f"{b.get('name', '?')}({inp_str})")
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, str):
                    n = len(c)
                elif isinstance(c, list):
                    n = sum(len(x.get("text", "")) for x in c if isinstance(x, dict))
                else:
                    n = 0
                parts.append(f"<tool_result {n}B>")
        return " | ".join(parts) if parts else f"<{role}>"
    return f"<{role}>"


def _bytes_of(messages: list[Message]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    for v in b.values():
                        n += len(str(v))
    return n


def build_timeline(
    messages: list[Message],
    *,
    config: AgentmwConfig | None = None,
    provider: LLMProvider | None = None,
    memory: ReasoningLibrary | None = None,
    use_judge: bool = False,
) -> TimelineReport:
    """Walk a saved trace and identify the first point at which any monitor fires.

    `provider` overrides the auto-selected LLM provider for monitors / judge.
    `use_judge` adds a per-assistant-turn judge call (slower, more thorough).
    """
    cfg = config or default_config()
    if use_judge and provider is None and cfg.pipeline.use_llm:
        provider = select_provider(cfg.provider)
    seen_fires: set[str] = set()
    steps: list[TimelineStep] = []
    first_fire_step: int | None = None

    for i in range(1, len(messages) + 1):
        prefix = messages[:i]
        last = messages[i - 1]
        report = run_monitors(prefix, config=cfg, provider=provider) if use_judge else run_monitors(
            prefix, config=_heuristics_only_cfg(cfg)
        )
        all_fires = report.triggered
        new_fires = [r for r in all_fires if r.name not in seen_fires]
        for r in new_fires:
            seen_fires.add(r.name)

        judge = None
        if use_judge and last.get("role") == "assistant" and provider is not None and getattr(provider, "available", False):
            from agentmw.core.llm_monitor import llm_review
            from agentmw.core.providers import ProviderError
            try:
                issues = llm_review(prefix, provider)
                judge = issues[0] if issues else None
            except ProviderError as e:
                judge = MonitorResult("llm_judge", False, reason=f"judge skipped: {e}")

        step = TimelineStep(
            idx=i,
            role=last.get("role", "?"),
            summary=_summarize(last),
            bytes_at_step=_bytes_of(prefix),
            new_fires=new_fires,
            all_fires=all_fires,
            judge_result=judge,
        )
        steps.append(step)
        if first_fire_step is None and (new_fires or (judge and judge.triggered)):
            first_fire_step = i

    total = _bytes_of(messages)
    wasted = 0
    wasted_msgs = 0
    if first_fire_step is not None:
        wasted = total - steps[first_fire_step - 1].bytes_at_step
        wasted_msgs = len(messages) - first_fire_step

    recalled: list[Pattern] = []
    if memory is not None:
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user:
            text = _text_of(first_user) if isinstance(first_user.get("content"), list) else (first_user.get("content") or "")
            if isinstance(text, str) and text.strip():
                recalled = memory.recall(text, limit=3)

    return TimelineReport(
        steps=steps,
        total_bytes=total,
        first_fire_step=first_fire_step,
        wasted_bytes_after_first_fire=wasted,
        wasted_messages_after_first_fire=wasted_msgs,
        recalled_patterns=recalled,
    )


def render_ascii(report: TimelineReport) -> str:
    """Render a timeline report as a human-readable ASCII tree."""
    lines: list[str] = []
    header = f"timeline · {len(report.steps)} steps · {report.total_bytes:,} bytes"
    lines.append("┌─ " + header + " " + "─" * max(0, 60 - len(header)))
    lines.append("│")
    for s in report.steps:
        role_tag = {"user": "user      ", "assistant": "assistant ", "tool": "tool      "}.get(s.role, s.role + " ")
        bar = f"│ #{s.idx:02d} [{role_tag}] "
        lines.append(bar + s.summary[:90])
        if s.judge_result and s.judge_result.triggered:
            lines.append(f"│        ⚖  llm_judge: {s.judge_result.reason}")
        for f in s.new_fires:
            lines.append(f"│        ⚠  {f.name} FIRED — {f.reason}")
        if s.idx == report.first_fire_step:
            lines.append(f"│        ⏵  POINT OF NO RETURN: {report.wasted_bytes_after_first_fire:,} bytes "
                         f"and {report.wasted_messages_after_first_fire} messages wasted after this step.")
    lines.append("│")
    if report.recalled_patterns:
        lines.append(f"└─ Reasoning-library patterns applicable to this task: {len(report.recalled_patterns)}")
        for p in report.recalled_patterns:
            score = f" (score {p.score:.2f})" if p.score else ""
            mark = "✓" if p.outcome == "success" else "✗"
            lines.append(f"   {mark} {p.pattern_text}{score}")
    else:
        lines.append("└─ No reasoning-library patterns matched.")
    return "\n".join(lines)


def _heuristics_only_cfg(base: AgentmwConfig) -> AgentmwConfig:
    """A clone of `base` with the LLM monitor disabled, for incremental
    step-by-step walks (where calling the LLM N times would be wasteful)."""
    import copy
    clone = copy.deepcopy(base)
    clone.pipeline.use_llm = False
    clone.pipeline.heuristics_prefilter = True
    clone.pipeline.use_heuristics_fallback = True
    return clone


def load_trace(path: str) -> list[Message]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise ValueError("trace JSON must be a list of messages or {messages: [...]}")
    return data
