"""Counterfactual replay: from the point of no return, ask a provider to
predict the next few steps assuming the agent had accepted the monitor's
correction. Render side-by-side with the original.

This is a *simulation* — no tools are actually executed. The goal is to give
a developer a vivid picture of where the run would have diverged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agentmw.core.heuristics import _text_of
from agentmw.core.monitors import MonitorResult
from agentmw.core.providers import LLMProvider, ProviderError
from agentmw.core.timeline import TimelineReport

logger = logging.getLogger("agentmw.replay")

Message = dict[str, Any]


SYSTEM = (
    "You are simulating a coding agent that has just received a correction "
    "from a runtime monitor. Predict the next 2 or 3 steps the agent would "
    "take if it accepted the correction. Stay concise and concrete."
)

PROMPT_TEMPLATE = """Original trace up to the point of correction:
{trace}

A runtime monitor flagged a problem here:
  type: {monitor_name}
  reason: {monitor_reason}
  suggested correction: {monitor_correction}

Reasoning patterns relevant to this task (from prior runs):
{patterns}

Predict what the agent should do next, accepting the correction. Return ONLY
JSON of the form:

{{"steps": [{{"action": "<tool name or 'reflect'>", "rationale": "<one sentence>"}}, ...]}}

Maximum 3 steps. Be concrete; cite specific files or symbols when possible."""


@dataclass
class GhostStep:
    action: str
    rationale: str


@dataclass
class Counterfactual:
    branch_at: int
    monitor: MonitorResult
    ghost_steps: list[GhostStep]
    raw_response: str = ""


def _condense(messages: list[Message], n: int = 8) -> str:
    selected = messages[-n:]
    out: list[str] = []
    for i, msg in enumerate(selected, start=max(1, len(messages) - len(selected) + 1)):
        role = msg.get("role", "?")
        text = _text_of(msg)
        content = msg.get("content")
        if isinstance(content, list):
            extras = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    inp = b.get("input", {})
                    try:
                        inp_s = json.dumps(inp, default=str)[:100]
                    except (TypeError, ValueError):
                        inp_s = str(inp)[:100]
                    extras.append(f"{b.get('name', '?')}({inp_s})")
                elif isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    sz = len(c) if isinstance(c, str) else 0
                    extras.append(f"<tool_result {sz}B>")
            if extras:
                text = (text + " | " + " ; ".join(extras)).strip(" |")
        out.append(f"#{i} [{role}] {text[:240]}")
    return "\n".join(out)


def build_counterfactual(
    messages: list[Message],
    report: TimelineReport,
    provider: LLMProvider,
) -> Counterfactual | None:
    """Use the provider to simulate the divergent branch from first-fire."""
    if not getattr(provider, "available", False):
        raise ProviderError(f"provider {provider.name} not available")
    if report.first_fire_step is None:
        return None
    step = report.steps[report.first_fire_step - 1]
    monitor = step.new_fires[0] if step.new_fires else (step.judge_result or MonitorResult("?", False))
    if not monitor.triggered:
        return None

    prefix = messages[: report.first_fire_step]
    patterns_block = "\n".join(
        f"- {p.pattern_text}" for p in report.recalled_patterns
    ) or "(none)"

    prompt = PROMPT_TEMPLATE.format(
        trace=_condense(prefix),
        monitor_name=monitor.name,
        monitor_reason=monitor.reason,
        monitor_correction=monitor.correction or monitor.reason,
        patterns=patterns_block,
    )
    raw = provider.judge(SYSTEM, prompt)

    ghost: list[GhostStep] = []
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(raw[start : end + 1])
            for s in (data.get("steps") or [])[:3]:
                if isinstance(s, dict):
                    ghost.append(GhostStep(
                        action=str(s.get("action", ""))[:120],
                        rationale=str(s.get("rationale", ""))[:240],
                    ))
        except (ValueError, json.JSONDecodeError):
            pass

    return Counterfactual(
        branch_at=report.first_fire_step,
        monitor=monitor,
        ghost_steps=ghost,
        raw_response=raw,
    )


def render_counterfactual(
    report: TimelineReport,
    counterfactual: Counterfactual,
) -> str:
    """ASCII side-by-side: original wasted steps vs the ghost branch."""
    lines: list[str] = []
    lines.append("=" * 76)
    lines.append(f"COUNTERFACTUAL at step #{counterfactual.branch_at}")
    lines.append(f"  monitor: {counterfactual.monitor.name}")
    lines.append(f"  reason:  {counterfactual.monitor.reason}")
    lines.append("=" * 76)
    lines.append("")
    original_tail = [s for s in report.steps[counterfactual.branch_at - 1 :]]
    left_label = "ORIGINAL (what actually happened)"
    right_label = "GHOST (with correction accepted)"
    lines.append(f"  {left_label:<36}│  {right_label}")
    lines.append("  " + "─" * 36 + "┼─" + "─" * 36)
    max_rows = max(len(original_tail), len(counterfactual.ghost_steps))
    for i in range(max_rows):
        left = ""
        right = ""
        if i < len(original_tail):
            s = original_tail[i]
            left = f"#{s.idx} [{s.role[:8]}] {s.summary[:24]}"
        if i < len(counterfactual.ghost_steps):
            g = counterfactual.ghost_steps[i]
            right = f"{g.action[:22]} — {g.rationale[:28]}"
        lines.append(f"  {left:<36}│  {right}")
    lines.append("")
    return "\n".join(lines)
