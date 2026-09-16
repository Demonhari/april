"""Primary monitor: a single LLM call that classifies the latest turn for
loops, redundancy, contradictions, hallucinations, or abandonment.

One call replaces N independent monitors. Output is structured JSON. If the
provider is unreachable or returns garbage, we surface that to the caller so
the heuristic fallback can kick in.

The trace fed to the model is a *compressed view* of the last few turns —
not the entire history — to keep cost predictable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentmw.core.heuristics import _text_of, _tool_signature
from agentmw.core.providers import LLMProvider, ProviderError

logger = logging.getLogger("agentmw.llm_monitor")

Message = dict[str, Any]


SYSTEM = (
    "You are a strict reviewer of an AI coding agent's recent turns. "
    "Your job is to identify, with high precision, when the agent is "
    "wasting tokens or going off-track. Be conservative: only flag when "
    "evidence is concrete in the supplied trace."
)

PROMPT_TEMPLATE = """Analyze this agent trace (most recent turns first).

For each issue category below, decide whether it is occurring and provide a
short reason citing concrete evidence from the trace. Return ONLY a JSON
object with this shape:

{{
  "issues": [
    {{"type": "<one of: loop | redundant_tool_call | contradiction | abandonment | hallucination>",
      "reason": "<one short sentence, must cite specific tool name / phrase / step>"}}
  ]
}}

Definitions:
- loop:               the same tool was called more than twice with identical inputs.
- redundant_tool_call: the latest tool call repeats one already in this trace.
- contradiction:       the assistant reversed a stated claim without new evidence.
- abandonment:         the assistant dropped a stated plan without justification.
- hallucination:       the assistant cited a file, line, or output that does not appear in any tool_result.

If nothing applies, return {{"issues": []}}.

Trace:
---
{trace}
---"""


def _condense_for_review(messages: list[Message], last_n: int = 8) -> str:
    """Render the last N turns as a compact text for the model to inspect."""
    selected = messages[-last_n:]
    lines: list[str] = []
    for i, msg in enumerate(selected, start=max(1, len(messages) - len(selected) + 1)):
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(f"#{i} [{role}] {content[:300]}")
            continue
        if not isinstance(content, list):
            lines.append(f"#{i} [{role}] <empty>")
            continue
        parts: list[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", "")[:300])
            elif t == "tool_use":
                inp = b.get("input", {})
                inp_str = json.dumps(inp, default=str)[:120] if isinstance(inp, dict) else str(inp)[:120]
                parts.append(f"TOOL_CALL {b.get('name', '?')}({inp_str})")
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    text = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                else:
                    text = str(c) if c is not None else ""
                lines_count = text.count("\n") + 1
                lines.append(f"#{i} [tool_result] {text[:240]}{'…' if len(text) > 240 else ''} ({lines_count} lines)")
                continue
        if parts:
            lines.append(f"#{i} [{role}] " + " | ".join(parts))
    return "\n".join(lines)


def _parse_issues(raw: str) -> list[dict]:
    if not raw:
        return []
    # Tolerate models that wrap JSON in markdown fences or chat.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    issues = parsed.get("issues", [])
    return issues if isinstance(issues, list) else []


def llm_review(
    messages: list[Message],
    provider: LLMProvider,
    last_n: int = 8,
) -> list:
    """Run a single LLM review pass. Returns a list of MonitorResult.

    Raises ProviderError on transport / authentication problems so the caller
    can decide whether to fall back to heuristics.
    """
    from agentmw.core.monitors import MonitorResult  # local import to avoid cycle

    if not getattr(provider, "available", False):
        raise ProviderError(f"provider {provider.name} not available")

    trace = _condense_for_review(messages, last_n=last_n)
    prompt = PROMPT_TEMPLATE.format(trace=trace)
    raw = provider.judge(SYSTEM, prompt)
    issues = _parse_issues(raw)

    results: list = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("type", "")).strip().lower()
        reason = str(issue.get("reason", "")).strip()
        if not kind or kind == "none" or not reason:
            continue
        results.append(MonitorResult(
            name=f"llm:{kind}",
            triggered=True,
            reason=reason,
            correction=(
                f"A reviewer flagged this turn as {kind}: {reason} "
                "Re-anchor on the original task, cite evidence explicitly, and avoid "
                "repeating actions you already performed."
            ),
        ))
    return results
