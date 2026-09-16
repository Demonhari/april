"""Heuristic fallback monitors.

These are rule-based, deterministic, dependency-free monitors. They are run
ONLY when the configured LLM provider is unavailable (`pipeline.use_llm` off,
or `provider == none`, or provider unreachable). They are also used as a
fast prefilter if `pipeline.heuristics_prefilter` is enabled.

The LLM monitor is the primary path; regex is the safety net.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable

from agentmw.core.config import MonitorConfig

Message = dict[str, Any]


# `MonitorResult` is defined in `monitors.py` to keep a single public type;
# importing it here would create a cycle. We import at call-site.

def _iter_tool_uses(messages: list[Message]) -> Iterable[tuple[int, dict[str, Any]]]:
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield i, block


def _tool_signature(block: dict[str, Any]) -> str:
    name = block.get("name", "")
    raw_input = block.get("input", {})
    try:
        serialized = json.dumps(raw_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = repr(raw_input)
    return f"{name}::{serialized}"


def _text_of(msg: Message) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


_SECOND_GUESS_PATTERNS = [
    r"\bactually,?\s+(let me|i should|wait)\b",
    r"\bwait,?\s+(no|that'?s wrong|let me reconsider)\b",
    r"\blet me reconsider\b",
    r"\bon second thought\b",
    r"\bi was wrong\b",
]
_SECOND_GUESS_RE = re.compile("|".join(_SECOND_GUESS_PATTERNS), re.IGNORECASE)


def loop_monitor(messages: list[Message], cfg: MonitorConfig):
    from agentmw.core.monitors import MonitorResult
    sigs = [_tool_signature(b) for _, b in _iter_tool_uses(messages)][-cfg.loop_window:]
    if not sigs:
        return MonitorResult("loop", False)
    counts: dict[str, int] = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1])
    if worst[1] >= cfg.loop_threshold:
        name = worst[0].split("::", 1)[0]
        return MonitorResult(
            "loop", True,
            reason=f"Tool `{name}` called {worst[1]}x with identical input in last {cfg.loop_window} calls.",
            correction=(
                f"You appear to be looping: `{name}` was called {worst[1]}x with the same arguments. "
                "Stop repeating it. Either change strategy, summarize what you know, or ask for clarification."
            ),
        )
    return MonitorResult("loop", False)


def redundant_tool_call_monitor(messages: list[Message], cfg: MonitorConfig):
    from agentmw.core.monitors import MonitorResult
    uses = list(_iter_tool_uses(messages))
    if len(uses) < 2:
        return MonitorResult("redundant_tool_call", False)
    last_sig = _tool_signature(uses[-1][1])
    earlier = [_tool_signature(b) for _, b in uses[:-1]]
    if last_sig in earlier:
        name = last_sig.split("::", 1)[0]
        return MonitorResult(
            "redundant_tool_call", True,
            reason=f"Tool `{name}` was already invoked with these exact arguments earlier in this run.",
            correction=(
                f"Tool `{name}` already returned a result for these arguments. "
                "Re-read the earlier tool_result instead of calling it again."
            ),
        )
    return MonitorResult("redundant_tool_call", False)


def second_guessing_monitor(messages: list[Message], cfg: MonitorConfig):
    from agentmw.core.monitors import MonitorResult
    hits = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        hits += len(_SECOND_GUESS_RE.findall(_text_of(msg)))
    if hits >= cfg.second_guess_min_count:
        return MonitorResult(
            "second_guessing", True,
            reason=f"Detected {hits} self-correction phrases in assistant turns.",
            correction=(
                "You are second-guessing yourself. Commit to one approach, state it explicitly, "
                "and execute it. If you have new evidence, name it; otherwise stop reversing course."
            ),
        )
    return MonitorResult("second_guessing", False)


def run_heuristics(messages: list[Message], cfg: MonitorConfig) -> list:
    return [
        loop_monitor(messages, cfg),
        redundant_tool_call_monitor(messages, cfg),
        second_guessing_monitor(messages, cfg),
    ]
