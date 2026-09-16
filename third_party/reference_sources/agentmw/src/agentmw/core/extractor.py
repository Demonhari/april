"""Pattern extractor: turn a completed session into reusable reasoning patterns.

This closes the loop. Without it, the reasoning library only grows when a
developer remembers to call `memory.save(...)`. With it, the agent's own
successful (or failed) runs are mined automatically and persisted.

Run heuristics:
    - A session is "completable" when the last assistant turn has stop_reason
      "end_turn" or the trace ends without a pending tool_use.
    - Extraction is best-effort: never raises, logs and moves on.
    - Patterns are deduped against existing memory by cosine similarity > 0.92.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from agentmw.core.heuristics import _text_of
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.providers import LLMProvider, ProviderError

logger = logging.getLogger("agentmw.extractor")

Message = dict[str, Any]


SYSTEM = (
    "You read completed agent traces and extract reusable reasoning patterns. "
    "A reusable pattern is a short, concrete heuristic that would help any "
    "future agent attempting a similar task. Avoid restating the task itself. "
    "Avoid hindsight platitudes ('be careful'). Cite concrete signals."
)


PROMPT = """Trace of a finished agent run:
---
{trace}
---

Run outcome: {outcome}

Extract 1 to 3 reusable patterns. Each pattern must be:
- one or two sentences
- actionable on a *different* task in the same domain
- grounded in concrete evidence from THIS trace (a specific tool / file / phrase)

Return ONLY JSON of the form:
{{"patterns": [
  {{"task": "<short description of the type of task this applies to>",
    "pattern": "<the heuristic itself>"}},
  ...
]}}

If the trace contains no transferable lesson, return {{"patterns": []}}."""


@dataclass
class ExtractedPattern:
    task: str
    pattern: str


def is_session_complete(messages: list[Message]) -> bool:
    """True when the trace looks done: ends with assistant text, no pending tool_use."""
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") == "user":
        # could be a pending tool_result waiting for assistant — not complete
        return False
    if last.get("role") != "assistant":
        return False
    content = last.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        # Complete only if no trailing tool_use awaits a result.
        has_text = False
        has_pending_tool = False
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text", "").strip():
                has_text = True
            if b.get("type") == "tool_use":
                has_pending_tool = True
        return has_text and not has_pending_tool
    return False


def _condense(messages: list[Message], max_chars: int = 4000) -> str:
    out: list[str] = []
    used = 0
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            line = f"#{i} [{role}] {content[:300]}"
        elif isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    parts.append(b.get("text", "")[:200])
                elif t == "tool_use":
                    parts.append(f"TOOL {b.get('name', '?')}({json.dumps(b.get('input', {}), default=str)[:80]})")
                elif t == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                    sz = len(c) if isinstance(c, str) else 0
                    parts.append(f"<tool_result {sz}B>")
            line = f"#{i} [{role}] " + " | ".join(parts)
        else:
            line = f"#{i} [{role}]"
        if used + len(line) > max_chars:
            out.append("... (trace truncated)")
            break
        out.append(line)
        used += len(line)
    return "\n".join(out)


def _parse(raw: str) -> list[ExtractedPattern]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    items = data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[ExtractedPattern] = []
    for it in items[:3]:
        if not isinstance(it, dict):
            continue
        task = str(it.get("task", "")).strip()
        pattern = str(it.get("pattern", "")).strip()
        if task and pattern:
            out.append(ExtractedPattern(task=task[:200], pattern=pattern[:600]))
    return out


class PatternExtractor:
    def __init__(self, provider: LLMProvider, memory: ReasoningLibrary, dedup_threshold: float = 0.92) -> None:
        self.provider = provider
        self.memory = memory
        self.dedup_threshold = dedup_threshold

    def _is_duplicate(self, pattern_text: str, task: str) -> bool:
        if not self.memory.semantic_enabled:
            return False
        existing = self.memory.recall(task, limit=5)
        for ex in existing:
            if ex.score >= self.dedup_threshold:
                if pattern_text.strip().lower() == ex.pattern_text.strip().lower():
                    return True
        return False

    def extract(self, messages: list[Message], outcome: str = "success") -> list[ExtractedPattern]:
        if not getattr(self.provider, "available", False):
            return []
        prompt = PROMPT.format(trace=_condense(messages), outcome=outcome)
        try:
            raw = self.provider.judge(SYSTEM, prompt)
        except ProviderError as e:
            logger.warning("extractor: provider failed (%s)", e)
            return []
        candidates = _parse(raw)
        saved: list[ExtractedPattern] = []
        for c in candidates:
            if self._is_duplicate(c.pattern, c.task):
                logger.info("extractor: skipped duplicate pattern")
                continue
            self.memory.save(task=c.task, pattern_text=c.pattern, outcome=outcome)
            saved.append(c)
        logger.info("extractor: saved %d new patterns (of %d candidates)", len(saved), len(candidates))
        return saved


def extract_in_background(
    provider: LLMProvider,
    memory: ReasoningLibrary,
    messages: list[Message],
    outcome: str = "success",
) -> threading.Thread:
    """Fire-and-forget extraction. Caller may join() the returned thread if desired."""

    def _run() -> None:
        try:
            PatternExtractor(provider, memory).extract(list(messages), outcome=outcome)
        except Exception as e:  # noqa: BLE001 — background worker must never raise
            logger.warning("extractor background worker failed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name="agentmw-extractor")
    t.start()
    return t
