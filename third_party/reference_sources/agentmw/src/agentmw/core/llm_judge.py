"""Backward-compatible single-turn judge.

New code should use `agentmw.core.llm_monitor.llm_review` with a configured
provider. This module remains as a stable shim for existing call sites and
tests; it transparently uses the provider abstraction underneath.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from agentmw.core.config import ProviderConfig
from agentmw.core.heuristics import _text_of
from agentmw.core.monitors import MonitorResult
from agentmw.core.providers import OllamaProvider, ProviderError

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AGENTMW_JUDGE_MODEL", "llama3.2:3b")

_PROMPT = """You are a strict reviewer of an AI agent's last turn.

Given the assistant's most recent message below, answer in JSON:
{{"problem": "<one of: none | contradiction | abandonment | hallucination>", "reason": "<one short sentence>"}}

- contradiction: the assistant reversed an earlier claim without new evidence.
- abandonment:  the assistant gave up on a stated plan without justification.
- hallucination: the assistant cited a tool output, file, or fact not visible in the turn.
- none: nothing wrong.

Assistant turn:
---
{text}
---

Respond with ONLY the JSON object."""


def _ollama_available(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 0.5) -> bool:
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _call_ollama(prompt: str, model: str, host: str, timeout: float) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "").strip()


def llm_judge_monitor(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 15.0,
) -> MonitorResult:
    """Single-turn judge backed by Ollama (kept for backward compatibility)."""
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if not last_assistant:
        return MonitorResult("llm_judge", False)
    text = _text_of(last_assistant)
    if not text.strip():
        return MonitorResult("llm_judge", False)
    if not _ollama_available(host):
        return MonitorResult("llm_judge", False, reason="Ollama unreachable; skipped.")
    try:
        raw = _call_ollama(_PROMPT.format(text=text[:2000]), model, host, timeout)
    except Exception as e:
        return MonitorResult("llm_judge", False, reason=f"judge error: {e}")
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return MonitorResult("llm_judge", False, reason=f"unparseable judge output: {raw[:80]}")
    problem = (parsed.get("problem") or "none").lower()
    if problem == "none":
        return MonitorResult("llm_judge", False)
    return MonitorResult(
        "llm_judge", True,
        reason=f"{problem}: {parsed.get('reason', '')}",
        correction=(
            f"A reviewer flagged this turn as {problem}. {parsed.get('reason', '')} "
            "Re-anchor on the original task before continuing."
        ),
    )
