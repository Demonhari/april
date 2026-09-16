"""LLM provider abstraction for monitors and judges.

Implementations:
    - OllamaProvider:    local or remote Ollama server
    - OpenAIProvider:    OpenAI Chat Completions (or any OpenAI-compatible endpoint via base_url)
    - AnthropicProvider: Anthropic Messages API
    - OpenRouterProvider: OpenAI-compatible relay; routes to many models

Each implementation:
    - reads its dependency lazily
    - never raises on `available` checks (returns False on any error)
    - retries transient HTTP errors per `ProviderConfig.max_retries`
    - returns the raw model text — callers parse JSON themselves
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from agentmw.core.config import ProviderConfig

logger = logging.getLogger("agentmw.providers")


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    available: bool
    model: str

    def judge(self, system: str, prompt: str) -> str: ...


@dataclass
class NoneProvider:
    """Explicit "no provider". Used when LLM monitoring is disabled."""

    name: str = "none"
    available: bool = False
    model: str = ""

    def judge(self, system: str, prompt: str) -> str:
        raise ProviderError("provider disabled")


def _retry(fn, retries: int, base_delay: float = 0.4):
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(base_delay * (2**attempt))
            else:
                raise ProviderError(f"transient failure after {retries+1} attempts: {e}") from e
    assert last_exc is not None  # for type checker
    raise ProviderError(str(last_exc))


def _http_json(
    url: str,
    body: dict,
    headers: dict[str, str],
    timeout: float,
    method: str = "POST",
) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------- Ollama ----------

class OllamaProvider:
    name = "ollama"

    def __init__(self, cfg: ProviderConfig) -> None:
        self._cfg = cfg
        self.host = cfg.base_url or "http://localhost:11434"
        self.model = cfg.model or "llama3.2:3b"

    @property
    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=1.0):
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def judge(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "prompt": f"{system}\n\n{prompt}",
            "stream": False,
        }
        result = _retry(
            lambda: _http_json(
                f"{self.host}/api/generate",
                body,
                {"Content-Type": "application/json"},
                self._cfg.timeout_seconds,
            ),
            retries=self._cfg.max_retries,
        )
        return str(result.get("response", "")).strip()


# ---------- OpenAI (and OpenAI-compatible) ----------

class OpenAIProvider:
    name = "openai"

    def __init__(self, cfg: ProviderConfig) -> None:
        self._cfg = cfg
        self.base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = cfg.model or "gpt-4o-mini"
        self.api_key = cfg.api_key

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def judge(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        result = _retry(
            lambda: _http_json(
                f"{self.base_url}/chat/completions",
                body,
                headers,
                self._cfg.timeout_seconds,
            ),
            retries=self._cfg.max_retries,
        )
        choices = result.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()


# ---------- Anthropic ----------

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, cfg: ProviderConfig) -> None:
        self._cfg = cfg
        self.base_url = (cfg.base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.model = cfg.model or "claude-haiku-4-5-20251001"
        self.api_key = cfg.api_key

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def judge(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": 512,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        result = _retry(
            lambda: _http_json(
                f"{self.base_url}/messages",
                body,
                headers,
                self._cfg.timeout_seconds,
            ),
            retries=self._cfg.max_retries,
        )
        blocks = result.get("content") or []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                return str(b.get("text", "")).strip()
        return ""


# ---------- OpenRouter (OpenAI-compatible relay) ----------

class OpenRouterProvider(OpenAIProvider):
    name = "openrouter"

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        self.base_url = (cfg.base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.model = cfg.model or "anthropic/claude-haiku-4.5"


# ---------- Factory ----------

_REGISTRY: dict[str, type] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
    "none": NoneProvider,
}


def _build(name: str, cfg: ProviderConfig) -> LLMProvider:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown provider: {name}")
    if cls is NoneProvider:
        return NoneProvider()
    return cls(cfg)  # type: ignore[return-value]


def select_provider(cfg: ProviderConfig) -> LLMProvider:
    """Build a provider per cfg.name, or auto-pick the first available."""
    if cfg.name != "auto":
        return _build(cfg.name, cfg)

    for candidate in ("ollama", "anthropic", "openai", "openrouter"):
        try:
            p = _build(candidate, cfg)
        except Exception as e:
            logger.debug("provider %s unavailable: %s", candidate, e)
            continue
        if getattr(p, "available", False):
            logger.info("auto-selected provider: %s (model=%s)", p.name, p.model)
            return p
    return NoneProvider()
