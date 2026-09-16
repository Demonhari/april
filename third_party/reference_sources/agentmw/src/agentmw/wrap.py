"""wrap() / wrap_async() — production middleware for LLM clients.

For every `messages.create(...)`:
    1. Run the monitor pipeline (LLM primary, heuristics fallback).
    2. Recall reasoning patterns from memory; inject as a system note.
    3. Compress stale tool_results.
    4. Forward to the inner client.
    5. Auto-record the session to disk (if configured).
    6. If the session looks complete, auto-extract reusable patterns in
       a background thread.
    7. Update telemetry counters.

The provider is wrapped in a circuit breaker so a failing monitor cannot
slow the inner LLM call. Anything not yet wired falls back to a no-op so
the wrapped client always behaves like the inner client on the happy path.

Async clients use `wrap_async(...)` which exposes the same `.messages.create`
coroutine surface.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentmw.core.breaker import BreakerConfig as BreakerCfg, CircuitBreakerProvider
from agentmw.core.compression import CompressionStats, compress_history
from agentmw.core.config import AgentmwConfig, default_config
from agentmw.core.extractor import extract_in_background, is_session_complete
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.monitors import MonitorReport, run_monitors
from agentmw.core.providers import LLMProvider, ProviderError, select_provider
from agentmw.core.sessions import Session, SessionStore
from agentmw.core.telemetry import Telemetry, global_telemetry

logger = logging.getLogger("agentmw.wrap")


@dataclass
class CallTrace:
    monitors: MonitorReport
    compression: CompressionStats
    recalled_patterns: int
    system_note: str = ""
    session_id: str = ""


@dataclass
class WrapConfig:
    """Per-instance toggles. Heavyweight config lives in AgentmwConfig."""

    enable_monitors: bool = True
    enable_compression: bool = True
    enable_memory: bool = True
    traces: list[CallTrace] = field(default_factory=list)


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        return b.get("text", "")
    return ""


def _merge_system(existing: Any, addition: str) -> Any:
    if not addition:
        return existing
    if existing is None or existing == "":
        return addition
    if isinstance(existing, str):
        return existing + "\n\n" + addition
    if isinstance(existing, list):
        return list(existing) + [{"type": "text", "text": addition}]
    return addition


def _response_to_message(response: Any) -> dict | None:
    """Normalize an Anthropic-style response into an assistant message dict.

    Returns None if the response does not look like an Anthropic Messages reply.
    """
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        try:
            data = response.model_dump()
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict) and "content" in data:
            return {"role": "assistant", "content": data["content"]}
    if isinstance(response, dict) and "content" in response:
        return {"role": "assistant", "content": response["content"]}
    return None


def _stop_reason(response: Any) -> str | None:
    if hasattr(response, "stop_reason"):
        return getattr(response, "stop_reason")
    if isinstance(response, dict):
        return response.get("stop_reason")
    return None


def _wrap_provider(provider: LLMProvider | None, cfg: AgentmwConfig, tel: Telemetry | None) -> LLMProvider | None:
    if provider is None or not cfg.breaker.enabled:
        return provider

    def _on_trip() -> None:
        if tel is not None:
            tel.record_breaker_trip()

    return CircuitBreakerProvider(
        provider,
        BreakerCfg(
            failure_threshold=cfg.breaker.failure_threshold,
            failure_window_seconds=cfg.breaker.failure_window_seconds,
            cooldown_seconds=cfg.breaker.cooldown_seconds,
        ),
        on_trip=_on_trip,
    )


class _Lifecycle:
    """Shared logic between sync and async proxies."""

    def __init__(
        self,
        wrap_cfg: WrapConfig,
        agentmw_cfg: AgentmwConfig,
        memory: ReasoningLibrary | None,
        provider: LLMProvider | None,
    ) -> None:
        self.wrap_cfg = wrap_cfg
        self.cfg = agentmw_cfg
        self.memory = memory
        self.provider = provider
        self.telemetry = global_telemetry() if agentmw_cfg.telemetry.enabled else None
        self.session_id = uuid.uuid4().hex[:12]
        self._session_messages: list[dict] = []
        self._calls_since_flush = 0
        self._store: SessionStore | None = (
            SessionStore(directory=agentmw_cfg.recorder.directory)
            if agentmw_cfg.recorder.enabled
            else None
        )

    # ---- pre-call (mutate messages + emit system note) ----
    def pre_call(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], CallTrace]:
        messages = list(kwargs.get("messages") or [])
        system = kwargs.get("system")
        extra_notes: list[str] = []
        recalled = 0

        if self.wrap_cfg.enable_memory and self.memory is not None and messages:
            note = self.memory.as_system_note(
                _first_user_text(messages),
                limit=self.cfg.memory.recall_limit,
            )
            if note:
                extra_notes.append(note)
                recalled = note.count("\n  ")

        monitor_report = MonitorReport()
        if self.wrap_cfg.enable_monitors:
            monitor_report = run_monitors(messages, config=self.cfg, provider=self.provider)
            corr = monitor_report.correction_text()
            if corr:
                extra_notes.append("[agentmw] Mid-run corrections:\n" + corr)

        stats = CompressionStats(0, 0, 0)
        if self.wrap_cfg.enable_compression:
            messages, stats = compress_history(
                messages,
                keep_recent=self.cfg.compression.keep_recent,
                max_tool_result_chars=self.cfg.compression.max_tool_result_chars,
            )

        system_note = "\n\n".join(extra_notes)
        new_system = _merge_system(system, system_note)

        new_kwargs = dict(kwargs)
        new_kwargs["messages"] = messages
        if new_system is not None and new_system != "":
            new_kwargs["system"] = new_system

        trace = CallTrace(
            monitors=monitor_report,
            compression=stats,
            recalled_patterns=recalled,
            system_note=system_note,
            session_id=self.session_id,
        )
        self.wrap_cfg.traces.append(trace)
        return new_kwargs, trace

    # ---- post-call (telemetry + record + auto-extract) ----
    def post_call(self, kwargs_sent: dict[str, Any], response: Any, trace: CallTrace) -> None:
        # Update rolling session messages (input + assistant reply).
        self._session_messages = list(kwargs_sent.get("messages") or [])
        reply = _response_to_message(response)
        if reply is not None:
            self._session_messages.append(reply)

        # Telemetry
        if self.telemetry is not None:
            self.telemetry.record_call(
                monitors_fired=[r.name for r in trace.monitors.triggered],
                tokens_saved=trace.compression.bytes_before - trace.compression.bytes_after,
                recalled=trace.recalled_patterns,
                provider_name=trace.monitors.provider_name,
            )
            self._calls_since_flush += 1
            if self._calls_since_flush >= self.cfg.telemetry.flush_every_calls:
                self.telemetry.save()
                self._calls_since_flush = 0

        # Auto-record
        if self._store is not None and self._session_messages:
            try:
                session = Session(
                    id=self.session_id,
                    task=_first_user_text(self._session_messages)[:200] or "(no task)",
                    messages=self._session_messages,
                )
                self._store.save(session)
            except Exception as e:  # noqa: BLE001
                logger.warning("auto-record failed: %s", e)

        # Auto-extract on session completion
        if (
            self.cfg.extractor.enabled
            and self.provider is not None
            and getattr(self.provider, "available", False)
            and self.memory is not None
            and self._session_messages
        ):
            should = (not self.cfg.extractor.on_completion_only) or (
                _stop_reason(response) == "end_turn" or is_session_complete(self._session_messages)
            )
            if should:
                try:
                    if self.cfg.extractor.background:
                        extract_in_background(self.provider, self.memory, list(self._session_messages))
                    else:
                        from agentmw.core.extractor import PatternExtractor
                        patterns = PatternExtractor(
                            self.provider, self.memory,
                            dedup_threshold=self.cfg.extractor.dedup_threshold,
                        ).extract(self._session_messages)
                        if self.telemetry is not None:
                            self.telemetry.record_extract(len(patterns))
                except Exception as e:  # noqa: BLE001
                    logger.warning("auto-extract failed: %s", e)


class _MessagesProxy:
    def __init__(self, inner: Any, lifecycle: _Lifecycle) -> None:
        self._inner = inner
        self._lc = lifecycle

    def create(self, **kwargs: Any) -> Any:
        new_kwargs, trace = self._lc.pre_call(kwargs)
        try:
            response = self._inner.create(**new_kwargs)
        except Exception:
            if self._lc.telemetry is not None:
                self._lc.telemetry.record_provider_call(success=False)
            raise
        if self._lc.telemetry is not None:
            self._lc.telemetry.record_provider_call(success=True)
        self._lc.post_call(new_kwargs, response, trace)
        return response

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _AsyncMessagesProxy:
    def __init__(self, inner: Any, lifecycle: _Lifecycle) -> None:
        self._inner = inner
        self._lc = lifecycle

    async def create(self, **kwargs: Any) -> Any:
        new_kwargs, trace = self._lc.pre_call(kwargs)
        try:
            response = await self._inner.create(**new_kwargs)
        except Exception:
            if self._lc.telemetry is not None:
                self._lc.telemetry.record_provider_call(success=False)
            raise
        if self._lc.telemetry is not None:
            self._lc.telemetry.record_provider_call(success=True)
        self._lc.post_call(new_kwargs, response, trace)
        return response

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class WrappedClient:
    def __init__(self, inner: Any, lifecycle: _Lifecycle) -> None:
        self._inner = inner
        self._lifecycle = lifecycle
        self.config = lifecycle.wrap_cfg
        self.agentmw_config = lifecycle.cfg
        self.memory = lifecycle.memory
        self.provider = lifecycle.provider
        self.session_id = lifecycle.session_id
        self.messages = _MessagesProxy(inner.messages, lifecycle)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class AsyncWrappedClient:
    def __init__(self, inner: Any, lifecycle: _Lifecycle) -> None:
        self._inner = inner
        self._lifecycle = lifecycle
        self.config = lifecycle.wrap_cfg
        self.agentmw_config = lifecycle.cfg
        self.memory = lifecycle.memory
        self.provider = lifecycle.provider
        self.session_id = lifecycle.session_id
        self.messages = _AsyncMessagesProxy(inner.messages, lifecycle)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def _build_lifecycle(
    config: WrapConfig | None,
    agentmw_config: AgentmwConfig | None,
    memory: ReasoningLibrary | None,
    provider: LLMProvider | None,
) -> _Lifecycle:
    wrap_cfg = config or WrapConfig()
    full_cfg = agentmw_config or default_config()
    if wrap_cfg.enable_memory and memory is None:
        memory = ReasoningLibrary(db_path=full_cfg.memory.db_path)
    tel = global_telemetry() if full_cfg.telemetry.enabled else None
    if provider is None and full_cfg.pipeline.use_llm:
        provider = select_provider(full_cfg.provider)
    provider = _wrap_provider(provider, full_cfg, tel)
    return _Lifecycle(wrap_cfg, full_cfg, memory, provider)


def wrap(
    client: Any,
    *,
    memory: ReasoningLibrary | None = None,
    config: WrapConfig | None = None,
    agentmw_config: AgentmwConfig | None = None,
    provider: LLMProvider | None = None,
) -> WrappedClient:
    """Wrap a sync Anthropic-style client (anthropic.Anthropic, openai.OpenAI, ...)."""
    return WrappedClient(client, _build_lifecycle(config, agentmw_config, memory, provider))


def wrap_async(
    client: Any,
    *,
    memory: ReasoningLibrary | None = None,
    config: WrapConfig | None = None,
    agentmw_config: AgentmwConfig | None = None,
    provider: LLMProvider | None = None,
) -> AsyncWrappedClient:
    """Wrap an async client (anthropic.AsyncAnthropic, openai.AsyncOpenAI, ...)."""
    return AsyncWrappedClient(client, _build_lifecycle(config, agentmw_config, memory, provider))
