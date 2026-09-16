"""Circuit breaker around a provider. Protects the wrapped client from
cascading slowdowns when the monitor provider is having a bad day.

States:
    closed:    requests pass through; failures increment.
    open:      requests short-circuit with ProviderError until cooldown elapses.
    half_open: one trial request is allowed; success → closed, failure → open.

Default: trip after 3 failures within 30 s; cool down for 60 s.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from agentmw.core.providers import LLMProvider, ProviderError

logger = logging.getLogger("agentmw.breaker")


@dataclass
class BreakerConfig:
    failure_threshold: int = 3
    failure_window_seconds: float = 30.0
    cooldown_seconds: float = 60.0


class CircuitBreakerProvider:
    """Wraps an LLMProvider with a fail-fast breaker."""

    def __init__(
        self,
        inner: LLMProvider,
        config: BreakerConfig | None = None,
        on_trip=None,
        on_recover=None,
    ) -> None:
        self._inner = inner
        self._cfg = config or BreakerConfig()
        self._lock = threading.Lock()
        self._failures: list[float] = []
        self._opened_at: float | None = None
        self._on_trip = on_trip
        self._on_recover = on_recover

    @property
    def name(self) -> str:
        return f"{self._inner.name}+breaker"

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "")

    @property
    def available(self) -> bool:
        if self._state == "open":
            return False
        return bool(getattr(self._inner, "available", False))

    @property
    def _state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.time() - self._opened_at >= self._cfg.cooldown_seconds:
                return "half_open"
            return "open"

    def _record_failure(self) -> None:
        with self._lock:
            now = time.time()
            self._failures.append(now)
            self._failures = [t for t in self._failures if now - t <= self._cfg.failure_window_seconds]
            if len(self._failures) >= self._cfg.failure_threshold and self._opened_at is None:
                self._opened_at = now
                logger.warning("circuit breaker tripped on provider=%s", self._inner.name)
                if self._on_trip:
                    try:
                        self._on_trip()
                    except Exception:  # noqa: BLE001
                        pass

    def _record_success(self) -> None:
        with self._lock:
            self._failures.clear()
            if self._opened_at is not None:
                self._opened_at = None
                logger.info("circuit breaker recovered on provider=%s", self._inner.name)
                if self._on_recover:
                    try:
                        self._on_recover()
                    except Exception:  # noqa: BLE001
                        pass

    def judge(self, system: str, prompt: str) -> str:
        state = self._state
        if state == "open":
            raise ProviderError("circuit breaker open")
        try:
            out = self._inner.judge(system, prompt)
        except ProviderError:
            self._record_failure()
            raise
        except Exception as e:
            self._record_failure()
            raise ProviderError(f"provider raised {type(e).__name__}: {e}") from e
        self._record_success()
        return out
