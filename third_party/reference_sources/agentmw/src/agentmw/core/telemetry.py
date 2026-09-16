"""Lightweight telemetry: counters persisted to JSON for `agentmw stats`.

No external deps. Atomic write via temp file + rename. Safe for parallel
processes (last writer wins for counters; this is intentional and OK for
soft telemetry).
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path


def _default_path() -> Path:
    base = os.environ.get("AGENTMW_HOME") or os.path.expanduser("~/.agentmw")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / "telemetry.json"


@dataclass
class Telemetry:
    started_at: float = field(default_factory=time.time)
    calls_total: int = 0
    calls_with_corrections: int = 0
    monitors_fired: dict[str, int] = field(default_factory=dict)
    tokens_compressed: int = 0
    patterns_recalled: int = 0
    patterns_extracted: int = 0
    provider_calls: int = 0
    provider_failures: int = 0
    breaker_trips: int = 0
    last_provider: str = ""
    last_provider_at: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _path: Path | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path | None = None) -> "Telemetry":
        path = path or _default_path()
        t = cls()
        t._path = path
        if path.is_file():
            try:
                with open(path) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(t, k) and not k.startswith("_"):
                        setattr(t, k, v)
            except (OSError, ValueError):
                pass
        return t

    def to_dict(self) -> dict:
        """Serializable dict without internal state (lock, path)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }

    def save(self) -> None:
        if self._path is None:
            self._path = _default_path()
        with self._lock:
            payload = self.to_dict()
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._path)

    def record_call(
        self,
        *,
        monitors_fired: list[str] | None = None,
        tokens_saved: int = 0,
        recalled: int = 0,
        provider_name: str = "",
    ) -> None:
        with self._lock:
            self.calls_total += 1
            if monitors_fired:
                self.calls_with_corrections += 1
                for name in monitors_fired:
                    self.monitors_fired[name] = self.monitors_fired.get(name, 0) + 1
            self.tokens_compressed += max(tokens_saved, 0)
            self.patterns_recalled += max(recalled, 0)
            if provider_name:
                self.last_provider = provider_name
                self.last_provider_at = time.time()

    def record_provider_call(self, success: bool) -> None:
        with self._lock:
            self.provider_calls += 1
            if not success:
                self.provider_failures += 1

    def record_extract(self, n_patterns: int) -> None:
        with self._lock:
            self.patterns_extracted += max(n_patterns, 0)

    def record_breaker_trip(self) -> None:
        with self._lock:
            self.breaker_trips += 1


_GLOBAL: Telemetry | None = None
_GLOBAL_LOCK = threading.Lock()


def global_telemetry() -> Telemetry:
    global _GLOBAL
    if _GLOBAL is None:
        with _GLOBAL_LOCK:
            if _GLOBAL is None:
                _GLOBAL = Telemetry.load()
                atexit.register(_safe_flush)
    return _GLOBAL


def _safe_flush() -> None:
    if _GLOBAL is None:
        return
    try:
        _GLOBAL.save()
    except Exception:  # noqa: BLE001
        pass


def reset_global_telemetry() -> None:
    """Reset the cached singleton — useful when AGENTMW_HOME changes mid-process."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            try:
                _GLOBAL.save()
            except Exception:  # noqa: BLE001
                pass
        _GLOBAL = None
