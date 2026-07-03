from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from april_common.settings import AprilSettings


@dataclass(frozen=True, slots=True)
class ResourceSignals:
    ram_headroom_gb: float
    cpu_load_percent: float
    on_ac_power: bool
    user_idle_seconds: float


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    min_ram_headroom_gb: float = 1.0
    max_cpu_load_percent: float = 90.0
    require_ac_power_for_background: bool = True
    min_idle_seconds_for_background: float = 300.0


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    allowed: bool
    reasons: tuple[str, ...]
    signals: ResourceSignals


class ResourceSignalProvider(Protocol):
    def sample(self) -> ResourceSignals: ...


class LocalResourceSignalProvider:
    """Best-effort local-only host signals.

    macOS-specific power/idle APIs are intentionally not required here. When a
    live signal is unavailable, the provider returns conservative-but-nonblocking
    defaults and leaves stricter platform probing to future adapters.
    """

    def sample(self) -> ResourceSignals:
        return ResourceSignals(
            ram_headroom_gb=_available_ram_gb(),
            cpu_load_percent=_cpu_load_percent(),
            on_ac_power=True,
            user_idle_seconds=0.0,
        )


class ResourceGovernor:
    def __init__(
        self,
        settings: AprilSettings,
        *,
        provider: ResourceSignalProvider | None = None,
        policy: ResourcePolicy | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider or LocalResourceSignalProvider()
        self.policy = policy or ResourcePolicy(
            min_ram_headroom_gb=max(1.0, settings.governor.max_resident_gb * 0.1)
        )

    def assess_resident(self) -> GovernorDecision:
        """Gate always-on resident services such as Runtime/API/Sentinel."""
        signals = self.provider.sample()
        reasons: list[str] = []
        if signals.ram_headroom_gb < self.policy.min_ram_headroom_gb:
            reasons.append("ram_headroom_below_policy")
        if signals.cpu_load_percent > self.policy.max_cpu_load_percent:
            reasons.append("cpu_load_above_policy")
        return GovernorDecision(allowed=not reasons, reasons=tuple(reasons), signals=signals)

    def assess_background(self) -> GovernorDecision:
        """Gate idle/background work such as future Dreamer evolution."""
        resident = self.assess_resident()
        reasons = list(resident.reasons)
        if self.policy.require_ac_power_for_background and not resident.signals.on_ac_power:
            reasons.append("ac_power_required")
        if resident.signals.user_idle_seconds < self.policy.min_idle_seconds_for_background:
            reasons.append("user_not_idle")
        return GovernorDecision(
            allowed=not reasons,
            reasons=tuple(reasons),
            signals=resident.signals,
        )


def _available_ram_gb() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 999.0
    return float(page_size * available_pages) / (1024**3)


def _cpu_load_percent() -> float:
    try:
        one_minute, _five, _fifteen = os.getloadavg()
        cpu_count = os.cpu_count() or 1
    except OSError:
        return 0.0
    return min(100.0, max(0.0, (one_minute / cpu_count) * 100.0))
