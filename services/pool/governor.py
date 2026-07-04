from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from april_common.settings import AprilSettings

# Signal source labels. "unknown" means the live probe was unavailable or
# failed; the governor then degrades safely (background work is refused with an
# explicit reason instead of guessing).
POWER_SOURCE_AC = "ac"
POWER_SOURCE_BATTERY = "battery"
SIGNAL_SOURCE_UNKNOWN = "unknown"
IDLE_SOURCE_HID = "hid"

_PMSET_AC_RE = re.compile(r"'AC Power'")
_PMSET_BATTERY_RE = re.compile(r"'Battery Power'")
_HID_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')

CommandRunner = Callable[[Sequence[str]], str]


@dataclass(frozen=True, slots=True)
class ResourceSignals:
    ram_headroom_gb: float
    cpu_load_percent: float
    on_ac_power: bool
    user_idle_seconds: float
    # Where the power/idle values came from. "unknown" marks a degraded probe
    # (non-macOS host, command failure, unparseable output); the governor then
    # refuses background work with an explicit reason instead of guessing.
    # When omitted, the values are treated as directly measured/injected
    # (fake providers in tests stay trusted).
    power_source: str = ""
    idle_source: str = ""

    def __post_init__(self) -> None:
        if not self.power_source:
            derived = POWER_SOURCE_AC if self.on_ac_power else POWER_SOURCE_BATTERY
            object.__setattr__(self, "power_source", derived)
        if not self.idle_source:
            object.__setattr__(self, "idle_source", IDLE_SOURCE_HID)


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


def _run_command(argv: Sequence[str]) -> str:
    """argv-only local probe; shell=False, bounded, never raises past caller."""
    completed = subprocess.run(
        list(argv),
        shell=False,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
    )
    return completed.stdout


def pmset_on_ac_power(runner: CommandRunner = _run_command) -> bool | None:
    """AC/battery from `pmset -g batt`. None when unavailable/unparseable."""
    try:
        output = runner(("pmset", "-g", "batt"))
    except Exception:
        return None
    if _PMSET_AC_RE.search(output):
        return True
    if _PMSET_BATTERY_RE.search(output):
        return False
    return None


def ioreg_user_idle_seconds(runner: CommandRunner = _run_command) -> float | None:
    """User idle seconds from IOHIDSystem HIDIdleTime (nanoseconds)."""
    try:
        output = runner(("ioreg", "-c", "IOHIDSystem", "-d", "4"))
    except Exception:
        return None
    match = _HID_IDLE_RE.search(output)
    if match is None:
        return None
    return int(match.group(1)) / 1_000_000_000.0


class LocalResourceSignalProvider:
    """Best-effort local-only host signals.

    RAM headroom and CPU load come from POSIX APIs everywhere. On macOS the
    power source is probed with `pmset -g batt` and user idle seconds with
    `ioreg` HIDIdleTime. On other platforms, or when either local command
    fails, the corresponding source is reported as "unknown" and the value
    falls back to a conservative default (battery / zero idle) so background
    work is refused with an explicit reason instead of silently running.
    No network is ever touched; both probes are local argv subprocesses.
    """

    def __init__(
        self,
        *,
        runner: CommandRunner = _run_command,
        platform_system: Callable[[], str] = platform.system,
    ) -> None:
        self.runner = runner
        self.platform_system = platform_system

    def sample(self) -> ResourceSignals:
        on_ac: bool | None = None
        idle_seconds: float | None = None
        if self.platform_system() == "Darwin":
            on_ac = pmset_on_ac_power(self.runner)
            idle_seconds = ioreg_user_idle_seconds(self.runner)
        power_source = SIGNAL_SOURCE_UNKNOWN
        if on_ac is True:
            power_source = POWER_SOURCE_AC
        elif on_ac is False:
            power_source = POWER_SOURCE_BATTERY
        return ResourceSignals(
            ram_headroom_gb=_available_ram_gb(),
            cpu_load_percent=_cpu_load_percent(),
            # Conservative fallbacks: unknown power reads as "not on AC" and
            # unknown idle reads as "user active", so unknown never unlocks
            # background work.
            on_ac_power=bool(on_ac),
            user_idle_seconds=idle_seconds if idle_seconds is not None else 0.0,
            power_source=power_source,
            idle_source=IDLE_SOURCE_HID if idle_seconds is not None else SIGNAL_SOURCE_UNKNOWN,
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
            min_ram_headroom_gb=max(1.0, settings.governor.max_resident_gb * 0.1),
            require_ac_power_for_background=settings.evolution.require_ac_power,
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
        """Gate idle/background work such as Dreamer evolution."""
        resident = self.assess_resident()
        reasons = list(resident.reasons)
        signals = resident.signals
        if self.policy.require_ac_power_for_background:
            if signals.power_source == SIGNAL_SOURCE_UNKNOWN:
                reasons.append("power_signal_unavailable")
            elif not signals.on_ac_power:
                reasons.append("ac_power_required")
        if signals.idle_source == SIGNAL_SOURCE_UNKNOWN:
            reasons.append("idle_signal_unavailable")
        elif signals.user_idle_seconds < self.policy.min_idle_seconds_for_background:
            reasons.append("user_not_idle")
        return GovernorDecision(
            allowed=not reasons,
            reasons=tuple(reasons),
            signals=signals,
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
