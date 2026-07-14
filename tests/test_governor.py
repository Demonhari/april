from __future__ import annotations

from collections.abc import Sequence

import pytest

from april_common.settings import GovernorSettings
from services.pool.governor import (
    LocalResourceSignalProvider,
    ResourceGovernor,
    ResourcePolicy,
    ResourceSignals,
    ioreg_user_idle_seconds,
    pmset_on_ac_power,
    vm_stat_available_ram_gb,
)

PMSET_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=1234567)\t100%; charged; 0:00 remaining present: true\n"
)
PMSET_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=1234567)\t93%; discharging; 4:12 remaining present: true\n"
)
IOREG_IDLE = '  | |   "HIDParameters" = {"stuff"=1}\n  | |   "HIDIdleTime" = 4500000000\n'
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12,345.
Pages active:                            98,765.
Pages inactive:                          23,456.
Pages speculative:                        1,234.
Pages throttled:                              0.
Pages wired down:                        45,678.
Pages purgeable:                          3,210.
"""


class FixedSignals:
    def __init__(self, signals: ResourceSignals) -> None:
        self.signals = signals

    def sample(self) -> ResourceSignals:
        return self.signals


def _fixed_runner(outputs: dict[str, str]):
    def runner(argv: Sequence[str]) -> str:
        return outputs[argv[0]]

    return runner


def _failing_runner(argv: Sequence[str]) -> str:
    raise OSError("command unavailable")


def test_pmset_parses_ac_and_battery() -> None:
    assert pmset_on_ac_power(_fixed_runner({"pmset": PMSET_AC})) is True
    assert pmset_on_ac_power(_fixed_runner({"pmset": PMSET_BATTERY})) is False
    assert pmset_on_ac_power(_fixed_runner({"pmset": "garbage"})) is None
    assert pmset_on_ac_power(_failing_runner) is None


def test_ioreg_parses_hid_idle_nanoseconds() -> None:
    assert ioreg_user_idle_seconds(_fixed_runner({"ioreg": IOREG_IDLE})) == 4.5
    assert ioreg_user_idle_seconds(_fixed_runner({"ioreg": "no match"})) is None
    assert ioreg_user_idle_seconds(_failing_runner) is None


def test_vm_stat_parses_apple_silicon_page_size_and_grouped_counts() -> None:
    expected_pages = 12_345 + 23_456 + 3_210 + 1_234
    assert vm_stat_available_ram_gb(_fixed_runner({"vm_stat": VM_STAT})) == pytest.approx(
        (16_384 * expected_pages) / (1024**3)
    )


def test_vm_stat_returns_none_for_malformed_or_failing_probe() -> None:
    malformed = "Mach Virtual Memory Statistics: (page size unknown)\nPages free: 1.\n"
    assert vm_stat_available_ram_gb(_fixed_runner({"vm_stat": malformed})) is None
    assert vm_stat_available_ram_gb(_failing_runner) is None


def test_local_provider_darwin_uses_real_probes() -> None:
    provider = LocalResourceSignalProvider(
        runner=_fixed_runner({"vm_stat": VM_STAT, "pmset": PMSET_AC, "ioreg": IOREG_IDLE}),
        platform_system=lambda: "Darwin",
    )
    signals = provider.sample()
    assert signals.on_ac_power is True
    assert signals.power_source == "ac"
    assert signals.user_idle_seconds == 4.5
    assert signals.idle_source == "hid"
    assert signals.ram_source == "vm_stat"


def test_local_provider_degrades_safely_on_probe_failure() -> None:
    provider = LocalResourceSignalProvider(
        runner=_failing_runner,
        platform_system=lambda: "Darwin",
    )
    signals = provider.sample()
    assert signals.on_ac_power is False
    assert signals.power_source == "unknown"
    assert signals.user_idle_seconds == 0.0
    assert signals.idle_source == "unknown"
    assert signals.ram_source == "unknown"


def test_local_provider_non_macos_reports_unknown_sources() -> None:
    provider = LocalResourceSignalProvider(
        runner=_failing_runner,
        platform_system=lambda: "Linux",
    )
    signals = provider.sample()
    assert signals.power_source == "unknown"
    assert signals.idle_source == "unknown"


def test_generation_threads_active_user_uses_smaller_budget(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=16.0,
                cpu_load_percent=5.0,
                on_ac_power=True,
                user_idle_seconds=10.0,
            )
        ),
    )
    assert governor.generation_thread_budget() == 6


def test_generation_threads_idle_user_uses_larger_budget(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=16.0,
                cpu_load_percent=5.0,
                on_ac_power=True,
                user_idle_seconds=600.0,
            )
        ),
    )
    assert governor.generation_thread_budget() == 8


def test_generation_threads_degraded_idle_probe_uses_safe_default(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=LocalResourceSignalProvider(
            runner=_failing_runner,
            platform_system=lambda: "Darwin",
        ),
    )
    assert governor.generation_thread_budget() == 6


def test_generation_thread_settings_require_active_budget_not_above_idle() -> None:
    with pytest.raises(ValueError, match="generation_threads_active"):
        GovernorSettings(generation_threads_active=9, generation_threads_idle=8)


def test_background_allowed_with_injected_ac_idle_signals(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=16.0,
                cpu_load_percent=5.0,
                on_ac_power=True,
                user_idle_seconds=600.0,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )
    decision = governor.assess_background()
    assert decision.allowed
    assert decision.reasons == ()


def test_background_refused_on_battery_with_reason(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=16.0,
                cpu_load_percent=5.0,
                on_ac_power=False,
                user_idle_seconds=600.0,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )
    decision = governor.assess_background()
    assert not decision.allowed
    assert "ac_power_required" in decision.reasons


def test_background_refused_when_signals_unknown(settings_tmp) -> None:
    provider = LocalResourceSignalProvider(
        runner=_failing_runner,
        platform_system=lambda: "Darwin",
    )
    governor = ResourceGovernor(
        settings_tmp,
        provider=provider,
        policy=ResourcePolicy(min_ram_headroom_gb=0.0, max_cpu_load_percent=100.0),
    )
    decision = governor.assess_background()
    assert not decision.allowed
    assert "power_signal_unavailable" in decision.reasons
    assert "idle_signal_unavailable" in decision.reasons
    assert "ram_signal_unavailable" in decision.reasons


def test_unknown_ram_only_advises_resident_and_model_load(settings_tmp) -> None:
    provider = LocalResourceSignalProvider(
        runner=_failing_runner,
        platform_system=lambda: "Darwin",
    )
    governor = ResourceGovernor(
        settings_tmp,
        provider=provider,
        policy=ResourcePolicy(min_ram_headroom_gb=64.0, max_cpu_load_percent=100.0),
    )

    resident = governor.assess_resident()
    assert resident.allowed
    assert resident.reasons == ()
    assert resident.advisories == ("ram_signal_unavailable",)

    model_load = governor.assess_model_load(projected_resident_gb=128.0)
    assert model_load.allowed
    assert model_load.reasons == ()
    assert model_load.advisories == ("ram_signal_unavailable",)


def test_resident_ignores_power_and_idle(settings_tmp) -> None:
    provider = LocalResourceSignalProvider(
        runner=_failing_runner,
        platform_system=lambda: "Linux",
    )
    governor = ResourceGovernor(
        settings_tmp,
        provider=provider,
        policy=ResourcePolicy(min_ram_headroom_gb=0.0, max_cpu_load_percent=100.0),
    )
    assert governor.assess_resident().allowed


def test_projected_model_memory_can_deny_interactive_load(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=2.5,
                cpu_load_percent=5.0,
                on_ac_power=True,
                user_idle_seconds=0.0,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )
    decision = governor.assess_model_load(projected_resident_gb=2.0)
    assert not decision.allowed
    assert "projected_ram_headroom_below_policy" in decision.reasons


def test_interactive_model_load_allows_battery_and_active_user(settings_tmp) -> None:
    governor = ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=16.0,
                cpu_load_percent=5.0,
                on_ac_power=False,
                user_idle_seconds=0.0,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=1.0, max_cpu_load_percent=90.0),
    )
    decision = governor.assess_model_load(projected_resident_gb=1.0)
    assert decision.allowed
    assert decision.reasons == ()


# ---------------------------------------------------------------------------
# Governor gating of specialist model loads (ModelLifecycle wiring)
# ---------------------------------------------------------------------------


def _lifecycle_with_governor(tmp_path, governor):
    from services.april_runtime.model_lifecycle import ModelLifecycle
    from services.april_runtime.model_registry import ModelRegistry

    base = {
        "name": "fake",
        "path": "missing.gguf",
        "backend": "fake",
        "chat_format": "generic",
        "threads": 1,
        "context_size": 1024,
        "temperature": 0.2,
        "max_output_tokens": 64,
    }
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "brain": {**base, "id": "april-brain", "role": "brain", "keep_loaded": True},
                "coding": {**base, "id": "april-coding", "role": "coding", "keep_loaded": False},
            }
        },
        root=tmp_path,
    )
    return ModelLifecycle(registry, root_backend="fake", governor=governor)


def _governor_with(settings_tmp, *, ram_gb: float, cpu: float, on_ac: bool, idle: float):
    return ResourceGovernor(
        settings_tmp,
        provider=FixedSignals(
            ResourceSignals(
                ram_headroom_gb=ram_gb,
                cpu_load_percent=cpu,
                on_ac_power=on_ac,
                user_idle_seconds=idle,
            )
        ),
        policy=ResourcePolicy(min_ram_headroom_gb=2.0, max_cpu_load_percent=90.0),
    )


async def test_low_ram_refuses_new_specialist_load(settings_tmp, tmp_path) -> None:
    from april_common.errors import ModelUnavailableError

    governor = _governor_with(settings_tmp, ram_gb=0.5, cpu=5.0, on_ac=True, idle=600.0)
    lifecycle = _lifecycle_with_governor(tmp_path, governor)
    with pytest.raises(ModelUnavailableError) as excinfo:
        await lifecycle.load_model("april-coding")
    assert "ram_headroom_below_policy" in str(excinfo.value.details)


async def test_high_cpu_refuses_new_specialist_load(settings_tmp, tmp_path) -> None:
    from april_common.errors import ModelUnavailableError

    governor = _governor_with(settings_tmp, ram_gb=16.0, cpu=99.0, on_ac=True, idle=600.0)
    lifecycle = _lifecycle_with_governor(tmp_path, governor)
    with pytest.raises(ModelUnavailableError) as excinfo:
        await lifecycle.load_model("april-coding")
    assert "cpu_load_above_policy" in str(excinfo.value.details)


async def test_battery_and_active_user_never_block_interactive_loads(
    settings_tmp, tmp_path
) -> None:
    # On battery with an active user: interactive model loads must still work;
    # only background (Dreamer) work is refused for those reasons.
    governor = _governor_with(settings_tmp, ram_gb=16.0, cpu=5.0, on_ac=False, idle=0.0)
    lifecycle = _lifecycle_with_governor(tmp_path, governor)
    state = await lifecycle.load_model("april-coding")
    assert state.state == "loaded"
    background = governor.assess_background()
    assert not background.allowed
    assert "ac_power_required" in background.reasons
    assert "user_not_idle" in background.reasons


async def test_unknown_signals_block_background_not_loads(settings_tmp, tmp_path) -> None:
    provider = LocalResourceSignalProvider(
        runner=_failing_runner,
        platform_system=lambda: "Darwin",
    )
    governor = ResourceGovernor(
        settings_tmp,
        provider=provider,
        policy=ResourcePolicy(min_ram_headroom_gb=0.0, max_cpu_load_percent=100.0),
    )
    lifecycle = _lifecycle_with_governor(tmp_path, governor)
    state = await lifecycle.load_model("april-coding")
    assert state.state == "loaded"
    background = governor.assess_background()
    assert not background.allowed
    assert "power_signal_unavailable" in background.reasons


async def test_keep_loaded_brain_model_loads_despite_pressure(settings_tmp, tmp_path) -> None:
    governor = _governor_with(settings_tmp, ram_gb=0.1, cpu=99.0, on_ac=False, idle=0.0)
    lifecycle = _lifecycle_with_governor(tmp_path, governor)
    state = await lifecycle.load_model("april-brain")
    assert state.state == "loaded"
