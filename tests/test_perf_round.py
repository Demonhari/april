from __future__ import annotations

import asyncio
import json
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from april_common.process_environment import ProcessCategory
from april_common.process_runner import run_restricted_process_sync
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.llama_cpp_backend import LlamaCppBackend, _build_prefix_adapter
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.perf_profile import (
    apply_tunable_overlay,
    load_matching_profile,
    load_matching_profile_async,
    profile_fingerprint,
    profile_inputs,
    profile_status,
)
from services.april_runtime.prefix_cache import PrefixStatePolicy
from services.pool.governor import ResourceGovernor, ResourceSignals


def test_prefix_policy_uses_longest_match_and_lru_caps() -> None:
    policy = PrefixStatePolicy(capacity_bytes=10, min_prefix_tokens=3, max_entries=2)
    family_a = tuple(range(8))
    family_b = (*tuple(range(5, 10)), 99, 100)
    assert policy.insert(family_a, "a", 4)
    assert policy.insert(family_b, "b", 4)
    entry = policy.lookup((*tuple(range(5, 10)), 99, 100, 7))
    assert entry is not None
    assert entry.state == "b"
    entry = policy.lookup((*family_a, 7))
    assert entry is not None
    assert entry.state == "a"
    assert policy.insert(tuple(range(20)), "too-large", 11) is False
    assert policy.stats()["rejected_oversize"] == 1
    assert policy.insert((40, 41, 42), "c", 4)
    assert policy.entries == 2
    assert policy.lookup(family_b) is None
    policy.clear()
    assert policy.stats()["entries"] == 0
    assert policy.stats()["bytes"] == 0
    assert policy.lookup((1, 2, 3)) is None
    assert policy.insert((1, 2), "short", 1)
    assert policy.insert((1, 2), "replacement", 1)
    assert policy.lookup(()) is None


def test_process_runner_sync_rejects_running_loop_without_creating_coroutine() -> None:
    async def check() -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="cannot run inside an event loop"):
                run_restricted_process_sync(
                    ["true"],
                    cwd=Path("."),
                    category=ProcessCategory.DAEMON,
                    timeout_seconds=1.0,
                )
        assert not any("never awaited" in str(item.message) for item in caught)

    asyncio.run(check())


@pytest.mark.asyncio
async def test_governor_async_samples_once_within_ttl(settings_tmp: Any) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> ResourceSignals:
            self.calls += 1
            return ResourceSignals(8.0, 10.0, True, 600.0)

    provider = Provider()
    governor = ResourceGovernor(settings_tmp, provider=provider)
    first, second = await asyncio.gather(
        governor.assess_background_async(), governor.assess_background_async()
    )
    assert first.allowed
    assert second.allowed
    assert provider.calls == 1
    await governor.assess_resident_async()
    assert provider.calls == 1


def test_perf_profile_overlay_only_changes_tunable_fields(tmp_path: Path) -> None:
    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="coding",
        threads=8,
        context_size=1024,
        temperature=0.2,
        max_output_tokens=32,
    )
    updated = apply_tunable_overlay(
        model,
        {
            "settings": {
                "threads": 6,
                "threads_batch": 6,
                "context_size": 256,
                "temperature": 1.0,
            }
        },
    )
    assert updated.threads == 6
    assert updated.threads_batch == 6
    assert updated.context_size == model.context_size
    assert updated.temperature == model.temperature
    model.path.write_bytes(b"model")
    inputs = profile_inputs(model, tmp_path)
    fingerprint = profile_fingerprint(inputs)
    profile_dir = tmp_path / "data" / "perf" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / f"{fingerprint}.json").write_text(
        json.dumps(
            {
                "model_id": model.id,
                "fingerprint": fingerprint,
                "fingerprint_inputs": inputs,
                "settings": {"threads": 4},
            }
        ),
        encoding="utf-8",
    )
    assert load_matching_profile(tmp_path, model).threads == 4
    assert profile_status(tmp_path, [model], "auto") == "active"


def test_perf_profile_rejects_invalid_and_stale_candidates(tmp_path: Path) -> None:
    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="coding",
        threads=8,
        context_size=1024,
        temperature=0.2,
        max_output_tokens=32,
    )
    model.path.write_bytes(b"model")
    directory = tmp_path / "data" / "perf" / "profiles"
    directory.mkdir(parents=True)
    (directory / "bad.json").write_text("not json", encoding="utf-8")
    (directory / "wrong.json").write_text(json.dumps({"model_id": "other"}), encoding="utf-8")
    (directory / "stale.json").write_text(
        json.dumps({"model_id": model.id, "fingerprint_inputs": {}, "fingerprint": "bad"}),
        encoding="utf-8",
    )
    assert apply_tunable_overlay(model, {"settings": []}) is model
    assert load_matching_profile(tmp_path, model) == model
    assert profile_status(tmp_path, [model], "off") == "off"
    assert profile_status(tmp_path, [model], "auto") == "stale"


def test_perf_profile_handles_adapter_identity_and_async_load(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    adapter_path = tmp_path / "adapter.bin"
    model_path.write_bytes(b"model")
    adapter_path.write_bytes(b"adapter")
    model = ModelDefinition(
        id="model",
        name="model",
        path=model_path,
        adapter_path=adapter_path,
        backend="fake",
        role="coding",
        threads=8,
        context_size=1024,
        temperature=0.2,
        max_output_tokens=32,
    )
    inputs = profile_inputs(model, tmp_path)
    assert inputs["adapter_identity"] is not None
    directory = tmp_path / "data" / "perf" / "profiles"
    directory.mkdir(parents=True)
    fingerprint = profile_fingerprint(inputs)
    (directory / f"{fingerprint}.json").write_text(
        json.dumps(
            {
                "model_id": model.id,
                "fingerprint_inputs": inputs,
                "fingerprint": fingerprint,
                "settings": {"flash_attn": True},
            }
        ),
        encoding="utf-8",
    )
    loaded = asyncio.run(load_matching_profile_async(tmp_path, model))
    assert loaded.flash_attn is True


def test_perf_profile_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import services.april_runtime.perf_profile as perf_profile

    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "missing.gguf",
        adapter_path=tmp_path / "missing.adapter",
        backend="fake",
        role="coding",
        threads=8,
        context_size=1024,
        temperature=0.2,
        max_output_tokens=32,
    )
    assert perf_profile.model_identity(model, tmp_path) is None
    assert perf_profile._adapter_identity(model, tmp_path) is None
    monkeypatch.setattr(
        perf_profile,
        "version",
        lambda _name: (_ for _ in ()).throw(perf_profile.PackageNotFoundError("missing")),
    )
    assert perf_profile.llama_cpp_version() is None
    directory = tmp_path / "data" / "perf" / "profiles"
    directory.mkdir(parents=True)
    inputs = profile_inputs(model, tmp_path)
    (directory / "mismatch.json").write_text(
        json.dumps({"model_id": model.id, "fingerprint_inputs": inputs, "fingerprint": "wrong"}),
        encoding="utf-8",
    )
    assert load_matching_profile(tmp_path, model) == model
    assert profile_status(tmp_path / "empty", [model], "auto") == "none"


@pytest.mark.asyncio
async def test_fake_perf_bench_is_redacted_and_has_two_brain_passes() -> None:
    from apps.runner.commands.runner_perf import _run_fake_bench

    report = await _run_fake_bench(Path.cwd(), role="brain", repeat=1)
    assert report["schema"] == "april.perf.bench.v1"
    assert report["simulated"] is True
    assert {case["pass"] for case in report["cases"]} == {1, 2}
    assert all("message" not in case and "text" not in case for case in report["cases"])
    assert all("secret" not in json.dumps(case).lower() for case in report["cases"])


@pytest.mark.asyncio
async def test_fake_perf_bench_includes_specialists() -> None:
    from apps.runner.commands.runner_perf import _run_fake_bench

    report = await _run_fake_bench(Path.cwd(), role="all", repeat=1)
    assert {case["role"] for case in report["cases"] if "role" in case} == {"coding", "reading"}


def test_perf_bench_command_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.runner.commands import runner_perf

    monkeypatch.setattr(
        runner_perf,
        "load_settings",
        lambda: types.SimpleNamespace(
            home=Path.cwd(), runtime=types.SimpleNamespace(backend="fake")
        ),
    )
    output = tmp_path / "bench.json"
    runner_perf.perf_bench(fake=True, role="brain", repeat=1, report=output)
    assert json.loads(output.read_text(encoding="utf-8"))["simulated"] is True


def test_perf_bench_rejects_invalid_role() -> None:
    from apps.runner.commands.runner_perf import perf_bench

    with pytest.raises(Exception, match="role must be brain"):
        perf_bench(fake=True, role="invalid", repeat=1, report=None)


def test_perf_tune_rejects_invalid_role() -> None:
    from apps.runner.commands.runner_perf import perf_tune

    with pytest.raises(Exception, match="role must be brain"):
        perf_tune(role="invalid", max_minutes=1.0, cooldown_seconds=0.0, dry_run=True, fake=False)


def test_runtime_backend_default_optional_hooks() -> None:
    class MinimalBackend(RuntimeBackend):
        async def load(self, model: ModelDefinition) -> None:
            del model

        async def unload(self) -> None:
            return None

        async def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
            del prompt, kwargs
            return GenerationResult(text="", input_tokens=0, output_tokens=0)

        def stream(self, prompt: str, **kwargs: Any):
            del prompt, kwargs
            if False:
                yield ""

        async def tokenize(self, text: str) -> list[int]:
            return list(text.encode())

        async def health(self) -> BackendHealth:
            return BackendHealth(ok=True, message="ok")

    backend = MinimalBackend()
    assert backend.apply_thread_budget(1, 1) is False
    assert backend.timing_diagnostics() == {}
    backend.finish_timing_diagnostics(1, 1)
    assert backend.prefix_cache_diagnostics() == {}


def test_perf_tune_fake_writes_simulated_report(
    settings_tmp: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.runner.commands import runner_perf

    monkeypatch.setattr(
        runner_perf,
        "load_settings",
        lambda: settings_tmp.model_copy(update={"home": Path.cwd()}),
    )
    monkeypatch.setattr(
        runner_perf,
        "datetime",
        types.SimpleNamespace(
            now=lambda _tz: types.SimpleNamespace(strftime=lambda _fmt: "fixed"), UTC="UTC"
        ),
    )
    runner_perf.perf_tune(
        role="brain", max_minutes=1.0, cooldown_seconds=0.0, dry_run=False, fake=True
    )
    output = Path.cwd() / "data" / "verification" / "perf-tune-fixed.json"
    assert json.loads(output.read_text(encoding="utf-8"))["simulated"] is True
    output.unlink()


def test_perf_tune_rejects_candidates() -> None:
    from apps.runner.commands.runner_perf import _accept_candidate

    baseline = {"outputs": ["same"], "peak_rss": 100, "metric": 10, "prompt_metric": 10}
    assert _accept_candidate(baseline, {**baseline, "outputs": ["different"]}, "threads") == (
        False,
        "semantic_drift",
    )
    assert _accept_candidate(baseline, {**baseline, "peak_rss": 116}, "threads") == (False, "rss")
    assert _accept_candidate(baseline, {**baseline, "metric": 10.4}, "threads") == (False, "slower")
    assert _accept_candidate(baseline, {**baseline, "prompt_metric": 10.5}, "flash_attn") == (
        True,
        None,
    )


def test_perf_tune_dry_run_lists_only_allowed_knobs(
    settings_tmp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.runner.commands import runner_perf

    monkeypatch.setattr(
        runner_perf,
        "load_settings",
        lambda: settings_tmp.model_copy(update={"home": Path.cwd()}),
    )
    runner_perf.perf_tune(
        role="brain",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        dry_run=True,
        fake=False,
    )


@pytest.mark.asyncio
async def test_llama_load_passes_batch_flash_and_applies_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)
            self.ctx = object()
            self.input_ids: list[int] = []
            self.n_tokens = 0

        def set_cache(self, value: object) -> None:
            self.cache = value

    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama  # type: ignore[attr-defined]
    module.llama_set_n_threads = lambda ctx, threads, batch: calls.update(  # type: ignore[attr-defined]
        {"applied": (ctx, threads, batch)}
    )
    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", module)
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fake")
    model = ModelDefinition(
        id="model",
        name="model",
        path=path,
        backend="llama_cpp",
        role="coding",
        threads=8,
        threads_batch=7,
        flash_attn=True,
        prefix_cache_mb=0,
        context_size=1024,
        temperature=0.2,
        max_output_tokens=32,
    )
    backend = LlamaCppBackend()
    await backend.load(model)
    assert calls["n_threads_batch"] == 7
    assert calls["flash_attn"] is True
    assert backend.apply_thread_budget(4, 3)
    assert calls["applied"][1:] == (4, 3)  # type: ignore[index]


def test_prefix_adapter_reports_hit_and_handles_adapter_errors() -> None:
    class BaseCache:
        def __init__(self, capacity_bytes: int) -> None:
            self.capacity_bytes = capacity_bytes

    class State:
        def __init__(self) -> None:
            self.llama_state = b"state"
            self.scores = np.zeros((1, 2), dtype=np.float32)
            self.input_ids = np.asarray(tuple(range(8)), dtype=np.int32)

    llm = types.SimpleNamespace(input_ids=np.asarray((0, 1, 99)), n_tokens=3)
    policy = PrefixStatePolicy(capacity_bytes=1024, min_prefix_tokens=3, max_entries=4)
    errors: list[str] = []
    adapter = _build_prefix_adapter(BaseCache, llm, policy, lambda: errors.append("error"))
    state = State()
    key = tuple(range(8))
    adapter[key] = state
    restored = adapter[key]
    assert restored is state
    assert adapter.last_lookup["hit"] is True
    llm.input_ids = np.asarray((50, 51, 52))
    with pytest.raises(KeyError):
        adapter[(50, 51, 52)]
    assert adapter.last_lookup["hit"] is False
    assert not errors


def test_llama_perf_counters_are_feature_detected() -> None:
    backend = LlamaCppBackend()
    backend._llm = types.SimpleNamespace(ctx=object())
    backend._llama_module = types.SimpleNamespace(
        llama_perf_context_reset=lambda _ctx: None,
        llama_perf_context=lambda _ctx: {
            "n_p_eval": 12,
            "t_p_eval": 3.5,
            "n_eval": 4,
            "t_eval": 2.0,
        },
    )
    backend._begin_perf()
    backend._finish_perf(12, 4)
    assert backend.timing_diagnostics()["prompt_eval_tokens"] == 12
    assert backend.timing_diagnostics()["eval_ms"] == 2.0
