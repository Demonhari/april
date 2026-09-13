from __future__ import annotations

import asyncio
import ctypes
import json
import time
import types
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import typer

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


def test_prefix_save_guard_models_snapshot_geometry_and_learns_kv_only() -> None:
    model = ModelDefinition(
        id="qwen",
        name="qwen",
        path=Path("qwen.gguf"),
        backend="llama_cpp",
        role="coding",
        threads=8,
        context_size=4096,
        temperature=0.0,
        max_output_tokens=2048,
        prefix_cache_mb=768,
        prefix_cache_min_tokens=128,
    )
    backend = LlamaCppBackend()
    cache_calls: list[object] = []
    backend._model = model
    backend._llm = types.SimpleNamespace(set_cache=cache_calls.append)
    backend._prefix_policy = PrefixStatePolicy(768 * 1024 * 1024, 128, 64)
    backend._prefix_adapter = types.SimpleNamespace(set_prompt_tokens=lambda _tokens: None)
    backend._prefix_n_batch = 128
    backend._prefix_n_vocab = 151_936
    backend._prefix_n_ctx = 4096

    backend._attach_prefix_cache(150, "synthetic", max_output_tokens=2048)
    assert backend._prefix_attached is True
    short_state = types.SimpleNamespace(
        n_tokens=150,
        llama_state=b"k" * (150 * 112 * 1024 + 151_936 * 4),
    )
    backend._prefix_adapter.prompt_tokens = 150
    backend._record_prefix_save_state((1,) * 150, 0, short_state)

    backend._attach_prefix_cache(2000, "synthetic", max_output_tokens=2048)
    assert backend._prefix_attached is True
    assert backend._attach_skipped_reason is None
    assert backend._prefix_projected_bytes is not None
    assert backend._prefix_projected_bytes < backend._prefix_policy.capacity_bytes


def test_prefix_save_guard_skips_a_genuinely_oversize_projection() -> None:
    model = ModelDefinition(
        id="small-cache",
        name="small-cache",
        path=Path("model.gguf"),
        backend="llama_cpp",
        role="coding",
        threads=4,
        context_size=4096,
        temperature=0.0,
        max_output_tokens=2048,
        prefix_cache_mb=1,
        prefix_cache_min_tokens=128,
    )
    backend = LlamaCppBackend()
    backend._model = model
    backend._llm = types.SimpleNamespace(set_cache=lambda _cache: None)
    backend._prefix_policy = PrefixStatePolicy(1024 * 1024, 128, 64)
    backend._prefix_adapter = types.SimpleNamespace(set_prompt_tokens=lambda _tokens: None)
    backend._prefix_n_batch = 128
    backend._prefix_n_vocab = 151_936
    backend._prefix_n_ctx = 4096
    backend._kv_bytes_per_token_estimate = 112 * 1024

    backend._attach_prefix_cache(2000, "synthetic", max_output_tokens=2048)
    assert backend._prefix_attached is False
    assert backend._attach_skipped_reason == "projected_oversize"
    assert backend._prefix_projected_bytes is not None
    assert backend._prefix_projected_bytes > backend._prefix_policy.capacity_bytes


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
                "schema": "april.perf.profile.v2",
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
                "schema": "april.perf.profile.v2",
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
    assert report["schema"] == "april.perf.bench.v2"
    assert report["simulated"] is True
    assert {case["pass"] for case in report["cases"]} == {1, 2}
    assert all("message" not in case and "text" not in case for case in report["cases"])
    assert all("secret" not in json.dumps(case).lower() for case in report["cases"])


@pytest.mark.asyncio
async def test_fake_perf_bench_includes_specialists() -> None:
    from apps.runner.commands.runner_perf import _run_fake_bench

    report = await _run_fake_bench(Path.cwd(), role="all", repeat=1)
    assert {case["role"] for case in report["cases"] if "role" in case} == {"coding", "reading"}


@pytest.mark.asyncio
async def test_fake_perf_bench_isolates_routing_stream_and_specialist_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.runner.commands import runner_perf

    class BrokenLifecycle:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def generate(self, _request: object) -> object:
            raise RuntimeError("synthetic runtime failure")

        async def stream(self, _request: object):
            raise RuntimeError("synthetic stream failure")
            yield ("unused", {})

    monkeypatch.setattr(runner_perf, "ModelLifecycle", BrokenLifecycle)
    report = await runner_perf._run_fake_bench(Path.cwd(), role="all", repeat=1)
    assert report["partial"] is True
    encoded = json.dumps(report)
    assert "synthetic runtime failure" not in encoded
    assert "_generated_text" not in encoded


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


@pytest.mark.asyncio
async def test_real_tune_wrapper_and_worker_result_are_fakeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.runner import perf_tune
    from apps.runner.commands import runner_perf

    model = types.SimpleNamespace(id="model", role="brain")
    registry = types.SimpleNamespace(list=lambda: [model])
    monkeypatch.setattr(
        runner_perf,
        "ModelRegistry",
        types.SimpleNamespace(from_file=lambda *_args, **_kwargs: registry),
    )

    async def implementation(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"schema": "april.perf.tune.v2", "profiles": []}

    monkeypatch.setattr(runner_perf, "_run_real_tune_impl", implementation)
    result = await runner_perf._run_real_tune(
        tmp_path, role="brain", max_minutes=1.0, cooldown_seconds=0.0
    )
    assert result["schema"] == "april.perf.tune.v2"

    settings = types.SimpleNamespace(
        home=tmp_path, runtime=types.SimpleNamespace(backend="llama_cpp")
    )
    monkeypatch.setattr(runner_perf, "load_settings", lambda: settings)
    monkeypatch.setattr(runner_perf, "_candidate_values", lambda _model: [("threads", [4])])
    monkeypatch.setattr(runner_perf, "_require_real_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner_perf,
        "datetime",
        types.SimpleNamespace(
            now=lambda _tz: types.SimpleNamespace(strftime=lambda _fmt: "tune-fixed"), UTC="UTC"
        ),
    )
    await asyncio.to_thread(
        runner_perf.perf_tune,
        role="brain",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        dry_run=False,
        fake=False,
    )
    assert (tmp_path / "data/verification/perf-tune-tune-fixed.json").is_file()

    def fake_process(*_args: object, **_kwargs: object) -> Any:
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "metrics": [{"metric": 3, "prompt_metric": 4, "semantic_digest": "digest"}],
                    "semantic_check_digest": "check",
                    "routing_decisions": [],
                    "peak_rss_bytes": 2048,
                    "valid_prefill": True,
                }
            ),
        )

    monkeypatch.setattr(perf_tune, "run_restricted_process_sync", fake_process)
    candidate = types.SimpleNamespace(model_dump=lambda mode: {"id": "model"})
    measured = await perf_tune.measure_real_candidate(tmp_path, candidate, candidate, runs=1)
    assert measured["metric"] == 3
    assert measured["prompt_metric"] == 4


def test_real_bench_uses_typed_client_and_stops_isolated_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_verify = types.ModuleType("apps.runner.verify")
    fake_verify.plan_multi_model_verification = lambda _home: []  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "apps.runner.verify", fake_verify)
    import apps.runner.verification.multi_model as multi_model
    from apps.runner.commands import runner_perf

    class Session:
        instances: ClassVar[list[Session]] = []

        def __init__(self, **_kwargs: object) -> None:
            self.runtime_url = "http://runtime"
            self.runtime_token = "runtime-token"
            self.runtime_log = tmp_path / "runtime.log"
            self.api_log = tmp_path / "api.log"
            self.temp = tmp_path / "isolated"
            self.temp.mkdir()
            self.stopped = False
            self.__class__.instances.append(self)

        def _prepare(self) -> None:
            return None

        def _env(self) -> dict[str, str]:
            return {}

        def _start(self, *_args: object, **_kwargs: object) -> object:
            return object()

        def _wait_json(self, *_args: object, **_kwargs: object) -> None:
            return None

        def _stop(self) -> None:
            self.stopped = True

    class Client:
        async def health(self) -> dict[str, int]:
            return {"process_rss_bytes": 100, "process_peak_rss_bytes": 120}

        async def load(self, _model_id: str) -> object:
            return object()

        async def chat(self, **_kwargs: object) -> object:
            return object()

        async def stream(self, **_kwargs: object):
            yield json.dumps({"event": "token", "payload": {"text": "synthetic"}})
            yield json.dumps(
                {
                    "event": "usage",
                    "payload": {
                        "timing": {
                            "total_ms": 3.0,
                            "ttft_ms": 1.0,
                            "prompt_eval_tokens_per_second": 4.0,
                            "prompt_reuse_ratio": 0.5,
                        },
                        "prefix_cache": {"restore_ms": 0.1, "save_ms": 0.2, "state_bytes": 8},
                    },
                }
            )
            yield json.dumps({"event": "done", "payload": {"finish_reason": "stop"}})

    class Router:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def route_result(self, _message: str, **_kwargs: object):
            async def result() -> object:
                return types.SimpleNamespace(
                    routing_latency_ms=2.0,
                    runtime_timing={"total_ms": 2.0},
                    runtime_prefix_cache={},
                    route_source=types.SimpleNamespace(value="model"),
                    decision=types.SimpleNamespace(
                        intent="normal_conversation", agent="general", tools_needed=[]
                    ),
                )

            return result()

    monkeypatch.setattr(multi_model, "AllConfiguredModelsVerifier", Session)
    monkeypatch.setattr(runner_perf, "RuntimeClient", lambda *_args, **_kwargs: Client())
    monkeypatch.setattr(runner_perf, "BrainRouter", Router)
    monkeypatch.setattr(runner_perf, "trusted_capability_summary", lambda **_kwargs: "capabilities")
    report = runner_perf._run_real_bench_mode(Path.cwd(), role="all", repeat=1, cache_enabled=True)
    assert report["isolation"] == "temporary_verify_services"
    assert report["cases"]
    assert Session.instances[-1].stopped is True


def test_real_bench_comparison_is_partial_when_one_mode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.runner.commands import runner_perf

    calls = 0

    def mode(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("isolated runtime failed")
        return {
            "schema": "april.perf.bench.v2",
            "simulated": False,
            "cases": [],
            "summary": {},
            "partial": False,
            "fallback_count": 0,
        }

    monkeypatch.setattr(runner_perf, "_run_real_bench_mode", mode)
    report = runner_perf._run_real_bench(
        Path.cwd(), role="brain", repeat=1, compare_prefix_cache=True
    )
    assert report["partial"] is True
    assert report["modes"]["off"]["error"]["error_type"] == "RuntimeError"
    assert calls == 2


def test_real_bench_layout_comparison_and_runtime_readiness_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.runner.commands import runner_perf

    monkeypatch.setattr(
        runner_perf,
        "_run_real_bench_mode",
        lambda *_args, **_kwargs: {
            "schema": "april.perf.bench.v2",
            "simulated": False,
            "cases": [],
            "summary": {},
            "partial": False,
            "fallback_count": 0,
        },
    )
    report = runner_perf._run_real_bench(tmp_path, role="brain", repeat=1, compare_layout=True)
    assert report["comparison_kind"] == "layout"
    plain = runner_perf._run_real_bench(tmp_path, role="brain", repeat=1)
    assert plain["simulated"] is False

    monkeypatch.setattr(runner_perf.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(typer.Exit):
        runner_perf._require_real_runtime(tmp_path, "llama_cpp", command="bench")
    monkeypatch.setattr(runner_perf.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        runner_perf,
        "ModelRegistry",
        types.SimpleNamespace(
            from_file=lambda *_args, **_kwargs: types.SimpleNamespace(
                list=lambda: [
                    types.SimpleNamespace(resolved_path=lambda _home: tmp_path / "missing")
                ]
            )
        ),
    )
    with pytest.raises(typer.Exit):
        runner_perf._require_real_runtime(tmp_path, "llama_cpp", command="bench")


def test_perf_bench_rejects_invalid_role() -> None:
    from apps.runner.commands.runner_perf import perf_bench

    with pytest.raises(Exception, match="role must be brain"):
        perf_bench(fake=True, role="invalid", repeat=1, report=None)


@pytest.mark.asyncio
async def test_perf_bench_stream_and_redaction_helpers() -> None:
    from apps.runner.commands import runner_perf

    class Client:
        async def stream(self, **_kwargs: object):
            for item in (
                "not-json",
                json.dumps({"event": "token", "payload": {"text": "a"}}),
                json.dumps({"event": "token", "payload": {"text": 3}}),
                json.dumps({"event": "usage", "payload": {"timing": {"total_ms": 4}}}),
                json.dumps({"event": "done", "payload": {"finish_reason": "length"}}),
                json.dumps({"event": "ignored", "payload": "not-a-map"}),
            ):
                yield item

    usage, text, finish = await runner_perf._stream_bench_call(
        Client(), model_id="model", messages=[], max_output_tokens=1, request_id="request"
    )
    assert usage["timing"]["total_ms"] == 4
    assert text == "a"
    assert finish == "length"
    assert runner_perf._timing_value({}, "missing") is None
    assert runner_perf._health_memory({}) == {"rss_bytes": None, "peak_rss_bytes": None}
    report = {
        "cases": [
            {"phase": "agent", "case_id": "one", "_generated_text": "secret"},
            {"phase": "agent", "case_id": "two", "_generated_text": "same"},
        ],
        "summary": {"coding": {"median_total_ms": 10.0, "calls": 2}},
    }
    other = {
        "cases": [
            {"phase": "agent", "case_id": "one", "_generated_text": "different"},
            {"phase": "agent", "case_id": "two", "_generated_text": "same"},
        ],
        "summary": {"coding": {"median_total_ms": 12.0, "calls": 2}},
    }
    assert runner_perf._agent_text_divergence(report, other) == 1
    assert "_generated_text" not in runner_perf._redact_bench_report(report)["cases"][0]
    assert runner_perf._summary_deltas(report, other)["coding"]["median_total_ms"] == 2.0
    assert runner_perf._summary_deltas({"summary": []}, {"summary": {}}) == {}
    assert runner_perf._routing_passes_identical([]) is None


@pytest.mark.asyncio
async def test_bench_workloads_are_token_sized_for_configured_models(settings_tmp: Any) -> None:
    from agents.registry import default_agent_registry
    from apps.runner.commands import runner_perf
    from services.april_runtime.model_lifecycle import ModelLifecycle
    from services.april_runtime.model_registry import ModelRegistry
    from services.brain.capabilities import trusted_capability_summary
    from services.brain.memory_policy import AgentMemoryContext
    from services.brain.request_context import RequestContext
    from skills.registry import default_registry

    home = Path.cwd()
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    lifecycle = ModelLifecycle(registry, root_backend="fake")
    client = runner_perf._LocalRuntimeClient(lifecycle)
    summary = trusted_capability_summary(
        settings=settings_tmp,
        agent_registry=default_agent_registry(),
        tool_registry=default_registry(),
        model_registry=registry,
        request_context=RequestContext.unknown(),
    )
    for model in registry.list():
        agent = default_agent_registry().get(
            {"brain": "general_agent", "coding": "coding_agent", "reading": "reading_agent"}.get(
                str(model.role), "general_agent"
            )
        )
        if agent is None:
            continue
        plan = await runner_perf._fit_bench_message_set(
            client=client,
            model_id=model.id,
            context_size=model.context_size,
            max_output_tokens=48,
            system_prompt=agent.system_prompt,
            capability_summary=summary,
            memory_context=AgentMemoryContext(history=runner_perf._synthetic_history()),
            question="Synthetic benchmark question.",
        )
        assert plan["skipped"] is False
        assert plan["token_count"] + 48 <= int(0.70 * model.context_size)
        if model.role == "reading":
            assert plan["messages"]
    await lifecycle.cleanup()

    error = runner_perf._bench_failure_report(
        role="brain", repeat=1, simulated=True, exc=ValueError("secret prompt")
    )
    assert error["partial"] is True
    assert "secret prompt" not in json.dumps(error)
    assert runner_perf._bench_error(ValueError()) == {
        "error_code": "BENCH_CALL_FAILED",
        "error_type": "ValueError",
    }


def test_perf_workload_filler_and_budget_helpers_are_token_based() -> None:
    from apps.runner.perf_workload import (
        FILLER_UNIT,
        build_filler,
        raw_workload_budget,
        workload_budget,
        workload_budget_for_fraction,
    )

    def count(text: str) -> int:
        return len(text.split())

    filler = build_filler(10, count)
    assert filler.startswith(FILLER_UNIT)
    assert count(filler) <= 10
    assert build_filler(0, count) == ""
    assert workload_budget(4096, 48, 100) == 2719
    assert workload_budget(128, 48, 100) == 128
    assert workload_budget_for_fraction(4096, 32, 10, 0.60) == 2415
    assert raw_workload_budget(128, 48, 100) < 128


@pytest.mark.asyncio
async def test_perf_workload_async_filler_and_skip_reasons() -> None:
    from apps.runner.perf_workload import build_filler_async, fit_bench_message_set
    from services.brain.memory_policy import AgentMemoryContext

    async def count_text(text: str) -> int:
        return len(text.split())

    assert (await build_filler_async(7, count_text)).split()
    assert await build_filler_async(0, count_text) == ""

    class CountingClient:
        async def count_message_tokens(self, *, model_id: str, messages: list[Any]) -> int:
            del model_id
            return sum(len(message.content.split()) for message in messages)

    fixed = await fit_bench_message_set(
        client=CountingClient(),
        model_id="test",
        context_size=128,
        max_output_tokens=1,
        system_prompt="word " * 100,
        capability_summary="capability",
        memory_context=AgentMemoryContext(),
        question="question",
    )
    assert fixed["skipped"] is True
    assert fixed["reason"] == "fixed_prompt_exceeds_budget"

    budget = await fit_bench_message_set(
        client=CountingClient(),
        model_id="test",
        context_size=128,
        max_output_tokens=32,
        system_prompt="system",
        capability_summary="capability",
        memory_context=AgentMemoryContext(),
        question="question",
    )
    assert budget["skipped"] is True
    assert budget["reason"] == "workload_budget_below_minimum"

    class OverheadClient(CountingClient):
        calls = 0

        async def count_message_tokens(self, *, model_id: str, messages: list[Any]) -> int:
            measured = await super().count_message_tokens(model_id=model_id, messages=messages)
            self.calls += 1
            return measured + (50 if self.calls > 1 and len(messages) > 1 else 0)

    sized = await fit_bench_message_set(
        client=OverheadClient(),
        model_id="test",
        context_size=512,
        max_output_tokens=32,
        system_prompt="system",
        capability_summary="capability",
        memory_context=AgentMemoryContext(),
        question="question",
    )
    assert sized["skipped"] is False
    assert sized["token_count"] + 32 <= int(0.70 * 512)


@pytest.mark.asyncio
async def test_fake_bench_workload_skip_paths_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.runner.perf_fake_bench as fake_bench

    class EmptyRegistry:
        def list(self) -> list[Any]:
            return []

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr(fake_bench, "default_agent_registry", lambda: EmptyRegistry())
    brain_report = await fake_bench.run_fake_bench(Path.cwd(), role="brain", repeat=1)
    assert any(case.get("phase") == "agent" and "error" in case for case in brain_report["cases"])

    from agents.registry import default_agent_registry

    monkeypatch.setattr(fake_bench, "default_agent_registry", default_agent_registry)

    async def skip_workload(**_kwargs: Any) -> dict[str, Any]:
        return {"skipped": True, "reason": "test_skip"}

    monkeypatch.setattr(fake_bench, "fit_agent_workload", skip_workload)
    brain_skip_report = await fake_bench.run_fake_bench(Path.cwd(), role="brain", repeat=1)
    assert any(case.get("phase") == "workload_skipped" for case in brain_skip_report["cases"])
    coding_report = await fake_bench.run_fake_bench(Path.cwd(), role="coding", repeat=1)
    assert coding_report["cases"][0]["phase"] == "workload_skipped"

    reading_report = await fake_bench.run_fake_bench(Path.cwd(), role="creative", repeat=1)
    assert reading_report["cases"] == []


def test_tune_workload_prompt_is_deterministic_and_token_sized() -> None:
    from apps.runner.perf_worker import _workload_prompt

    def count_tokens(text: str) -> int:
        return len(text.split())

    prompt = _workload_prompt("nonce", 1024, count_tokens, max_output_tokens=32)
    assert prompt == _workload_prompt("nonce", 1024, count_tokens, max_output_tokens=32)
    assert count_tokens(prompt) + 32 <= int(0.60 * 1024)
    assert _workload_prompt("nonce", 128, count_tokens, max_output_tokens=32).startswith(
        "Measurement nonce: nonce"
    )
    assert _workload_prompt("nonce", 512).startswith("Measurement nonce: nonce")


@pytest.mark.asyncio
async def test_tune_sized_workload_uses_tokenizer_and_fails_closed() -> None:
    from apps.runner.perf_worker import _sized_workload_prompt

    class CountingLifecycle:
        def __init__(self, *, oversized: bool = False) -> None:
            self.calls = 0
            self.oversized = oversized

        async def count_message_tokens(self, model_id: str, messages: list[Any]) -> int:
            del model_id
            self.calls += 1
            text = messages[0].content
            if self.oversized and self.calls > 1 and "\n" in text:
                return 10_000
            return len(text.split())

    skipped, _, reason = await _sized_workload_prompt(
        CountingLifecycle(), "model", "nonce", 128, 32
    )
    assert skipped is None
    assert reason == "workload_budget_below_minimum"

    lifecycle = CountingLifecycle(oversized=True)
    skipped, _, reason = await _sized_workload_prompt(lifecycle, "model", "nonce", 512, 32)
    assert skipped is None
    assert reason == "sized_prompt_exceeds_budget"
    assert lifecycle.calls > 4


def test_perf_bench_writes_failure_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.runner.commands import runner_perf

    settings = types.SimpleNamespace(home=tmp_path, runtime=types.SimpleNamespace(backend="fake"))
    monkeypatch.setattr(runner_perf, "load_settings", lambda: settings)

    async def fail(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("private prompt")

    monkeypatch.setattr(runner_perf, "_run_fake_bench", fail)
    output = tmp_path / "failure.json"
    runner_perf.perf_bench(fake=True, role="brain", repeat=1, report=output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["partial"] is True
    assert "private prompt" not in output.read_text(encoding="utf-8")


def test_real_perf_bench_writes_failure_report_after_readiness_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.runner.commands import runner_perf

    settings = types.SimpleNamespace(
        home=tmp_path, runtime=types.SimpleNamespace(backend="llama_cpp")
    )
    monkeypatch.setattr(runner_perf, "load_settings", lambda: settings)
    monkeypatch.setattr(
        runner_perf,
        "_require_real_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime unavailable")),
    )
    output = tmp_path / "real-failure.json"
    runner_perf.perf_bench(fake=False, role="brain", repeat=1, report=output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["partial"] is True
    assert report["error"]["error_type"] == "RuntimeError"


def test_stable_layout_message_helpers_keep_volatile_content_out_of_system_prefix() -> None:
    from agents.registry import default_agent_registry
    from services.brain.agent_loop import StructuredAgentLoop
    from services.brain.memory_policy import AgentMemoryContext
    from services.brain.orchestration.finalization_flow import conversation_chat_messages

    agent = default_agent_registry().get("general_agent")
    assert agent is not None
    loop = StructuredAgentLoop.__new__(StructuredAgentLoop)
    messages = loop._initial_messages(
        agent, "question", [], ["volatile request evidence"], "stable identity"
    )
    assert "stable identity" in messages[0].content
    assert "volatile request evidence" in messages[-1].content
    assert "volatile request evidence" not in messages[0].content

    messages = conversation_chat_messages(
        system_prompt="agent",
        memory_context=AgentMemoryContext(),
        current_prompt="volatile",
        stable_prefix="only stable",
    )
    assert messages[0].content == "agent\n\nonly stable"
    assert messages[-1].content == "volatile"


def test_perf_tune_rejects_invalid_role() -> None:
    from apps.runner.commands.runner_perf import perf_tune

    with pytest.raises(Exception, match="role must be brain"):
        perf_tune(role="invalid", max_minutes=1.0, cooldown_seconds=0.0, dry_run=True, fake=False)


def test_perf_commands_refuse_non_real_backend(
    monkeypatch: pytest.MonkeyPatch, settings_tmp: Any
) -> None:
    from apps.runner.commands import runner_perf

    monkeypatch.setattr(runner_perf, "load_settings", lambda: settings_tmp)
    monkeypatch.setattr(
        runner_perf,
        "ModelRegistry",
        types.SimpleNamespace(
            from_file=lambda *_args, **_kwargs: types.SimpleNamespace(list=lambda: [])
        ),
    )
    with pytest.raises(typer.Exit):
        runner_perf.perf_bench(fake=False, role="brain", repeat=1, report=None)
    with pytest.raises(typer.Exit):
        runner_perf.perf_tune(
            role="brain", max_minutes=1.0, cooldown_seconds=0.0, dry_run=False, fake=False
        )


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
    from apps.runner.commands import runner_perf
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
    assert _accept_candidate(
        {**baseline, "routing_decisions": [("a", "b", "c")]},
        {**baseline, "routing_decisions": [("x", "b", "c")]},
        "threads",
    ) == (False, "semantic_drift")
    assert _accept_candidate(
        {**baseline, "routing_decisions": []},
        {**baseline, "routing_decisions": []},
        "threads",
    ) != (False, "semantic_drift")
    assert _accept_candidate({**baseline, "valid_prefill": False}, baseline, "threads") == (
        False,
        "unmeasured",
    )
    assert _accept_candidate({**baseline, "metric": 0}, {**baseline, "metric": 10}, "threads") == (
        False,
        "unmeasured",
    )
    model = types.SimpleNamespace(context_size=200, threads=8, threads_batch=8)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runner_perf, "_physical_cores_for_plan", lambda: None)
    values = runner_perf._candidate_values(model)
    assert values[0][0] == "n_batch=n_ubatch"
    monkeypatch.setattr(runner_perf, "_physical_cores_for_plan", lambda: 8)
    assert {name for name, _values in runner_perf._candidate_values(model)} == {
        "threads_batch",
        "n_batch=n_ubatch",
        "flash_attn",
        "threads",
    }
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_tune_coordinate_sweep_uses_fake_worker_and_writes_profile(tmp_path: Path) -> None:
    from apps.runner import perf_tune
    from apps.runner.commands.runner_perf import _accept_candidate

    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="coding",
        threads=8,
        threads_batch=8,
        context_size=1024,
        temperature=0.0,
        max_output_tokens=32,
    )
    registry = types.SimpleNamespace(list=lambda: [model])

    async def fake_measure(
        _home: Path,
        _baseline: ModelDefinition,
        candidate: ModelDefinition,
        *,
        runs: int,
        nonce_prefix: str,
    ) -> dict[str, Any]:
        del runs, nonce_prefix
        return {
            "metric": 20.0 if candidate.threads == 4 else 10.0,
            "prompt_metric": 10.0,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 100,
            "valid_prefill": True,
        }

    output = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        registry=registry,
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda baseline, candidate, knob: _accept_candidate(
            baseline, candidate, knob
        ),
        profile_inputs=lambda _model, _root: {"identity": "fixed"},
        profile_fingerprint=lambda _inputs: "fixed-fingerprint",
        tunable_fields={"threads", "threads_batch", "n_batch", "n_ubatch", "flash_attn"},
        measure_candidate=fake_measure,
    )
    assert output["profiles"] == [{"model_id": "model", "fingerprint": "fixed-fingerprint"}]
    assert (tmp_path / "data/perf/profiles/fixed-fingerprint.json").exists()


@pytest.mark.asyncio
async def test_tune_budget_and_final_recheck_never_write_unaccepted_profile(
    tmp_path: Path,
) -> None:
    from apps.runner import perf_tune

    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="coding",
        threads=8,
        threads_batch=8,
        context_size=1024,
        temperature=0.0,
        max_output_tokens=32,
    )
    registry = types.SimpleNamespace(list=lambda: [model])

    async def budget_measure(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise perf_tune.TuneBudgetExceeded

    exhausted = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        registry=registry,
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (True, None),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "budget",
        tunable_fields={"threads"},
        measure_candidate=budget_measure,
    )
    assert exhausted["stopped_due_to_budget"] is True
    assert exhausted["profiles"] == []

    calls = 0

    async def good_measure(
        _home: Path,
        _baseline: ModelDefinition,
        candidate: ModelDefinition,
        *,
        runs: int,
        nonce_prefix: str,
    ) -> dict[str, Any]:
        del runs, nonce_prefix
        return {
            "metric": 20.0 if candidate.threads == 4 else 10.0,
            "prompt_metric": 10.0,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 100,
            "valid_prefill": True,
        }

    def reject_final(
        _baseline: dict[str, Any], _candidate: dict[str, Any], _knob: str
    ) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return (calls == 1, None if calls == 1 else "slower")

    rejected = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        registry=registry,
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=reject_final,
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "rejected",
        tunable_fields={"threads"},
        measure_candidate=good_measure,
    )
    assert rejected["profiles"] == []
    assert rejected["results"][-1]["knob"] == "final_recheck"


@pytest.mark.asyncio
async def test_tune_helpers_cover_pair_cooldown_and_knob_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.runner import perf_tune

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(perf_tune.asyncio, "sleep", fake_sleep)
    model = types.SimpleNamespace(
        threads=8, threads_batch=8, n_batch=128, n_ubatch=128, flash_attn=False
    )

    async def sample(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {
            "metric": 1,
            "prompt_metric": 2,
            "outputs": [],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    left, right = await perf_tune.measure_abab_pair(
        Path("."),
        model,
        model,
        cooldown_seconds=1.0,
        deadline=10**12,
        measure_candidate=sample,
        pairs=1,
    )
    assert left["metric"] == right["metric"] == 1
    assert slept == [1.0, 1.0]
    assert perf_tune._aggregate_measurements([])["metric"] == 0.0
    assert (
        perf_tune.changed_tunable_knob(
            model, types.SimpleNamespace(**{**vars(model), "threads_batch": 4})
        )
        == "threads_batch"
    )
    assert (
        perf_tune.changed_tunable_knob(
            model, types.SimpleNamespace(**{**vars(model), "n_batch": 256})
        )
        == "n_batch=n_ubatch"
    )
    assert (
        perf_tune.changed_tunable_knob(
            model, types.SimpleNamespace(**{**vars(model), "flash_attn": True})
        )
        == "flash_attn"
    )
    assert perf_tune.changed_tunable_knob(model, model) == "threads"


@pytest.mark.asyncio
async def test_tune_pair_and_sweep_isolate_worker_failures(tmp_path: Path) -> None:
    from apps.runner import perf_tune

    model = object()
    candidate_calls = 0

    async def candidate_failure(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal candidate_calls
        candidate_calls += 1
        if candidate_calls == 2:
            raise RuntimeError("candidate worker")
        return {"metric": 1, "prompt_metric": 1, "valid_prefill": True}

    with pytest.raises(perf_tune.TuneWorkerFailed, match="candidate"):
        await perf_tune.measure_abab_pair(
            tmp_path,
            object(),
            object(),
            cooldown_seconds=0,
            deadline=time.monotonic() + 1,
            measure_candidate=candidate_failure,
            pairs=1,
            runs=1,
        )

    async def successful_measure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "metric": 1,
            "prompt_metric": 1,
            "outputs": [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    with pytest.raises(perf_tune.TuneBudgetExceeded):
        await perf_tune.measure_abab_pair(
            tmp_path,
            model,
            model,
            cooldown_seconds=0.01,
            deadline=time.monotonic() + 0.001,
            measure_candidate=successful_measure,
            pairs=1,
            runs=1,
        )

    async def baseline_failure(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("baseline worker")

    with pytest.raises(perf_tune.TuneWorkerFailed, match="baseline"):
        await perf_tune.measure_abab_pair(
            tmp_path,
            object(),
            object(),
            cooldown_seconds=0,
            deadline=time.monotonic() + 1,
            measure_candidate=baseline_failure,
            pairs=1,
            runs=1,
        )

    def model_with_copy(model_id: str) -> Any:
        model = types.SimpleNamespace(id=model_id, role="coding", threads=8)
        model.model_copy = lambda update: types.SimpleNamespace(
            id=model.id,
            role=model.role,
            threads=update.get("threads", model.threads),
        )
        return model

    model_a = model_with_copy("baseline-failed")
    model_b = model_with_copy("candidate-failed")

    async def sweep_measure(
        _home: Path,
        baseline: Any,
        candidate: Any,
        **_kwargs: object,
    ) -> dict[str, Any]:
        if baseline.id == "baseline-failed":
            raise RuntimeError("baseline")
        if candidate is not baseline:
            raise RuntimeError("candidate")
        return {
            "metric": 1,
            "prompt_metric": 1,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    output = await perf_tune.run_real_tune(
        tmp_path,
        role="all",
        max_minutes=1,
        cooldown_seconds=0,
        registry=types.SimpleNamespace(list=lambda: [model_a, model_b]),
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (False, "slower"),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "unused",
        tunable_fields={"threads"},
        measure_candidate=sweep_measure,
    )
    assert [item["reason"] for item in output["results"]] == [
        "baseline_failed",
        "worker_failed",
    ]


@pytest.mark.asyncio
async def test_tune_worker_rejects_nonzero_and_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.runner import perf_tune

    candidate = types.SimpleNamespace(model_dump=lambda mode: {"id": "model"})
    result = types.SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr(perf_tune, "run_restricted_process_sync", lambda *_a, **_k: result)
    with pytest.raises(perf_tune.TuneWorkerFailed):
        await perf_tune.measure_real_candidate(tmp_path, candidate, candidate)
    result.returncode = 0
    result.stdout = "not json"
    with pytest.raises(perf_tune.TuneWorkerFailed):
        await perf_tune.measure_real_candidate(tmp_path, candidate, candidate)


@pytest.mark.asyncio
async def test_tune_real_path_runs_brain_routing_checks_and_passes_sandbox_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.runner import perf_tune

    model = ModelDefinition(
        id="brain",
        name="brain",
        path=tmp_path / "brain.gguf",
        backend="llama_cpp",
        role="brain",
        threads=8,
        context_size=4096,
        temperature=0.0,
        max_output_tokens=32,
    )
    calls: list[dict[str, Any]] = []

    async def measure(
        _home: Path,
        _baseline: Any,
        candidate: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append({"candidate": candidate, **kwargs})
        metric = 2.0 if candidate.threads == 4 else 1.0
        return {
            "metric": metric,
            "prompt_metric": metric,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": [("normal_conversation", "general", "none")],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    monkeypatch.setattr(perf_tune, "measure_real_candidate", measure)
    output = await perf_tune.run_real_tune(
        tmp_path,
        role="brain",
        max_minutes=1,
        cooldown_seconds=0,
        registry=types.SimpleNamespace(list=lambda: [model]),
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda baseline, candidate, _knob: (
            candidate["metric"] >= baseline["metric"] * 1.05,
            None,
        ),
        profile_inputs=lambda *_args: {"identity": "fixed"},
        profile_fingerprint=lambda _inputs: "brain-profile",
        tunable_fields={"threads"},
        development_unsandboxed_override=True,
    )
    assert output["profiles"]
    assert sum(call.get("routing_check") is True for call in calls) == 2
    assert output["routing_cases_compared"] == 1
    assert sum(call.get("routing_check") is not True for call in calls) > 0
    assert all(call.get("development_unsandboxed_override") is True for call in calls)


@pytest.mark.asyncio
async def test_tune_routing_mismatch_blocks_profile_and_runs_only_two_route_workers(
    tmp_path: Path,
) -> None:
    from apps.runner import perf_tune

    model = ModelDefinition(
        id="brain",
        name="brain",
        path=tmp_path / "brain.gguf",
        backend="llama_cpp",
        role="brain",
        threads=8,
        context_size=4096,
        temperature=0.0,
        max_output_tokens=32,
    )
    calls: list[bool] = []

    async def measure(
        _home: Path,
        _baseline: Any,
        candidate: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        routing = bool(kwargs.get("routing_check"))
        calls.append(routing)
        return {
            "metric": 2.0 if candidate.threads == 4 else 1.0,
            "prompt_metric": 2.0,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": (
                [("changed", "general", "none")]
                if candidate.threads == 4
                else [("same", "general", "none")]
            )
            if routing
            else [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    output = await perf_tune.run_real_tune(
        tmp_path,
        role="brain",
        max_minutes=1,
        cooldown_seconds=0,
        registry=types.SimpleNamespace(list=lambda: [model]),
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (True, None),
        profile_inputs=lambda *_args: {"identity": "fixed"},
        profile_fingerprint=lambda _inputs: "blocked",
        tunable_fields={"threads"},
        measure_candidate=measure,
    )
    assert sum(calls) == 2
    assert output["profiles"] == []
    routing_result = output["results"][-1]
    assert routing_result["reason"] == "semantic_drift"
    assert routing_result["routing_cases_compared"] == 1


@pytest.mark.asyncio
async def test_tune_final_recheck_worker_failure_and_deadline_are_isolated(
    tmp_path: Path,
) -> None:
    from apps.runner import perf_tune

    model = ModelDefinition(
        id="coding",
        name="coding",
        path=tmp_path / "coding.gguf",
        backend="fake",
        role="coding",
        threads=8,
        context_size=2048,
        temperature=0.0,
        max_output_tokens=32,
    )
    calls = 0

    async def measure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls > 4:
            raise perf_tune.TuneWorkerFailed("candidate")
        return {
            "metric": 2 if calls % 2 == 0 else 1,
            "prompt_metric": 2 if calls % 2 == 0 else 1,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    output = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=1,
        cooldown_seconds=0,
        registry=types.SimpleNamespace(list=lambda: [model]),
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (True, None),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "unused",
        tunable_fields={"threads"},
        measure_candidate=measure,
    )
    assert output["results"][-1]["reason"] == "worker_failed"
    with pytest.raises(perf_tune.TuneBudgetExceeded):
        await perf_tune.measure_abab_pair(
            tmp_path,
            model,
            model,
            cooldown_seconds=0,
            deadline=time.monotonic() - 1,
            measure_candidate=measure,
        )


@pytest.mark.asyncio
async def test_tune_routing_failure_and_candidate_exception_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.runner import perf_tune

    model = ModelDefinition(
        id="brain",
        name="brain",
        path=tmp_path / "brain.gguf",
        backend="fake",
        role="brain",
        threads=8,
        context_size=2048,
        temperature=0.0,
        max_output_tokens=32,
    )

    async def route_failure(
        _home: Path, _baseline: Any, candidate: Any, **kwargs: Any
    ) -> dict[str, Any]:
        if kwargs.get("routing_check"):
            raise RuntimeError("routing worker")
        metric = 2 if candidate.threads == 4 else 1
        return {
            "metric": metric,
            "prompt_metric": metric,
            "outputs": ["same"],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    monkeypatch.setattr(perf_tune, "measure_real_candidate", route_failure)
    output = await perf_tune.run_real_tune(
        tmp_path,
        role="brain",
        max_minutes=1,
        cooldown_seconds=0,
        registry=types.SimpleNamespace(list=lambda: [model]),
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (True, None),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "unused",
        tunable_fields={"threads"},
    )
    assert output["results"][-1]["knob"] == "routing_check"

    explicit_calls = 0

    async def explicit_candidate_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal explicit_calls
        explicit_calls += 1
        if explicit_calls == 2:
            raise perf_tune.TuneWorkerFailed("candidate")
        return {"metric": 1, "prompt_metric": 1, "valid_prefill": True}

    with pytest.raises(perf_tune.TuneWorkerFailed, match="candidate"):
        await perf_tune.measure_abab_pair(
            tmp_path,
            model,
            model,
            cooldown_seconds=0,
            deadline=time.monotonic() + 1,
            measure_candidate=explicit_candidate_failure,
            pairs=1,
            runs=1,
        )


@pytest.mark.asyncio
async def test_tune_deadline_paths_are_explicit_and_do_not_write_profiles(tmp_path: Path) -> None:
    from apps.runner import perf_tune

    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="coding",
        threads=8,
        threads_batch=8,
        context_size=1024,
        temperature=0.0,
        max_output_tokens=32,
    )
    registry = types.SimpleNamespace(list=lambda: [model])
    empty = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=0.0,
        cooldown_seconds=0.0,
        registry=registry,
        candidate_values=lambda _model: [],
        accept_candidate=lambda *_args: (True, None),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "unused",
        tunable_fields={"threads"},
    )
    assert empty["stopped_due_to_budget"] is True

    async def sample(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {
            "metric": 1,
            "prompt_metric": 1,
            "outputs": [],
            "semantic_check": "same",
            "routing_decisions": [],
            "peak_rss": 1,
            "valid_prefill": True,
        }

    rejected = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        registry=registry,
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (False, "slower"),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "unused",
        tunable_fields={"threads"},
        measure_candidate=sample,
    )
    assert rejected["results"][0]["reason"] == "slower"

    original_pair = perf_tune.measure_abab_pair

    async def fail_final(*args: object, **kwargs: object) -> Any:
        if kwargs.get("first") is not None:
            raise perf_tune.TuneBudgetExceeded
        return await original_pair(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(perf_tune, "measure_abab_pair", fail_final)
    budget_final = await perf_tune.run_real_tune(
        tmp_path,
        role="coding",
        max_minutes=1.0,
        cooldown_seconds=0.0,
        registry=registry,
        candidate_values=lambda _model: [("threads", [4])],
        accept_candidate=lambda *_args: (True, None),
        profile_inputs=lambda *_args: {},
        profile_fingerprint=lambda _inputs: "unused",
        tunable_fields={"threads"},
        measure_candidate=sample,
    )
    assert budget_final["stopped_due_to_budget"] is True
    assert budget_final["profiles"] == []
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_perf_worker_measurement_is_fakeable_without_llama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.runner import perf_worker

    class FakeLifecycle:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def generate(self, _request: Any) -> Any:
            from services.april_runtime.schemas import ChatResponse, Usage

            return ChatResponse(
                request_id="worker",
                model_id="model",
                content="ok",
                usage=Usage(
                    input_tokens=100,
                    output_tokens=2,
                    total_tokens=102,
                    timing={
                        "prompt_tokens": 100,
                        "prompt_eval_tokens": 100,
                        "prompt_eval_tokens_per_second": 10.0,
                        "eval_tokens_per_second": 2.0,
                    },
                ),
            )

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(perf_worker, "ModelLifecycle", FakeLifecycle)
    from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat

    client = perf_worker._LifecycleClient(FakeLifecycle())
    client_response = await client.chat(
        model_id="model",
        messages=[ChatMessage(role="user", content="synthetic")],
        options=GenerationOptions(temperature=0.0, max_output_tokens=1),
        response_format=ResponseFormat(),
    )
    assert client_response.content == "ok"
    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="coding",
        threads=4,
        context_size=1024,
        temperature=0.0,
        max_output_tokens=32,
    )
    result = await perf_worker.measure(
        {"home": str(tmp_path), "model": model.model_dump(mode="json"), "runs": 1}
    )
    assert result["valid_prefill"] is True
    assert result["metrics"][0]["prompt_metric"] == 10.0


@pytest.mark.asyncio
async def test_perf_worker_brain_routing_and_entrypoint_are_fakeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from apps.runner import perf_worker

    class FakeLifecycle:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def generate(self, _request: Any) -> Any:
            from services.april_runtime.schemas import ChatResponse, Usage

            return ChatResponse(
                request_id="worker",
                model_id="model",
                content="ok",
                usage=Usage(
                    input_tokens=100,
                    output_tokens=2,
                    total_tokens=102,
                    timing={
                        "prompt_tokens": 100,
                        "prompt_eval_tokens": 100,
                        "prompt_eval_tokens_per_second": 10.0,
                        "eval_tokens_per_second": 2.0,
                    },
                ),
            )

        async def cleanup(self) -> None:
            return None

    class FakeRouter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def route_result(self, _message: str) -> Any:
            return types.SimpleNamespace(
                decision=types.SimpleNamespace(intent="chat", agent="general", tools_needed=[])
            )

    monkeypatch.setattr(perf_worker, "ModelLifecycle", FakeLifecycle)
    monkeypatch.setattr(perf_worker, "BrainRouter", FakeRouter)
    fixture = tmp_path / "tests/fixtures/evals/brain_routes.yaml"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("cases:\n  - message: synthetic\n", encoding="utf-8")
    model = ModelDefinition(
        id="model",
        name="model",
        path=tmp_path / "model.gguf",
        backend="fake",
        role="brain",
        threads=4,
        context_size=1024,
        temperature=0.0,
        max_output_tokens=32,
    )
    result = await perf_worker.measure(
        {"home": str(tmp_path), "model": model.model_dump(mode="json"), "runs": 1}
    )
    assert result["routing_decisions"] == [("chat", "general", "none")]
    monkeypatch.setattr(perf_worker.asyncio, "run", lambda _coroutine: {"ok": True})
    monkeypatch.setattr(perf_worker, "measure", lambda _payload: {"ok": True})
    monkeypatch.setattr(__import__("sys"), "argv", ["perf_worker", "--payload", "{}"])
    perf_worker.main()
    assert json.loads(capsys.readouterr().out) == {"ok": True}


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
    class PerfContextData(ctypes.Structure):
        _fields_ = [
            ("t_start_ms", ctypes.c_double),
            ("t_load_ms", ctypes.c_double),
            ("t_p_eval_ms", ctypes.c_double),
            ("t_eval_ms", ctypes.c_double),
            ("n_p_eval", ctypes.c_int32),
            ("n_eval", ctypes.c_int32),
            ("n_reused", ctypes.c_int32),
        ]

    backend = LlamaCppBackend()
    backend._llm = types.SimpleNamespace(ctx=object())
    backend._llama_module = types.SimpleNamespace(
        llama_perf_context_reset=lambda _ctx: None,
        llama_perf_context=lambda _ctx: PerfContextData(0.0, 1.0, 3.5, 2.0, 12, 4, 0),
    )
    backend._begin_perf()
    backend._finish_perf(12, 4)
    assert backend.timing_diagnostics()["prompt_eval_tokens"] == 12
    assert backend.timing_diagnostics()["eval_ms"] == 2.0


def test_perf_report_helpers_and_hardware_success_only_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import april_common.hardware_profile as hardware
    from apps.runner.commands import runner_perf
    from apps.runner.commands.runner_perf import (
        _bench_summary,
        _compare_bench_modes,
        _routing_modes_identical,
        _routing_passes_identical,
    )

    cases = [
        {
            "workload": "brain",
            "phase": "routing",
            "pass": 1,
            "operation": "a",
            "context": "b",
            "tool_class": "none",
            "timing": {"total_ms": 10.0, "prompt_eval_tokens_per_second": 5.0},
        },
        {
            "workload": "brain",
            "phase": "routing",
            "pass": 2,
            "operation": "a",
            "context": "b",
            "tool_class": "none",
            "timing": {"total_ms": 20.0, "prompt_eval_tokens_per_second": 7.0},
        },
    ]
    assert _bench_summary(cases)["brain"]["median_total_ms"] == 15.0
    assert _routing_passes_identical(cases) is True
    off = {"cases": cases, "fallback_count": 0}
    on = {"cases": cases, "fallback_count": 1}
    assert _routing_modes_identical(off, on)
    comparison = _compare_bench_modes(role="brain", repeat=1, off=off, on=on)
    assert comparison["comparison"]["fallback_or_failure_count"]["on"] == 1
    assert (
        _bench_summary(
            [
                {
                    "workload": "brain",
                    "phase": "agent",
                    "pass": 1,
                    "timing": {"total_ms": 4},
                }
            ]
        )["brain"]["first_pass"]["median_total_ms"]
        == 4
    )
    assert (
        runner_perf._summary_deltas(
            {"summary": {"brain": {"first_pass": {"median_total_ms": 4}}}},
            {"summary": {"brain": {"first_pass": {"median_total_ms": 6}}}},
        )["brain"]["median_total_ms"]
        == 2
    )
    assert runner_perf._summary_deltas({"summary": {"bad": []}}, {"summary": {"bad": {}}}) == {}

    monkeypatch.setattr(hardware, "_physical_cpu_cache", None)
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware, "_darwin_sysctl_uint", lambda _name: 8)
    assert hardware.physical_cpu_count() == 8
    monkeypatch.setattr(hardware, "_physical_cpu_cache", None)
    monkeypatch.setattr(hardware, "_darwin_sysctl_uint", lambda _name: None)
    assert hardware.physical_cpu_count() is None
    monkeypatch.setattr(hardware, "_darwin_sysctl_string", lambda _name: "Intel CPU")
    monkeypatch.setattr(hardware, "_cpu_brand_cache", None)
    assert hardware.cpu_brand() == "Intel CPU"
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(
        hardware.Path,
        "read_text",
        lambda *_args, **_kwargs: (
            "processor: 0\nphysical id: 0\ncore id: 0\n\n"
            "processor: 1\nphysical id: 0\ncore id: 1\n\n"
            "model name: Synthetic CPU\n"
        ),
    )
    monkeypatch.setattr(hardware, "_physical_cpu_cache", None)
    monkeypatch.setattr(hardware, "_cpu_brand_cache", None)
    assert hardware.physical_cpu_count() == 2
    assert hardware.cpu_brand() == "Synthetic CPU"
    assert not (tmp_path / "unused").exists()


@pytest.mark.asyncio
async def test_router_prewarm_audits_success_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.api.application import _router_prewarm

    class Audit:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def write(self, event: dict[str, object]) -> None:
            self.events.append(event)

    class Client:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.requests: list[dict[str, object]] = []

        async def chat(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if self.failures:
                self.failures -= 1
                raise RuntimeError("offline")
            return object()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("services.api.application.asyncio.sleep", no_sleep)
    audit = Audit()
    client = Client(0)
    active = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            runtime=types.SimpleNamespace(backend="llama_cpp", prefix_prewarm=True)
        ),
        governor=types.SimpleNamespace(
            assess_resident_async=lambda: _allowed_decision(),
        ),
        orchestrator=types.SimpleNamespace(
            brain_router=types.SimpleNamespace(
                router_model_id="brain", router_system_prompt="router-system"
            )
        ),
        runtime_client=client,
        approvals=types.SimpleNamespace(audit=audit),
    )
    await _router_prewarm(active)
    assert audit.events[0]["status"] == "loaded"
    assert client.requests[0]["messages"][0].content == "router-system"

    audit.events.clear()
    failing = Client(3)
    active.runtime_client = failing
    await _router_prewarm(active)
    assert audit.events == [
        {
            "event_type": "router_prefix_prewarm",
            "actor": "core_api",
            "status": "failed",
            "reason": "RuntimeError",
        }
    ]


async def _allowed_decision() -> object:
    return types.SimpleNamespace(allowed=True, reasons=())
