from __future__ import annotations

import json
from pathlib import Path

from apps.runner.mac_report import (
    EnvironmentSnapshot,
    RealModelReport,
    ReportThresholds,
    RoutingReport,
    SkippedCheck,
    build_mac_report,
    environment_snapshot,
    quantization_from_basename,
    routing_report_from_results,
    write_report,
)
from apps.runner.verify import TargetMacValidator

ENV = EnvironmentSnapshot(
    generated_at="2026-06-25T00:00:00Z",
    os="Darwin 24.0.0",
    cpu_architecture="arm64",
    python_version="3.11.15",
)


def _passing_real_model() -> RealModelReport:
    return RealModelReport(
        attempted=True,
        model_id="april-brain",
        role="brain",
        path_basename="granite-3.3-2b-q4_k_m.gguf",
        quantization="q4_k_m",
        context_size=1024,
        load_success=True,
        load_duration_seconds=1.5,
        chat_success=True,
        structured_brain_json_success=True,
        streaming_success=True,
        first_token_latency_seconds=0.3,
        unload_success=True,
        output_token_count=32,
        tokens_per_second=18.0,
        process_rss_bytes=512_000_000,
        process_peak_rss_bytes=600_000_000,
    )


# --- pure builder ----------------------------------------------------------


def test_quantization_from_basename() -> None:
    assert quantization_from_basename("granite-3.3-2b-q4_k_m.gguf") == "q4_k_m"
    assert quantization_from_basename("model-f16.gguf") == "f16"
    assert quantization_from_basename("plain.gguf") is None
    assert quantization_from_basename(None) is None


def test_routing_report_from_results() -> None:
    class _R:
        def __init__(self, ok: bool) -> None:
            self.ok = ok

    report = routing_report_from_results([_R(True), _R(True), _R(False)])
    assert report.total == 3
    assert report.passed == 2
    assert report.accuracy == round(2 / 3, 4)


class _EvalLike:
    """Minimal stand-in for a BrainEvalResult with the fields the report reads."""

    def __init__(
        self,
        case_id: str,
        *,
        ok: bool,
        schema_valid: bool,
        routing_ok: bool,
        routing_method: str | None,
    ) -> None:
        self.id = case_id
        self.ok = ok
        self.schema_valid = schema_valid
        self.routing_ok = routing_ok
        self.actual = {"routing_method": routing_method}


def test_routing_report_tracks_valid_failures_and_per_case() -> None:
    results = [
        _EvalLike(
            "normal_chat", ok=True, schema_valid=True, routing_ok=True, routing_method="model"
        ),
        _EvalLike(
            "repair_case",
            ok=True,
            schema_valid=True,
            routing_ok=True,
            routing_method="model_repair",
        ),
        _EvalLike(
            "fallback_case",
            ok=False,
            schema_valid=True,
            routing_ok=False,
            routing_method="fallback",
        ),
        _EvalLike(
            "bad_json",
            ok=False,
            schema_valid=False,
            routing_ok=False,
            routing_method=None,
        ),
    ]
    report = routing_report_from_results(results)
    assert report.total == 4
    assert report.passed == 2
    assert report.failures == 2
    # Three of four decisions were structurally valid (only bad_json was not).
    assert report.schema_valid_count == 3
    assert report.fallback_count == 1
    assert report.model_repair_count == 1
    assert [case.id for case in report.cases] == [
        "normal_chat",
        "repair_case",
        "fallback_case",
        "bad_json",
    ]
    fallback = next(case for case in report.cases if case.id == "fallback_case")
    assert fallback.ok is False
    assert fallback.schema_valid is True
    assert fallback.routing_method == "fallback"


def test_routing_report_per_case_is_redacted() -> None:
    # Per-case results must never carry prompt text, decision text, or paths — only
    # the fixture id, booleans, and the routing method enum string.
    results = [
        _EvalLike("c1", ok=True, schema_valid=True, routing_ok=True, routing_method="model"),
    ]
    serialized = routing_report_from_results(results).model_dump_json()
    for secret in ("/Users/", "Bearer ", "sk-", "ignore previous instructions"):
        assert secret not in serialized


def test_report_all_pass() -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=_passing_real_model(),
        routing=RoutingReport(total=38, passed=38, accuracy=1.0),
        skipped=[],
        checks_passed=12,
        checks_failed=0,
    )
    assert report.summary == "pass"
    assert report.threshold_failures == []
    assert report.real_model.attempted is True


def test_report_no_real_model_is_degraded() -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=RealModelReport(attempted=False),
        routing=RoutingReport(total=38, passed=38, accuracy=1.0),
        skipped=[SkippedCheck(name="model load", reason="No readable GGUF.")],
        checks_passed=6,
        checks_failed=0,
    )
    # A run that never exercised a real model can never be a pass.
    assert report.summary == "degraded"
    assert report.real_model.attempted is False
    assert any(item.name == "model load" for item in report.skipped)


def test_report_threshold_failure_degrades_but_does_not_fail() -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=_passing_real_model(),
        routing=None,
        skipped=[],
        checks_passed=12,
        checks_failed=0,
        thresholds=ReportThresholds(min_tokens_per_second=100.0),
    )
    assert report.summary == "degraded"
    assert any("tokens_per_second" in failure for failure in report.threshold_failures)


def test_report_structural_failure_is_fail() -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=_passing_real_model(),
        routing=None,
        skipped=[],
        checks_passed=10,
        checks_failed=2,
    )
    assert report.summary == "fail"


def test_report_require_real_model_without_model_fails() -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=RealModelReport(attempted=False),
        routing=None,
        skipped=[SkippedCheck(name="model load", reason="required but missing")],
        checks_passed=2,
        checks_failed=0,
        require_real_model=True,
    )
    assert report.summary == "fail"


def test_report_is_redacted_by_construction() -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=_passing_real_model(),
        routing=RoutingReport(total=1, passed=1, accuracy=1.0),
        skipped=[],
        checks_passed=12,
        checks_failed=0,
    )
    data = json.loads(report.model_dump_json())
    # Allowlist of fields: no field outside this set can ever leak (e.g. prompt
    # text, generated content, raw tokens, secrets, or absolute paths).
    assert set(data) == {
        "schema_version",
        "generated_at",
        "os",
        "cpu_architecture",
        "python_version",
        "runtime_backend",
        "real_model",
        "routing",
        "thresholds",
        "threshold_failures",
        "skipped",
        "checks_passed",
        "checks_failed",
        "summary",
    }
    assert set(data["real_model"]) == {
        "attempted",
        "model_id",
        "role",
        "path_basename",
        "quantization",
        "context_size",
        "load_success",
        "load_duration_seconds",
        "chat_success",
        "structured_brain_json_success",
        "streaming_success",
        "first_token_latency_seconds",
        "unload_success",
        "output_token_count",
        "tokens_per_second",
        "process_rss_bytes",
        "process_peak_rss_bytes",
    }
    serialized = report.model_dump_json()
    for secret in ("local-dev-token", "local-dev-runtime-token", "sk-", "Bearer ", "/Users/"):
        assert secret not in serialized
    # Only the basename is exposed, never a directory path.
    assert "/" not in (report.real_model.path_basename or "")


def test_environment_snapshot_is_populated() -> None:
    env = environment_snapshot()
    assert env.generated_at
    assert env.cpu_architecture
    assert env.python_version


def test_write_report_creates_file(tmp_path: Path) -> None:
    report = build_mac_report(
        environment=ENV,
        runtime_backend="llama_cpp",
        real_model=RealModelReport(attempted=False),
        routing=None,
        skipped=[],
        checks_passed=0,
        checks_failed=0,
    )
    out = tmp_path / "nested" / "mac-report.json"
    written = write_report(report, out)
    assert written.exists()
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["summary"] == "degraded"
    assert loaded["schema_version"] == 1


# --- TargetMacValidator integration (no real model required) ---------------


def test_target_mac_build_report_no_model_skips_real_checks(monkeypatch) -> None:
    monkeypatch.setattr("apps.runner.verify.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verify.platform.machine", lambda: "arm64")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    monkeypatch.setattr(
        "apps.runner.verify.query_audio_devices",
        lambda: {
            "sounddevice_installed": False,
            "input_devices": [],
            "output_devices": [],
            "error": "missing",
        },
    )
    monkeypatch.delenv("APRIL_TEST_GGUF_PATH", raising=False)
    validator = TargetMacValidator(
        home=Path.cwd(),
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=5.0,
    )
    validator.run()
    report = validator.build_report()
    assert report.real_model.attempted is False
    assert report.summary == "degraded"
    assert report.runtime_backend  # configured backend reported, not a fake fallback
    # Real-model checks are clearly skipped with reasons rather than faked.
    assert any(item.name == "model load" for item in report.skipped)
    # The deterministic routing evaluation still runs and is reported.
    assert report.routing is not None
    assert report.routing.total > 0


def test_target_mac_report_does_not_claim_real_when_simulated(monkeypatch) -> None:
    # Even with require_real_model unset, a skipped real model must not produce a
    # "pass" summary; honesty requires "degraded".
    monkeypatch.setattr("apps.runner.verify.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verify.platform.machine", lambda: "arm64")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    monkeypatch.setattr(
        "apps.runner.verify.query_audio_devices",
        lambda: {"sounddevice_installed": False, "input_devices": [], "output_devices": []},
    )
    monkeypatch.delenv("APRIL_TEST_GGUF_PATH", raising=False)
    validator = TargetMacValidator(
        home=Path.cwd(),
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=5.0,
    )
    validator.run()
    report = validator.build_report()
    assert report.summary != "pass"


# --- voice opt-in semantics for target-Mac verification --------------------


def _voice_components(
    *, whisper: str = "degraded", piper: str = "degraded", wake: str = "degraded"
) -> dict:
    """A scripted ``voice_doctor`` payload so target-Mac tests do not depend on
    whether voice artifacts happen to exist on the developer machine."""
    return {
        "components": [
            {"name": "whisper binary", "status": whisper, "message": "whisper binary"},
            {"name": "whisper model", "status": whisper, "message": "whisper model"},
            {"name": "piper binary", "status": piper, "message": "piper binary"},
            {"name": "piper model", "status": piper, "message": "piper model"},
            {"name": "wake-word model", "status": wake, "message": "wake-word model"},
        ]
    }


def _patch_target_mac(monkeypatch, *, voice_components: dict, devices_present: bool) -> None:
    monkeypatch.setattr("apps.runner.verify.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.verify.platform.machine", lambda: "arm64")
    monkeypatch.setattr("apps.runner.verify._llama_cpp_installed", lambda: False)
    devices = (
        {
            "sounddevice_installed": True,
            "input_devices": [{"index": 0, "name": "Mic"}],
            "output_devices": [{"index": 1, "name": "Speaker"}],
        }
        if devices_present
        else {
            "sounddevice_installed": False,
            "input_devices": [],
            "output_devices": [],
            "error": "missing",
        }
    )
    monkeypatch.setattr("apps.runner.verify.query_audio_devices", lambda: devices)
    monkeypatch.setattr("apps.runner.verify.voice_doctor", lambda settings: voice_components)
    monkeypatch.delenv("APRIL_TEST_GGUF_PATH", raising=False)


def test_target_mac_default_config_voice_disabled_skips_voice(monkeypatch) -> None:
    # On a clean target Mac with no voice artifacts present, the shipped default
    # (voice off) must SKIP the voice checks, never fail them. With no real model
    # the report is degraded — not fail — and no `run april setup voice` is forced.
    monkeypatch.delenv("APRIL_VOICE_ENABLED", raising=False)
    _patch_target_mac(monkeypatch, voice_components=_voice_components(), devices_present=False)
    validator = TargetMacValidator(
        home=Path.cwd(),
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=5.0,
    )
    validator.run()
    report = validator.build_report()
    assert report.real_model.attempted is False
    assert report.checks_failed == 0
    assert report.summary == "degraded"
    assert all(check.status != "fail" for check in validator.checks)
    skipped = {item.name for item in report.skipped}
    assert "whisper.cpp executable availability" in skipped
    assert "Piper voice availability" in skipped


def test_target_mac_voice_enabled_missing_artifacts_block(monkeypatch) -> None:
    # With voice explicitly enabled, missing whisper.cpp / Piper artifacts are
    # hard failures, so the report fails honestly rather than hiding the gap.
    monkeypatch.setenv("APRIL_VOICE_ENABLED", "true")
    _patch_target_mac(monkeypatch, voice_components=_voice_components(), devices_present=True)
    validator = TargetMacValidator(
        home=Path.cwd(),
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=5.0,
    )
    validator.run()
    report = validator.build_report()
    failed = {check.name for check in validator.checks if check.status == "fail"}
    assert "whisper.cpp executable availability" in failed
    assert "whisper.cpp model availability" in failed
    assert "Piper executable availability" in failed
    assert "Piper voice availability" in failed
    # A missing wake-word model stays a manual path, never a hard failure.
    assert "wake-word model availability" not in failed
    assert report.summary == "fail"


def test_target_mac_voice_enabled_missing_wake_word_only_is_manual(monkeypatch) -> None:
    # Push-to-talk works without a wake-word model: with whisper/Piper present and
    # only the wake-word model missing, voice contributes no hard failure — the
    # wake-word check is a manual/optional path, and the run stays degraded.
    monkeypatch.setenv("APRIL_VOICE_ENABLED", "true")
    _patch_target_mac(
        monkeypatch,
        voice_components=_voice_components(whisper="ok", piper="ok", wake="degraded"),
        devices_present=True,
    )
    validator = TargetMacValidator(
        home=Path.cwd(),
        model_path=None,
        require_real_model=False,
        max_output_tokens=8,
        timeout=5.0,
    )
    validator.run()
    report = validator.build_report()
    assert {check.name for check in validator.checks if check.status == "fail"} == set()
    wake = next(c for c in validator.checks if c.name == "wake-word model availability")
    assert wake.status == "manual"
    assert report.summary == "degraded"
