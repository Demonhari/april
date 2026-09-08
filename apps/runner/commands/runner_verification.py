from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.acceptance import (
    AcceptanceReport,
)
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.mac_report import ReportThresholds, write_report
from apps.runner.multi_model_report import (
    write_multi_model_report,
    write_routing_only_report,
)
from apps.runner.soak import write_soak_report
from apps.runner.verify import (
    VerifyCheck,
    build_workflow_report,
    run_local_security_integrity_verification,
    write_workflow_report,
)
from apps.runner.voice_live import VoiceLiveReport
from apps.runner.wake_live import WakeWordLiveReport, run_sentinel_live_verification
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.settings import load_settings

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


_ACCEPTANCE_STATUS_STYLE = {
    "pass": "[green]PASS[/green]",
    "warning": "[yellow]WARNING[/yellow]",
    "fail": "[red]FAIL[/red]",
}


@_registry.april_app.command()
def verify(
    model_path: Path | None = typer.Argument(None),
    fake: bool = typer.Option(False, "--fake", help="Run deterministic fake-backend verification."),
    development_unsandboxed_override: bool = typer.Option(
        False,
        "--development-unsandboxed-override",
        help=(
            "Development-only Linux/CI verification escape hatch; explicitly permits "
            "restricted subprocesses without OS-enforced isolation."
        ),
    ),
    real_model: bool = typer.Option(False, "--real-model"),
    workflow: bool = typer.Option(False, "--workflow"),
    target_mac: bool = typer.Option(False, "--target-mac"),
    all_configured_models: bool = typer.Option(
        False,
        "--all-configured-models",
        "--mac-readiness",
        help="Verify every configured real GGUF model (load/chat/stream/unload + switching).",
    ),
    routing_only: bool = typer.Option(
        False,
        "--routing-only",
        help="Run isolated real Brain routing diagnostics without specialist or tool workflows.",
    ),
    soak: bool = typer.Option(False, "--soak", help="Run a bounded fake-backend soak check."),
    minutes: float = typer.Option(10.0, "--minutes", min=0.01, max=240.0),
    soak_interval_seconds: float = typer.Option(
        1.0,
        "--soak-interval-seconds",
        min=0.1,
        max=60.0,
        help="Delay between fake soak iterations.",
    ),
    cycle_fake_models: bool = typer.Option(False, "--cycle-fake-models"),
    require_real_model: bool = typer.Option(False, "--require-real-model"),
    json_output: bool = typer.Option(False, "--json"),
    report: Path | None = typer.Option(
        None, "--report", help="Write a redacted machine-readable verification report JSON here."
    ),
    min_tokens_per_second: float | None = typer.Option(None, "--min-tokens-per-second", min=0.0),
    max_load_seconds: float | None = typer.Option(None, "--max-load-seconds", min=0.0),
    max_first_token_latency_seconds: float | None = typer.Option(
        None, "--max-first-token-latency-seconds", min=0.0
    ),
    max_rss_mb: float | None = typer.Option(None, "--max-rss-mb", min=0.0),
    min_routing_accuracy: float = typer.Option(0.90, "--min-routing-accuracy", min=0.0, max=1.0),
    min_model_only_routing_accuracy: float = typer.Option(
        0.75, "--min-model-only-routing-accuracy", min=0.0, max=1.0
    ),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    timeout: float = typer.Option(180.0, "--timeout", min=1.0),
    candidate_adapter_model_id: str | None = typer.Option(
        None,
        "--candidate-adapter-model-id",
        help="Temporarily verify this model with --candidate-adapter-path.",
    ),
    candidate_adapter_path: Path | None = typer.Option(
        None,
        "--candidate-adapter-path",
        help="Temporary LoRA adapter path for all-configured-model verification.",
    ),
) -> None:
    thresholds = ReportThresholds(
        min_tokens_per_second=min_tokens_per_second,
        max_load_seconds=max_load_seconds,
        max_first_token_latency_seconds=max_first_token_latency_seconds,
        max_rss_mb=max_rss_mb,
        min_routing_accuracy=min_routing_accuracy,
        min_model_only_routing_accuracy=min_model_only_routing_accuracy,
    )
    if development_unsandboxed_override:
        manager = _composition_api._manager()
        manager_settings = getattr(manager, "settings", None)
        environment = getattr(manager_settings, "environment", None)
        if environment is None:
            try:
                environment = load_settings(root=manager.home).environment
            except Exception:
                environment = None
        if environment != "development":
            console.print(
                "[red]--development-unsandboxed-override is valid only in development "
                "with a readable configuration.[/red]"
            )
            raise typer.Exit(1)
    candidate_adapter_requested = (
        candidate_adapter_model_id is not None or candidate_adapter_path is not None
    )
    if candidate_adapter_requested:
        if not all_configured_models:
            console.print(
                "[red]Candidate adapter verification is only supported with "
                "--all-configured-models.[/red]"
            )
            raise typer.Exit(1)
        if candidate_adapter_model_id is None or candidate_adapter_path is None:
            console.print(
                "[red]Use both --candidate-adapter-model-id and --candidate-adapter-path.[/red]"
            )
            raise typer.Exit(1)
    if soak:
        soak_report = _composition_api.run_fake_soak(
            _composition_api._manager().home,
            minutes=minutes,
            interval_seconds=soak_interval_seconds,
            cycle_models=cycle_fake_models,
        )
        checks = [
            VerifyCheck(
                name="fake soak",
                ok=soak_report.summary == "pass",
                detail=f"iterations={soak_report.iterations}, failures={len(soak_report.failures)}",
            )
        ]
        if json_output:
            console.print_json(data=soak_report.model_dump())
        else:
            _composition_api._print_verification_table("APRIL Fake Soak Verification", checks)
        if report is not None:
            written = write_soak_report(soak_report, report)
            console.print(
                f"[green]Wrote fake soak report to {written}[/green] "
                f"(summary: {soak_report.summary}, real_model_verified: false)"
            )
        if soak_report.summary != "pass":
            raise typer.Exit(1)
        raise typer.Exit(0)
    if routing_only:
        routing_report = _composition_api.run_routing_only_verification(
            _composition_api._manager().home,
            max_output_tokens=max(max_output_tokens, 192),
            timeout=timeout,
            thresholds=thresholds,
        )
        if json_output:
            console.print_json(data=routing_report.model_dump())
        else:
            console.print(
                "Isolated routing-only validation is diagnostic evidence, not full readiness."
            )
            console.print_json(data=routing_report.model_dump())
        if report is not None:
            written = write_routing_only_report(routing_report, report)
            console.print(f"[green]Wrote routing-only report to {written}[/green]")
        if routing_report.summary != "pass":
            raise typer.Exit(1)
        raise typer.Exit(0)
    if all_configured_models:
        verifier = _composition_api.run_all_configured_models_verification(
            _composition_api._manager().home,
            require_real_model=require_real_model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            thresholds=thresholds,
            candidate_adapter_model_id=candidate_adapter_model_id,
            candidate_adapter_path=candidate_adapter_path,
            routing_evaluation=require_real_model,
        )
        checks = verifier.checks
        multi_report = verifier.build_report()
        if json_output:
            console.print_json(data=multi_report.model_dump())
        else:
            _composition_api._print_verification_table(
                "APRIL All-Configured-Model Verification", checks
            )
            _print_routing_summary(multi_report)
        if report is not None:
            written = write_multi_model_report(multi_report, report)
            console.print(
                f"[green]Wrote multi-model verification report to {written}[/green] "
                f"(summary: {multi_report.summary}, "
                f"verification_level: {multi_report.verification_level})"
            )
        if not all(check.ok for check in checks) or multi_report.summary == "fail":
            raise typer.Exit(1)
        raise typer.Exit(0)
    if target_mac:
        validator = _composition_api.TargetMacValidator(
            home=_composition_api._manager().home,
            model_path=model_path,
            require_real_model=require_real_model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        checks = validator.run()
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _composition_api._print_verification_table("APRIL Target Mac Validation", checks)
        if report is not None:
            rendered = validator.build_report(thresholds=thresholds)
            written = write_report(rendered, report)
            console.print(
                f"[green]Wrote verification report to {written}[/green] "
                f"(summary: {rendered.summary})"
            )
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if workflow:
        checks = _composition_api.run_workflow_verification(
            _composition_api._manager().home,
            real_model=real_model,
            model_path=model_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _composition_api._print_verification_table("APRIL Workflow Verification", checks)
        if report is not None:
            workflow_report = build_workflow_report(
                checks,
                real_model_requested=real_model,
                timeout_seconds=timeout,
                max_output_tokens=max_output_tokens,
                config_fingerprint=config_fingerprint_digest(_composition_api._manager().home),
            )
            written = write_workflow_report(workflow_report, report)
            console.print(
                f"[green]Wrote workflow verification report to {written}[/green] "
                f"(summary: {workflow_report.summary}, "
                f"real_model_verified: {str(workflow_report.real_model_verified).lower()})"
            )
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if real_model:
        configured_path = model_path or (
            Path(os.environ["APRIL_TEST_GGUF_PATH"])
            if os.environ.get("APRIL_TEST_GGUF_PATH")
            else None
        )
        if configured_path is None:
            console.print(
                "[yellow]Skipping real-model verification: no GGUF path provided.[/yellow]"
            )
            raise typer.Exit(0)
        if not configured_path.expanduser().exists():
            console.print(f"[red]GGUF path does not exist: {configured_path}[/red]")
            raise typer.Exit(1)
        checks = _composition_api.run_real_model_verification(
            _composition_api._manager().home,
            configured_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _composition_api._print_verification_table("APRIL Real Model Verification", checks)
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if not fake:
        checks = run_local_security_integrity_verification(_composition_api._manager().home)
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _composition_api._print_verification_table("APRIL Local Security and Integrity", checks)
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if development_unsandboxed_override:
        checks = _composition_api.run_fake_verification(
            _composition_api._manager().home,
            development_unsandboxed_override=True,
        )
    else:
        checks = _composition_api.run_fake_verification(_composition_api._manager().home)
    if json_output:
        console.print_json(
            data={
                "checks": [asdict(check) for check in checks],
                "evidence_scope": {
                    "audit_home": "isolated_temporary_april_home",
                    "actual_home_audit_verified": False,
                    "model_evidence": "fake_only",
                    "sandbox_capability": "current_host_capability",
                },
            }
        )
    else:
        console.print(
            "Fake verification uses an isolated temporary APRIL home; its audit check "
            "does not verify or repair this installation's audit chain."
        )
        console.print(
            "Fake-model results are plumbing/security evidence, not real-model quality "
            "or performance evidence. Sandbox checks report this host's capabilities."
        )
        _composition_api._print_verification_table("APRIL Verification", checks)
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


def _print_routing_summary(report: object) -> None:
    """Render bounded Brain-routing counters and reason codes."""
    table = Table(title="Brain routing evaluation")
    table.add_column("Metric")
    table.add_column("Value")
    brain = next(
        (
            model
            for model in getattr(report, "models", [])
            if getattr(model, "role", None) == "brain"
        ),
        None,
    )
    routing = getattr(brain, "routing", None)
    model_only = getattr(brain, "model_only_routing", None)
    end_to_end_categories = _routing_failure_categories(routing)
    model_only_categories = _routing_failure_categories(model_only)
    rows = {
        "end-to-end cases": _routing_counts(routing),
        "end-to-end semantic intent": _routing_semantic_counts(routing),
        "end-to-end schema-valid": getattr(routing, "schema_valid_count", 0) if routing else 0,
        "end-to-end deterministic/model/repair/fallback": _routing_provenance_counts(routing),
        "end-to-end failure categories": end_to_end_categories or "none",
        "end-to-end stage codes": _routing_stage_codes(routing),
        "end-to-end coercions": _routing_counter(routing, "coercions"),
        "end-to-end contract rejections": _routing_counter(routing, "first_rejection_code"),
        "end-to-end inference failures": getattr(routing, "inference_failed_count", 0)
        if routing
        else 0,
        "end-to-end downstream errors": _routing_counter(routing, "downstream_error_code"),
        "end-to-end downstream runtime errors": _routing_counter(
            routing, "downstream_runtime_error_code"
        ),
        "model-only cases": _routing_counts(model_only),
        "model-only semantic intent": _routing_semantic_counts(model_only),
        "model-only schema-valid": getattr(model_only, "schema_valid_count", 0)
        if model_only
        else 0,
        "model-only deterministic/model/repair/fallback": _routing_provenance_counts(model_only),
        "model-only failure categories": model_only_categories or "none",
        "model-only stage codes": _routing_stage_codes(model_only),
        "model-only coercions": _routing_counter(model_only, "coercions"),
        "model-only contract rejections": _routing_counter(model_only, "first_rejection_code"),
        "model-only inference failures": getattr(model_only, "inference_failed_count", 0)
        if model_only
        else 0,
        "model-only downstream errors": _routing_counter(model_only, "downstream_error_code"),
        "model-only downstream runtime errors": _routing_counter(
            model_only, "downstream_runtime_error_code"
        ),
        "runtime process": _runtime_process_summary(report),
        "preserved logs": getattr(report, "log_directory_basename", None) or "none",
        "threshold failures": ", ".join(getattr(report, "threshold_failures", [])) or "none",
        "routing reason": getattr(brain, "routing_error_code", None) or "none",
    }
    for key, value in rows.items():
        table.add_row(key, str(value))
    console.print(table)


def _runtime_process_summary(report: object) -> str:
    process = getattr(report, "runtime_process", None)
    if not isinstance(process, dict):
        return "not recorded"
    runtime = process.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("alive") is not False:
        return str(process)
    if getattr(report, "runtime_error", None) is False:
        return "stopped by verifier (SIGTERM)"
    return str(process)


def _routing_counts(report: object) -> str:
    if report is None:
        return "0/0 (0.00)"
    passed = getattr(report, "passed", 0)
    total = getattr(report, "total", 0)
    accuracy = getattr(report, "accuracy", 0.0)
    return f"{passed}/{total} ({accuracy:.2f})"


def _routing_provenance_counts(report: object) -> str:
    if report is None:
        return "deterministic=0 model=0 repair=0 fallback=0 unknown=0"
    return (
        f"deterministic={getattr(report, 'deterministic_count', 0)} "
        f"model={getattr(report, 'model_count', 0)} "
        f"repair={getattr(report, 'model_repair_count', 0)} "
        f"fallback={getattr(report, 'fallback_count', 0)} "
        f"unknown={getattr(report, 'unknown_provenance_count', 0)}"
    )


def _routing_semantic_counts(report: object) -> str:
    if report is None:
        return "0/0 (0.00)"
    return (
        f"{getattr(report, 'semantic_passed', 0)}/{getattr(report, 'total', 0)} "
        f"({getattr(report, 'semantic_accuracy', 0.0):.2f}); "
        f"normalized={getattr(report, 'passed', 0)}/{getattr(report, 'total', 0)}"
    )


def _routing_failure_categories(report: object) -> str:
    if report is None:
        return ""
    from collections import Counter

    counts: Counter[str] = Counter()
    for case in getattr(report, "cases", []):
        for code in getattr(case, "mismatch_codes", []):
            counts[str(code)] += 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _routing_stage_codes(report: object) -> str:
    if report is None:
        return "none"
    from collections import Counter

    counts: Counter[str] = Counter(
        str(stage)
        for case in getattr(report, "cases", [])
        if (stage := getattr(case, "stage_code", None))
    )
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _routing_counter(report: object, field: str) -> str:
    if report is None:
        return "none"
    from collections import Counter

    counts: Counter[str] = Counter()
    for case in getattr(report, "cases", []):
        values = getattr(case, field, None)
        if field == "coercions":
            values = values or []
            for value in values:
                if isinstance(value, str):
                    counts[value] += 1
        elif isinstance(values, str) and values:
            counts[values] += 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _voice_live_runner(settings: Any) -> Callable[[], VoiceLiveReport]:
    def run() -> VoiceLiveReport:
        doctor = _composition_api.collect_voice_doctor(settings)
        console.print(f"Voice doctor status: {doctor['status']}")
        guidance = doctor.get("macos_microphone_permission_guidance")
        if guidance:
            console.print(str(guidance))

        def confirm(message: str) -> bool:
            return typer.confirm(message, default=False)

        return asyncio.run(
            _composition_api.run_voice_live_verification(
                settings=settings,
                confirm_recording=confirm,
                confirm_transcription=confirm,
                confirm_playback=confirm,
            )
        )

    return run


def _wake_word_live_runner(settings: Any) -> Callable[[], WakeWordLiveReport]:
    def run() -> WakeWordLiveReport:
        doctor = _composition_api.collect_voice_doctor(settings)
        console.print(f"Voice doctor status: {doctor['status']}")
        for key in ("macos_microphone_permission_guidance", "wake_word_guidance"):
            guidance = doctor.get(key)
            if guidance:
                console.print(str(guidance))

        def confirm(message: str) -> bool:
            return typer.confirm(message, default=False)

        return asyncio.run(
            _composition_api.run_wake_word_live_verification(
                settings=settings,
                confirm_microphone=confirm,
            )
        )

    return run


def _print_acceptance(report: AcceptanceReport) -> None:
    status = _ACCEPTANCE_STATUS_STYLE.get(report.final_status, report.final_status)
    env = report.environment
    console.print(
        f"APRIL acceptance — {status} "
        f"(level={report.acceptance_level}, backend={report.runtime_backend}, "
        f"env={env.deployment}, arch={env.cpu_architecture})"
    )
    table = Table(title="Acceptance gates")
    table.add_column("Gate")
    table.add_column("Result")
    table.add_column("Detail")
    table.add_row(
        "configuration",
        "valid" if report.config_valid else "invalid",
        "ok" if report.config_valid else "; ".join(report.config_errors) or "invalid",
    )
    fake = report.fake_verification
    table.add_row(
        "fake verification",
        fake.summary,
        (
            f"{fake.checks_passed}/{fake.checks_total} checks passed"
            if fake.ran
            else "skipped (configuration invalid)"
        ),
    )
    readiness = report.readiness
    table.add_row(
        "readiness preflight",
        "ready" if readiness.real_model_preflight_ready else "not ready",
        f"{len(readiness.blockers)} blocker(s), {len(readiness.warnings)} warning(s)",
    )
    if report.real_model_verification is not None:
        real = report.real_model_verification
        table.add_row(
            "real models",
            real.summary,
            f"level={real.verification_level}, {real.models_passed}/{real.models_attempted} passed",
        )
    if report.voice_live is not None:
        voice = report.voice_live
        table.add_row(
            "voice (push-to-talk)",
            voice.summary,
            f"recording={voice.recording_success}, stt={voice.stt_success}, "
            f"playback_confirmed={voice.playback_user_confirmed}",
        )
    if report.wake_word_live is not None:
        wake = report.wake_word_live
        table.add_row(
            "voice (wake word)",
            wake.summary,
            f"wake_detected={wake.wake_word_detected}, stt={wake.stt_success}, "
            f"api={wake.api_success}, playback_confirmed={wake.playback_user_confirmed}",
        )
    services = report.services
    if services.requested:
        table.add_row(
            "services",
            services.mode,
            f"startup={services.startup_status}, shutdown={services.shutdown_status}, "
            f"api={services.api_reachable}, runtime={services.runtime_reachable}",
        )
    console.print(table)
    if report.next_actions:
        console.print("[bold]Next actions:[/bold]")
        for action in report.next_actions:
            # markup=False so command tokens like '.[runtime]' are not parsed as tags.
            console.print(f"  {action}", markup=False)
