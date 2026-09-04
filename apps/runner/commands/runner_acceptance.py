from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.acceptance import (
    AcceptanceFlagError,
    AcceptanceReport,
    FakeVerificationSummary,
    ServicesSummary,
    default_acceptance_report_path,
    validate_acceptance_flags,
    write_acceptance_report,
)
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.go_live import (
    GoLiveReport,
    build_go_live_report,
    default_go_live_report_path,
    write_go_live_report,
)
from apps.runner.mac_report import ReportThresholds, redact_reason
from apps.runner.multi_model_report import (
    MultiModelVerificationReport,
)
from apps.runner.preflight import PreflightReport
from apps.runner.service_manager import AprilServiceManager
from apps.runner.voice_live import VoiceLiveReport
from apps.runner.wake_live import WakeWordLiveReport, run_sentinel_live_verification
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.errors import ConfigError
from april_common.settings import load_settings

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


_GO_LIVE_STATUS_STYLE = {
    "pass": "[green]PASS[/green]",
    "warning": "[yellow]WARNING[/yellow]",
    "fail": "[red]FAIL[/red]",
}

_PREFLIGHT_STATUS_STYLE = {
    "pass": "[green]pass[/green]",
    "warning": "[yellow]warning[/yellow]",
    "fail": "[red]fail[/red]",
}


def _run_acceptance_with_services(
    *,
    manager: AprilServiceManager,
    require_real_models: bool,
    allow_sanity_pass: bool,
    start_services: bool,
    fake_services: bool,
    keep_services_running: bool,
    service_timeout: float,
    max_output_tokens: int,
    timeout: float,
    thresholds: ReportThresholds,
    voice_live_runner: Callable[[], VoiceLiveReport] | None,
    wake_word_live_runner: Callable[[], WakeWordLiveReport] | None,
) -> AcceptanceReport:
    """Orchestrate APRIL services around an acceptance run and record the lifecycle.

    Services are only touched when ``start_services`` is set. Services that
    acceptance itself started are always stopped at the end (success, failure,
    timeout, cancellation, or KeyboardInterrupt) unless ``keep_services_running``.
    The redacted lifecycle is embedded in the returned report's ``services``.
    """
    services = ServicesSummary(
        requested=start_services,
        mode=("fake" if fake_services else "real") if start_services else "none",
    )
    started_by_acceptance = False
    try:
        if start_services:
            manager.startup_timeout_seconds = service_timeout
            before = manager.status()
            if before.ok:
                services.startup_status = "already_running"
            else:
                status = manager.start(fake_backend=fake_services)
                started_by_acceptance = True
                services.started_by_acceptance = True
                services.startup_status = "ok" if status.ok else "failed"
            current = manager.status()
            services.api_reachable = current.api.healthy
            services.runtime_reachable = current.runtime.healthy
        report = _composition_api.run_acceptance(
            manager.home,
            require_real_models=require_real_models,
            allow_sanity_pass=allow_sanity_pass,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            thresholds=thresholds,
            services=services,
            voice_live_runner=voice_live_runner,
            wake_word_live_runner=wake_word_live_runner,
        )
    except BaseException:
        # Always release services acceptance started, even on timeout/cancel/Ctrl-C.
        if started_by_acceptance and not keep_services_running:
            with contextlib.suppress(Exception):
                manager.stop()
        raise

    # Mutate report.services (not the local) so the lifecycle is recorded regardless
    # of how pydantic stored the embedded summary.
    if started_by_acceptance and not keep_services_running:
        try:
            manager.stop()
            report.services.stopped_after_acceptance = True
            report.services.shutdown_status = "stopped"
        except Exception:
            report.services.shutdown_status = "failed"
    elif started_by_acceptance:
        report.services.shutdown_status = "kept_running"
    else:
        report.services.shutdown_status = "not_applicable"
    return report


@_registry.april_app.command()
def acceptance(
    require_real_models: bool = typer.Option(
        False,
        "--require-real-models",
        help="Fail unless every configured chat GGUF model loads, chats, streams, and unloads.",
    ),
    allow_sanity_pass: bool = typer.Option(
        False,
        "--allow-sanity-pass",
        help="Allow a clean fake-only run to report pass (otherwise fake-only is a warning).",
    ),
    voice_live: bool = typer.Option(
        False, "--voice-live", help="Also run interactive push-to-talk voice verification."
    ),
    wake_word_live: bool = typer.Option(
        False, "--wake-word-live", help="Also run interactive live wake-word verification."
    ),
    start_services: bool = typer.Option(
        False,
        "--start-services",
        help="Start missing APRIL services before live voice/wake-word checks.",
    ),
    fake_services: bool = typer.Option(
        False,
        "--fake-services",
        help="With --start-services, start services with the fake runtime (plumbing only).",
    ),
    keep_services_running: bool = typer.Option(
        False,
        "--keep-services-running",
        help="Leave services that acceptance started running afterwards.",
    ),
    service_timeout: float = typer.Option(20.0, "--service-timeout", min=1.0),
    write_report: bool = typer.Option(
        False,
        "--write-report",
        help="Write a redacted report to data/verification/acceptance-<timestamp>.json.",
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Write the redacted acceptance report JSON to this path instead."
    ),
    json_output: bool = typer.Option(False, "--json"),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    timeout: float = typer.Option(180.0, "--timeout", min=1.0),
    min_tokens_per_second: float | None = typer.Option(None, "--min-tokens-per-second", min=0.0),
    max_load_seconds: float | None = typer.Option(None, "--max-load-seconds", min=0.0),
    max_first_token_latency_seconds: float | None = typer.Option(
        None, "--max-first-token-latency-seconds", min=0.0
    ),
    max_rss_mb: float | None = typer.Option(None, "--max-rss-mb", min=0.0),
) -> None:
    """Single MacBook Pro acceptance gate: config + readiness + fake (+ real/voice).

    ``run april acceptance`` is fake/local sanity only and reports a warning unless
    ``--require-real-models`` is supplied (or ``--allow-sanity-pass`` for a clean
    fake-only pass). Add ``--start-services`` so live voice/wake-word checks can
    reach the Core API; ``--fake-services`` uses the fake runtime for plumbing only.
    """
    try:
        validate_acceptance_flags(
            require_real_models=require_real_models,
            start_services=start_services,
            fake_services=fake_services,
        )
    except AcceptanceFlagError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    manager = _composition_api._manager()
    settings = manager.settings
    thresholds = ReportThresholds(
        min_tokens_per_second=min_tokens_per_second,
        max_load_seconds=max_load_seconds,
        max_first_token_latency_seconds=max_first_token_latency_seconds,
        max_rss_mb=max_rss_mb,
    )
    report_obj = _composition_api._run_acceptance_with_services(
        manager=manager,
        require_real_models=require_real_models,
        allow_sanity_pass=allow_sanity_pass,
        start_services=start_services,
        fake_services=fake_services,
        keep_services_running=keep_services_running,
        service_timeout=service_timeout,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        thresholds=thresholds,
        voice_live_runner=_composition_api._voice_live_runner(settings) if voice_live else None,
        wake_word_live_runner=_composition_api._wake_word_live_runner(settings)
        if wake_word_live
        else None,
    )

    target = report
    if target is None and write_report:
        target = default_acceptance_report_path(manager.home)

    if json_output:
        console.print_json(data=report_obj.model_dump())
    else:
        _composition_api._print_acceptance(report_obj)

    if target is not None:
        written = write_acceptance_report(report_obj, target)
        console.print(
            f"[green]Wrote acceptance report to {written}[/green] "
            f"(final_status: {report_obj.final_status})"
        )

    if report_obj.final_status == "fail":
        raise typer.Exit(1)


def _go_live_fake_summary(home: Path, *, config_valid: bool) -> FakeVerificationSummary:
    """Deterministic fake-backend sanity, summarized and redacted.

    Mirrors the acceptance fake summary but keeps ``run_fake_verification`` as a
    main-module patch point so the go-live CLI is testable without real services.
    """
    if not config_valid:
        return FakeVerificationSummary(ran=False, summary="skipped")
    checks = _composition_api.run_fake_verification(home)
    total = len(checks)
    passed = sum(1 for check in checks if check.ok)
    failed = total - passed
    failures = [f"{check.name}: {redact_reason(check.detail)}" for check in checks if not check.ok]
    return FakeVerificationSummary(
        ran=True,
        checks_total=total,
        checks_passed=passed,
        checks_failed=failed,
        failures=failures,
        summary="pass" if failed == 0 else "fail",
    )


def _collect_go_live_report(
    home: Path,
    *,
    services: ServicesSummary,
    max_output_tokens: int,
    timeout: float,
    thresholds: ReportThresholds,
) -> GoLiveReport:
    """Compose the real-model go-live proof from existing verification primitives.

    Read-only: validates config, builds offline readiness, runs deterministic fake
    sanity, and — only when the local preflight can actually support it — spawns the
    all-configured real-model verifier (with strict-JSON routing evals enabled) which
    loads/chats/streams/unloads every configured GGUF and exercises specialist
    switching. Never downloads a model, installs a package, opens the microphone, or
    mutates configuration.
    """
    errors = list(_composition_api.validate_configuration(home))
    config_valid = not errors
    readiness = _composition_api.build_readiness_report(home)
    fake = _go_live_fake_summary(home, config_valid=config_valid)

    chat_models = [
        model
        for model in readiness.models
        if model.backend == "llama_cpp" and model.role != "embedding"
    ]
    preflight_ok = (
        config_valid
        and not readiness.runtime_is_fake
        and readiness.llama_cpp_python_available
        and bool(chat_models)
        and all(model.path_exists for model in chat_models)
    )

    multi_model: MultiModelVerificationReport | None = None
    if preflight_ok:
        verifier = _composition_api.run_all_configured_models_verification(
            home,
            require_real_model=True,
            routing_evaluation=True,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            thresholds=thresholds,
        )
        multi_model = verifier.build_report()

    try:
        settings = load_settings(root=home)
        embedding_provider = settings.memory.embedding_provider
        embedding_model_id = settings.memory.embedding_model_id
    except ConfigError:
        embedding_provider = "unknown"
        embedding_model_id = None

    return build_go_live_report(
        home=home,
        config_valid=config_valid,
        config_errors=errors,
        readiness=readiness,
        fake_verification=fake,
        real_requested=True,
        multi_model=multi_model,
        embedding_provider=embedding_provider,
        embedding_model_id=embedding_model_id,
        services=services,
        config_fingerprint=config_fingerprint_digest(home),
    )


def _run_go_live_with_services(
    *,
    manager: AprilServiceManager,
    start_services: bool,
    keep_services_running: bool,
    service_timeout: float,
    max_output_tokens: int,
    timeout: float,
    thresholds: ReportThresholds,
) -> GoLiveReport:
    """Orchestrate APRIL's main services around a go-live proof and record the
    lifecycle. Services are only touched with ``--start-services``; anything go-live
    started is always stopped at the end unless ``--keep-services-running``. The
    real-model verifier still runs in its own ephemeral runtime regardless — the main
    services here exist to additionally prove they start and stop cleanly on this Mac.
    """
    services = ServicesSummary(
        requested=start_services,
        mode="real" if start_services else "none",
    )
    started_by_go_live = False
    try:
        if start_services:
            manager.startup_timeout_seconds = service_timeout
            before = manager.status()
            if before.ok:
                services.startup_status = "already_running"
            else:
                status = manager.start(fake_backend=False)
                started_by_go_live = True
                services.started_by_go_live = True
                services.startup_status = "ok" if status.ok else "failed"
            current = manager.status()
            services.api_reachable = current.api.healthy
            services.runtime_reachable = current.runtime.healthy
        report = _collect_go_live_report(
            manager.home,
            services=services,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            thresholds=thresholds,
        )
    except BaseException:
        if started_by_go_live and not keep_services_running:
            with contextlib.suppress(Exception):
                manager.stop()
        raise

    if started_by_go_live and not keep_services_running:
        try:
            manager.stop()
            report.services.stopped_after_go_live = True
            report.services.shutdown_status = "stopped"
        except Exception:
            report.services.shutdown_status = "failed"
    elif started_by_go_live:
        report.services.shutdown_status = "kept_running"
    else:
        report.services.shutdown_status = "not_applicable"
    # The final-status gate must see the recorded service lifecycle: a requested
    # service that failed to stop cleanly is a hard fail. Rebuild over the same
    # primitives is avoided — only re-derive the final status from services.
    return _refinalize_go_live(report)


def _refinalize_go_live(report: GoLiveReport) -> GoLiveReport:
    """Re-evaluate the go-live final status after the service shutdown is recorded.

    ``_collect_go_live_report`` runs before services are stopped, so a shutdown
    failure recorded afterwards must still be able to fail the run.
    """
    if report.services.shutdown_status == "failed" and report.final_status != "fail":
        blocker = "services failed to start or stop cleanly"
        if blocker not in report.blockers:
            report.blockers.append(blocker)
            report.final.blockers.append(blocker)
        report.final_status = "fail"
        report.final.status = "fail"
    return report


def _print_go_live(report: GoLiveReport) -> None:
    status = _GO_LIVE_STATUS_STYLE.get(report.final_status, report.final_status)
    env = report.environment
    console.print(
        f"APRIL go-live proof — {status} "
        f"(level={report.acceptance_level}, backend={report.runtime_backend}, "
        f"env={env.deployment}, arch={env.cpu_architecture})"
    )
    # The headline distinction: a working real-model core is reported separately
    # from the hardened go-live rung so a hardening advisory never hides it.
    core_label = {"ready": "ready", "fail": "fail", "not_run": "not run"}.get(
        report.real_model_core_status, report.real_model_core_status
    )
    console.print(f"Real model core: {core_label}")
    if report.hardened_go_live_ready:
        console.print("Hardened go-live: ready")
    elif report.hardening_blockers:
        reason = "; ".join(report.hardening_blockers)
        console.print(f"Hardened go-live: blocked because {reason}", markup=False)
    elif report.hardening_warnings:
        reason = "; ".join(report.hardening_warnings)
        console.print(f"Hardened go-live: warning because {reason}", markup=False)
    else:
        console.print("Hardened go-live: not ready")
    table = Table(title="Go-live proof gates")
    table.add_column("Gate")
    table.add_column("Result")
    table.add_column("Detail")
    table.add_row(
        "real GGUF models present",
        f"{report.configured_chat_models_present_count}/{report.configured_chat_models_count}",
        "all configured chat models present"
        if report.configured_chat_models_count
        and report.configured_chat_models_present_count == report.configured_chat_models_count
        else "missing model file(s)",
    )
    acc = report.acceptance
    table.add_row(
        "real models load/chat/stream/unload",
        acc.summary,
        f"verified={acc.real_model_verified}, {acc.models_passed}/{acc.models_attempted} passed",
    )
    routing = report.routing
    if routing.passed_without_fallback:
        routing_result = "pass"
    elif not routing.report_exists:
        routing_result = "n/a"
    else:
        routing_result = "fail"
    table.add_row(
        "brain strict-JSON routing",
        routing_result,
        f"{routing.routing_cases_passed}/{routing.routing_cases_total} cases, "
        f"fallback={routing.routing_fallback_count}",
    )
    spec = report.specialist
    table.add_row(
        "specialist switching",
        ("verified" if spec.verified else "fail") if spec.applicable else "n/a",
        f"chat_models={spec.chat_models_count}, attempted={spec.attempted}",
    )
    services = report.services
    if services.requested:
        table.add_row(
            "services",
            services.mode,
            f"startup={services.startup_status}, shutdown={services.shutdown_status}",
        )
    console.print(table)
    if report.warnings:
        console.print("[bold]Warnings:[/bold]")
        for warning in report.warnings:
            console.print(f"  {warning}", markup=False)
    if report.next_actions:
        console.print("[bold]Next actions:[/bold]")
        for action in report.next_actions:
            # markup=False so command tokens like '.[runtime]' are not parsed as tags.
            console.print(f"  {action}", markup=False)


@_registry.april_app.command("go-live")
def go_live(
    start_services: bool = typer.Option(
        False,
        "--start-services",
        help="Also start/stop APRIL's main services to prove a clean service lifecycle.",
    ),
    keep_services_running: bool = typer.Option(
        False,
        "--keep-services-running",
        help="Leave services that go-live started running afterwards.",
    ),
    service_timeout: float = typer.Option(20.0, "--service-timeout", min=1.0),
    write_report: bool = typer.Option(
        False,
        "--write-report",
        help="Write a redacted report to data/verification/go-live-<timestamp>.json.",
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Write the redacted go-live report JSON to this path instead."
    ),
    json_output: bool = typer.Option(False, "--json"),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    timeout: float = typer.Option(180.0, "--timeout", min=1.0),
    min_tokens_per_second: float | None = typer.Option(None, "--min-tokens-per-second", min=0.0),
    max_load_seconds: float | None = typer.Option(None, "--max-load-seconds", min=0.0),
    max_first_token_latency_seconds: float | None = typer.Option(
        None, "--max-first-token-latency-seconds", min=0.0
    ),
    max_rss_mb: float | None = typer.Option(None, "--max-rss-mb", min=0.0),
) -> None:
    """Real-Mac go-live proof: prove real models, strict-JSON routing, and switching.

    Unlike ``run april acceptance`` (which can report a fake/local sanity pass),
    go-live is real-model-only. It reuses APRIL's verification primitives to prove
    real GGUF models load/chat/stream/unload, the Brain produces strict-JSON routing
    with the real brain model (no fallback), and specialist switching keeps the brain
    resident. Voice is opt-in and out of scope for the first go-live milestone: a
    disabled voice stack is a skipped/warning note, never a failure.

    Read-only by construction: no config mutation, no model download, no package
    install, no microphone, no wake-word listening, no TTS, and no external network.
    The written report is redacted (basenames, counts, booleans, statuses only).
    """
    manager = _composition_api._manager()
    thresholds = ReportThresholds(
        min_tokens_per_second=min_tokens_per_second,
        max_load_seconds=max_load_seconds,
        max_first_token_latency_seconds=max_first_token_latency_seconds,
        max_rss_mb=max_rss_mb,
    )
    report_obj = _run_go_live_with_services(
        manager=manager,
        start_services=start_services,
        keep_services_running=keep_services_running,
        service_timeout=service_timeout,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        thresholds=thresholds,
    )

    target = report
    if target is None and write_report:
        target = default_go_live_report_path(manager.home)

    if json_output:
        console.print_json(data=report_obj.model_dump())
    else:
        _print_go_live(report_obj)

    if target is not None:
        written = write_go_live_report(report_obj, target)
        console.print(
            f"[green]Wrote go-live report to {written}[/green] "
            f"(final_status: {report_obj.final_status})"
        )

    if report_obj.final_status == "fail":
        raise typer.Exit(1)


def _print_preflight(report: PreflightReport) -> None:
    verdict = "[green]ready[/green]" if report.ok else "[red]blocked[/red]"
    console.print(
        f"APRIL startup preflight — {verdict} "
        f"(env={report.environment}, fake={str(report.fake).lower()})"
    )
    table = Table(title="Startup preflight")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(
            check.name,
            _PREFLIGHT_STATUS_STYLE.get(check.status, check.status),
            check.detail,
        )
    console.print(table)
