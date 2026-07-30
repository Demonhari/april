from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.acceptance import (
    AcceptanceReport,
)
from apps.runner.bootstrap import bootstrap as run_bootstrap
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.mac_activation import (
    ActivationFlagError,
    MacActivationReport,
    default_activation_report_path,
    run_mac_activation,
    validate_activation_flags,
    voice_paths_complete,
    write_activation_report,
)
from apps.runner.mac_report import ReportThresholds
from apps.runner.model_tools import (
    create_macos_app_stub,
    setup_embedding_model,
    setup_model_set,
    setup_voice_stack,
)
from apps.runner.setup_checklist import SetupChecklist, build_setup_checklist
from apps.runner.wake_live import run_sentinel_live_verification
from april_common.errors import ConfigError
from april_common.settings import load_settings, project_root
from april_common.token_setup import (
    legacy_plaintext_credentials_detected,
    provision_credentials,
    write_credential_store_reference,
)

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


_CHECKLIST_STATUS_STYLE = {
    "done": "[green]done[/green]",
    "warning": "[yellow]warning[/yellow]",
    "blocker": "[red]blocker[/red]",
    "next": "[cyan]next[/cyan]",
}

_ACTIVATION_STATUS_STYLE = {
    "validated": "[green]VALIDATED[/green]",
    "applied": "[green]APPLIED[/green]",
    "incomplete": "[yellow]INCOMPLETE[/yellow]",
    "failed": "[red]FAILED[/red]",
}


def _print_setup_checklist(checklist: SetupChecklist) -> None:
    console.print("APRIL first-run setup checklist (read-only)")
    table = Table(title="Setup steps")
    table.add_column("#")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Detail")
    for step in checklist.steps:
        table.add_row(
            str(step.number),
            step.title,
            _CHECKLIST_STATUS_STYLE.get(step.status, step.status),
            step.detail,
        )
    console.print(table)
    if checklist.next_command:
        console.print("[bold]Next command:[/bold]")
        console.print(f"  {checklist.next_command}", markup=False)
    else:
        console.print("[green]All recommended setup steps are complete.[/green]")
    console.print("[bold]Security and integrity follow-up:[/bold]")
    for command in checklist.security_integrity_commands:
        console.print(f"  {command}", markup=False)


@_registry.setup_app.command("checklist")
def setup_checklist(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show the recommended setup order and what is already done (read-only).

    Never installs, downloads, mutates config, starts a service, loads a model, or
    opens the microphone — it only detects state and prints the next command.
    """
    checklist = build_setup_checklist(_composition_api._manager().home)
    if json_output:
        console.print_json(data=checklist.model_dump())
        return
    _print_setup_checklist(checklist)


@_registry.setup_app.command("models")
def setup_models(
    brain: Path | None = typer.Option(None, "--brain", help="Local brain GGUF path."),
    coding: Path | None = typer.Option(None, "--coding", help="Local coding GGUF path."),
    reading: Path | None = typer.Option(None, "--reading", help="Local reading GGUF path."),
    reasoning: Path | None = typer.Option(
        None, "--reasoning", help="Optional reasoning GGUF path."
    ),
    brain_id: str | None = typer.Option(None, "--brain-id"),
    coding_id: str | None = typer.Option(None, "--coding-id"),
    reading_id: str | None = typer.Option(None, "--reading-id"),
    reasoning_id: str | None = typer.Option(None, "--reasoning-id"),
    copy_into_models: bool = typer.Option(False, "--copy-into-models"),
    apply_changes: bool = typer.Option(False, "--apply"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Safely validate and optionally configure APRIL's local GGUF model set."""
    if apply_changes and dry_run:
        console.print("[red]Use either --apply or --dry-run, not both.[/red]")
        raise typer.Exit(1)
    if apply_changes:
        console.print(
            "[red]Synchronous setup mutation is retired. Use one exact-approved "
            "`run april model import ... --sha256 EXPECTED_SHA256` job per model.[/red]"
        )
        raise typer.Exit(1)
    try:
        result = setup_model_set(
            home=_composition_api._manager().home,
            role_paths={
                "brain": brain,
                "coding": coding,
                "reading": reading,
                "reasoning": reasoning,
            },
            role_ids={
                "brain": brain_id,
                "coding": coding_id,
                "reading": reading_id,
                "reasoning": reasoning_id,
            },
            copy_into_models=copy_into_models,
            apply=apply_changes,
            force=force,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[green]Model setup applied.[/green]"
        if result["applied"]
        else "[yellow]Model setup dry run; no files were changed.[/yellow]"
    )
    for entry in result["entries"]:
        console.print(
            f"{entry['role']}: {entry['source_basename']} -> {entry['model_id']} "
            f"(copy_into_models={entry['copy_into_models']})"
        )
    if result["backup_basename"]:
        console.print(f"Config backup: {result['backup_basename']}")
    console.print("Next commands:")
    for command in result["next_commands"]:
        console.print(f"  {command}")


@_registry.setup_app.command("embeddings")
def setup_embeddings(
    model: Path = typer.Option(
        ..., "--model", help="Local embedding GGUF path (never downloaded)."
    ),
    model_id: str = typer.Option("april-embedding", "--id", help="Embedding model id."),
    name: str | None = typer.Option(None, "--name", help="Optional display name."),
    copy_into_models: bool = typer.Option(False, "--copy-into-models"),
    apply_changes: bool = typer.Option(False, "--apply"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Register a local runtime-local embedding model (dry-run unless --apply).

    Never downloads a model. Switching to runtime-local embeddings changes the
    vector space, so the printed next commands always include the reindex command.
    """
    if apply_changes and dry_run:
        console.print("[red]Use either --apply or --dry-run, not both.[/red]")
        raise typer.Exit(1)
    if apply_changes:
        console.print(
            "[red]Synchronous embedding import/provider mutation is retired. "
            "Import with `run april model import --role embedding ... "
            "--sha256 EXPECTED_SHA256`, then select the provider explicitly and "
            "run `run april memory reindex --wait`.[/red]"
        )
        raise typer.Exit(1)
    try:
        result = setup_embedding_model(
            home=_composition_api._manager().home,
            source_path=model,
            model_id=model_id,
            name=name,
            copy_into_models=copy_into_models,
            apply=apply_changes,
            force=force,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[green]Embedding setup applied; memory.embedding_provider=runtime-local.[/green]"
        if result["applied"]
        else "[yellow]Embedding setup dry run; no files were changed.[/yellow]"
    )
    plan = result["plan"]
    console.print(
        f"embedding: {plan['source_basename']} -> {plan['model_id']} "
        f"(copy_into_models={plan['copy_into_models']})"
    )
    for backup in result["backup_basenames"]:
        console.print(f"Config backup: {backup}")
    console.print("[bold]Switching providers requires a reindex.[/bold] Next commands:")
    for command in result["next_commands"]:
        console.print(f"  {command}", markup=False)


@_registry.setup_app.command("voice")
def setup_voice(
    whisper_binary: Path = typer.Option(..., "--whisper-binary"),
    whisper_model: Path = typer.Option(..., "--whisper-model"),
    piper_binary: Path = typer.Option(..., "--piper-binary"),
    piper_model: Path = typer.Option(..., "--piper-model"),
    wake_word_model: Path | None = typer.Option(None, "--wake-word-model"),
    apply_changes: bool = typer.Option(False, "--apply"),
    enable: bool = typer.Option(
        False,
        "--enable",
        help="Turn voice ON after required paths validate. Voice stays OFF without this flag.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Validate and optionally configure local voice tools without recording."""
    if apply_changes and dry_run:
        console.print("[red]Use either --apply or --dry-run, not both.[/red]")
        raise typer.Exit(1)
    try:
        result = setup_voice_stack(
            home=_composition_api._manager().home,
            whisper_binary=whisper_binary,
            whisper_model=whisper_model,
            piper_binary=piper_binary,
            piper_model=piper_model,
            wake_word_model=wake_word_model,
            apply=apply_changes,
            enable=enable,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[green]Voice setup applied.[/green]"
        if result["applied"]
        else "[yellow]Voice setup dry run; no files were changed.[/yellow]"
    )
    for artifact in result["artifacts"]:
        label = artifact["basename"] or "not configured"
        console.print(f"{artifact['name']}: {label}")
    for warning in result["warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    # Voice is never enabled by surprise: state the enabled/disabled outcome plainly.
    if result["voice_enabled"]:
        console.print("[green]Voice is now ENABLED.[/green]")
        if result["wake_word_available"]:
            console.print(
                "Push-to-talk is available. Wake-word listening stays UNVERIFIED until "
                "`run april voice verify-live` passes on this Mac."
            )
        else:
            console.print(
                "Push-to-talk is available. No wake-word model is configured, so wake-word "
                "listening is UNAVAILABLE; push-to-talk works without one."
            )
    elif apply_changes and enable:
        # enable was requested but apply did not run (should not happen, but be honest).
        console.print("[yellow]Voice remains DISABLED.[/yellow]")
    else:
        console.print(
            "[yellow]Voice remains DISABLED.[/yellow] Paths are validated only; voice stays OFF "
            "unless you re-run with --apply --enable."
        )
    if result["backup_basename"]:
        console.print(f"Config backup: {result['backup_basename']}")
    console.print("Next commands:")
    for command in result["next_commands"]:
        console.print(f"  {command}")


def _print_activation(report: MacActivationReport) -> None:
    status = _ACTIVATION_STATUS_STYLE.get(report.final_status, report.final_status)
    console.print(f"APRIL Mac activation — {status} (mode={report.mode})")
    models = report.models
    if models.error:
        console.print(f"[red]Models: {models.error}[/red]")
    else:
        console.print(
            "Models: "
            f"validated={models.validated}, applied={models.applied}, "
            f"core_complete={models.core_model_set_complete}, "
            f"partial={models.partial_model_set}"
        )
        if models.supplied_roles:
            console.print(f"  supplied: {', '.join(models.supplied_roles)}")
        console.print(f"  optional: {', '.join(models.optional_roles)}")
        if models.missing_required_roles:
            console.print(
                f"[yellow]  missing required: {', '.join(models.missing_required_roles)}[/yellow]"
            )
        for entry in models.entries:
            console.print(f"  {entry.role}: {entry.basename}")
    voice = report.voice
    if voice.skipped:
        console.print("Voice: skipped (voice is opt-in; no voice flags supplied)")
    elif voice.error:
        console.print(f"[red]Voice: {voice.error}[/red]")
    else:
        console.print(
            f"Voice: validated={voice.validated}, applied={voice.applied}, enabled={voice.enabled}"
        )
        for artifact in voice.artifacts:
            console.print(f"  {artifact.name}: {artifact.basename or 'not configured'}")
        for warning in voice.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    transaction = report.transaction
    if transaction.requested:
        if transaction.committed:
            console.print(
                f"Transaction: committed (backup={transaction.backup_basename or 'none'})"
            )
        elif transaction.rolled_back:
            console.print(
                f"[yellow]Transaction: rolled back ({transaction.rollback_status}) — "
                f"{transaction.rollback_reason}[/yellow]"
            )
        else:
            console.print(
                f"[yellow]Transaction: not committed ({transaction.rollback_status})[/yellow]"
            )
    acceptance_link = report.acceptance
    if acceptance_link.ran:
        console.print(
            f"Acceptance: {acceptance_link.final_status} "
            f"(level={acceptance_link.acceptance_level}, "
            f"backend={acceptance_link.runtime_backend}, "
            f"voice={acceptance_link.voice_live_summary or 'n/a'}, "
            f"wake={acceptance_link.wake_word_live_summary or 'n/a'})"
        )
    elif acceptance_link.skipped_reason:
        console.print(f"Acceptance: skipped — {acceptance_link.skipped_reason}")
    if report.next_actions:
        console.print("[bold]Next commands:[/bold]")
        for action in report.next_actions:
            # markup=False so command tokens like '.[runtime]' are not parsed as tags.
            console.print(f"  {action}", markup=False)


@_registry.setup_app.command("mac-activation", context_settings={"allow_extra_args": True})
def setup_mac_activation(
    ctx: typer.Context,
    brain: Path | None = typer.Option(None, "--brain", help="Local brain GGUF path."),
    coding: Path | None = typer.Option(None, "--coding", help="Local coding GGUF path."),
    reading: Path | None = typer.Option(None, "--reading", help="Local reading GGUF path."),
    reasoning: Path | None = typer.Option(
        None, "--reasoning", help="Optional reasoning GGUF path."
    ),
    reasoning_id: str | None = typer.Option(None, "--reasoning-id"),
    whisper_binary: Path | None = typer.Option(None, "--whisper-binary"),
    whisper_model: Path | None = typer.Option(None, "--whisper-model"),
    piper_binary: Path | None = typer.Option(None, "--piper-binary"),
    piper_model: Path | None = typer.Option(None, "--piper-model"),
    wake_word_model: Path | None = typer.Option(None, "--wake-word-model"),
    skip_voice: bool = typer.Option(
        False,
        "--skip-voice",
        help="Explicit models-only override. Voice is already opt-in; "
        "incompatible with any voice flag.",
    ),
    enable_voice: bool = typer.Option(
        False,
        "--enable-voice",
        help="Turn voice ON after all required voice artifacts validate (with --apply).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    apply_changes: bool = typer.Option(False, "--apply"),
    no_rollback: bool = typer.Option(
        False, "--no-rollback", help="Debug only: leave partial config if an apply step fails."
    ),
    run_acceptance_after: bool = typer.Option(
        False, "--run-acceptance", help="After --apply, run real-model acceptance."
    ),
    acceptance_voice_live: bool = typer.Option(
        False,
        "--acceptance-voice-live",
        help="Run push-to-talk voice acceptance (needs --run-acceptance).",
    ),
    acceptance_wake_word_live: bool = typer.Option(
        False,
        "--acceptance-wake-word-live",
        help="Run wake-word acceptance (needs --run-acceptance).",
    ),
    start_services: bool = typer.Option(
        False, "--start-services", help="Start missing services for live acceptance checks."
    ),
    fake_services: bool = typer.Option(
        False, "--fake-services", help="Start fake services (incompatible with real acceptance)."
    ),
    allow_partial_model_set: bool = typer.Option(
        False,
        "--allow-partial-model-set",
        help="Register supplied models even when brain/coding/reading are not all complete.",
    ),
    keep_services_running: bool = typer.Option(
        False, "--keep-services-running", help="Leave services acceptance started running."
    ),
    service_timeout: float = typer.Option(20.0, "--service-timeout", min=1.0),
    write_report: bool = typer.Option(
        False,
        "--write-report",
        help="Write a redacted report to data/verification/mac-activation-<timestamp>.json.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Guided, transactional local activation: validate models + voice, apply, verify.

    Dry-run by default. Never downloads models, installs packages, uses sudo or
    Homebrew, or records audio. Config is written only with --apply, all paths are
    validated first, and a failed apply step is rolled back automatically.
    """
    if apply_changes and any(path is not None for path in (brain, coding, reading, reasoning)):
        console.print(
            "[red]Synchronous model registration through mac-activation is retired. "
            "Import each local GGUF through exact-approved `run april model import "
            "... --sha256 EXPECTED_SHA256`; activation remains a separate manual phase.[/red]"
        )
        raise typer.Exit(1)
    activation_voice_paths: dict[str, Path | None] = {
        "whisper_binary": whisper_binary,
        "whisper_model": whisper_model,
        "piper_binary": piper_binary,
        "piper_model": piper_model,
        "wake_word_model": wake_word_model,
    }
    any_voice_path_supplied = any(path is not None for path in activation_voice_paths.values())
    try:
        validate_activation_flags(
            apply=apply_changes,
            dry_run=dry_run,
            skip_voice=skip_voice,
            enable_voice=enable_voice,
            run_acceptance_after=run_acceptance_after,
            acceptance_voice_live=acceptance_voice_live,
            acceptance_wake_word_live=acceptance_wake_word_live,
            start_services=start_services,
            fake_services=fake_services,
            voice_paths_supplied=any_voice_path_supplied,
            voice_required_complete=voice_paths_complete(activation_voice_paths),
        )
    except ActivationFlagError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    manager = _composition_api._manager()
    home = manager.home
    settings = manager.settings

    acceptance_runner: Callable[[], AcceptanceReport] | None = None
    if run_acceptance_after and apply_changes:

        def _activation_acceptance() -> AcceptanceReport:
            # Reuse the acceptance service orchestration verbatim; activation
            # acceptance is always real-model and may add live voice/wake checks.
            return _composition_api._run_acceptance_with_services(
                manager=manager,
                require_real_models=True,
                allow_sanity_pass=False,
                start_services=start_services,
                fake_services=fake_services,
                keep_services_running=keep_services_running,
                service_timeout=service_timeout,
                max_output_tokens=32,
                timeout=180.0,
                thresholds=ReportThresholds(),
                voice_live_runner=(
                    _composition_api._voice_live_runner(settings) if acceptance_voice_live else None
                ),
                wake_word_live_runner=(
                    _composition_api._wake_word_live_runner(settings)
                    if acceptance_wake_word_live
                    else None
                ),
            )

        acceptance_runner = _activation_acceptance

    report_obj = run_mac_activation(
        home,
        model_paths={"brain": brain, "coding": coding, "reading": reading, "reasoning": reasoning},
        model_ids={"reasoning": reasoning_id},
        voice_paths=activation_voice_paths,
        skip_voice=skip_voice,
        apply=apply_changes,
        enable_voice=enable_voice,
        run_acceptance_after=run_acceptance_after,
        allow_partial_model_set=allow_partial_model_set,
        no_rollback=no_rollback,
        acceptance_runner=acceptance_runner,
    )

    if ctx.args and not write_report:
        console.print("[red]Unexpected extra argument. Did you mean --write-report PATH?[/red]")
        raise typer.Exit(1)
    if len(ctx.args) > 1:
        console.print("[red]Only one --write-report path may be supplied.[/red]")
        raise typer.Exit(1)
    target = (
        Path(ctx.args[0])
        if write_report and ctx.args
        else default_activation_report_path(home)
        if write_report
        else None
    )

    if json_output:
        console.print_json(data=report_obj.model_dump())
    else:
        _print_activation(report_obj)

    if target is not None:
        written = write_activation_report(report_obj, target)
        console.print(
            f"[green]Wrote activation report to {written}[/green] "
            f"(final_status: {report_obj.final_status})"
        )

    if report_obj.final_status == "failed":
        raise typer.Exit(1)
    if report_obj.acceptance.ran and report_obj.acceptance.final_status == "fail":
        raise typer.Exit(1)


@_registry.setup_app.command("app-stub")
def setup_app_stub(
    output: Path = typer.Option(Path("dist/APRIL.app"), "--output"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Create the unsigned local-development macOS APRIL.app launcher."""
    try:
        result = create_macos_app_stub(
            home=_composition_api._manager().home, output=output, force=force
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Created unsigned APRIL development launcher: {result.output_path}[/green]"
    )
    console.print("Unsigned local development only. No models, tokens, signing, or notarization.")


@_registry.setup_app.command("tokens")
def setup_tokens(
    output: Path = typer.Option(
        Path(".env"),
        "--output",
        help="Non-secret credential-store identifiers file.",
    ),
    backend: str | None = typer.Option(None, "--store"),
    credential_file: Path | None = typer.Option(None, "--credential-file"),
) -> None:
    from apps.runner.security_commands import _store_for_command
    from april_common.credentials import CredentialStoreError

    home = Path(os.environ.get("APRIL_HOME", project_root())).expanduser().resolve()
    settings = load_settings(root=home, legacy_credential_migration=True)
    target = output if output.is_absolute() else home / output
    try:
        if legacy_plaintext_credentials_detected(home):
            console.print(
                "[red]Legacy plaintext credentials detected. Run "
                "`run april security credentials migrate` first.[/red]"
            )
            raise typer.Exit(1)
        store = _store_for_command(
            settings,
            backend=backend,
            file_path=credential_file,
        )
        result = provision_credentials(store)
        write_credential_store_reference(target, store)
    except (ConfigError, CredentialStoreError, OSError) as exc:
        console.print(f"[red]Credential setup failed ({type(exc).__name__}).[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"Generated APRIL API and Runtime credentials in {result.store}; "
        f"wrote non-secret identifiers to {target}."
    )
    console.print("Credential values were not printed or written to .env.")


@_registry.setup_app.command("bootstrap")
def setup_bootstrap(
    env_file: Path = typer.Option(Path(".env"), "--env-file", help="Local env file for tokens."),
    force: bool = typer.Option(False, "--force", help="Regenerate tokens even if they exist."),
    apply_profile: bool = typer.Option(
        False, "--apply-profile", help="Apply the recommended model profile (mutates configs)."
    ),
    no_auto_profile: bool = typer.Option(
        False,
        "--no-auto-profile",
        help="Suppress Intel first-run automatic conservative profile selection.",
    ),
    show_paths: bool = typer.Option(
        False, "--show-paths", help="Include absolute local paths in bootstrap output."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Safe, non-destructive local first-run setup. Never prints full tokens."""
    home = _composition_api._manager().home
    target_env = env_file if env_file.is_absolute() else home / env_file
    report = run_bootstrap(
        home,
        env_file=target_env,
        force=force,
        apply_profile=apply_profile,
        no_auto_profile=no_auto_profile,
        show_paths=show_paths,
    )
    if json_output:
        console.print_json(data=report)
    else:
        _print_bootstrap(report)
    if not report["config_valid"]:
        raise typer.Exit(1)


def _print_bootstrap(report: dict[str, Any]) -> None:
    console.print(f"[bold]APRIL bootstrap[/bold] — home: {report['home']}")
    created = sum(1 for item in report["directories"] if item["created"])
    console.print(f"Directories: {len(report['directories'])} ensured ({created} newly created).")
    tokens = report["tokens"]
    console.print(f"Tokens ({report['env_file']}): {tokens['action']} (values not printed).")
    for warning in report["dev_token_warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    machine = report["machine"]
    console.print(
        f"Machine: {machine['architecture']} · {machine['cpu_count']} CPUs · "
        f"{machine['available_memory']} RAM"
    )
    console.print(
        f"Recommended profile: {report['recommended_profile']} "
        f"({report['expected_backend']}); "
        + (
            f"applied {report['applied_profile']}."
            if report["profile_applied"]
            else "not applied (use --apply-profile)."
        )
    )
    console.print(
        f"llama-cpp-python: {'available' if report['llama_cpp_available'] else 'not installed'}; "
        f"models missing files: {len(report['missing_model_paths'])}."
    )
    console.print(f"Allowed filesystem roots: {report['allowed_filesystem_roots']}")
    console.print(f"Config valid: {report['config_valid']}")
    console.print("Next commands:")
    for command in report["next_commands"]:
        console.print(f"  {command}")
