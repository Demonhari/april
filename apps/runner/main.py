from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.evals import run_fake_brain_eval, run_real_brain_eval
from apps.runner.install import is_april_wrapper, path_contains_dir
from apps.runner.model_tools import (
    apply_model_profile,
    import_model,
    load_model_profiles,
    model_doctor,
)
from apps.runner.service_manager import AprilServiceManager, ServiceStatus
from apps.runner.verify import (
    BenchmarkResult,
    VerifyCheck,
    run_fake_verification,
    run_model_benchmark,
    run_real_model_verification,
    run_workflow_verification,
)
from april_common.config_validation import validate_configuration
from april_common.effective_config import load_agents_file, load_permissions_file, load_tools_file
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry

app = typer.Typer(help="Global command dispatcher.")
april_app = typer.Typer(help="Run APRIL from any folder.", invoke_without_command=True)
model_app = typer.Typer(help="Model operations.")
project_app = typer.Typer(help="Project operations.")
memory_app = typer.Typer(help="Memory operations.")
conversation_app = typer.Typer(help="Conversation operations.")
config_app = typer.Typer(help="Configuration operations.")
agent_app = typer.Typer(help="Direct specialist agent operations.")
voice_app = typer.Typer(help="Voice operations.")
reminder_app = typer.Typer(help="Reminder operations.")
task_app = typer.Typer(help="Task inspection operations.")
eval_app = typer.Typer(help="Local evaluation operations.")
app.add_typer(april_app, name="april")
april_app.add_typer(model_app, name="model")
april_app.add_typer(project_app, name="project")
april_app.add_typer(memory_app, name="memory")
april_app.add_typer(conversation_app, name="conversation")
april_app.add_typer(config_app, name="config")
april_app.add_typer(agent_app, name="agent")
april_app.add_typer(voice_app, name="voice")
april_app.add_typer(reminder_app, name="reminder")
april_app.add_typer(task_app, name="task")
april_app.add_typer(eval_app, name="eval")


def _manager() -> AprilServiceManager:
    return AprilServiceManager()


def _effective_fake(ctx: typer.Context, explicit: bool) -> bool:
    inherited = bool((ctx.obj or {}).get("fake", False))
    return inherited or explicit


def _effective_oneshot(ctx: typer.Context) -> bool:
    return bool((ctx.obj or {}).get("oneshot", False))


def _run_april_cli(args: list[str]) -> int:
    env = dict(os.environ)
    env.setdefault("APRIL_HOME", str(_manager().home))
    completed = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", *args],
        cwd=env["APRIL_HOME"],
        env=env,
        check=False,
    )
    return completed.returncode


def _ensure_services(fake: bool) -> ServiceStatus:
    try:
        status = _manager().start(fake_backend=fake)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not status.ok:
        console.print("[red]APRIL services are not healthy.[/red]")
        _print_status(status)
        raise typer.Exit(1)
    return status


def _delegate(args: list[str], *, fake: bool, oneshot: bool = False) -> None:
    manager = _manager()
    before = manager.status()
    try:
        _ensure_services(fake)
        if oneshot:
            console.print(
                "[yellow]APRIL oneshot mode: services will stop after this command.[/yellow]"
            )
        else:
            console.print("[green]APRIL services are running and will remain running.[/green]")
        code = _run_april_cli(args)
    finally:
        if oneshot and not before.ok:
            console.print("[yellow]Stopping APRIL services started for oneshot mode.[/yellow]")
            _print_status(manager.stop())
    raise typer.Exit(code)


def _print_status(status: ServiceStatus) -> None:
    table = Table(title="APRIL Services")
    table.add_column("Service")
    table.add_column("PID")
    table.add_column("Running")
    table.add_column("Healthy")
    table.add_column("Log")
    for info in (status.runtime, status.api):
        table.add_row(
            info.name,
            str(info.pid or "-"),
            "yes" if info.running else "no",
            "yes" if info.healthy else "no",
            str(info.log_path),
        )
    console.print(table)


def _status_payload(status: ServiceStatus) -> dict[str, Any]:
    return {
        "runtime": {
            "pid": status.runtime.pid,
            "running": status.runtime.running,
            "healthy": status.runtime.healthy,
            "log_path": str(status.runtime.log_path),
        },
        "api": {
            "pid": status.api.pid,
            "running": status.api.running,
            "healthy": status.api.healthy,
            "log_path": str(status.api.log_path),
        },
        "ok": status.ok,
    }


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve() or left.samefile(right)
    except FileNotFoundError:
        return left.resolve() == right.resolve()


def _doctor() -> None:
    manager = _manager()
    home = manager.home
    local_bin = Path.home() / ".local" / "bin"
    run_path = local_bin / "run"
    april_run_path = local_bin / "april-run"
    command_run = shutil.which("run")
    command_path = Path(command_run) if command_run else None
    run_found = command_path is not None
    command_is_april = bool(command_path and is_april_wrapper(command_path))
    command_points_to_expected = bool(
        command_path and run_path.exists() and _same_file(command_path, run_path)
    )

    table = Table(title="APRIL Launcher Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("APRIL_HOME", str(home))
    python_exists = (home / ".venv/bin/python").exists()
    table.add_row(".venv/bin/python exists", "yes" if python_exists else "no")
    table.add_row(f"{run_path} exists", "yes" if run_path.exists() else "no")
    table.add_row(f"{april_run_path} exists", "yes" if april_run_path.exists() else "no")
    table.add_row("run wrapper APRIL-owned", "yes" if is_april_wrapper(run_path) else "no")
    table.add_row(
        "april-run wrapper APRIL-owned",
        "yes" if is_april_wrapper(april_run_path) else "no",
    )
    table.add_row("run wrapper executable", "yes" if os.access(run_path, os.X_OK) else "no")
    table.add_row(
        "april-run wrapper executable",
        "yes" if os.access(april_run_path, os.X_OK) else "no",
    )
    table.add_row(f"{local_bin} in PATH", "yes" if path_contains_dir(local_bin) else "no")
    table.add_row("command -v run", command_run or "not found")
    table.add_row("command -v run is APRIL wrapper", "yes" if command_is_april else "no")
    table.add_row(
        "command -v run points to ~/.local/bin/run",
        "yes" if command_points_to_expected else "no",
    )
    console.print(table)
    _print_status(manager.status())

    if not run_found:
        console.print("[yellow]run was not found in PATH.[/yellow]")
        console.print(f"cd {home}")
        console.print("make install-global")
        console.print('export PATH="$HOME/.local/bin:$PATH"')
        console.print("run april --fake")
    elif not command_is_april:
        console.print("[yellow]run resolves to a non-APRIL command.[/yellow]")
        console.print(f"cd {home}")
        console.print("make install-global-force")
    elif path_contains_dir(local_bin):
        console.print("[green]OK: run resolves to an APRIL wrapper visible in PATH.[/green]")


def _print_verification_table(title: str, checks: list[VerifyCheck]) -> None:
    table = Table(title=title)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        table.add_row(check.name, "pass" if check.ok else "fail", check.detail)
    console.print(table)


def _print_model_doctor(payload: dict[str, Any]) -> None:
    table = Table(title="APRIL Model Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Python", str(payload["python_version"]))
    table.add_row("APRIL_HOME", str(payload["april_home"]))
    table.add_row("Runtime backend", str(payload["runtime_backend"]))
    table.add_row(
        "llama-cpp-python installed",
        "yes" if payload["llama_cpp_python_installed"] else "no",
    )
    table.add_row("API token", str(payload["api_token"]))
    table.add_row("Runtime token", str(payload["runtime_token"]))
    table.add_row("Machine", str(payload["machine"]))
    table.add_row("CPU count", str(payload["cpu_count"]))
    table.add_row("Estimated RAM", str(payload["estimated_ram"]))
    console.print(table)

    models = Table(title="Configured Models")
    for column in (
        "ID",
        "Role",
        "Path",
        "Exists",
        "Size",
        "Ctx",
        "Threads",
        "Batch",
        "Keep",
        "Idle unload",
        "Realism",
    ):
        models.add_column(column)
    for model in payload["models"]:
        models.add_row(
            str(model["id"]),
            str(model["role"]),
            str(model["path"]),
            "yes" if model["path_exists"] else "no",
            str(model["file_size"]),
            str(model["context_size"]),
            str(model["threads"]),
            str(model["n_batch"] or "-"),
            "yes" if model["keep_loaded"] else "no",
            str(model["idle_unload_seconds"] or "-"),
            str(model["realism"]),
        )
    console.print(models)


def _print_benchmark(results: list[BenchmarkResult]) -> None:
    table = Table(title="APRIL Model Benchmark")
    table.add_column("Run")
    table.add_column("Load")
    table.add_column("First token")
    table.add_column("Generation")
    table.add_column("Tokens")
    table.add_column("Tokens/sec")
    table.add_column("Unload")
    table.add_column("Detail")
    for result in results:
        table.add_row(
            str(result.run_index),
            f"{result.load_time_seconds:.2f}s",
            "n/a"
            if result.first_token_latency_seconds is None
            else f"{result.first_token_latency_seconds:.2f}s",
            f"{result.generation_time_seconds:.2f}s",
            str(result.output_tokens),
            f"{result.tokens_per_second:.2f}",
            "yes" if result.unload_success else "no",
            result.detail,
        )
    console.print(table)
    console.print(
        "CPU-only recommendation: keep contexts conservative, use small batch sizes, "
        "and unload non-brain models when not in active use."
    )


def _print_brain_eval(results: list[Any]) -> None:
    table = Table(title="APRIL Brain Eval")
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Detail")
    for result in results:
        actual_intent = result.actual.get("intent", "-")
        actual_agent = result.actual.get("agent", "-")
        table.add_row(
            result.id,
            "pass" if result.ok else "fail",
            f"{result.expected_intent}/{result.expected_agent}",
            f"{actual_intent}/{actual_agent}",
            result.detail,
        )
    console.print(table)


@april_app.callback(invoke_without_command=True)
def april(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    oneshot: bool = typer.Option(
        False,
        "--oneshot",
        help="Stop services after the delegated command when this invocation started them.",
    ),
) -> None:
    ctx.obj = {"fake": fake, "oneshot": oneshot}
    if ctx.invoked_subcommand is None:
        _delegate(["chat"], fake=fake, oneshot=oneshot)


@april_app.command()
def chat(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["chat"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def ask(
    ctx: typer.Context,
    message: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
) -> None:
    args = ["ask", message]
    if project_id:
        args.extend(["--project-id", project_id])
    if repo_path:
        args.extend(["--repo-path", repo_path])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def status(json_output: bool = typer.Option(False, "--json")) -> None:
    status_value = _manager().status()
    if json_output:
        console.print_json(data=_status_payload(status_value))
        return
    _print_status(status_value)


@app.command()
def doctor() -> None:
    _doctor()


@april_app.command("doctor")
def april_doctor() -> None:
    _doctor()


@april_app.command()
def health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["health"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def models(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["models"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def approvals(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["approvals"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@april_app.command()
def approve(
    ctx: typer.Context,
    approval_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["approve", approval_id], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx)
    )


@april_app.command()
def deny(
    ctx: typer.Context,
    approval_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["deny", approval_id], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx)
    )


@april_app.command()
def projects(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["projects"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@model_app.command("load")
def model_load(
    ctx: typer.Context,
    model_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["model", "load", model_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@model_app.command("unload")
def model_unload(
    ctx: typer.Context,
    model_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["model", "unload", model_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@model_app.command("doctor")
def model_doctor_command(json_output: bool = typer.Option(False, "--json")) -> None:
    payload = model_doctor(_manager().home)
    if json_output:
        console.print_json(data=payload)
        return
    _print_model_doctor(payload)


@model_app.command("import")
def model_import_command(
    role: str = typer.Option(..., "--role"),
    model_id: str = typer.Option(..., "--id"),
    name: str = typer.Option(..., "--name"),
    path: Path = typer.Option(..., "--path"),
    copy_into_models: bool = typer.Option(False, "--copy-into-models"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    try:
        result = import_model(
            home=_manager().home,
            role=role,
            model_id=model_id,
            name=name,
            source_path=path,
            copy_into_models=copy_into_models,
            force=force,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Registered {result.model_id} for role {result.role}.[/green]")
    console.print(f"Model path: {result.path}")
    console.print(result.next_command)


@model_app.command("benchmark")
def model_benchmark_command(
    model_path: Path,
    prompt: str = typer.Option("Reply with one short sentence.", "--prompt"),
    runs: int = typer.Option(1, "--runs", min=1, max=20),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    keep_loaded: bool = typer.Option(False, "--keep-loaded"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if not model_path.expanduser().exists():
        console.print(f"[red]GGUF path does not exist: {model_path}[/red]")
        raise typer.Exit(1)
    results = run_model_benchmark(
        _manager().home,
        model_path,
        prompt=prompt,
        runs=runs,
        max_output_tokens=max_output_tokens,
        keep_loaded=keep_loaded,
    )
    if json_output:
        console.print_json(data={"runs": [result.model_dump() for result in results]})
    else:
        _print_benchmark(results)
    if not all(result.ok for result in results):
        raise typer.Exit(1)


profile_app = typer.Typer(help="Model profile operations.")
model_app.add_typer(profile_app, name="profile")


@profile_app.command("list")
def model_profile_list() -> None:
    profiles = load_model_profiles(_manager().home)
    table = Table(title="APRIL Model Profiles")
    table.add_column("Profile")
    table.add_column("Description")
    for name, profile in profiles.items():
        description = profile.get("description", "") if isinstance(profile, dict) else ""
        table.add_row(str(name), str(description))
    console.print(table)


@profile_app.command("apply")
def model_profile_apply(profile_name: str) -> None:
    try:
        backup = apply_model_profile(home=_manager().home, profile_name=profile_name)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Applied model profile: {profile_name}[/green]")
    console.print(f"Backup: {backup}")


@project_app.command("add")
def project_add(
    ctx: typer.Context,
    path: str,
    name: str | None = typer.Option(None, "--name"),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["project", "add", path]
    if name:
        args.extend(["--name", name])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@project_app.command("index")
def project_index(
    ctx: typer.Context,
    project_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["project", "index", project_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("list")
def memory_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["memory", "list"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@memory_app.command("search")
def memory_search(
    ctx: typer.Context,
    query: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "search", query],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("delete")
def memory_delete(
    ctx: typer.Context,
    memory_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "delete", memory_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("export")
def memory_export(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["memory", "export"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@memory_app.command("doctor")
def memory_doctor() -> None:
    settings = load_settings(root=_manager().home)
    enabled = settings.memory.embedding_provider == "hashed-token"
    data = {
        "embedding_provider": settings.memory.embedding_provider,
        "embedding_model_id": settings.memory.embedding_model_id,
        "semantic_embeddings_enabled": enabled,
        "status": "ok"
        if enabled
        else "disabled: runtime-local requires explicit local embedding backend support",
    }
    console.print_json(data=data)


@conversation_app.command("delete")
def conversation_delete(
    ctx: typer.Context,
    conversation_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["conversation", "delete", conversation_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@reminder_app.command("list")
def reminder_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["reminder", "list"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@reminder_app.command("create")
def reminder_create(
    ctx: typer.Context,
    content: str,
    due_at: str | None = typer.Option(None, "--due-at"),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["reminder", "create", content]
    if due_at:
        args.extend(["--due-at", due_at])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@reminder_app.command("delete")
def reminder_delete(
    ctx: typer.Context,
    reminder_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["reminder", "delete", reminder_id],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@task_app.command("list")
def task_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(["task", "list"], fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@eval_app.command("brain")
def eval_brain(
    fake: bool = typer.Option(False, "--fake", help="Run deterministic fake Brain eval."),
    real_model: Path | None = typer.Option(None, "--real-model"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if fake:
        results = run_fake_brain_eval(_manager().home)
    elif real_model is not None:
        if not real_model.expanduser().exists():
            console.print(f"[red]GGUF path does not exist: {real_model}[/red]")
            raise typer.Exit(1)
        results = run_real_brain_eval(_manager().home, real_model)
    else:
        console.print("[red]Use --fake or --real-model /path/to/model.gguf.[/red]")
        raise typer.Exit(1)
    if json_output:
        console.print_json(data={"results": [result.model_dump() for result in results]})
    else:
        _print_brain_eval(results)
    if not all(result.ok for result in results):
        raise typer.Exit(1)


@voice_app.command("health")
def voice_health(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "health"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("doctor")
def voice_doctor(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "doctor"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("devices")
def voice_devices(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "devices"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("ptt")
def voice_ptt(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    seconds: float | None = typer.Option(None, "--seconds", min=0.1, max=300.0),
) -> None:
    args = ["voice", "ptt"]
    if seconds is not None:
        args.extend(["--seconds", str(seconds)])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@voice_app.command("test-record")
def voice_test_record(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    seconds: float = typer.Option(3.0, "--seconds", min=0.1, max=30.0),
) -> None:
    _delegate(
        ["voice", "test-record", "--seconds", str(seconds)],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("test-stt")
def voice_test_stt(
    ctx: typer.Context,
    audio_path: Path,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "test-stt", str(audio_path)],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("test-tts")
def voice_test_tts(
    ctx: typer.Context,
    text: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "test-tts", text],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@voice_app.command("listen")
def voice_listen(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _delegate(
        ["voice", "listen"],
        fake=_effective_fake(ctx, fake),
        oneshot=_effective_oneshot(ctx),
    )


@agent_app.command("run")
def agent_run(
    ctx: typer.Context,
    agent: str,
    message: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    project_id: str | None = typer.Option(None, "--project-id"),
    repo_path: str | None = typer.Option(None, "--repo-path"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
) -> None:
    args = ["agent", "run", agent, message]
    if project_id:
        args.extend(["--project-id", project_id])
    if repo_path:
        args.extend(["--repo-path", repo_path])
    if conversation_id:
        args.extend(["--conversation-id", conversation_id])
    _delegate(args, fake=_effective_fake(ctx, fake), oneshot=_effective_oneshot(ctx))


@config_app.command("validate")
def config_validate() -> None:
    errors = validate_configuration(_manager().home)
    if errors:
        console.print("[red]APRIL configuration is invalid.[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    console.print("[green]APRIL configuration is valid.[/green]")


@config_app.command("inspect")
def config_inspect() -> None:
    errors = validate_configuration(_manager().home)
    if errors:
        console.print("[red]APRIL configuration is invalid.[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(1)
    settings = load_settings(root=_manager().home)
    home = _manager().home
    settings_data = settings.model_dump(mode="json")
    if isinstance(settings_data.get("api"), dict):
        settings_data["api"]["token"] = "[REDACTED]"
    if isinstance(settings_data.get("runtime"), dict):
        settings_data["runtime"]["token"] = "[REDACTED]"
    models = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    data = {
        "settings": settings_data,
        "models": [model.model_dump(mode="json") for model in models.list()],
        "agents": load_agents_file(home).model_dump(mode="json"),
        "tools": load_tools_file(home).model_dump(mode="json"),
        "permissions": load_permissions_file(home).model_dump(mode="json"),
    }
    console.print_json(data=data)


@april_app.command()
def verify(
    model_path: Path | None = typer.Argument(None),
    fake: bool = typer.Option(False, "--fake", help="Run deterministic fake-backend verification."),
    real_model: bool = typer.Option(False, "--real-model"),
    workflow: bool = typer.Option(False, "--workflow"),
    json_output: bool = typer.Option(False, "--json"),
    max_output_tokens: int = typer.Option(32, "--max-output-tokens", min=1, max=4096),
    timeout: float = typer.Option(180.0, "--timeout", min=1.0),
) -> None:
    if workflow:
        checks = run_workflow_verification(
            _manager().home,
            real_model=real_model,
            model_path=model_path,
        )
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _print_verification_table("APRIL Workflow Verification", checks)
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
        checks = run_real_model_verification(
            _manager().home,
            configured_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        if json_output:
            console.print_json(data={"checks": [asdict(check) for check in checks]})
        else:
            _print_verification_table("APRIL Real Model Verification", checks)
        if not all(check.ok for check in checks):
            raise typer.Exit(1)
        raise typer.Exit(0)
    if not fake:
        console.print("[red]Use --fake for deterministic local verification.[/red]")
        raise typer.Exit(1)
    checks = run_fake_verification(_manager().home)
    if json_output:
        console.print_json(data={"checks": [asdict(check) for check in checks]})
    else:
        _print_verification_table("APRIL Verification", checks)
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


@april_app.command()
def stop() -> None:
    _print_status(_manager().stop())


@april_app.command()
def restart(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start services with fake runtime."),
) -> None:
    _print_status(_manager().restart(fake_backend=_effective_fake(ctx, fake)))


@april_app.command()
def logs(
    lines: int = typer.Option(80, "--lines", min=1, max=1000),
    tail: int | None = typer.Option(None, "--tail", min=1, max=1000),
) -> None:
    console.print(_manager().logs(lines=tail or lines))


if __name__ == "__main__":
    app()
