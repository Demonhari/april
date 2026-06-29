"""Startup preflight for ``run april start --preflight``.

A redacted, side-effect-free set of checks run *before* APRIL's services are
started, so a misconfigured or unsafe environment fails fast instead of leaving
half-started services behind. Preflight never starts a service, never opens the
microphone, never loads a model, and never mutates configuration — it only reads
settings, probes paths for writability, checks port availability, and inspects
the managed pid files for stale locks.

The verdict is intentionally conservative: ``ok`` is true only when no check is a
``fail``. ``--fake`` relaxes exactly two checks (a fake runtime backend and the
absence of real GGUF files become acceptable for development), nothing else.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.mac_report import redact_reason
from april_common.errors import ConfigError
from april_common.settings import (
    KNOWN_DEFAULT_API_TOKENS,
    KNOWN_DEFAULT_RUNTIME_TOKENS,
    PLACEHOLDER_API_TOKENS,
    PLACEHOLDER_RUNTIME_TOKENS,
    AprilSettings,
    load_settings,
)
from services.april_runtime.model_registry import ModelRegistry

PreflightStatus = Literal["pass", "warning", "fail"]

PortChecker = Callable[[str, int], bool]
PidAlive = Callable[[int], bool]


class PreflightCheck(BaseModel):
    name: str
    status: PreflightStatus
    detail: str


class PreflightReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["startup_preflight"] = "startup_preflight"
    fake: bool = False
    environment: str = "development"
    ok: bool = False
    checks: list[PreflightCheck] = Field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [check.name for check in self.checks if check.status == "fail"]


def _default_port_in_use(host: str, port: int) -> bool:
    target = host if host not in {"", "0.0.0.0"} else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((target, port)) == 0


def _default_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _path_writable(path: Path) -> bool:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return os.access(probe, os.W_OK)


def _read_pid(pid_path: Path) -> int | None:
    try:
        raw = pid_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _token_default(value: str | None, defaults: set[str], placeholders: set[str]) -> bool:
    return (not value) or value in defaults or value in placeholders


def build_preflight_report(
    home: Path,
    *,
    fake: bool = False,
    port_in_use: PortChecker | None = None,
    pid_alive: PidAlive | None = None,
) -> PreflightReport:
    """Build the startup preflight verdict. Pure over injected probes for testing."""
    home = home.expanduser().resolve()
    port_in_use = port_in_use or _default_port_in_use
    pid_alive = pid_alive or _default_pid_alive
    checks: list[PreflightCheck] = []

    try:
        settings = load_settings(root=home)
    except ConfigError as exc:
        return PreflightReport(
            fake=fake,
            ok=False,
            checks=[
                PreflightCheck(
                    name="config valid", status="fail", detail=redact_reason(str(exc))[:240]
                )
            ],
        )

    environment = settings.environment
    hardened = environment == "production"

    # --- config valid --------------------------------------------------------
    config_errors = _config_errors(home)
    checks.append(
        PreflightCheck(
            name="config valid",
            status="pass" if not config_errors else "fail",
            detail="configuration is valid"
            if not config_errors
            else "; ".join(redact_reason(error) for error in config_errors)[:240],
        )
    )

    # --- tokens not default in hardened mode --------------------------------
    api_default = _token_default(
        settings.api.token, KNOWN_DEFAULT_API_TOKENS, PLACEHOLDER_API_TOKENS
    )
    runtime_default = _token_default(
        settings.runtime.token, KNOWN_DEFAULT_RUNTIME_TOKENS, PLACEHOLDER_RUNTIME_TOKENS
    )
    if api_default or runtime_default:
        checks.append(
            PreflightCheck(
                name="tokens hardened",
                # In production, default/placeholder tokens are a hard fail; in
                # development they are an accepted warning.
                status="fail" if hardened else "warning",
                detail="default/placeholder tokens are active"
                + (" (blocked in production)" if hardened else " (acceptable in development)"),
            )
        )
    else:
        checks.append(
            PreflightCheck(name="tokens hardened", status="pass", detail="tokens are configured")
        )

    # --- runtime backend -----------------------------------------------------
    backend = settings.runtime.backend
    if backend == "llama_cpp":
        checks.append(PreflightCheck(name="runtime backend", status="pass", detail="llama_cpp"))
    elif fake:
        checks.append(
            PreflightCheck(
                name="runtime backend",
                status="warning",
                detail=f"backend '{backend}' allowed by --fake (development)",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="runtime backend",
                status="fail",
                detail=f"backend '{backend}' is fake; pass --fake to allow development startup",
            )
        )

    # --- model files present (real mode only) -------------------------------
    checks.append(_model_files_check(settings, home, fake=fake))

    # --- writable paths ------------------------------------------------------
    for name, path in (
        ("database path writable", settings.database_path),
        ("vector index path writable", settings.vector_index_path),
        ("log path writable", settings.logs_path),
        ("report directory writable", home / "data" / "verification"),
    ):
        writable = _path_writable(path)
        checks.append(
            PreflightCheck(
                name=name,
                status="pass" if writable else "fail",
                detail="writable" if writable else "not writable",
            )
        )

    # --- ports available + stale locks --------------------------------------
    run_dir = home / "data" / "run"
    checks.extend(
        _port_and_lock_checks(
            settings=settings,
            run_dir=run_dir,
            port_in_use=port_in_use,
            pid_alive=pid_alive,
        )
    )

    ok = not any(check.status == "fail" for check in checks)
    return PreflightReport(fake=fake, environment=environment, ok=ok, checks=checks)


def _config_errors(home: Path) -> list[str]:
    from april_common.config_validation import validate_configuration

    try:
        return list(validate_configuration(home))
    except Exception as exc:  # pragma: no cover - defensive
        return [str(exc)]


def _model_files_check(settings: AprilSettings, home: Path, *, fake: bool) -> PreflightCheck:
    if fake or settings.runtime.backend != "llama_cpp":
        return PreflightCheck(
            name="model files present",
            status="pass",
            detail="real model files not required in fake/development mode",
        )
    try:
        registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    except ConfigError as exc:
        return PreflightCheck(
            name="model files present", status="fail", detail=redact_reason(str(exc))[:240]
        )
    chat_models = [
        model
        for model in registry.list()
        if model.backend == "llama_cpp" and model.role != "embedding"
    ]
    missing = [model.id for model in chat_models if not model.resolved_path(registry.root).exists()]
    if not chat_models:
        return PreflightCheck(
            name="model files present",
            status="fail",
            detail="no llama_cpp chat models are configured",
        )
    if missing:
        return PreflightCheck(
            name="model files present",
            status="fail",
            detail="missing model files: " + ", ".join(sorted(missing)),
        )
    return PreflightCheck(
        name="model files present",
        status="pass",
        detail=f"{len(chat_models)} configured chat GGUF(s) present",
    )


def _port_and_lock_checks(
    *,
    settings: AprilSettings,
    run_dir: Path,
    port_in_use: PortChecker,
    pid_alive: PidAlive,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    services = (
        ("api", settings.api.host, settings.api.port, run_dir / "api.pid"),
        ("runtime", settings.runtime.host, settings.runtime.port, run_dir / "runtime.pid"),
    )
    stale_locks: list[str] = []
    for name, host, port, pid_path in services:
        pid = _read_pid(pid_path)
        owned_alive = pid is not None and pid_alive(pid)
        in_use = port_in_use(host, port)
        if not in_use:
            checks.append(
                PreflightCheck(
                    name=f"{name} port available", status="pass", detail=f"port {port} is free"
                )
            )
        elif owned_alive:
            checks.append(
                PreflightCheck(
                    name=f"{name} port available",
                    status="pass",
                    detail=f"port {port} held by an already-running APRIL {name}",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name=f"{name} port available",
                    status="fail",
                    detail=f"port {port} is in use by a foreign process",
                )
            )
        # A pid file whose process is gone is a stale lock (auto-cleaned on start).
        if pid is not None and not pid_alive(pid):
            stale_locks.append(f"{name}.pid")
    checks.append(
        PreflightCheck(
            name="no stale lock files",
            status="warning" if stale_locks else "pass",
            detail=(
                "stale pid files will be cleaned on start: " + ", ".join(sorted(stale_locks))
                if stale_locks
                else "no stale pid files"
            ),
        )
    )
    return checks
