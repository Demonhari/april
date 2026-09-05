from __future__ import annotations

import platform
from pathlib import Path

from apps.runner.verification.types import VerifyCheck, VerifyStatus
from april_common.audit import audit_logger_for_settings
from april_common.credentials import CredentialStore
from april_common.errors import ConfigError
from april_common.process_sandbox import SandboxBackend, sandbox_capabilities
from april_common.settings import load_settings
from april_common.token_setup import legacy_plaintext_credentials_detected
from services.evolution.rollouts import inspect_rollout_state
from services.memory.maintenance import check_database


def run_local_sandbox_verification(home: Path) -> list[VerifyCheck]:
    try:
        settings = load_settings(root=home)
    except (ConfigError, RuntimeError) as exc:
        return [
            VerifyCheck(
                name="Tool Worker sandbox capability",
                ok=False,
                detail=f"unavailable ({type(exc).__name__})",
            )
        ]
    report = sandbox_capabilities(
        environment=settings.environment,
        development_override=settings.workers.development_unsandboxed_override,
    )
    backend_available = report.backend is not SandboxBackend.UNAVAILABLE
    production = settings.environment == "production"
    unavailable_status: VerifyStatus = "fail" if production else "skip"
    return [
        VerifyCheck(
            "sandbox backend",
            backend_available or not production,
            report.backend.value,
            status="pass" if backend_available else unavailable_status,
        ),
        VerifyCheck(
            "sandbox network denial",
            report.network_denial_available or not production,
            "available" if report.network_denial_available else "unavailable",
            status="pass" if report.network_denial_available else unavailable_status,
        ),
        VerifyCheck(
            "sandbox filesystem policy",
            report.filesystem_policy_available or not production,
            "available" if report.filesystem_policy_available else "unavailable",
            status="pass" if report.filesystem_policy_available else unavailable_status,
        ),
        VerifyCheck(
            "sandbox production fail closed",
            report.production_fail_closed,
            "enabled" if report.production_fail_closed else "disabled",
        ),
        VerifyCheck(
            "sandbox development override",
            not report.development_override_enabled or not production,
            report.warning or "disabled",
            status="skip" if report.development_override_enabled else "pass",
        ),
    ]


def run_local_security_integrity_verification(
    home: Path, *, credential_store: CredentialStore | None = None
) -> list[VerifyCheck]:
    """Run redacted local security checks without exposing credential values."""
    try:
        settings = load_settings(root=home, credential_store=credential_store)
    except (ConfigError, RuntimeError) as exc:
        return [
            VerifyCheck(
                name="security configuration",
                ok=False,
                detail=f"unavailable ({type(exc).__name__})",
            )
        ]
    store_name: str = settings.security.credential_store
    if store_name == "auto":
        store_name = (
            "macos-keychain"
            if settings.environment == "production" and platform.system() == "Darwin"
            else "legacy-development-default"
        )
    legacy = legacy_plaintext_credentials_detected(settings.home)
    audit_result = audit_logger_for_settings(settings, credential_store=credential_store).verify()
    database = check_database(settings.database_path, home=settings.home, full=False)
    backup = database.last_successful_backup
    backup_detail = str(backup.get("creation_timestamp", "known")) if backup else "none recorded"
    rollout_state = inspect_rollout_state(settings)
    rollout_safe = rollout_state["status"] in {"disabled", "ok", "not_initialized"}
    return [
        *run_local_sandbox_verification(home),
        VerifyCheck("credential store selected", True, store_name),
        VerifyCheck(
            "API credential available",
            bool(settings.api.token),
            "available" if settings.api.token else "missing",
        ),
        VerifyCheck(
            "Runtime credential available",
            bool(settings.runtime.token),
            "available" if settings.runtime.token else "missing",
        ),
        VerifyCheck(
            "legacy plaintext credential",
            not legacy,
            "detected; run security credentials migrate" if legacy else "not detected",
        ),
        VerifyCheck("audit chain", audit_result.valid, audit_result.status),
        VerifyCheck("database quick_check", database.quick_check == "ok", database.quick_check),
        VerifyCheck(
            "database foreign keys",
            database.foreign_key_consistent,
            (
                "ok"
                if database.foreign_key_consistent
                else f"{database.foreign_key_violations} violation(s)"
            ),
        ),
        VerifyCheck("database WAL state", database.journal_mode == "wal", database.journal_mode),
        VerifyCheck("last successful backup", True, backup_detail),
        VerifyCheck(
            "evolution rollout safety",
            rollout_safe,
            (
                "disabled by configuration"
                if rollout_state["status"] == "disabled"
                else str(rollout_state["status"])
            ),
        ),
    ]
