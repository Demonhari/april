from __future__ import annotations

import platform
from dataclasses import dataclass

from apps.runner.readiness_models import CheckStatus, ReadinessCheck
from april_common.audit import audit_logger_for_settings, audit_startup_decision
from april_common.credentials import CredentialStore, CredentialStoreError
from april_common.errors import AprilError
from april_common.process_sandbox import SandboxBackend, sandbox_capabilities
from april_common.settings import (
    KNOWN_DEFAULT_API_TOKENS,
    KNOWN_DEFAULT_RUNTIME_TOKENS,
    PLACEHOLDER_API_TOKENS,
    PLACEHOLDER_RUNTIME_TOKENS,
    AprilSettings,
)
from april_common.token_setup import legacy_plaintext_credentials_detected

_OFFLINE_AUDIT_MAX_BYTES = 4 * 1024 * 1024


def _token_status(value: str | None, defaults: set[str], placeholders: set[str]) -> str:
    if not value:
        return "missing"
    if value in placeholders:
        return "placeholder-insecure"
    if value in defaults:
        return "default-development"
    return "configured"


def _audit_readiness_check(
    settings: AprilSettings,
    *,
    credential_store: CredentialStore | None,
) -> tuple[str, list[str], int, bool, ReadinessCheck]:
    """Verify a bounded audit log without exposing payloads or paths."""
    try:
        audit_size = settings.audit_path.stat().st_size if settings.audit_path.exists() else 0
        if audit_size > _OFFLINE_AUDIT_MAX_BYTES:
            status = "unverified_size_limit"
            issue_codes = ["verification_skipped_size_limit"]
            issue_details = issue_codes
            record_count = 0
            verification_required = True
        else:
            decision = audit_startup_decision(
                settings,
                credential_store=credential_store,
                logger_factory=audit_logger_for_settings,
            )
            status = decision.status
            issue_details = list(decision.issue_lines)
            issue_codes = list(decision.issue_codes)
            record_count = decision.record_count
            verification_required = status == "unavailable"
    except (AprilError, CredentialStoreError, OSError, RuntimeError):
        status = "unavailable"
        issue_codes = ["verification_unavailable"]
        issue_details = issue_codes
        record_count = 0
        verification_required = True

    if status in {"valid", "anchor_lagged"}:
        check_status: CheckStatus = "ok"
        action = None
    else:
        check_status = "blocker"
        action = "run april audit verify --json"
        if status == "corrupt":
            action += (
                '; then review with run april audit recover --reason "owner-reviewed recovery"'
            )
    if status == "corrupt":
        detail = "corrupt; historical records remain unverified"
    elif status == "unavailable":
        detail = "unavailable; verification could not be completed"
    elif status == "unverified_size_limit":
        detail = (
            "unverified; offline inspection skipped this audit because it exceeds the 4 MiB bound"
        )
    else:
        detail = status
    if issue_details:
        detail += "; issues=" + ",".join(issue_details)[:180]
    return (
        status,
        issue_codes,
        record_count,
        verification_required,
        ReadinessCheck(name="audit chain", status=check_status, detail=detail, action=action),
    )


@dataclass(frozen=True)
class RuntimeSecurityReadiness:
    api_status: str
    runtime_status: str
    credential_store_selected: str
    legacy_plaintext: bool


def _append_runtime_security_checks(
    settings: AprilSettings,
    checks: list[ReadinessCheck],
) -> RuntimeSecurityReadiness:
    """Append token, credential-store, and Tool Worker sandbox checks."""
    api_status = _token_status(
        settings.api.token,
        KNOWN_DEFAULT_API_TOKENS,
        PLACEHOLDER_API_TOKENS,
    )
    runtime_status = _token_status(
        settings.runtime.token,
        KNOWN_DEFAULT_RUNTIME_TOKENS,
        PLACEHOLDER_RUNTIME_TOKENS,
    )
    token_statuses = {api_status, runtime_status}
    if "placeholder-insecure" in token_statuses:
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="Insecure placeholder tokens from .env.example are still active.",
                action="run april setup tokens",
            )
        )
    elif "default-development" in token_statuses:
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="Default development tokens are still active.",
                action="run april setup tokens",
            )
        )
    elif "missing" in token_statuses:
        checks.append(
            ReadinessCheck(
                name="api/runtime tokens",
                status="warning",
                detail="A loopback token is not configured.",
                action="run april setup tokens",
            )
        )
    else:
        checks.append(ReadinessCheck(name="api/runtime tokens", status="ok", detail="configured"))

    credential_store_selected: str = settings.security.credential_store
    if credential_store_selected == "auto":
        credential_store_selected = (
            "macos-keychain"
            if settings.environment == "production" and platform.system() == "Darwin"
            else "legacy-development-default"
        )
    legacy_plaintext = legacy_plaintext_credentials_detected(settings.home)
    checks.append(
        ReadinessCheck(
            name="credential store",
            status="warning" if legacy_plaintext else "ok",
            detail=(
                f"{credential_store_selected}; legacy plaintext credential detected"
                if legacy_plaintext
                else credential_store_selected
            ),
            action=("run april security credentials migrate" if legacy_plaintext else None),
        )
    )

    sandbox = sandbox_capabilities(
        environment=settings.environment,
        development_override=settings.workers.development_unsandboxed_override,
    )
    sandbox_status: CheckStatus = "ok"
    if sandbox.backend is SandboxBackend.UNAVAILABLE:
        sandbox_status = "blocker" if settings.environment == "production" else "warning"
    elif sandbox.development_override_enabled:
        sandbox_status = "warning"
    checks.append(
        ReadinessCheck(
            name="Tool Worker OS sandbox",
            status=sandbox_status,
            detail=(
                sandbox.warning
                or f"{sandbox.backend.value}; network denial and filesystem policy are OS-enforced"
            ),
            action=(
                "Run APRIL on macOS with /usr/bin/sandbox-exec available."
                if sandbox.backend is SandboxBackend.UNAVAILABLE
                else None
            ),
        )
    )
    return RuntimeSecurityReadiness(
        api_status=api_status,
        runtime_status=runtime_status,
        credential_store_selected=credential_store_selected,
        legacy_plaintext=legacy_plaintext,
    )
