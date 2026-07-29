from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from april_common.path_security import deny_sensitive_path
from april_common.process_environment import ProcessCategory

SANDBOX_POLICY_VERSION = "seatbelt-v1"
DEFAULT_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


class SandboxOperation(StrEnum):
    RESTRICTED_COMMAND = "restricted_command"
    CONFIGURED_TEST = "configured_test"
    PATCH = "patch"
    GIT_MUTATION = "git_mutation"
    REPOSITORY_INDEXING = "repository_indexing"
    DOCUMENT_PROCESSING = "document_processing"
    FINETUNE = "finetune"
    MODEL_VERIFICATION = "model_verification"
    BENCHMARKING = "benchmarking"


class NetworkPolicy(StrEnum):
    DENY_ALL = "deny_all"
    LOOPBACK_ONLY = "loopback_only"


class SandboxBackend(StrEnum):
    MACOS_SEATBELT = "macos_seatbelt"
    DEVELOPMENT_OVERRIDE = "development_unsandboxed_override"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    operation: SandboxOperation
    network: NetworkPolicy
    readable_roots: tuple[Path, ...]
    writable_roots: tuple[Path, ...]
    temporary_roots: tuple[Path, ...] = ()
    fail_closed_in_production: bool = True


@dataclass(frozen=True, slots=True)
class SandboxCapabilities:
    backend: SandboxBackend
    network_denial_available: bool
    filesystem_policy_available: bool
    production_fail_closed: bool
    development_override_enabled: bool
    warning: str | None
    policy_version: str = SANDBOX_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class SandboxLaunch:
    argv: tuple[str, ...]
    capabilities: SandboxCapabilities
    profile: str | None


class SandboxProvider(Protocol):
    def capabilities(
        self,
        *,
        environment: str,
        development_override: bool,
    ) -> SandboxCapabilities: ...

    def wrap(
        self,
        argv: Sequence[str],
        *,
        policy: SandboxPolicy,
        environment: str,
        development_override: bool,
    ) -> SandboxLaunch: ...


class SandboxUnavailableError(RuntimeError):
    pass


class HostProcessSandbox:
    """Truthful host sandbox selection with injectable platform discovery."""

    def __init__(
        self,
        *,
        system: str | None = None,
        sandbox_exec: Path = DEFAULT_SANDBOX_EXEC,
    ) -> None:
        self.system = system or platform.system()
        self.sandbox_exec = sandbox_exec

    def capabilities(
        self,
        *,
        environment: str,
        development_override: bool,
    ) -> SandboxCapabilities:
        seatbelt = self.system == "Darwin" and _is_executable(self.sandbox_exec)
        override_active = not seatbelt and environment == "development" and development_override
        if seatbelt:
            return SandboxCapabilities(
                backend=SandboxBackend.MACOS_SEATBELT,
                network_denial_available=True,
                filesystem_policy_available=True,
                production_fail_closed=True,
                development_override_enabled=False,
                warning=None,
            )
        if override_active:
            return SandboxCapabilities(
                backend=SandboxBackend.DEVELOPMENT_OVERRIDE,
                network_denial_available=False,
                filesystem_policy_available=False,
                production_fail_closed=True,
                development_override_enabled=True,
                warning=(
                    "DEVELOPMENT ONLY: restricted subprocesses are running without "
                    "OS-enforced network or filesystem isolation."
                ),
            )
        return SandboxCapabilities(
            backend=SandboxBackend.UNAVAILABLE,
            network_denial_available=False,
            filesystem_policy_available=False,
            production_fail_closed=True,
            development_override_enabled=False,
            warning=(
                "No OS-enforced subprocess sandbox is available; restricted "
                "operations fail closed. Environment filtering is credential "
                "minimization, not network isolation."
            ),
        )

    def wrap(
        self,
        argv: Sequence[str],
        *,
        policy: SandboxPolicy,
        environment: str,
        development_override: bool,
    ) -> SandboxLaunch:
        capabilities = self.capabilities(
            environment=environment,
            development_override=development_override,
        )
        normalized = tuple(argv)
        if capabilities.backend is SandboxBackend.MACOS_SEATBELT:
            profile = generate_seatbelt_profile(policy, executable=normalized[0])
            return SandboxLaunch(
                argv=(str(self.sandbox_exec), "-p", profile, *normalized),
                capabilities=capabilities,
                profile=profile,
            )
        if capabilities.backend is SandboxBackend.DEVELOPMENT_OVERRIDE:
            return SandboxLaunch(
                argv=normalized,
                capabilities=capabilities,
                profile=None,
            )
        if environment == "production" and policy.fail_closed_in_production:
            raise SandboxUnavailableError("sandbox_unavailable_production_fail_closed")
        raise SandboxUnavailableError("sandbox_unavailable")


def operation_policy(
    category: ProcessCategory,
    *,
    project_root: Path,
    allowed_roots: Sequence[Path] = (),
    temporary_roots: Sequence[Path] = (),
) -> SandboxPolicy | None:
    operation = _operation_for_category(category)
    if operation is None:
        return None
    roots = _normalize_roots((project_root, *allowed_roots))
    temporary = _normalize_roots(temporary_roots)
    network = (
        NetworkPolicy.LOOPBACK_ONLY
        if operation
        in {
            SandboxOperation.MODEL_VERIFICATION,
            SandboxOperation.BENCHMARKING,
        }
        else NetworkPolicy.DENY_ALL
    )
    writable = roots
    if operation in {
        SandboxOperation.REPOSITORY_INDEXING,
        SandboxOperation.DOCUMENT_PROCESSING,
        SandboxOperation.MODEL_VERIFICATION,
        SandboxOperation.BENCHMARKING,
    }:
        writable = _normalize_roots((project_root, *temporary))
    return SandboxPolicy(
        operation=operation,
        network=network,
        readable_roots=roots,
        writable_roots=writable,
        temporary_roots=temporary,
    )


def sandbox_capabilities(
    *,
    environment: str,
    development_override: bool = False,
    provider: SandboxProvider | None = None,
) -> SandboxCapabilities:
    selected = provider or HostProcessSandbox()
    return selected.capabilities(
        environment=environment,
        development_override=development_override,
    )


def generate_seatbelt_profile(
    policy: SandboxPolicy,
    *,
    executable: str,
) -> str:
    """Generate an argv-safe Seatbelt profile without mutable profile files."""
    executable_path = _resolve_executable(executable)
    readable = _normalize_roots((*policy.readable_roots, *policy.temporary_roots))
    writable = _normalize_roots((*policy.writable_roots, *policy.temporary_roots))
    system_read_paths = (
        Path("/System"),
        Path("/usr/lib"),
        Path("/usr/share"),
        Path("/Library/Apple/System/Library"),
        Path("/private/var/db/timezone"),
        Path("/dev/null"),
        Path("/dev/urandom"),
        executable_path,
        executable_path.parent,
        executable_path.parent.parent,
    )
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow file-read-metadata)",
    ]
    for path in _normalize_roots(system_read_paths):
        lines.append(_seatbelt_rule("allow", "file-read*", path))
    for path in readable:
        lines.append(_seatbelt_rule("allow", "file-read*", path))
    for path in writable:
        lines.append(_seatbelt_rule("allow", "file-write*", path))
    if policy.network is NetworkPolicy.LOOPBACK_ONLY:
        lines.extend(
            [
                '(allow network-outbound (remote ip "localhost:*"))',
                '(allow network-inbound (local ip "localhost:*"))',
            ]
        )
    else:
        lines.append("(deny network*)")
    for path in _sensitive_paths((*readable, *writable)):
        lines.append(_seatbelt_rule("deny", "file-read*", path))
        lines.append(_seatbelt_rule("deny", "file-write*", path))
    return "\n".join(lines)


def _operation_for_category(category: ProcessCategory) -> SandboxOperation | None:
    return {
        ProcessCategory.RESTRICTED_COMMAND: SandboxOperation.RESTRICTED_COMMAND,
        ProcessCategory.TEST_RUNNER: SandboxOperation.CONFIGURED_TEST,
        ProcessCategory.GIT: SandboxOperation.GIT_MUTATION,
        ProcessCategory.REPOSITORY_INDEXING: SandboxOperation.REPOSITORY_INDEXING,
        ProcessCategory.DOCUMENT_PROCESSING: SandboxOperation.DOCUMENT_PROCESSING,
        ProcessCategory.FINETUNE: SandboxOperation.FINETUNE,
        ProcessCategory.MODEL_VERIFICATION: SandboxOperation.MODEL_VERIFICATION,
        ProcessCategory.BENCHMARKING: SandboxOperation.BENCHMARKING,
    }.get(category)


def _normalize_roots(values: Sequence[Path]) -> tuple[Path, ...]:
    unique: dict[str, Path] = {}
    for value in values:
        path = value.expanduser().resolve(strict=False)
        deny_sensitive_path(path)
        unique[str(path)] = path
    return tuple(unique[key] for key in sorted(unique))


def _sensitive_paths(roots: Sequence[Path]) -> tuple[Path, ...]:
    relative = (
        Path(".ssh"),
        Path(".aws"),
        Path(".azure"),
        Path(".gnupg"),
        Path(".config/gcloud"),
        Path(".env"),
        Path(".netrc"),
        Path("Library/Keychains"),
        Path("Library/Application Support/Google/Chrome"),
        Path("Library/Application Support/Firefox"),
    )
    return _normalize_sensitive(root / item for root in roots for item in relative)


def _normalize_sensitive(values: Iterable[Path]) -> tuple[Path, ...]:
    unique = {str(path.resolve(strict=False)): path.resolve(strict=False) for path in values}
    return tuple(unique[key] for key in sorted(unique))


def _resolve_executable(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.expanduser().resolve(strict=False)
    located = shutil.which(value)
    if located is None:
        return candidate.resolve(strict=False)
    return Path(located).resolve(strict=False)


def _seatbelt_rule(action: str, operation: str, path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    filter_name = "literal" if path.is_file() else "subpath"
    return f'({action} {operation} ({filter_name} "{escaped}"))'


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
