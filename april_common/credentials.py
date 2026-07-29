from __future__ import annotations

import json
import os
import platform
import secrets
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from april_common.errors import ConfigError

KEYCHAIN_EXECUTABLE = Path("/usr/bin/security")
KEYCHAIN_SERVICE = "com.april.local-assistant.credentials"
KEYCHAIN_ACCOUNTS: dict[CredentialKey, str]
_FILE_FORMAT_VERSION = 1


class CredentialKey(StrEnum):
    API_TOKEN = "core-api-token"
    RUNTIME_TOKEN = "runtime-auth-token"
    AUDIT_ANCHOR = "audit-terminal-anchor"


KEYCHAIN_ACCOUNTS = {
    CredentialKey.API_TOKEN: "april.core-api",
    CredentialKey.RUNTIME_TOKEN: "april.runtime",
    CredentialKey.AUDIT_ANCHOR: "april.audit-anchor",
}


class CredentialStoreError(RuntimeError):
    """A deliberately redacted credential-store failure."""


class CredentialStore(Protocol):
    @property
    def backend_name(self) -> str: ...

    def get(self, key: CredentialKey) -> str | None: ...

    def set(self, key: CredentialKey, value: str) -> None: ...

    def delete(self, key: CredentialKey) -> None: ...

    def exists(self, key: CredentialKey) -> bool: ...

    def rotate(self, key: CredentialKey, value: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[
    [Sequence[str], Mapping[str, str], float],
    CommandResult,
]


def _run_keychain_command(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class MacOSKeychainCredentialStore:
    """Small, testable adapter around the macOS generic-password CLI."""

    def __init__(
        self,
        *,
        runner: CommandRunner = _run_keychain_command,
        executable: Path = KEYCHAIN_EXECUTABLE,
        service: str = KEYCHAIN_SERVICE,
        timeout_seconds: float = 5.0,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._runner = runner
        self._executable = executable
        self._service = service
        self._timeout_seconds = max(0.1, min(timeout_seconds, 30.0))
        self._environment = dict(environment or _minimal_keychain_environment())

    @property
    def backend_name(self) -> str:
        return "macos-keychain"

    def get(self, key: CredentialKey) -> str | None:
        result = self._execute(
            (
                str(self._executable),
                "find-generic-password",
                "-s",
                self._service,
                "-a",
                KEYCHAIN_ACCOUNTS[key],
                "-w",
            ),
            allow_missing=True,
        )
        if result is None:
            return None
        return result.stdout.rstrip("\r\n")

    def set(self, key: CredentialKey, value: str) -> None:
        _validate_credential_value(value)
        self._execute(
            (
                str(self._executable),
                "add-generic-password",
                "-U",
                "-s",
                self._service,
                "-a",
                KEYCHAIN_ACCOUNTS[key],
                "-w",
                value,
            )
        )

    def delete(self, key: CredentialKey) -> None:
        self._execute(
            (
                str(self._executable),
                "delete-generic-password",
                "-s",
                self._service,
                "-a",
                KEYCHAIN_ACCOUNTS[key],
            ),
            allow_missing=True,
        )

    def exists(self, key: CredentialKey) -> bool:
        return self.get(key) is not None

    def rotate(self, key: CredentialKey, value: str) -> str | None:
        previous = self.get(key)
        try:
            self.set(key, value)
            if not secrets.compare_digest(self.get(key) or "", value):
                raise CredentialStoreError("Credential store read-back verification failed.")
        except Exception:
            if previous is None:
                self.delete(key)
            else:
                self.set(key, previous)
            raise
        return previous

    def _execute(
        self,
        argv: Sequence[str],
        *,
        allow_missing: bool = False,
    ) -> CommandResult | None:
        if self._runner is _run_keychain_command and not self._executable.is_file():
            raise CredentialStoreError("macOS Keychain is unavailable.")
        try:
            result = self._runner(argv, self._environment, self._timeout_seconds)
        except (OSError, subprocess.SubprocessError) as exc:
            raise CredentialStoreError(
                f"macOS Keychain command failed ({type(exc).__name__})."
            ) from None
        if result.returncode == 0:
            return result
        # `security` uses 44 for an item that does not exist. Some macOS
        # releases return the signed OSStatus form instead.
        if allow_missing and result.returncode in {44, -25300}:
            return None
        raise CredentialStoreError(
            f"macOS Keychain command failed (exit {result.returncode}); output redacted."
        )


def _minimal_keychain_environment() -> dict[str, str]:
    environment = {"LANG": "C", "LC_ALL": "C"}
    home = os.environ.get("HOME")
    if home:
        environment["HOME"] = home
    return environment


class InMemoryCredentialStore:
    def __init__(self, initial: Mapping[CredentialKey, str] | None = None) -> None:
        self._values = dict(initial or {})

    @property
    def backend_name(self) -> str:
        return "memory"

    def get(self, key: CredentialKey) -> str | None:
        return self._values.get(key)

    def set(self, key: CredentialKey, value: str) -> None:
        _validate_credential_value(value)
        self._values[key] = value

    def delete(self, key: CredentialKey) -> None:
        self._values.pop(key, None)

    def exists(self, key: CredentialKey) -> bool:
        return key in self._values

    def rotate(self, key: CredentialKey, value: str) -> str | None:
        previous = self.get(key)
        self.set(key, value)
        return previous


class FileCredentialStore:
    """Explicit development fallback stored outside the APRIL repository."""

    def __init__(self, path: Path, *, repository_root: Path) -> None:
        self.path = path.expanduser().resolve(strict=False)
        self.repository_root = repository_root.expanduser().resolve(strict=False)
        if self.path == self.repository_root or self.path.is_relative_to(self.repository_root):
            raise CredentialStoreError(
                "File credential store must be located outside the APRIL repository."
            )

    @property
    def backend_name(self) -> str:
        return "file"

    def get(self, key: CredentialKey) -> str | None:
        return self._read().get(key.value)

    def set(self, key: CredentialKey, value: str) -> None:
        _validate_credential_value(value)
        values = self._read()
        values[key.value] = value
        self._write(values)

    def delete(self, key: CredentialKey) -> None:
        values = self._read()
        if values.pop(key.value, None) is not None:
            self._write(values)

    def exists(self, key: CredentialKey) -> bool:
        return self.get(key) is not None

    def rotate(self, key: CredentialKey, value: str) -> str | None:
        previous = self.get(key)
        self.set(key, value)
        if not secrets.compare_digest(self.get(key) or "", value):
            if previous is None:
                self.delete(key)
            else:
                self.set(key, previous)
            raise CredentialStoreError("Credential store read-back verification failed.")
        return previous

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode & 0o077:
            raise CredentialStoreError(
                "Credential file permissions are insecure; expected owner-only access."
            )
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CredentialStoreError("Credential file is unreadable or malformed.") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != _FILE_FORMAT_VERSION
            or not isinstance(payload.get("credentials"), dict)
        ):
            raise CredentialStoreError("Credential file has an unsupported format.")
        credentials = payload["credentials"]
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in credentials.items()
        ):
            raise CredentialStoreError("Credential file contains invalid entries.")
        return dict(credentials)

    def _write(self, values: Mapping[str, str]) -> None:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format_version": _FILE_FORMAT_VERSION,
                        "credentials": dict(values),
                    },
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def select_credential_store(
    *,
    backend: str,
    environment: str,
    repository_root: Path,
    file_path: Path | None = None,
    keychain_runner: CommandRunner = _run_keychain_command,
) -> CredentialStore:
    selected = backend
    if selected == "auto":
        if platform.system() == "Darwin":
            selected = "keychain"
        elif environment == "test":
            selected = "memory"
        else:
            raise ConfigError(
                "No credential store was selected. Configure an explicit development "
                "file store outside the repository on non-macOS systems."
            )
    if selected == "keychain":
        if platform.system() != "Darwin" and keychain_runner is _run_keychain_command:
            raise CredentialStoreError("macOS Keychain is unavailable on this platform.")
        store = MacOSKeychainCredentialStore(runner=keychain_runner)
        if environment == "production":
            # A harmless lookup proves the adapter is callable. Missing items are
            # handled later as unavailable credentials.
            store.exists(CredentialKey.API_TOKEN)
        return store
    if selected == "memory":
        if environment != "test":
            raise ConfigError("The in-memory credential store is allowed only in tests.")
        return InMemoryCredentialStore()
    if selected == "file":
        if environment == "production":
            raise ConfigError("The file credential store is not allowed in production.")
        if file_path is None:
            raise ConfigError("An explicit credential file path is required.")
        return FileCredentialStore(file_path, repository_root=repository_root)
    raise ConfigError(f"Unsupported credential store backend: {selected}")


def generate_credential_token() -> str:
    return secrets.token_urlsafe(32)


def _validate_credential_value(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise CredentialStoreError("Credential values must be non-empty strings.")
    if "\x00" in value:
        raise CredentialStoreError("Credential values cannot contain NUL bytes.")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
