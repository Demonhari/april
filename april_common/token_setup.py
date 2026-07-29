from __future__ import annotations

import os
import secrets
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from april_common.credentials import (
    CredentialKey,
    CredentialStore,
    CredentialStoreError,
    FileCredentialStore,
    generate_credential_token,
)
from april_common.errors import AprilError

if TYPE_CHECKING:
    from april_common.audit import AuditLogger

_LEGACY_KEYS = {
    "APRIL_API_TOKEN": CredentialKey.API_TOKEN,
    "APRIL_RUNTIME_TOKEN": CredentialKey.RUNTIME_TOKEN,
}


@dataclass(frozen=True, slots=True)
class GeneratedTokens:
    api_token: str
    runtime_token: str


@dataclass(frozen=True, slots=True)
class MigrationResult:
    status: str
    migrated: tuple[str, ...]
    source: str | None
    store: str


@dataclass(frozen=True, slots=True)
class RotationResult:
    rotated: tuple[str, ...]
    restart_services: tuple[str, ...]
    store: str


def generate_tokens() -> GeneratedTokens:
    return GeneratedTokens(
        api_token=generate_credential_token(),
        runtime_token=generate_credential_token(),
    )


def provision_credentials(store: CredentialStore) -> RotationResult:
    """Create both service credentials as one rollback-capable operation."""
    previous = {
        key: store.get(key) for key in (CredentialKey.API_TOKEN, CredentialKey.RUNTIME_TOKEN)
    }
    generated = {
        key: generate_credential_token()
        for key in (CredentialKey.API_TOKEN, CredentialKey.RUNTIME_TOKEN)
    }
    changed: list[CredentialKey] = []
    try:
        for key, value in generated.items():
            store.set(key, value)
            if not secrets.compare_digest(store.get(key) or "", value):
                raise CredentialStoreError("Credential setup read-back verification failed.")
            changed.append(key)
    except Exception:
        for key in changed:
            old = previous[key]
            if old is None:
                store.delete(key)
            else:
                store.set(key, old)
        raise
    return RotationResult(
        rotated=(CredentialKey.API_TOKEN.value, CredentialKey.RUNTIME_TOKEN.value),
        restart_services=("April Runtime", "Core API"),
        store=store.backend_name,
    )


def write_credential_store_reference(path: Path, store: CredentialStore) -> None:
    """Write non-secret backend identifiers so child processes resolve the store."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    names = {
        "APRIL_CREDENTIAL_STORE",
        "APRIL_CREDENTIAL_FILE_PATH",
        "APRIL_API_CREDENTIAL_ID",
        "APRIL_RUNTIME_CREDENTIAL_ID",
        "APRIL_AUDIT_ANCHOR_CREDENTIAL_ID",
        *_LEGACY_KEYS,
    }
    retained = [
        line
        for line in lines
        if line.strip().removeprefix("export ").partition("=")[0].strip() not in names
    ]
    backend = store.backend_name.replace("macos-", "")
    retained.extend(
        [
            f"APRIL_CREDENTIAL_STORE={backend}",
            f"APRIL_API_CREDENTIAL_ID={CredentialKey.API_TOKEN.value}",
            f"APRIL_RUNTIME_CREDENTIAL_ID={CredentialKey.RUNTIME_TOKEN.value}",
            f"APRIL_AUDIT_ANCHOR_CREDENTIAL_ID={CredentialKey.AUDIT_ANCHOR.value}",
        ]
    )
    if isinstance(store, FileCredentialStore):
        retained.append(f"APRIL_CREDENTIAL_FILE_PATH={store.path}")
    _atomic_write_text(path, "\n".join(retained) + "\n", mode=0o600)


def write_token_env_file(path: Path, tokens: GeneratedTokens) -> None:
    """Legacy development helper retained solely for explicit migration tests.

    New setup paths write to a CredentialStore. This function remains so older
    callers can construct legacy input without using unsafe file primitives.
    """
    path = path.expanduser()
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replacements = {
        "APRIL_API_TOKEN": tokens.api_token,
        "APRIL_RUNTIME_TOKEN": tokens.runtime_token,
    }
    written: set[str] = set()
    lines: list[str] = []
    for line in existing:
        key, separator, _value = line.partition("=")
        if separator and key in replacements:
            lines.append(f"{key}={replacements[key]}")
            written.add(key)
        else:
            lines.append(line)
    for key, value in replacements.items():
        if key not in written:
            lines.append(f"{key}={value}")
    _atomic_write_text(path, "\n".join(lines) + "\n", mode=0o600)


def migrate_legacy_credentials(
    *,
    home: Path,
    store: CredentialStore,
    env_file: Path | None = None,
    config_file: Path | None = None,
    legacy_audit_anchor_file: Path | None = None,
) -> MigrationResult:
    """Move legacy tokens only after store write and read-back verification."""
    root = home.expanduser().resolve()
    dotenv_path = (env_file or root / ".env").expanduser().resolve(strict=False)
    yaml_path = (config_file or root / "configs" / "april.yaml").expanduser().resolve(strict=False)
    dotenv_tokens = _legacy_dotenv_tokens(dotenv_path)
    yaml_tokens = _legacy_yaml_tokens(yaml_path)
    tokens = dict(yaml_tokens)
    tokens.update(dotenv_tokens)
    anchor_path = (
        legacy_audit_anchor_file.expanduser().resolve(strict=False)
        if legacy_audit_anchor_file is not None
        else root / "logs" / "audit.jsonl.anchor"
    )
    anchor_value = _legacy_anchor_value(anchor_path)
    values = dict(tokens)
    if anchor_value is not None:
        values[CredentialKey.AUDIT_ANCHOR] = anchor_value
    if not values:
        existing = tuple(
            key.value
            for key in (CredentialKey.API_TOKEN, CredentialKey.RUNTIME_TOKEN)
            if store.exists(key)
        )
        return MigrationResult(
            status="already_migrated" if existing else "no_legacy_credentials",
            migrated=(),
            source=None,
            store=store.backend_name,
        )

    previous = {key: store.get(key) for key in values}
    rollback_files: dict[Path, Path] = {}
    source = dotenv_path if dotenv_tokens else yaml_path if yaml_tokens else anchor_path
    try:
        for key, value in values.items():
            store.set(key, value)
        for key, value in values.items():
            stored = store.get(key)
            if stored is None or not secrets.compare_digest(stored, value):
                raise CredentialStoreError("Credential migration read-back verification failed.")

        if dotenv_tokens:
            rollback_files[dotenv_path] = _rollback_copy(dotenv_path)
            _sanitize_dotenv(dotenv_path, store)
        if yaml_tokens:
            rollback_files[yaml_path] = _rollback_copy(yaml_path)
            _sanitize_yaml(yaml_path, store.backend_name)
        if anchor_value is not None:
            anchor_path.unlink()
    except Exception:
        for key, previous_value in previous.items():
            if previous_value is None:
                store.delete(key)
            else:
                store.set(key, previous_value)
        for original, rollback in rollback_files.items():
            if rollback.exists():
                _atomic_write_text(
                    original,
                    rollback.read_text(encoding="utf-8"),
                    mode=0o600,
                )
        raise
    finally:
        for rollback in rollback_files.values():
            rollback.unlink(missing_ok=True)

    return MigrationResult(
        status="migrated",
        migrated=tuple(sorted(key.value for key in values)),
        source=source.name,
        store=store.backend_name,
    )


def rotate_credentials(
    *,
    store: CredentialStore,
    rotate_api: bool,
    rotate_runtime: bool,
    audit: AuditLogger | None = None,
    commit_callback: Callable[[], None] | None = None,
) -> RotationResult:
    selected: list[CredentialKey] = []
    if rotate_api:
        selected.append(CredentialKey.API_TOKEN)
    if rotate_runtime:
        selected.append(CredentialKey.RUNTIME_TOKEN)
    if not selected:
        raise CredentialStoreError("Select --api, --runtime, or --all.")
    missing = [key.value for key in selected if not store.exists(key)]
    if missing:
        raise CredentialStoreError(
            "Rotation refused because one or more selected credentials are unavailable."
        )

    previous = {key: store.get(key) for key in selected}
    generated = {key: generate_credential_token() for key in selected}
    committed: list[CredentialKey] = []
    try:
        for key in selected:
            store.rotate(key, generated[key])
            if not secrets.compare_digest(store.get(key) or "", generated[key]):
                raise CredentialStoreError("Credential rotation read-back verification failed.")
            committed.append(key)
        if commit_callback is not None:
            commit_callback()
        if audit is not None:
            audit.write(
                {
                    "event_type": "credential_rotation_succeeded",
                    "credentials": [key.value for key in selected],
                    "store": store.backend_name,
                }
            )
    except Exception:
        for key in committed:
            old = previous[key]
            if old is None:
                store.delete(key)
            else:
                store.set(key, old)
        if audit is not None:
            with suppress(AprilError):
                audit.write(
                    {
                        "event_type": "credential_rotation_failed",
                        "credentials": [key.value for key in selected],
                        "store": store.backend_name,
                    }
                )
        raise

    restarts: list[str] = []
    if CredentialKey.API_TOKEN in selected:
        restarts.append("Core API")
    if CredentialKey.RUNTIME_TOKEN in selected:
        restarts.extend(["April Runtime", "Core API"])
    return RotationResult(
        rotated=tuple(key.value for key in selected),
        restart_services=tuple(dict.fromkeys(restarts)),
        store=store.backend_name,
    )


def legacy_plaintext_credentials_detected(
    home: Path,
    *,
    env_file: Path | None = None,
    config_file: Path | None = None,
) -> bool:
    root = home.expanduser().resolve()
    return bool(
        _legacy_dotenv_tokens(env_file or root / ".env")
        or _legacy_yaml_tokens(config_file or root / "configs" / "april.yaml")
    )


def _legacy_dotenv_tokens(path: Path) -> dict[CredentialKey, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return {}
    values: dict[CredentialKey, str] = {}
    for raw in lines:
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        key = _LEGACY_KEYS.get(name.strip())
        parsed = value.strip().strip("'\"") if separator else ""
        if key is not None and parsed:
            values[key] = parsed
    return values


def _legacy_yaml_tokens(path: Path) -> dict[CredentialKey, str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    if not isinstance(payload, dict):
        return {}
    values: dict[CredentialKey, str] = {}
    for section, key in (
        ("api", CredentialKey.API_TOKEN),
        ("runtime", CredentialKey.RUNTIME_TOKEN),
    ):
        section_payload = payload.get(section)
        value = section_payload.get("token") if isinstance(section_payload, dict) else None
        if isinstance(value, str) and value:
            values[key] = value
    return values


def _legacy_anchor_value(path: Path) -> str | None:
    try:
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise CredentialStoreError("Legacy audit anchor permissions are not owner-only.")
        return path.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise CredentialStoreError("Legacy audit anchor is unreadable.") from exc


def _sanitize_dotenv(path: Path, store: CredentialStore) -> None:
    write_credential_store_reference(path, store)


def _sanitize_yaml(path: Path, backend: str) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise CredentialStoreError("Legacy configuration is not a mapping.")
    for section in ("api", "runtime"):
        section_payload = payload.get(section)
        if isinstance(section_payload, dict):
            section_payload.pop("token", None)
    security = payload.setdefault("security", {})
    if not isinstance(security, dict):
        raise CredentialStoreError("Security configuration is not a mapping.")
    security.update(
        {
            "credential_store": backend.replace("macos-", ""),
            "api_credential_id": CredentialKey.API_TOKEN.value,
            "runtime_credential_id": CredentialKey.RUNTIME_TOKEN.value,
            "audit_anchor_credential_id": CredentialKey.AUDIT_ANCHOR.value,
        }
    )
    _atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False), mode=0o600)


def _rollback_copy(path: Path) -> Path:
    rollback = path.with_name(f".{path.name}.april-credentials-rollback")
    _atomic_write_text(rollback, path.read_text(encoding="utf-8"), mode=0o600)
    return rollback


def _atomic_write_text(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
