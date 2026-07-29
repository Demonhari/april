from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from april_common.path_security import is_path_within_roots
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import (
    ModelDefinition,
    ModelRegistry,
    UniqueKeyLoader,
)

COPY_CHUNK_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL_JOURNAL_STATUSES = frozenset({"completed", "failed", "cancelled", "reconciled"})

Progress = Callable[[int, str], Awaitable[None]]


class ModelImportError(RuntimeError):
    pass


class ModelImportService:
    def __init__(
        self,
        settings: AprilSettings,
        *,
        atomic_writer: Callable[[Path, bytes], None] | None = None,
    ) -> None:
        self.settings = settings
        self.home = settings.home.resolve()
        self.models_dir = self.home / "models"
        self.config_path = self.home / "configs" / "models.yaml"
        self.state_dir = self.home / "data" / "model-import"
        self.journal_dir = self.state_dir / "journals"
        self.staging_dir = self.home / ".april_tmp" / "model-import"
        self.lock_path = self.state_dir / "models-config.lock"
        self._atomic_writer = atomic_writer or _atomic_write_bytes

    def prepare_payload(
        self,
        *,
        source_path: str,
        model_id: str,
        role: str,
        name: str,
        expected_sha256: str | None = None,
        requested_verification: bool = False,
    ) -> dict[str, Any]:
        """Validate and bind an approval payload without mutating model state."""
        source = self._validate_source(source_path)
        self._validate_identifiers(model_id=model_id, role=role, name=name)
        info = source.stat()
        destination = self.models_dir / source.name
        return {
            "source_path": str(source),
            "model_id": model_id,
            "role": role,
            "name": name,
            "expected_sha256": (
                _validate_expected_sha(expected_sha256)
                if expected_sha256 is not None
                else _sha256_file(source)
            ),
            "source_identity": {
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
                "size": int(info.st_size),
                "modified_ns": int(info.st_mtime_ns),
            },
            "format": "gguf",
            "destination": str(destination.relative_to(self.home)),
            "requested_verification": requested_verification,
        }

    async def run(
        self,
        *,
        operation_id: str,
        source_path: str,
        model_id: str,
        role: str,
        name: str,
        expected_sha256: str,
        cancellation_event: asyncio.Event,
        progress: Progress,
        source_identity: dict[str, Any] | None = None,
        destination: str | None = None,
        model_format: str = "gguf",
        requested_verification: bool = False,
    ) -> dict[str, Any]:
        existing = self._completed_result(operation_id)
        if existing is not None:
            return existing
        source = self._validate_source(source_path)
        self._validate_identifiers(model_id=model_id, role=role, name=name)
        expected = _validate_expected_sha(expected_sha256)
        if model_format != "gguf":
            raise ModelImportError("model_import_invalid_format")
        if source_identity is not None:
            self._validate_source_identity(source, source_identity)
        basename = source.name
        target = self.models_dir / basename
        expected_destination = str(target.relative_to(self.home))
        if destination is not None and destination != expected_destination:
            raise ModelImportError("model_import_destination_mismatch")
        staging = self.staging_dir / f"{operation_id}-{basename}.part"
        snapshot = self.journal_dir / f"{operation_id}.models.yaml.before"
        journal_path = self._journal_path(operation_id)
        self._prepare_directories()
        if target.exists() or target.is_symlink():
            raise ModelImportError("model_import_overwrite_rejected")
        if self._model_id_exists(model_id):
            raise ModelImportError("model_import_identifier_exists")
        journal = {
            "schema_version": 1,
            "operation_id": operation_id,
            "status": "planned",
            "source_path": str(source),
            "staging_path": str(staging),
            "destination_path": str(target),
            "snapshot_path": str(snapshot),
            "model_id": model_id,
            "role": role,
            "basename": basename,
            "byte_count": 0,
            "sha256": None,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self._write_journal(journal_path, journal)
        self._atomic_writer(snapshot, self.config_path.read_bytes())
        journal = self._update_journal(journal_path, journal, status="copying")
        copied = 0
        digest = hashlib.sha256()
        published = False
        try:
            await progress(5, "model_import_validated")
            descriptor = os.open(
                staging,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with (
                    source.open("rb") as source_handle,
                    os.fdopen(descriptor, "wb", closefd=True) as staged_handle,
                ):
                    descriptor = -1
                    total = source.stat().st_size
                    while True:
                        _raise_if_cancelled(cancellation_event)
                        chunk = await asyncio.to_thread(source_handle.read, COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        await asyncio.to_thread(staged_handle.write, chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                        percent = 5 + int(70 * copied / max(1, total))
                        await progress(min(percent, 75), "model_import_copying")
                    await asyncio.to_thread(staged_handle.flush)
                    await asyncio.to_thread(os.fsync, staged_handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            _raise_if_cancelled(cancellation_event)
            actual_sha = digest.hexdigest()
            if expected is not None and actual_sha != expected:
                raise ModelImportError("model_import_hash_mismatch")
            journal = self._update_journal(
                journal_path,
                journal,
                status="staged",
                byte_count=copied,
                sha256=actual_sha,
            )
            await progress(80, "model_import_staged")
            with self._config_write_lock():
                if target.exists() or target.is_symlink():
                    raise ModelImportError("model_import_overwrite_rejected")
                if self._model_id_exists(model_id):
                    raise ModelImportError("model_import_identifier_exists")
                if self._artifact_sha_exists(actual_sha, operation_id=operation_id):
                    raise ModelImportError("model_import_artifact_exists")
                _raise_if_cancelled(cancellation_event)
                os.replace(staging, target)
                published = True
                _fsync_directory(self.models_dir)
                journal = self._update_journal(
                    journal_path,
                    journal,
                    status="configuring",
                )
                config_bytes = self._registered_config(
                    model_id=model_id,
                    role=role,
                    name=name,
                    destination=target,
                )
                try:
                    self._atomic_writer(self.config_path, config_bytes)
                    ModelRegistry.from_file(self.config_path, root=self.home)
                except Exception:
                    self._atomic_writer(self.config_path, snapshot.read_bytes())
                    raise
            await progress(95, "model_import_registered")
            result = {
                "model_id": model_id,
                "logical_role": role,
                "basename": basename,
                "byte_count": copied,
                "sha256": actual_sha,
                "registration_status": "registered_inactive",
            }
            if requested_verification:
                result["requested_verification"] = True
            self._update_journal(
                journal_path,
                journal,
                status="completed",
                result=result,
            )
            snapshot.unlink(missing_ok=True)
            await progress(100, "model_import_completed")
            return result
        except asyncio.CancelledError:
            self._rollback(
                destination=target,
                staging=staging,
                snapshot=snapshot,
                restore_config=published,
            )
            self._update_journal(journal_path, journal, status="cancelled")
            raise
        except Exception:
            self._rollback(
                destination=target,
                staging=staging,
                snapshot=snapshot,
                restore_config=published,
            )
            self._update_journal(journal_path, journal, status="failed")
            raise

    def reconcile(self) -> list[str]:
        self._prepare_directories()
        reconciled: list[str] = []
        for journal_path in sorted(self.journal_dir.glob("*.json")):
            try:
                journal = _read_json_object(journal_path)
                if journal.get("status") in _TERMINAL_JOURNAL_STATUSES:
                    continue
                operation_id = str(journal["operation_id"])
                staging = _owned_path(str(journal["staging_path"]), self.staging_dir)
                destination = _owned_path(
                    str(journal["destination_path"]),
                    self.models_dir,
                )
                snapshot = _owned_path(str(journal["snapshot_path"]), self.journal_dir)
                expected_sha = journal.get("sha256")
                with self._config_write_lock():
                    if snapshot.is_file() and journal.get("status") == "configuring":
                        self._atomic_writer(self.config_path, snapshot.read_bytes())
                    if (
                        destination.is_file()
                        and isinstance(expected_sha, str)
                        and _sha256_file(destination) == expected_sha
                    ):
                        destination.unlink()
                        _fsync_directory(destination.parent)
                    staging.unlink(missing_ok=True)
                snapshot.unlink(missing_ok=True)
                self._update_journal(journal_path, journal, status="reconciled")
                reconciled.append(operation_id)
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
        return reconciled

    def _completed_result(self, operation_id: str) -> dict[str, Any] | None:
        path = self._journal_path(operation_id)
        if not path.is_file():
            return None
        journal = _read_json_object(path)
        result = journal.get("result")
        if journal.get("status") != "completed" or not isinstance(result, dict):
            return None
        completed = {
            "model_id": str(result["model_id"]),
            "logical_role": str(result["logical_role"]),
            "basename": str(result["basename"]),
            "byte_count": int(result["byte_count"]),
            "sha256": str(result["sha256"]),
            "registration_status": str(result["registration_status"]),
        }
        if result.get("requested_verification") is True:
            completed["requested_verification"] = True
        return completed

    def _validate_source(self, value: str) -> Path:
        requested = Path(value).expanduser()
        if not requested.is_absolute():
            raise ModelImportError("model_import_source_must_be_absolute")
        if requested.is_symlink():
            raise ModelImportError("model_import_symlink_rejected")
        try:
            info = requested.lstat()
            source = requested.resolve(strict=True)
        except OSError as exc:
            raise ModelImportError("model_import_source_unavailable") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ModelImportError("model_import_source_not_regular_file")
        if source.suffix.casefold() != ".gguf":
            raise ModelImportError("model_import_source_not_gguf")
        if not is_path_within_roots(source, [self.home, *self.settings.allowed_roots]):
            raise ModelImportError("model_import_source_outside_allowed_roots")
        try:
            with source.open("rb") as handle:
                if handle.read(4) != b"GGUF":
                    raise ModelImportError("model_import_invalid_gguf_magic")
        except OSError as exc:
            raise ModelImportError("model_import_source_unreadable") from exc
        return source

    def _validate_identifiers(self, *, model_id: str, role: str, name: str) -> None:
        if not _IDENTIFIER.fullmatch(model_id):
            raise ModelImportError("model_import_invalid_model_id")
        if role not in ModelDefinition.VALID_ROLES:
            raise ModelImportError("model_import_invalid_role")
        if not name.strip() or len(name) > 160 or "\x00" in name:
            raise ModelImportError("model_import_invalid_name")

    @staticmethod
    def _validate_source_identity(source: Path, expected: dict[str, Any]) -> None:
        info = source.stat()
        actual = {
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "size": int(info.st_size),
            "modified_ns": int(info.st_mtime_ns),
        }
        normalized = {
            key: int(expected.get(key, -1)) for key in ("device", "inode", "size", "modified_ns")
        }
        if actual != normalized:
            raise ModelImportError("model_import_source_identity_changed")

    def _artifact_sha_exists(self, sha256: str, *, operation_id: str) -> bool:
        for path in self.journal_dir.glob("*.json"):
            if path.stem == operation_id:
                continue
            try:
                journal = _read_json_object(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            result = journal.get("result")
            completed_match = (
                journal.get("status") == "completed"
                and isinstance(result, dict)
                and result.get("sha256") == sha256
            )
            publishing_match = (
                journal.get("status") == "configuring" and journal.get("sha256") == sha256
            )
            if completed_match or publishing_match:
                return True
        return False

    def _registered_config(
        self,
        *,
        model_id: str,
        role: str,
        name: str,
        destination: Path,
    ) -> bytes:
        data = _read_yaml_mapping(self.config_path)
        models = data.get("models")
        if not isinstance(models, dict):
            raise ModelImportError("model_import_registry_invalid")
        if any(
            isinstance(value, dict) and value.get("id") == model_id for value in models.values()
        ):
            raise ModelImportError("model_import_identifier_exists")
        key = model_id
        if key in models:
            raise ModelImportError("model_import_identifier_exists")
        models[key] = {
            "id": model_id,
            "name": name,
            "path": str(destination.relative_to(self.home)),
            "backend": "llama_cpp",
            "role": role,
            "threads": max(1, min(os.cpu_count() or 4, 8)),
            "context_size": 4096,
            "temperature": 0.2,
            "max_output_tokens": 1024,
            "keep_loaded": False,
            "idle_unload_seconds": 300,
            "priority": -100,
        }
        ModelRegistry.from_dict(data, root=self.home)
        return yaml.safe_dump(data, sort_keys=False).encode("utf-8")

    def _model_id_exists(self, model_id: str) -> bool:
        registry = ModelRegistry.from_file(self.config_path, root=self.home)
        return registry.exists(model_id)

    def _prepare_directories(self) -> None:
        for path in (self.models_dir, self.state_dir, self.journal_dir, self.staging_dir):
            path.mkdir(parents=True, mode=0o700, exist_ok=True)

    def _journal_path(self, operation_id: str) -> Path:
        if not _IDENTIFIER.fullmatch(operation_id):
            raise ModelImportError("model_import_invalid_operation_id")
        return self.journal_dir / f"{operation_id}.json"

    def _write_journal(self, path: Path, journal: dict[str, Any]) -> None:
        _atomic_write_bytes(
            path,
            json.dumps(journal, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )

    def _update_journal(
        self,
        path: Path,
        journal: dict[str, Any],
        *,
        status: str,
        **updates: Any,
    ) -> dict[str, Any]:
        updated = {
            **journal,
            **updates,
            "status": status,
            "updated_at": utc_now_iso(),
        }
        self._write_journal(path, updated)
        return updated

    def _rollback(
        self,
        *,
        destination: Path,
        staging: Path,
        snapshot: Path,
        restore_config: bool,
    ) -> None:
        rollback_error: Exception | None = None
        with self._config_write_lock():
            if restore_config and snapshot.is_file():
                try:
                    self._atomic_writer(self.config_path, snapshot.read_bytes())
                except Exception as exc:
                    rollback_error = exc
            if restore_config:
                destination.unlink(missing_ok=True)
            staging.unlink(missing_ok=True)
            if destination.parent.exists():
                _fsync_directory(destination.parent)
        snapshot.unlink(missing_ok=True)
        if rollback_error is not None:
            raise rollback_error

    @contextmanager
    def _config_write_lock(self) -> Any:
        try:
            import fcntl
        except ImportError as exc:
            raise ModelImportError("model_import_config_lock_unavailable") from exc
        self.lock_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


async def run_model_import_job(
    settings: AprilSettings,
    *,
    operation_id: str,
    payload: dict[str, Any],
    cancellation_event: asyncio.Event,
    progress: Progress,
) -> dict[str, Any]:
    return await ModelImportService(settings).run(
        operation_id=operation_id,
        source_path=str(payload["source_path"]),
        model_id=str(payload["model_id"]),
        role=str(payload["role"]),
        name=str(payload["name"]),
        expected_sha256=(str(payload["expected_sha256"])),
        cancellation_event=cancellation_event,
        progress=progress,
        source_identity=dict(payload["source_identity"]),
        destination=str(payload["destination"]),
        model_format=str(payload["format"]),
        requested_verification=bool(payload["requested_verification"]),
    )


def reconcile_model_imports(settings: AprilSettings) -> list[str]:
    return ModelImportService(settings).reconcile()


def _validate_expected_sha(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ModelImportError("model_import_invalid_expected_sha256")
    return normalized


def _raise_if_cancelled(event: asyncio.Event) -> None:
    if event.is_set():
        raise asyncio.CancelledError


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ModelImportError("model_import_registry_invalid") from exc
    if not isinstance(data, dict):
        raise ModelImportError("model_import_registry_invalid")
    return data


def _read_json_object(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("journal_not_object")
    return decoded


def _owned_path(value: str, root: Path) -> Path:
    path = Path(value).resolve(strict=False)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("journal_path_outside_owned_root") from exc
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
