from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, Field, field_validator

from apps.runner.model_tools import setup_model_set
from april_common.errors import ConfigError
from april_common.time import utc_now, utc_now_iso
from services.april_runtime.model_registry import UniqueKeyLoader

CORE_DOWNLOAD_ROLES = ("brain", "coding", "reading")
VALID_DOWNLOAD_ROLES = {"brain", "coding", "reading", "reasoning", "embedding"}
MIN_GGUF_BYTES = 16

DownloadFunction = Callable[[str, Path, str | None], None]
DownloadMode = Literal["dry_run", "apply"]


class ModelDownloadEntry(BaseModel):
    role: str
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    target_path: Path
    required_for_full_activation: bool = False
    license_hint: str = Field(min_length=1)
    source: Literal["huggingface"]

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in VALID_DOWNLOAD_ROLES:
            raise ValueError(f"unsupported model download role: {value}")
        return value

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        if "://" in value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("repo_id must be a Hugging Face repo id, not a URL")
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("filename must be a relative Hugging Face filename")
        if path.suffix.lower() != ".gguf":
            raise ValueError("filename must point to a .gguf file")
        return value

    @field_validator("target_path")
    @classmethod
    def validate_target_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("target_path must be a relative path inside APRIL_HOME")
        if value.suffix.lower() != ".gguf":
            raise ValueError("target_path must end with .gguf")
        return value


class ModelDownloadReportEntry(BaseModel):
    role: str
    id: str
    name: str
    repo_id: str
    filename: str
    target_path: str
    target_basename: str
    source: Literal["huggingface"]
    required_for_full_activation: bool
    license_hint: str
    status: str
    size_bytes: int | None = None
    sha256: str | None = None


class ModelDownloadReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["model_download"] = "model_download"
    generated_at: str
    mode: DownloadMode
    applied: bool = False
    selected_roles: list[str] = Field(default_factory=list)
    entries: list[ModelDownloadReportEntry] = Field(default_factory=list)
    registration_applied: bool = False
    registration_backup_basename: str | None = None
    real_model_ready: bool = False
    real_model_verified: bool = False
    next_commands: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid model download manifest YAML: {path.name}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Model download manifest must be a mapping: {path.name}")
    return loaded


def load_model_download_manifest(home: Path) -> dict[str, ModelDownloadEntry]:
    path = home.expanduser().resolve() / "configs" / "model_downloads.yaml"
    if not path.exists():
        raise ConfigError("configs/model_downloads.yaml is missing.")
    raw = _read_yaml(path)
    entries: dict[str, ModelDownloadEntry] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ConfigError("Each model download manifest entry must be a mapping.")
        try:
            entry = ModelDownloadEntry.model_validate(value)
        except ValueError as exc:
            raise ConfigError(f"Invalid model download manifest entry: {key}") from exc
        if key != entry.role:
            raise ConfigError(f"Model download entry key must match role: {key}")
        entries[key] = entry
    return entries


def default_model_download_report_path(home: Path) -> Path:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return home.expanduser() / "data" / "verification" / f"model-download-{stamp}.json"


def write_model_download_report(report: ModelDownloadReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


def _target_path(home: Path, entry: ModelDownloadEntry) -> Path:
    root = home.expanduser().resolve()
    target = (root / entry.target_path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"Download target escapes APRIL_HOME: {entry.target_path}") from exc
    return target


def _download_url(entry: ModelDownloadEntry) -> str:
    repo_id = quote(entry.repo_id, safe="/")
    filename = quote(entry.filename, safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _stdlib_download(url: str, part_path: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response, part_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def validate_gguf_file(
    path: Path, *, min_bytes: int = MIN_GGUF_BYTES, allow_part_suffix: bool = False
) -> None:
    if not path.exists():
        raise ConfigError(f"Downloaded model is missing: {path.name}")
    if not path.is_file():
        raise ConfigError(f"Downloaded model is not a file: {path.name}")
    suffix_ok = path.suffix.lower() == ".gguf" or (
        allow_part_suffix and path.name.endswith(".gguf.part")
    )
    if not suffix_ok:
        raise ConfigError(f"Downloaded model is not a GGUF file: {path.name}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ConfigError(f"Downloaded GGUF is too small: {path.name}")
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise ConfigError(f"Downloaded file does not start with GGUF magic bytes: {path.name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _select_entries(
    manifest: dict[str, ModelDownloadEntry], *, all_core: bool, role: str | None
) -> list[ModelDownloadEntry]:
    if all_core and role is not None:
        raise ConfigError("Use either --all-core or --role, not both.")
    if not all_core and role is None:
        raise ConfigError("Choose --all-core or --role.")
    if all_core:
        missing = [
            download_role for download_role in CORE_DOWNLOAD_ROLES if download_role not in manifest
        ]
        if missing:
            raise ConfigError(
                "Model download manifest is missing required core roles: " + ", ".join(missing)
            )
        return [manifest[download_role] for download_role in CORE_DOWNLOAD_ROLES]
    if role not in VALID_DOWNLOAD_ROLES:
        raise ConfigError(f"Unsupported model download role: {role}")
    if role not in manifest:
        raise ConfigError(f"No model download manifest entry for role: {role}")
    return [manifest[role]]


def _report_entry(
    entry: ModelDownloadEntry,
    *,
    status: str,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> ModelDownloadReportEntry:
    return ModelDownloadReportEntry(
        role=entry.role,
        id=entry.id,
        name=entry.name,
        repo_id=entry.repo_id,
        filename=entry.filename,
        target_path=entry.target_path.as_posix(),
        target_basename=entry.target_path.name,
        source=entry.source,
        required_for_full_activation=entry.required_for_full_activation,
        license_hint=entry.license_hint,
        status=status,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _next_commands(manifest: dict[str, ModelDownloadEntry]) -> list[str]:
    brain = manifest["brain"].target_path.as_posix()
    coding = manifest["coding"].target_path.as_posix()
    reading = manifest["reading"].target_path.as_posix()
    return [
        "run april model doctor",
        # Voice is opt-in, so model-only activation needs no --skip-voice here.
        "run april setup mac-activation "
        f"--brain {brain} --coding {coding} --reading {reading} "
        "--apply --run-acceptance --start-services",
        "run april verify --all-configured-models --require-real-model "
        "--report data/verification/mac-readiness.json",
    ]


def _cleanup_created_targets(targets: list[Path]) -> None:
    for target in targets:
        try:
            if target.exists() and target.is_file():
                target.unlink()
        except OSError:
            pass


def run_model_downloads(
    home: Path,
    *,
    all_core: bool = False,
    role: str | None = None,
    apply: bool = False,
    yes: bool = False,
    force: bool = False,
    skip_existing: bool = False,
    download_func: DownloadFunction | None = None,
) -> ModelDownloadReport:
    root = home.expanduser().resolve()
    if force and skip_existing:
        raise ConfigError("Use either --force or --skip-existing, not both.")
    if apply and not yes:
        raise ConfigError("Model downloads require --yes when --apply is used.")
    manifest = load_model_download_manifest(root)
    entries = _select_entries(manifest, all_core=all_core, role=role)
    mode: DownloadMode = "apply" if apply else "dry_run"
    report_entries: list[ModelDownloadReportEntry] = []
    selected_roles = [entry.role for entry in entries]
    download = download_func or _stdlib_download

    if not apply:
        for entry in entries:
            target = _target_path(root, entry)
            if target.exists() and skip_existing:
                status = "would_skip_existing"
            elif target.exists() and force:
                status = "would_overwrite"
            elif target.exists():
                status = "exists"
            else:
                status = "would_download"
            report_entries.append(_report_entry(entry, status=status))
        return ModelDownloadReport(
            generated_at=utc_now_iso(),
            mode=mode,
            applied=False,
            selected_roles=selected_roles,
            entries=report_entries,
            next_commands=_next_commands(manifest),
        )

    created_targets: list[Path] = []
    role_paths: dict[str, Path | None] = {}
    role_ids: dict[str, str | None] = {}
    try:
        for entry in entries:
            target = _target_path(root, entry)
            role_paths[entry.role] = target
            role_ids[entry.role] = entry.id
            existed_before = target.exists()
            if existed_before:
                if skip_existing:
                    report_entries.append(_report_entry(entry, status="skipped_existing"))
                    continue
                if not force:
                    raise ConfigError(
                        f"Model target already exists: {entry.target_path.as_posix()}. "
                        "Use --skip-existing or --force."
                    )
            target.parent.mkdir(parents=True, exist_ok=True)
            part_path = target.with_name(f"{target.name}.part")
            if part_path.exists():
                raise ConfigError(f"Partial download already exists: {part_path.name}")
            try:
                download(_download_url(entry), part_path, os.environ.get("HF_TOKEN"))
                validate_gguf_file(part_path, allow_part_suffix=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                part_path.replace(target)
                validate_gguf_file(target)
                digest = sha256_file(target)
                if not existed_before:
                    created_targets.append(target)
                report_entries.append(
                    _report_entry(
                        entry,
                        status="downloaded",
                        size_bytes=target.stat().st_size,
                        sha256=digest,
                    )
                )
            except Exception:
                if part_path.exists():
                    part_path.unlink()
                raise

        registration = setup_model_set(
            home=root,
            role_paths=role_paths,
            role_ids=role_ids,
            apply=True,
            force=True,
        )
    except Exception:
        _cleanup_created_targets(created_targets)
        raise

    backup = registration.get("backup_basename")
    return ModelDownloadReport(
        generated_at=utc_now_iso(),
        mode=mode,
        applied=True,
        selected_roles=selected_roles,
        entries=report_entries,
        registration_applied=True,
        registration_backup_basename=str(backup) if backup else None,
        next_commands=_next_commands(manifest),
    )
