from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from april_common.config_validation import validate_configuration
from april_common.errors import ConfigError
from april_common.path_security import is_path_within_roots
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry, UniqueKeyLoader

MODEL_RUNTIME_FIELDS = {
    "context_size",
    "threads",
    "n_batch",
    "n_ubatch",
    "n_gpu_layers",
    "use_mmap",
    "use_mlock",
    "keep_loaded",
    "idle_unload_seconds",
    "temperature",
    "max_output_tokens",
}


@dataclass(frozen=True, slots=True)
class ModelImportResult:
    model_id: str
    role: str
    path: Path
    copied: bool
    config_path: Path
    next_command: str


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"YAML file must be a mapping: {path}")
    return loaded


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _config_path(home: Path) -> Path:
    return home / "configs" / "models.yaml"


def _validate_after_write(home: Path) -> None:
    errors = validate_configuration(home)
    if errors:
        raise ConfigError("Configuration validation failed after edit.", {"errors": errors})


def _role_key(data: dict[str, Any], role: str, model_id: str) -> str:
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        raise ConfigError("configs/models.yaml models field must be a mapping.")
    for key, value in models.items():
        if isinstance(value, dict) and (value.get("role") == role or value.get("id") == model_id):
            return str(key)
    return role


def import_model(
    *,
    home: Path,
    role: str,
    model_id: str,
    name: str,
    source_path: Path,
    copy_into_models: bool = False,
    force: bool = False,
) -> ModelImportResult:
    root = home.expanduser().resolve()
    source = source_path.expanduser().resolve()
    if not source.exists():
        raise ConfigError(f"Model path does not exist: {source}")
    if not source.is_file():
        raise ConfigError(f"Model path is not a file: {source}")
    if source.suffix.lower() != ".gguf":
        raise ConfigError("Model import only accepts existing .gguf files.")

    settings = load_settings(root=root)
    allowed_roots = [root, *settings.allowed_roots]
    if not copy_into_models and not is_path_within_roots(source, allowed_roots):
        raise ConfigError(
            "Model path is outside APRIL allowed roots. Use --copy-into-models to copy it locally.",
            {"path": str(source)},
        )

    target = source
    copied = False
    stored_path: str | Path = str(source)
    if copy_into_models:
        models_dir = root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / source.name
        if target.exists() and not force:
            raise ConfigError(f"Model file already exists: {target}. Pass --force to overwrite.")
        if source != target:
            shutil.copy2(source, target)
            copied = True
        stored_path = target.relative_to(root)

    config_path = _config_path(root)
    data = _read_yaml(config_path)
    key = _role_key(data, role, model_id)
    models = data.setdefault("models", {})
    existing = models.get(key, {}) if isinstance(models.get(key, {}), dict) else {}
    models[key] = {
        **existing,
        "id": model_id,
        "name": name,
        "path": str(stored_path),
        "backend": existing.get("backend", "llama_cpp"),
        "role": role,
        "threads": int(existing.get("threads") or max(1, min(os.cpu_count() or 4, 8))),
        "context_size": int(existing.get("context_size") or 4096),
        "temperature": float(existing.get("temperature") or 0.2),
        "max_output_tokens": int(existing.get("max_output_tokens") or 1024),
        "keep_loaded": bool(existing.get("keep_loaded", role == "brain")),
        "idle_unload_seconds": existing.get(
            "idle_unload_seconds", None if role == "brain" else 300
        ),
        "priority": int(existing.get("priority") or (100 if role == "brain" else 50)),
    }
    _write_yaml(config_path, data)
    try:
        ModelRegistry.from_file(config_path, root=root)
        _validate_after_write(root)
    except Exception:
        if copy_into_models and copied and target.exists():
            target.unlink()
        raise

    return ModelImportResult(
        model_id=model_id,
        role=role,
        path=target,
        copied=copied,
        config_path=config_path,
        next_command=f"run april verify --real-model {target}",
    )


def load_model_profiles(home: Path) -> dict[str, Any]:
    path = home / "configs" / "model_profiles.yaml"
    data = _read_yaml(path)
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ConfigError("configs/model_profiles.yaml must contain a profiles mapping.")
    return profiles


def apply_model_profile(*, home: Path, profile_name: str) -> Path:
    root = home.expanduser().resolve()
    profiles = load_model_profiles(root)
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ConfigError(f"Unknown model profile: {profile_name}")
    config_path = _config_path(root)
    backup = config_path.with_suffix(f".yaml.bak-{time.strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(config_path, backup)
    data = _read_yaml(config_path)
    models = data.get("models")
    if not isinstance(models, dict):
        raise ConfigError("configs/models.yaml models field must be a mapping.")
    for role, settings in profile.items():
        if role == "description" or not isinstance(settings, dict):
            continue
        for model in models.values():
            if not isinstance(model, dict) or model.get("role") != role:
                continue
            for field, value in settings.items():
                if field in MODEL_RUNTIME_FIELDS:
                    model[field] = value
    _write_yaml(config_path, data)
    try:
        ModelRegistry.from_file(config_path, root=root)
        _validate_after_write(root)
    except Exception:
        shutil.copy2(backup, config_path)
        raise
    return backup


def redact_token(value: str | None) -> str:
    if not value:
        return "not configured"
    if len(value) <= 8:
        return "[REDACTED]"
    return f"{value[:4]}...{value[-4:]}"


def machine_kind() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macOS Apple Silicon"
    if system == "darwin" and machine in {"x86_64", "amd64", "i386", "i686"}:
        return "macOS Intel"
    if system == "linux":
        return "Linux"
    return "unknown"


def estimate_ram_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            return None
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return pages * page_size
    return None


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _model_realism(model_size: int | None, ram_bytes: int | None) -> str:
    if model_size is None:
        return "unknown: model file is missing"
    if ram_bytes is None:
        return "unknown: RAM estimate unavailable"
    if model_size * 3 > ram_bytes:
        return "risky: model may exceed comfortable CPU-only RAM headroom"
    if model_size * 2 > ram_bytes:
        return "tight: close other apps and use conservative context/batch settings"
    return "ok"


def model_doctor(home: Path) -> dict[str, Any]:
    root = home.expanduser().resolve()
    settings = load_settings(root=root)
    registry = ModelRegistry.from_file(root / "configs" / "models.yaml", root=root)
    ram = estimate_ram_bytes()
    models: list[dict[str, Any]] = []
    for model in registry.list():
        path = model.resolved_path(root)
        exists = path.exists()
        size = path.stat().st_size if exists else None
        models.append(
            {
                "id": model.id,
                "role": model.role,
                "name": model.name,
                "backend": model.backend,
                "path": str(path),
                "path_exists": exists,
                "file_size_bytes": size,
                "file_size": format_bytes(size),
                "context_size": model.context_size,
                "threads": model.threads,
                "n_batch": model.n_batch,
                "n_ubatch": model.n_ubatch,
                "n_gpu_layers": model.n_gpu_layers,
                "keep_loaded": model.keep_loaded,
                "idle_unload_seconds": model.idle_unload_seconds,
                "realism": _model_realism(size, ram),
            }
        )
    return {
        "python_version": sys.version.split()[0],
        "april_home": str(root),
        "runtime_backend": settings.runtime.backend,
        "llama_cpp_python_installed": importlib.util.find_spec("llama_cpp") is not None,
        "api_token": redact_token(settings.api.token),
        "runtime_token": redact_token(settings.runtime.token),
        "machine": machine_kind(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "estimated_ram_bytes": ram,
        "estimated_ram": format_bytes(ram),
        "models": models,
    }
