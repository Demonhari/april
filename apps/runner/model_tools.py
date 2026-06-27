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


@dataclass(frozen=True, slots=True)
class AppStubResult:
    output_path: Path
    launcher_path: Path
    unsigned: bool = True


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"YAML file must be a mapping: {path}")
    return loaded


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _config_path(home: Path) -> Path:
    return home / "configs" / "models.yaml"


def _settings_config_path(home: Path) -> Path:
    return home / "configs" / "april.yaml"


def _timestamped_backup(path: Path) -> Path:
    backup = path.with_suffix(f"{path.suffix}.bak-{time.strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(path, backup)
    return backup


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


def _existing_model_for_role(data: dict[str, Any], role: str) -> dict[str, Any]:
    models = data.get("models")
    if not isinstance(models, dict):
        return {}
    for value in models.values():
        if isinstance(value, dict) and value.get("role") == role:
            return value
    return {}


def _validate_model_source(
    *,
    home: Path,
    source_path: Path,
    copy_into_models: bool,
) -> Path:
    root = home.expanduser().resolve()
    source_raw = source_path.expanduser()
    if not source_raw.exists():
        raise ConfigError(f"Model path does not exist: {source_raw.name}")
    if not source_raw.is_file():
        raise ConfigError(f"Model path is not a file: {source_raw.name}")
    if source_raw.suffix.lower() != ".gguf":
        raise ConfigError("Model setup only accepts existing .gguf files.")
    source = source_raw.resolve()

    settings = load_settings(root=root)
    allowed_roots = [root, *settings.allowed_roots]
    if source_raw.is_symlink() and not is_path_within_roots(source, allowed_roots):
        raise ConfigError("Model symlink target is outside APRIL allowed roots.")
    if not copy_into_models and not is_path_within_roots(source, allowed_roots):
        raise ConfigError(
            "Model path is outside APRIL allowed roots. Use --copy-into-models to copy it locally.",
            {"path": str(source)},
        )
    return source


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


def setup_model_set(
    *,
    home: Path,
    role_paths: dict[str, Path | None],
    role_ids: dict[str, str | None] | None = None,
    copy_into_models: bool = False,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Validate and optionally register APRIL's intended local GGUF model set.

    This wrapper is dry-run by default and reuses ``import_model`` for the
    actual safe copy/config mutation path when ``apply`` is true.
    """
    root = home.expanduser().resolve()
    supplied = {role: path for role, path in role_paths.items() if path is not None}
    if not supplied:
        raise ConfigError("At least one model path must be supplied.")
    ids = role_ids or {}
    config_path = _config_path(root)
    data = _read_yaml(config_path)
    entries: list[dict[str, Any]] = []
    for role, source_path in supplied.items():
        source = _validate_model_source(
            home=root,
            source_path=source_path,
            copy_into_models=copy_into_models,
        )
        existing = _existing_model_for_role(data, role)
        model_id = ids.get(role) or str(existing.get("id") or f"april-{role}")
        name = str(existing.get("name") or source.stem)
        entries.append(
            {
                "role": role,
                "model_id": model_id,
                "name": name,
                "source_basename": source.name,
                "copy_into_models": copy_into_models,
                "would_write": apply,
            }
        )

    backup: Path | None = None
    imported: list[ModelImportResult] = []
    if apply:
        backup = _timestamped_backup(config_path)
        try:
            for entry in entries:
                imported.append(
                    import_model(
                        home=root,
                        role=str(entry["role"]),
                        model_id=str(entry["model_id"]),
                        name=str(entry["name"]),
                        source_path=supplied[str(entry["role"])],
                        copy_into_models=copy_into_models,
                        force=force,
                    )
                )
        except Exception:
            shutil.copy2(backup, config_path)
            raise

    by_role = {result.role: result for result in imported}
    rendered_entries: list[dict[str, Any]] = []
    for entry in entries:
        result = by_role.get(str(entry["role"]))
        rendered = dict(entry)
        if result is not None:
            rendered["copied"] = result.copied
            rendered["registered_basename"] = result.path.name
        else:
            rendered["copied"] = False
            rendered["registered_basename"] = entry["source_basename"]
        rendered_entries.append(rendered)

    return {
        "applied": apply,
        "backup_basename": backup.name if backup is not None else None,
        "entries": rendered_entries,
        "next_commands": [
            "run april model doctor",
            "run april verify --all-configured-models --require-real-model "
            "--report data/verification/mac-readiness.json",
        ],
        "mutating": apply,
    }


def setup_voice_stack(
    *,
    home: Path,
    whisper_binary: Path,
    whisper_model: Path,
    piper_binary: Path,
    piper_model: Path,
    wake_word_model: Path | None = None,
    apply: bool = False,
    enable: bool = False,
) -> dict[str, Any]:
    """Validate and optionally configure local voice assets without recording.

    Voice stays OFF by default. ``enable`` flips ``voice.enabled`` true, but only
    when ``apply`` actually writes the config and only after every required path
    has validated above. A missing wake-word model never blocks enabling: push-to-
    talk stays available, while wake-word listening remains explicitly unverified.
    """
    root = home.expanduser().resolve()
    config_path = _settings_config_path(root)
    required = {
        "whisper_binary_path": whisper_binary,
        "whisper_model_path": whisper_model,
        "piper_binary_path": piper_binary,
        "piper_model_path": piper_model,
    }
    resolved_required: dict[str, Path] = {}
    for key, path in required.items():
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            if apply:
                _write_voice_enabled(root, config_path, enabled=False)
            raise ConfigError(f"Voice path does not exist: {path.name}")
        if not resolved.is_file():
            if apply:
                _write_voice_enabled(root, config_path, enabled=False)
            raise ConfigError(f"Voice path is not a file: {path.name}")
        resolved_required[key] = resolved

    warnings: list[str] = []
    resolved_wake: Path | None = None
    if wake_word_model is not None:
        candidate = wake_word_model.expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            resolved_wake = candidate
        else:
            warnings.append("wake-word model missing; wake-word remains unconfigured")

    backup: Path | None = None
    if apply:
        backup = _timestamped_backup(config_path)
        data = _read_yaml(config_path)
        voice = data.setdefault("voice", {})
        if not isinstance(voice, dict):
            raise ConfigError("configs/april.yaml voice field must be a mapping.")
        for key, path in resolved_required.items():
            voice[key] = str(path)
        if resolved_wake is not None:
            voice["wake_word_model_path"] = str(resolved_wake)
        # Reached only after every required path validated above. Applying without
        # --enable is an explicit safe-off write, even if the existing config was on.
        voice["enabled"] = bool(enable)
        _write_yaml(config_path, data)
        try:
            _validate_after_write(root)
        except Exception:
            shutil.copy2(backup, config_path)
            raise

    artifacts = [
        {"name": key, "basename": path.name, "configured": True}
        for key, path in resolved_required.items()
    ]
    artifacts.append(
        {
            "name": "wake_word_model_path",
            "basename": resolved_wake.name if resolved_wake is not None else None,
            "configured": resolved_wake is not None,
        }
    )
    voice_enabled = bool(apply and enable)
    wake_word_available = resolved_wake is not None
    return {
        "applied": apply,
        # Whether voice.enabled was actually set true (only on apply + enable).
        "voice_enabled": voice_enabled,
        # The caller asked to enable; useful to message a dry run truthfully.
        "enable_requested": enable,
        "wake_word_available": wake_word_available,
        # Push-to-talk needs no wake-word model; wake-word listening does.
        "push_to_talk_available": True,
        "wake_word_verified": False,
        "backup_basename": backup.name if backup is not None else None,
        "artifacts": artifacts,
        "warnings": warnings,
        "next_commands": [
            "run april voice verify-live --report data/verification/voice-live.json",
        ],
        "mutating": apply,
    }


def _write_voice_enabled(root: Path, config_path: Path, *, enabled: bool) -> None:
    """Best-effort safe-off write when an applying voice setup cannot validate."""
    if not config_path.exists():
        return
    data = _read_yaml(config_path)
    voice = data.setdefault("voice", {})
    if not isinstance(voice, dict):
        return
    voice["enabled"] = enabled
    _write_yaml(config_path, data)


def create_macos_app_stub(
    *, home: Path, output: Path | None = None, force: bool = False
) -> AppStubResult:
    """Create an unsigned local APRIL.app launcher without tokens or models."""
    root = home.expanduser().resolve()
    settings = load_settings(root=root)
    target = output or root / "dist" / "APRIL.app"
    resolved = (
        (root / target).resolve() if not target.is_absolute() else target.expanduser().resolve()
    )
    if resolved.suffix != ".app":
        raise ConfigError("App stub output must end with .app.")
    allowed_roots = [root, *settings.allowed_roots]
    parent = resolved.parent
    if not is_path_within_roots(parent.resolve(), allowed_roots):
        raise ConfigError("App stub output is outside APRIL allowed roots.")
    parent.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink():
        raise ConfigError("App stub output must not be a symlink.")
    if resolved.exists():
        if not force:
            raise ConfigError(f"App stub already exists: {resolved}. Pass --force to overwrite.")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

    contents = resolved / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    launcher = macos / "APRIL"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    (contents / "Info.plist").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>APRIL</string>
  <key>CFBundleDisplayName</key>
  <string>APRIL</string>
  <key>CFBundleIdentifier</key>
  <string>local.april.dev</string>
  <key>CFBundleVersion</key>
  <string>0.1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleExecutable</key>
  <string>APRIL</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHumanReadableCopyright</key>
  <string>Unsigned local development launcher. No models, tokens, or secrets are bundled.</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    launcher.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

cd "{root}"
if command -v run >/dev/null 2>&1; then
  exec run april desktop "$@"
elif [ -x ".venv/bin/python" ]; then
  exec ".venv/bin/python" -m apps.runner.main april desktop "$@"
else
  exec python -m apps.runner.main april desktop "$@"
fi
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return AppStubResult(output_path=resolved, launcher_path=launcher)


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


def recommend_model_profile(home: Path) -> dict[str, Any]:
    """Report a safe, non-mutating model-profile recommendation for this machine.

    Inspects only local hardware (architecture, CPU count, physical memory) and
    the bundled profiles. It never installs packages, downloads models, edits
    shell startup files, switches configuration, or sends data anywhere.
    """
    root = home.expanduser().resolve()
    architecture = machine_kind()
    machine = platform.machine().lower()
    is_arm64 = machine in {"arm64", "aarch64"}
    ram = estimate_ram_bytes()
    profiles = load_model_profiles(root)

    if architecture == "macOS Apple Silicon":
        profile = "apple_silicon_macbook"
        expected_backend = "llama.cpp with Metal acceleration"
        notes = [
            'Use an arm64 Python: `python3 -c "import platform; print(platform.machine())"`'
            " should print arm64.",
            "Install a Metal-enabled llama-cpp-python build so layers offload to the GPU.",
            "Unified memory is shared with the GPU; leave headroom for other apps.",
            "Specialists are evicted on idle (idle_unload_seconds); the brain stays resident.",
        ]
    elif architecture == "macOS Intel":
        profile = "intel_macbook_cpu_low"
        expected_backend = "llama.cpp CPU-only (no Metal)"
        notes = [
            "Intel Macs run CPU-only; keep context and batch sizes conservative.",
            "One small brain model stays resident; specialists load on demand.",
        ]
    else:
        profile = "intel_macbook_cpu_low"
        expected_backend = "llama.cpp CPU-only"
        notes = [
            f"No tuned profile for {architecture}; the conservative CPU profile is a safe default.",
        ]

    if architecture == "macOS Apple Silicon" and not is_arm64:
        notes.insert(
            0,
            "WARNING: this Python is not arm64 (likely Rosetta); reinstall an arm64 Python "
            "to get Metal acceleration.",
        )

    if profile not in profiles:
        # Never recommend a profile that is not actually defined.
        profile = next(iter(profiles), profile)

    manual_commands = [
        f"run april model profile apply {profile}",
        "pip install -e '.[runtime]'",
        "run april model import --role brain --id april-brain --name <name> "
        "--path /path/to/model.gguf",
        "run april model doctor",
        "run april verify --real-model /path/to/model.gguf",
    ]
    return {
        "architecture": architecture,
        "platform": platform.platform(),
        "python_machine": platform.machine(),
        "arm64_python": is_arm64,
        "cpu_count": os.cpu_count(),
        "available_memory_bytes": ram,
        "available_memory": format_bytes(ram),
        "recommended_profile": profile,
        "available_profiles": sorted(profiles),
        "expected_backend": expected_backend,
        "manual_commands": manual_commands,
        "notes": notes,
        "mutating": False,
    }


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
