"""Safe, non-destructive first-run bootstrap for APRIL.

``run april setup bootstrap`` prepares a local APRIL home: it creates the data /
logs / models / index / audit / audio-cache directories, generates local API and
Runtime tokens into a chosen ``.env`` (without printing them or overwriting
existing secrets unless ``--force``), inspects the machine, recommends (but does
not silently apply) a model profile, and reports model / voice / llama.cpp / root
status plus the exact next verification commands.

It never installs Homebrew, Python packages, models, or voice binaries, never
downloads anything, and never edits shell startup files. It is fully
non-interactive so it can run in tests against a temporary ``APRIL_HOME``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path
from typing import Any

from apps.runner.model_tools import apply_model_profile, model_doctor, recommend_model_profile
from april_common.config_validation import validate_configuration
from april_common.settings import (
    KNOWN_DEFAULT_API_TOKENS,
    KNOWN_DEFAULT_RUNTIME_TOKENS,
    AprilSettings,
    load_settings,
)
from april_common.token_setup import GeneratedTokens, generate_tokens, write_token_env_file

TOKEN_KEYS = ("APRIL_API_TOKEN", "APRIL_RUNTIME_TOKEN")


def bootstrap(
    home: Path,
    *,
    env_file: Path | None = None,
    force: bool = False,
    apply_profile: bool = False,
) -> dict[str, Any]:
    """Run the bootstrap and return a structured, secret-free report."""
    root = home.expanduser().resolve()
    target_env = (env_file or root / ".env").expanduser()
    if not target_env.is_absolute():
        target_env = root / target_env

    directories = _ensure_directories_for(root)
    tokens = _ensure_tokens(target_env, force=force)

    # Reload after writing .env so the report reflects the post-bootstrap state
    # (when the env file is the home .env that load_settings reads).
    settings = load_settings(root=root)
    doctor = model_doctor(root)
    recommendation = recommend_model_profile(root)

    applied_profile: str | None = None
    profile_error: str | None = None
    if apply_profile:
        try:
            apply_model_profile(home=root, profile_name=recommendation["recommended_profile"])
            applied_profile = recommendation["recommended_profile"]
        except Exception as exc:  # surfaced, never raised, so bootstrap stays safe
            profile_error = str(exc)

    config_errors = validate_configuration(root)

    return {
        "home": str(root),
        "directories": directories,
        "env_file": str(target_env),
        "tokens": tokens,
        "dev_token_warnings": _dev_token_warnings(settings),
        "machine": {
            "architecture": recommendation["architecture"],
            "platform": recommendation["platform"],
            "cpu_count": recommendation["cpu_count"],
            "available_memory": recommendation["available_memory"],
            "arm64_python": recommendation["arm64_python"],
        },
        "recommended_profile": recommendation["recommended_profile"],
        "available_profiles": recommendation["available_profiles"],
        "expected_backend": recommendation["expected_backend"],
        "profile_applied": applied_profile is not None,
        "applied_profile": applied_profile,
        "profile_error": profile_error,
        "llama_cpp_available": bool(doctor["llama_cpp_python_installed"]),
        "models": [
            {
                "id": model["id"],
                "role": model["role"],
                "path_basename": Path(model["path"]).name,
                "exists": model["path_exists"],
            }
            for model in doctor["models"]
        ],
        "missing_model_paths": [
            Path(model["path"]).name for model in doctor["models"] if not model["path_exists"]
        ],
        "voice": _voice_report(settings),
        "allowed_filesystem_roots": [str(path) for path in settings.allowed_roots],
        "config_valid": not config_errors,
        "config_errors": config_errors,
        "next_commands": _next_commands(recommendation["recommended_profile"], target_env),
    }


def _ensure_directories_for(root: Path) -> list[dict[str, Any]]:
    settings = load_settings(root=root)
    raw = [
        settings.resolve_path(Path("data")),
        settings.resolve_path(Path("data/run")),
        settings.resolve_path(Path("data/artifacts/patches")),
        settings.resolve_path(Path("data/artifacts/cleanup")),
        settings.logs_path,
        settings.resolve_path(Path("models")),
        settings.vector_index_path,
        settings.audit_path.parent,
        settings.audio_cache_path,
    ]
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for path in raw:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        created = not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        # Local-only, owner-accessible directories (db, audit, audio, models).
        with contextlib.suppress(OSError):
            os.chmod(path, 0o700)
        results.append({"path": key, "created": created})
    return results


def _parse_env_tokens(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() in TOKEN_KEYS and value.strip():
            values[key.strip()] = value.strip()
    return values


def _ensure_tokens(env_file: Path, *, force: bool) -> dict[str, Any]:
    existing = _parse_env_tokens(env_file)
    has_api = "APRIL_API_TOKEN" in existing
    has_runtime = "APRIL_RUNTIME_TOKEN" in existing
    if has_api and has_runtime and not force:
        # Non-destructive: never overwrite existing secrets without --force.
        return {"action": "kept", "api_token_set": True, "runtime_token_set": True}
    new = generate_tokens()
    api = new.api_token if (force or not has_api) else existing["APRIL_API_TOKEN"]
    runtime = new.runtime_token if (force or not has_runtime) else existing["APRIL_RUNTIME_TOKEN"]
    write_token_env_file(env_file, GeneratedTokens(api_token=api, runtime_token=runtime))
    return {
        "action": "regenerated" if force else "generated",
        "api_token_set": True,
        "runtime_token_set": True,
    }


def _dev_token_warnings(settings: AprilSettings) -> list[str]:
    warnings: list[str] = []
    if settings.api.token in KNOWN_DEFAULT_API_TOKENS:
        warnings.append(
            "Effective API token is a known development token. Load the generated .env "
            "or set APRIL_API_TOKEN before any non-development use."
        )
    if settings.runtime.token is None or settings.runtime.token in KNOWN_DEFAULT_RUNTIME_TOKENS:
        warnings.append(
            "Effective Runtime token is unset or a known development token. Load the generated "
            ".env or set APRIL_RUNTIME_TOKEN before any non-development use."
        )
    return warnings


def _voice_report(settings: AprilSettings) -> dict[str, Any]:
    voice = settings.voice
    return {
        "enabled": voice.enabled,
        "sounddevice_available": importlib.util.find_spec("sounddevice") is not None,
        "openwakeword_available": importlib.util.find_spec("openwakeword") is not None,
        "paths": [
            _path_status("whisper_binary", voice.whisper_binary_path),
            _path_status("whisper_model", voice.whisper_model_path),
            _path_status("piper_binary", voice.piper_binary_path),
            _path_status("piper_model", voice.piper_model_path),
            _path_status("wake_word_model", voice.wake_word_model_path),
        ],
    }


def _path_status(name: str, value: Any) -> dict[str, Any]:
    configured = value is not None and str(value) != ""
    exists = configured and Path(str(value)).expanduser().exists()
    return {"name": name, "configured": configured, "exists": exists}


def _next_commands(profile: str, env_file: Path) -> list[str]:
    return [
        "run april config validate",
        "run april verify --fake",
        f"run april model profile apply {profile}",
        "pip install -e '.[runtime]'  # optional: real llama.cpp runtime",
        "run april model import --role brain --id april-brain --path /absolute/path/model.gguf",
        "run april model doctor",
        "run april verify --real-model /absolute/path/model.gguf",
        "run april verify --target-mac --require-real-model /absolute/path/model.gguf "
        "--report data/verification/mac-report.json",
    ]
