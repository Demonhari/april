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
import re
from pathlib import Path
from typing import Any

from apps.runner.model_tools import apply_model_profile, model_doctor, recommend_model_profile
from april_common.config_validation import validate_configuration
from april_common.settings import (
    INSECURE_API_TOKENS,
    INSECURE_RUNTIME_TOKENS,
    KNOWN_DEFAULT_API_TOKENS,
    KNOWN_DEFAULT_RUNTIME_TOKENS,
    PLACEHOLDER_API_TOKENS,
    PLACEHOLDER_RUNTIME_TOKENS,
    AprilSettings,
    load_settings,
)
from april_common.token_setup import GeneratedTokens, generate_tokens, write_token_env_file

TOKEN_KEYS = ("APRIL_API_TOKEN", "APRIL_RUNTIME_TOKEN")
_PATH_TEXT_RE = re.compile(r"~?(?:/[\w.\-]+){2,}/?")


def bootstrap(
    home: Path,
    *,
    env_file: Path | None = None,
    force: bool = False,
    apply_profile: bool = False,
    show_paths: bool = False,
) -> dict[str, Any]:
    """Run the bootstrap and return a structured, secret-free report."""
    root = home.expanduser().resolve()
    target_env = (env_file or root / ".env").expanduser()
    if not target_env.is_absolute():
        target_env = root / target_env
    target_env = target_env.resolve(strict=False)

    directories = _ensure_directories_for(root, show_paths=show_paths)
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
            profile_error = _display_text(str(exc), show_paths=show_paths)

    config_errors = [
        _display_text(error, show_paths=show_paths) for error in validate_configuration(root)
    ]

    return {
        "home": _display_path(root, show_paths=show_paths),
        "directories": directories,
        "env_file": _display_path(target_env, show_paths=show_paths),
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
        "voice": _voice_report(settings, show_paths=show_paths),
        "allowed_filesystem_roots": [
            _display_path(path, show_paths=show_paths) for path in settings.allowed_roots
        ],
        "config_valid": not config_errors,
        "config_errors": config_errors,
        "next_commands": _next_commands(recommendation["recommended_profile"], target_env),
    }


def _ensure_directories_for(root: Path, *, show_paths: bool) -> list[dict[str, Any]]:
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
        results.append({"path": _display_path(path, show_paths=show_paths), "created": created})
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
    api_token = settings.api.token
    runtime_token = settings.runtime.token
    if not api_token:
        warnings.append(
            "Effective APRIL_API_TOKEN is blank. Load the generated .env or set a "
            "strong local token before any non-development use."
        )
    elif api_token in KNOWN_DEFAULT_API_TOKENS:
        warnings.append(
            "Effective APRIL_API_TOKEN is a known development token. Load the generated .env "
            "or set a strong local token before any non-development use."
        )
    elif api_token in PLACEHOLDER_API_TOKENS or api_token in INSECURE_API_TOKENS:
        warnings.append(
            "Effective APRIL_API_TOKEN is a placeholder token. Load the generated .env "
            "or set a strong local token before any non-development use."
        )
    if runtime_token is None or runtime_token == "":
        warnings.append(
            "Effective APRIL_RUNTIME_TOKEN is blank or missing. Load the generated .env "
            "or set a strong local runtime token before any non-development use."
        )
    elif runtime_token in KNOWN_DEFAULT_RUNTIME_TOKENS:
        warnings.append(
            "Effective APRIL_RUNTIME_TOKEN is a known development token. Load the generated "
            ".env or set a strong local runtime token before any non-development use."
        )
    elif runtime_token in PLACEHOLDER_RUNTIME_TOKENS or runtime_token in INSECURE_RUNTIME_TOKENS:
        warnings.append(
            "Effective APRIL_RUNTIME_TOKEN is a placeholder token. Load the generated "
            ".env or set a strong local runtime token before any non-development use."
        )
    return warnings


def _voice_report(settings: AprilSettings, *, show_paths: bool) -> dict[str, Any]:
    voice = settings.voice
    return {
        "enabled": voice.enabled,
        "sounddevice_available": importlib.util.find_spec("sounddevice") is not None,
        "openwakeword_available": importlib.util.find_spec("openwakeword") is not None,
        "paths": [
            _path_status(settings, "whisper_binary", voice.whisper_binary_path, show_paths),
            _path_status(settings, "whisper_model", voice.whisper_model_path, show_paths),
            _path_status(settings, "piper_binary", voice.piper_binary_path, show_paths),
            _path_status(settings, "piper_model", voice.piper_model_path, show_paths),
            _path_status(settings, "wake_word_model", voice.wake_word_model_path, show_paths),
        ],
    }


def _path_status(
    settings: AprilSettings, name: str, value: Any, show_paths: bool
) -> dict[str, Any]:
    configured = value is not None and str(value) != ""
    resolved = settings.resolve_path(Path(str(value))) if configured else None
    exists = bool(resolved and resolved.exists())
    status: dict[str, Any] = {"name": name, "configured": configured, "exists": exists}
    if resolved is not None:
        status["path"] = _display_path(resolved, show_paths=show_paths)
    return status


def _display_path(path: Path, *, show_paths: bool) -> str:
    resolved = path.expanduser()
    if show_paths:
        return str(resolved.resolve(strict=False))
    return resolved.name or str(resolved)


def _display_text(text: str, *, show_paths: bool) -> str:
    if show_paths:
        return text

    def _basename(match: re.Match[str]) -> str:
        name = Path(match.group(0)).name
        return name or match.group(0)

    return _PATH_TEXT_RE.sub(_basename, text)


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
