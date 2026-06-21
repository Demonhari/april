from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from april_common.errors import ConfigError


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = "local-dev-token"
    cors_enabled: bool = False
    max_request_bytes: int = 1_048_576


class RuntimeSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8766
    url: str = "http://127.0.0.1:8766"
    token: str | None = None
    backend: str = "llama_cpp"
    preload_keep_loaded: bool = True
    request_timeout_seconds: float = 120.0
    max_loaded_specialist_models: int = 2


class MemorySettings(BaseModel):
    database_path: Path = Path("data/april.db")
    vector_index_path: Path = Path("data/vector_index")
    embedding_provider: str = "hashed-token"
    embedding_model_id: str | None = None

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        if value not in {"hashed-token", "runtime-local"}:
            raise ValueError("embedding_provider must be hashed-token or runtime-local")
        return value


class PathSettings(BaseModel):
    logs_path: Path = Path("logs")
    audit_path: Path = Path("logs/audit.jsonl")
    allowed_filesystem_roots: list[Path] = Field(default_factory=lambda: [Path(".")])
    max_file_read_bytes: int = 1_048_576
    max_file_write_bytes: int = 1_048_576


class PermissionSettings(BaseModel):
    approval_expiry_seconds: int = 900
    maximum_agent_tool_iterations: int = 5
    external_actions_enabled: bool = False
    tool_timeout_seconds: float = 15.0


class BrainSettings(BaseModel):
    model_id: str = "april-brain"


class VoiceSettings(BaseModel):
    enabled: bool = False
    audio_cache_path: Path = Path("data/audio_cache")
    retain_debug_audio: bool = False
    input_device: str | int | None = None
    output_device: str | int | None = None
    max_record_seconds: float = 30.0
    vad_energy_threshold: float = 0.01
    vad_required_frames: int = 3
    wake_word_threshold: float = 0.5
    wake_word_cooldown_seconds: float = 2.0
    whisper_binary_path: Path | None = None
    whisper_model_path: Path | None = None
    piper_binary_path: Path | None = None
    piper_model_path: Path | None = None
    wake_word_model_path: Path | None = None


class SchedulerSettings(BaseModel):
    enabled: bool = False
    poll_interval_seconds: float = 30.0
    notification_sink: str = "log"
    briefing_enabled: bool = False
    briefing_time: str = "08:00"
    repo_monitor_enabled: bool = False

    @field_validator("notification_sink")
    @classmethod
    def validate_notification_sink(cls, value: str) -> str:
        if value not in {"log", "macos"}:
            raise ValueError("notification_sink must be log or macos")
        return value

    @field_validator("briefing_time")
    @classmethod
    def validate_briefing_time(cls, value: str) -> str:
        hours, _, minutes = value.partition(":")
        try:
            hour, minute = int(hours), int(minutes)
        except ValueError as exc:
            raise ValueError("briefing_time must be HH:MM") from exc
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError("briefing_time must be within 00:00-23:59")
        return f"{hour:02d}:{minute:02d}"


class AprilSettings(BaseModel):
    home: Path
    api: ApiSettings = Field(default_factory=ApiSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    brain: BrainSettings = Field(default_factory=BrainSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)

    @field_validator("home")
    @classmethod
    def resolve_home(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    def resolve_path(self, value: Path) -> Path:
        expanded = value.expanduser()
        if expanded.is_absolute():
            return expanded.resolve()
        return (self.home / expanded).resolve()

    @property
    def database_path(self) -> Path:
        return self.resolve_path(self.memory.database_path)

    @property
    def vector_index_path(self) -> Path:
        return self.resolve_path(self.memory.vector_index_path)

    @property
    def logs_path(self) -> Path:
        return self.resolve_path(self.paths.logs_path)

    @property
    def audit_path(self) -> Path:
        return self.resolve_path(self.paths.audit_path)

    @property
    def audio_cache_path(self) -> Path:
        return self.resolve_path(self.voice.audio_cache_path)

    @property
    def scheduler_log_path(self) -> Path:
        return self.resolve_path(self.paths.logs_path / "scheduler.log")

    @property
    def allowed_roots(self) -> list[Path]:
        return [self.resolve_path(path) for path in self.paths.allowed_filesystem_roots]


ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "APRIL_HOME": ("home",),
    "APRIL_API_HOST": ("api", "host"),
    "APRIL_API_PORT": ("api", "port"),
    "APRIL_API_TOKEN": ("api", "token"),
    "APRIL_API_CORS_ENABLED": ("api", "cors_enabled"),
    "APRIL_API_MAX_REQUEST_BYTES": ("api", "max_request_bytes"),
    "APRIL_RUNTIME_HOST": ("runtime", "host"),
    "APRIL_RUNTIME_PORT": ("runtime", "port"),
    "APRIL_RUNTIME_URL": ("runtime", "url"),
    "APRIL_RUNTIME_TOKEN": ("runtime", "token"),
    "APRIL_RUNTIME_BACKEND": ("runtime", "backend"),
    "APRIL_RUNTIME_PRELOAD_KEEP_LOADED": ("runtime", "preload_keep_loaded"),
    "APRIL_RUNTIME_MAX_LOADED_SPECIALIST_MODELS": (
        "runtime",
        "max_loaded_specialist_models",
    ),
    "APRIL_DATABASE_PATH": ("memory", "database_path"),
    "APRIL_VECTOR_INDEX_PATH": ("memory", "vector_index_path"),
    "APRIL_MEMORY_EMBEDDING_PROVIDER": ("memory", "embedding_provider"),
    "APRIL_MEMORY_EMBEDDING_MODEL_ID": ("memory", "embedding_model_id"),
    "APRIL_LOGS_PATH": ("paths", "logs_path"),
    "APRIL_AUDIT_PATH": ("paths", "audit_path"),
    "APRIL_ALLOWED_FILESYSTEM_ROOTS": ("paths", "allowed_filesystem_roots"),
    "APRIL_MAX_FILE_READ_SIZE": ("paths", "max_file_read_bytes"),
    "APRIL_MAX_FILE_READ_BYTES": ("paths", "max_file_read_bytes"),
    "APRIL_TOOL_TIMEOUT": ("permissions", "tool_timeout_seconds"),
    "APRIL_APPROVAL_EXPIRY": ("permissions", "approval_expiry_seconds"),
    "APRIL_MAXIMUM_AGENT_TOOL_ITERATIONS": ("permissions", "maximum_agent_tool_iterations"),
    "APRIL_EXTERNAL_ACTIONS_ENABLED": ("permissions", "external_actions_enabled"),
    "APRIL_BRAIN_MODEL_ID": ("brain", "model_id"),
    "APRIL_VOICE_ENABLED": ("voice", "enabled"),
    "APRIL_AUDIO_CACHE_PATH": ("voice", "audio_cache_path"),
    "APRIL_VOICE_INPUT_DEVICE": ("voice", "input_device"),
    "APRIL_VOICE_OUTPUT_DEVICE": ("voice", "output_device"),
    "APRIL_VOICE_MAX_RECORD_SECONDS": ("voice", "max_record_seconds"),
    "APRIL_WHISPER_BINARY_PATH": ("voice", "whisper_binary_path"),
    "APRIL_WHISPER_MODEL_PATH": ("voice", "whisper_model_path"),
    "APRIL_PIPER_BINARY_PATH": ("voice", "piper_binary_path"),
    "APRIL_PIPER_MODEL_PATH": ("voice", "piper_model_path"),
    "APRIL_WAKE_WORD_MODEL_PATH": ("voice", "wake_word_model_path"),
    "APRIL_SCHEDULER_ENABLED": ("scheduler", "enabled"),
    "APRIL_SCHEDULER_POLL_INTERVAL_SECONDS": ("scheduler", "poll_interval_seconds"),
    "APRIL_SCHEDULER_NOTIFICATION_SINK": ("scheduler", "notification_sink"),
    "APRIL_SCHEDULER_BRIEFING_ENABLED": ("scheduler", "briefing_enabled"),
    "APRIL_SCHEDULER_BRIEFING_TIME": ("scheduler", "briefing_time"),
    "APRIL_SCHEDULER_REPO_MONITOR_ENABLED": ("scheduler", "repo_monitor_enabled"),
}


def _parse_env_value(raw: str) -> Any:
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration file must be a mapping: {path}")
    return loaded


def load_settings(config_path: Path | None = None, *, root: Path | None = None) -> AprilSettings:
    home = Path(os.environ.get("APRIL_HOME", root or project_root()))
    path = config_path or home / "configs" / "april.yaml"
    data = _read_yaml(path)
    data.setdefault("home", str(home))
    for env_name, field_path in ENV_OVERRIDES.items():
        if env_name in os.environ:
            if env_name == "APRIL_ALLOWED_FILESYSTEM_ROOTS":
                value = [part.strip() for part in os.environ[env_name].split(",") if part.strip()]
            else:
                value = _parse_env_value(os.environ[env_name])
            _set_nested(data, field_path, value)
    settings = AprilSettings.model_validate(data)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> AprilSettings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
