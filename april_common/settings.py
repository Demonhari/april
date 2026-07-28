from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from april_common.errors import ConfigError

KNOWN_DEFAULT_API_TOKENS = {"local-dev-token"}
KNOWN_DEFAULT_RUNTIME_TOKENS = {"local-dev-runtime-token"}
# Placeholder values shipped in .env.example. They are never secret and must be
# treated exactly like the known development defaults: warn in dev/test, reject
# in production. Keeping them distinct lets readiness say *why* a token is unsafe.
PLACEHOLDER_API_TOKENS = {"change-me-local-token"}
PLACEHOLDER_RUNTIME_TOKENS = {"change-me-runtime-token"}
INSECURE_API_TOKENS = KNOWN_DEFAULT_API_TOKENS | PLACEHOLDER_API_TOKENS
INSECURE_RUNTIME_TOKENS = KNOWN_DEFAULT_RUNTIME_TOKENS | PLACEHOLDER_RUNTIME_TOKENS
SAFE_TOKEN_ENVIRONMENTS = {"development", "test"}

# A POSIX-style environment variable name.
_DOTENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    # repr=False keeps the bearer token out of repr()/str() so it cannot leak
    # into logs, tracebacks or diagnostics. model_dump() still includes it so
    # callers that need it (and redact explicitly) keep working.
    token: str = Field(default="local-dev-token", repr=False)
    cors_enabled: bool = False
    max_request_bytes: int = 1_048_576


class RuntimeSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8766
    url: str = "http://127.0.0.1:8766"
    token: str | None = Field(default=None, repr=False)
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
    router_model_id: str | None = None
    routing_reliability_min_samples: int = Field(default=5, ge=1, le=1000)
    routing_reliability_prior_successes: float = Field(default=4.0, gt=0.0, le=100.0)
    routing_reliability_prior_failures: float = Field(default=4.0, gt=0.0, le=100.0)


class ConversationContextSettings(BaseModel):
    """Core-side character pre-bounds; Runtime remains the exact token authority."""

    summary_enabled: bool = True
    recent_turns_preserved: int = Field(default=4, ge=4, le=32)
    older_turns_before_summary: int = Field(default=3, ge=1, le=100)
    max_turns_per_summary: int = Field(default=8, ge=1, le=50)
    summary_max_output_tokens: int = Field(default=512, ge=64, le=2048)
    summary_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    rendered_summary_max_chars: int = Field(default=4000, ge=500, le=20_000)
    conversation_history_max_chars: int = Field(default=8000, ge=1000, le=40_000)
    durable_memory_max_chars: int = Field(default=4000, ge=0, le=20_000)
    file_document_max_chars: int = Field(default=6000, ge=0, le=40_000)
    tool_output_max_chars: int = Field(default=3000, ge=0, le=20_000)


class VoiceSettings(BaseModel):
    enabled: bool = False
    audio_cache_path: Path = Path("data/audio_cache")
    retain_debug_audio: bool = False
    input_device: str | int | None = None
    output_device: str | int | None = None
    max_record_seconds: float = 30.0
    vad_energy_threshold: float = Field(default=0.01, gt=0.0, le=0.5)
    # ``vad_required_frames`` is the deprecated onset alias.  A before
    # validator copies it only when the explicit setting is absent.
    vad_onset_frames: int = Field(default=3, ge=1, le=50)
    vad_required_frames: int = Field(default=3, ge=1, le=50)
    endpoint_silence_ms: int = Field(default=650, ge=300, le=2_000)
    minimum_utterance_ms: int = Field(default=300, ge=50, le=5_000)
    noise_calibration_ms: int = Field(default=300, ge=0, le=2_000)
    noise_threshold_multiplier: float = Field(default=2.5, ge=1.0, le=10.0)
    noise_threshold_margin: float = Field(default=0.002, ge=0.0, le=0.1)
    vad_hangover_ms: int = Field(default=100, ge=0, le=1_000)
    barge_in_trigger: Literal["wake_word", "speech", "off"] = "wake_word"
    barge_in_action: Literal["stop", "duck"] = "stop"
    barge_in_speech_onset_frames: int = Field(default=6, ge=3, le=100)
    barge_in_playback_grace_ms: int = Field(default=400, ge=0, le=5_000)
    wake_word_threshold: float = 0.5
    wake_word_cooldown_seconds: float = 2.0
    # Separate, independently bounded timeouts for the wake/listen lifecycle.
    wake_wait_seconds: float = 30.0
    utterance_max_seconds: float = Field(default=15.0, gt=0.0, le=60.0)
    # Bounded pre-roll so the onset of speech is not lost while VAD confirms it.
    wake_pre_roll_frames: int = 8
    whisper_binary_path: Path | None = None
    whisper_model_path: Path | None = None
    # Optional two-stage STT paths. Missing values inherit the legacy Whisper
    # paths so existing configurations keep identical behaviour.
    wake_confirmation_whisper_binary_path: Path | None = None
    wake_confirmation_whisper_model_path: Path | None = None
    transcription_whisper_binary_path: Path | None = None
    transcription_whisper_model_path: Path | None = None
    piper_binary_path: Path | None = None
    piper_model_path: Path | None = None
    wake_word_model_path: Path | None = None
    # v2: the Sentinel supports several wake models at once. The legacy singular
    # wake_word_model_path stays honoured; effective_wake_word_model_paths merges
    # both without duplicates so old configs keep working unchanged.
    wake_word_model_paths: list[Path] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _honour_deprecated_vad_onset_alias(cls, value: object) -> object:
        if isinstance(value, dict):
            normalized = dict(value)
            if "vad_onset_frames" not in normalized and "vad_required_frames" in normalized:
                normalized["vad_onset_frames"] = normalized["vad_required_frames"]
            return normalized
        return value

    @model_validator(mode="after")
    def _validate_turn_taking(self) -> VoiceSettings:
        values = (
            self.vad_energy_threshold,
            self.noise_threshold_multiplier,
            self.noise_threshold_margin,
            self.utterance_max_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Voice turn-taking numeric settings must be finite.")
        maximum_ms = self.utterance_max_seconds * 1_000
        if self.minimum_utterance_ms > maximum_ms:
            raise ValueError("minimum_utterance_ms cannot exceed utterance_max_seconds.")
        if self.vad_hangover_ms > self.endpoint_silence_ms:
            raise ValueError("vad_hangover_ms cannot exceed endpoint_silence_ms.")
        if self.noise_calibration_ms > maximum_ms:
            raise ValueError("noise_calibration_ms cannot exceed utterance_max_seconds.")
        return self

    @property
    def effective_confirmation_whisper_binary_path(self) -> Path | None:
        return self.wake_confirmation_whisper_binary_path or self.whisper_binary_path

    @property
    def effective_confirmation_whisper_model_path(self) -> Path | None:
        return self.wake_confirmation_whisper_model_path or self.whisper_model_path

    @property
    def effective_transcription_whisper_binary_path(self) -> Path | None:
        return self.transcription_whisper_binary_path or self.whisper_binary_path

    @property
    def effective_transcription_whisper_model_path(self) -> Path | None:
        return self.transcription_whisper_model_path or self.whisper_model_path

    @property
    def effective_wake_word_model_paths(self) -> list[Path]:
        merged: list[Path] = []
        for candidate in [self.wake_word_model_path, *self.wake_word_model_paths]:
            if candidate is not None and candidate not in merged:
                merged.append(candidate)
        return merged


class WakeSettings(BaseModel):
    """Always-on wake (Sentinel) policy. OFF by default; entirely local."""

    enabled: bool = False
    # Two-stage wake: a cheap openWakeWord score above candidate_threshold makes
    # a candidate; accept_threshold accepts outright, otherwise STT confirms.
    candidate_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    accept_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    confirm_with_stt: bool = True
    # With confirm_with_stt on, instant_accept lets a score at or above
    # accept_threshold wake without waiting for STT; lower-confidence
    # candidates are still confirmed. Off = every candidate is confirmed.
    instant_accept: bool = True
    # Maximum normalized edit distance for fuzzy wake-word matching in STT
    # transcripts (0.25 allows one edit on a five-letter word).
    fuzzy_max_distance: float = Field(default=0.25, ge=0.0, le=0.5)
    ring_buffer_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    follow_up_seconds: float = Field(default=8.0, ge=0.0, le=120.0)
    earcon_enabled: bool = True
    strict_address: bool = False
    speaker_gate: str = "off"
    speaker_verifier_model_path: Path | None = None

    @field_validator("speaker_gate", mode="before")
    @classmethod
    def coerce_speaker_gate(cls, value: object) -> object:
        # YAML 1.1 parses an unquoted `off` as boolean False; accept it as "off".
        if value is False:
            return "off"
        return value

    @field_validator("speaker_gate")
    @classmethod
    def validate_speaker_gate(cls, value: str) -> str:
        if value not in {"off", "soft"}:
            raise ValueError("speaker_gate must be off or soft")
        return value

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> WakeSettings:
        if self.accept_threshold < self.candidate_threshold:
            raise ValueError("wake.accept_threshold must be >= wake.candidate_threshold")
        return self


class SessionSettings(BaseModel):
    """Cross-surface conversation continuity for wake events."""

    continuity_minutes: float = Field(default=10.0, ge=0.0, le=24 * 60.0)


class DaemonSettings(BaseModel):
    """apriald supervisor behaviour."""

    autostart_on_cli: bool = True
    startup_timeout_seconds: float = Field(default=15.0, gt=0.0, le=300.0)
    shutdown_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    health_poll_seconds: float = Field(default=0.2, gt=0.0, le=5.0)
    child_startup_grace_seconds: float = Field(default=10.0, ge=0.0, le=300.0)
    child_health_failure_threshold: int = Field(default=3, ge=2, le=20)
    child_crash_loop_threshold: int = Field(default=5, ge=1, le=50)


class GovernorSettings(BaseModel):
    """Resource Governor budgets consulted by Sentinel/Dreamer/prewarm."""

    max_resident_gb: float = Field(default=12.0, gt=0.0)
    dreamer_nice: int = Field(default=10, ge=0, le=20)
    generation_threads_active: int = Field(default=6, ge=1, le=256)
    generation_threads_idle: int = Field(default=8, ge=1, le=256)

    @model_validator(mode="after")
    def generation_thread_budgets_are_ordered(self) -> GovernorSettings:
        if self.generation_threads_active > self.generation_threads_idle:
            raise ValueError(
                "governor.generation_threads_active must be <= generation_threads_idle"
            )
        return self


class EvolutionSettings(BaseModel):
    """Nightly Dreamer self-evolution. OFF by default; writes are fenced."""

    enabled: bool = False
    window: str = "02:30-06:00"
    require_ac_power: bool = True
    max_minutes: int = Field(default=90, ge=1, le=24 * 60)
    recheck_governor_between_phases: bool = True
    daily_memory_cap: int = Field(default=30, ge=0)
    # Archive reflection candidates below this confidence are discarded
    # (architecture default: 0.5).
    archive_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    prompt_overlay_max_chars: int = Field(default=1200, ge=0, le=20_000)
    model_drafted_overlays: bool = False
    user_model_autoapply: str = "safe_sections_only"

    @field_validator("window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        start, separator, end = value.partition("-")
        if not separator:
            raise ValueError("evolution.window must be HH:MM-HH:MM")
        for part in (start, end):
            hours, _, minutes = part.strip().partition(":")
            try:
                hour, minute = int(hours), int(minutes)
            except ValueError as exc:
                raise ValueError("evolution.window must be HH:MM-HH:MM") from exc
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("evolution.window times must be within 00:00-23:59")
        return value

    @field_validator("user_model_autoapply")
    @classmethod
    def validate_autoapply(cls, value: str) -> str:
        if value not in {"off", "safe_sections_only"}:
            raise ValueError("user_model_autoapply must be off or safe_sections_only")
        return value


class DeepModeSettings(BaseModel):
    """Budgets for the intelligence ladder's deep/council rungs."""

    max_seconds: float = Field(default=45.0, gt=0.0, le=600.0)
    council_n: int = Field(default=3, ge=2, le=5)
    council_mode: Literal["reasoning_n", "multi_agent"] = "reasoning_n"
    deep_confidence_threshold: float = 0.4
    verified_confidence_threshold: float = 0.7
    verified_draft_tokens: int = Field(default=1024, ge=64, le=8192)
    verified_critique_tokens: int = Field(default=512, ge=64, le=8192)
    verified_revision_tokens: int = Field(default=1024, ge=64, le=8192)
    deep_tokens: int = Field(default=1536, ge=64, le=16384)
    council_candidate_tokens: int = Field(default=1024, ge=64, le=8192)
    council_judge_tokens: int = Field(default=512, ge=64, le=4096)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> DeepModeSettings:
        deep = self.deep_confidence_threshold
        verified = self.verified_confidence_threshold
        if not (0.0 < deep < verified < 1.0):
            raise ValueError(
                "deep_mode thresholds must satisfy "
                "0 < deep_confidence_threshold < verified_confidence_threshold < 1"
            )
        return self


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
    environment: Literal["development", "test", "production"] = "development"
    api: ApiSettings = Field(default_factory=ApiSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    brain: BrainSettings = Field(default_factory=BrainSettings)
    conversation_context: ConversationContextSettings = Field(
        default_factory=ConversationContextSettings
    )
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    wake: WakeSettings = Field(default_factory=WakeSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    daemon: DaemonSettings = Field(default_factory=DaemonSettings)
    governor: GovernorSettings = Field(default_factory=GovernorSettings)
    evolution: EvolutionSettings = Field(default_factory=EvolutionSettings)
    deep_mode: DeepModeSettings = Field(default_factory=DeepModeSettings)

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

    @property
    def wake_socket_path(self) -> Path:
        return self.resolve_path(Path("data/wake.sock"))

    @property
    def mute_flag_path(self) -> Path:
        return self.resolve_path(Path("data/voice.mute"))

    @property
    def evolution_path(self) -> Path:
        return self.resolve_path(Path("data/evolution"))

    @property
    def playbooks_path(self) -> Path:
        return self.resolve_path(Path("data/playbooks"))


ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "APRIL_HOME": ("home",),
    "APRIL_ENV": ("environment",),
    "APRIL_ENVIRONMENT": ("environment",),
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
    "APRIL_ROUTER_MODEL_ID": ("brain", "router_model_id"),
    "APRIL_ROUTING_RELIABILITY_MIN_SAMPLES": (
        "brain",
        "routing_reliability_min_samples",
    ),
    "APRIL_VOICE_ENABLED": ("voice", "enabled"),
    "APRIL_AUDIO_CACHE_PATH": ("voice", "audio_cache_path"),
    "APRIL_VOICE_INPUT_DEVICE": ("voice", "input_device"),
    "APRIL_VOICE_OUTPUT_DEVICE": ("voice", "output_device"),
    "APRIL_VOICE_MAX_RECORD_SECONDS": ("voice", "max_record_seconds"),
    "APRIL_VOICE_VAD_ENERGY_THRESHOLD": ("voice", "vad_energy_threshold"),
    "APRIL_VOICE_VAD_ONSET_FRAMES": ("voice", "vad_onset_frames"),
    "APRIL_VOICE_VAD_REQUIRED_FRAMES": ("voice", "vad_required_frames"),
    "APRIL_VOICE_ENDPOINT_SILENCE_MS": ("voice", "endpoint_silence_ms"),
    "APRIL_VOICE_MINIMUM_UTTERANCE_MS": ("voice", "minimum_utterance_ms"),
    "APRIL_VOICE_NOISE_CALIBRATION_MS": ("voice", "noise_calibration_ms"),
    "APRIL_VOICE_NOISE_THRESHOLD_MULTIPLIER": (
        "voice",
        "noise_threshold_multiplier",
    ),
    "APRIL_VOICE_NOISE_THRESHOLD_MARGIN": ("voice", "noise_threshold_margin"),
    "APRIL_VOICE_VAD_HANGOVER_MS": ("voice", "vad_hangover_ms"),
    "APRIL_VOICE_UTTERANCE_MAX_SECONDS": ("voice", "utterance_max_seconds"),
    "APRIL_VOICE_BARGE_IN_TRIGGER": ("voice", "barge_in_trigger"),
    "APRIL_VOICE_BARGE_IN_ACTION": ("voice", "barge_in_action"),
    "APRIL_VOICE_BARGE_IN_SPEECH_ONSET_FRAMES": (
        "voice",
        "barge_in_speech_onset_frames",
    ),
    "APRIL_VOICE_BARGE_IN_PLAYBACK_GRACE_MS": (
        "voice",
        "barge_in_playback_grace_ms",
    ),
    "APRIL_WHISPER_BINARY_PATH": ("voice", "whisper_binary_path"),
    "APRIL_WHISPER_MODEL_PATH": ("voice", "whisper_model_path"),
    "APRIL_WAKE_CONFIRMATION_WHISPER_BINARY_PATH": (
        "voice",
        "wake_confirmation_whisper_binary_path",
    ),
    "APRIL_WAKE_CONFIRMATION_WHISPER_MODEL_PATH": (
        "voice",
        "wake_confirmation_whisper_model_path",
    ),
    "APRIL_TRANSCRIPTION_WHISPER_BINARY_PATH": (
        "voice",
        "transcription_whisper_binary_path",
    ),
    "APRIL_TRANSCRIPTION_WHISPER_MODEL_PATH": (
        "voice",
        "transcription_whisper_model_path",
    ),
    "APRIL_PIPER_BINARY_PATH": ("voice", "piper_binary_path"),
    "APRIL_PIPER_MODEL_PATH": ("voice", "piper_model_path"),
    "APRIL_WAKE_WORD_MODEL_PATH": ("voice", "wake_word_model_path"),
    "APRIL_WAKE_WORD_MODEL_PATHS": ("voice", "wake_word_model_paths"),
    "APRIL_WAKE_ENABLED": ("wake", "enabled"),
    "APRIL_WAKE_CANDIDATE_THRESHOLD": ("wake", "candidate_threshold"),
    "APRIL_WAKE_ACCEPT_THRESHOLD": ("wake", "accept_threshold"),
    "APRIL_WAKE_CONFIRM_WITH_STT": ("wake", "confirm_with_stt"),
    "APRIL_WAKE_INSTANT_ACCEPT": ("wake", "instant_accept"),
    "APRIL_WAKE_FUZZY_MAX_DISTANCE": ("wake", "fuzzy_max_distance"),
    "APRIL_WAKE_RING_BUFFER_SECONDS": ("wake", "ring_buffer_seconds"),
    "APRIL_WAKE_FOLLOW_UP_SECONDS": ("wake", "follow_up_seconds"),
    "APRIL_WAKE_EARCON_ENABLED": ("wake", "earcon_enabled"),
    "APRIL_WAKE_STRICT_ADDRESS": ("wake", "strict_address"),
    "APRIL_WAKE_SPEAKER_GATE": ("wake", "speaker_gate"),
    "APRIL_WAKE_SPEAKER_VERIFIER_MODEL_PATH": (
        "wake",
        "speaker_verifier_model_path",
    ),
    "APRIL_SESSION_CONTINUITY_MINUTES": ("session", "continuity_minutes"),
    "APRIL_DAEMON_AUTOSTART_ON_CLI": ("daemon", "autostart_on_cli"),
    "APRIL_GOVERNOR_MAX_RESIDENT_GB": ("governor", "max_resident_gb"),
    "APRIL_GOVERNOR_DREAMER_NICE": ("governor", "dreamer_nice"),
    "APRIL_GOVERNOR_GENERATION_THREADS_ACTIVE": (
        "governor",
        "generation_threads_active",
    ),
    "APRIL_GOVERNOR_GENERATION_THREADS_IDLE": (
        "governor",
        "generation_threads_idle",
    ),
    "APRIL_EVOLUTION_ENABLED": ("evolution", "enabled"),
    "APRIL_EVOLUTION_WINDOW": ("evolution", "window"),
    "APRIL_EVOLUTION_REQUIRE_AC_POWER": ("evolution", "require_ac_power"),
    "APRIL_EVOLUTION_MAX_MINUTES": ("evolution", "max_minutes"),
    "APRIL_EVOLUTION_RECHECK_GOVERNOR_BETWEEN_PHASES": (
        "evolution",
        "recheck_governor_between_phases",
    ),
    "APRIL_EVOLUTION_DAILY_MEMORY_CAP": ("evolution", "daily_memory_cap"),
    "APRIL_EVOLUTION_PROMPT_OVERLAY_MAX_CHARS": ("evolution", "prompt_overlay_max_chars"),
    "APRIL_EVOLUTION_MODEL_DRAFTED_OVERLAYS": ("evolution", "model_drafted_overlays"),
    "APRIL_DEEP_MODE_MAX_SECONDS": ("deep_mode", "max_seconds"),
    "APRIL_DEEP_MODE_COUNCIL_N": ("deep_mode", "council_n"),
    "APRIL_DEEP_MODE_COUNCIL_MODE": ("deep_mode", "council_mode"),
    "APRIL_DEEP_MODE_DEEP_CONFIDENCE_THRESHOLD": (
        "deep_mode",
        "deep_confidence_threshold",
    ),
    "APRIL_DEEP_MODE_VERIFIED_CONFIDENCE_THRESHOLD": (
        "deep_mode",
        "verified_confidence_threshold",
    ),
    "APRIL_SCHEDULER_ENABLED": ("scheduler", "enabled"),
    "APRIL_SCHEDULER_POLL_INTERVAL_SECONDS": ("scheduler", "poll_interval_seconds"),
    "APRIL_SCHEDULER_NOTIFICATION_SINK": ("scheduler", "notification_sink"),
    "APRIL_SCHEDULER_BRIEFING_ENABLED": ("scheduler", "briefing_enabled"),
    "APRIL_SCHEDULER_BRIEFING_TIME": ("scheduler", "briefing_time"),
    "APRIL_SCHEDULER_REPO_MONITOR_ENABLED": ("scheduler", "repo_monitor_enabled"),
}


# Keys honoured from ${APRIL_HOME}/.env. APRIL_HOME itself is excluded: the file
# lives inside APRIL_HOME, so it must not be able to relocate its own directory.
_DOTENV_SUPPORTED_KEYS = frozenset(ENV_OVERRIDES) - {"APRIL_HOME"}

# Env vars that map to an Optional[...] setting. A blank value from the process
# env or ${APRIL_HOME}/.env is an explicit *unset* and must become None — never
# Path(".") (the repo root) or an empty string. Non-empty values are untouched,
# so a legitimate relative path still resolves normally. The API token is
# deliberately excluded: its field is a non-optional str, so a blank stays "" and
# is reported as missing/insecure rather than silently becoming the default.
_OPTIONAL_BLANK_IS_NONE: frozenset[str] = frozenset(
    {
        "APRIL_RUNTIME_TOKEN",
        "APRIL_MEMORY_EMBEDDING_MODEL_ID",
        "APRIL_VOICE_INPUT_DEVICE",
        "APRIL_VOICE_OUTPUT_DEVICE",
        "APRIL_WHISPER_BINARY_PATH",
        "APRIL_WHISPER_MODEL_PATH",
        "APRIL_PIPER_BINARY_PATH",
        "APRIL_PIPER_MODEL_PATH",
        "APRIL_WAKE_WORD_MODEL_PATH",
    }
)


def _strip_inline_comment(value: str) -> str:
    # Drop a trailing `# comment` only when it follows whitespace, matching common
    # dotenv behaviour. A bare `#` inside an unquoted value stays literal.
    for index, char in enumerate(value):
        if char == "#" and index > 0 and value[index - 1] in " \t":
            return value[:index].rstrip()
    return value


def _parse_dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        inner = value[1:-1]
        if value[0] == '"':
            inner = (
                inner.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return inner
    return _strip_inline_comment(value)


def _read_dotenv(path: Path, supported_keys: frozenset[str]) -> dict[str, str]:
    """Parse a ${APRIL_HOME}/.env file into supported APRIL_* keys.

    The parser is intentionally inert: it never executes, interpolates, or
    shell-evaluates content. Lines that are blank, comments, or malformed are
    skipped silently rather than raising, and only known APRIL keys are returned.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not _DOTENV_KEY_RE.match(key) or key not in supported_keys:
            continue
        values[key] = _parse_dotenv_value(value)
    return values


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
    # Effective-settings precedence, highest first:
    #   1. process environment   2. ${APRIL_HOME}/.env   3. YAML   4. model defaults
    dotenv = _read_dotenv(home / ".env", _DOTENV_SUPPORTED_KEYS)
    path = config_path or home / "configs" / "april.yaml"
    data = _read_yaml(path)
    data.setdefault("home", str(home))
    for env_name, field_path in ENV_OVERRIDES.items():
        # os.environ always wins; .env only fills keys absent from the process env.
        raw = os.environ[env_name] if env_name in os.environ else dotenv.get(env_name)
        if raw is None:
            continue
        if not raw.strip() and env_name in _OPTIONAL_BLANK_IS_NONE:
            # Explicit blank for an optional setting means "unset", not Path(".").
            value: Any = None
        elif env_name in {"APRIL_ALLOWED_FILESYSTEM_ROOTS", "APRIL_WAKE_WORD_MODEL_PATHS"}:
            value = [part.strip() for part in raw.split(",") if part.strip()]
        else:
            value = _parse_env_value(raw)
        _set_nested(data, field_path, value)
    settings = AprilSettings.model_validate(data)
    _validate_default_tokens(settings)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> AprilSettings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def _validate_default_tokens(settings: AprilSettings) -> None:
    if settings.environment in SAFE_TOKEN_ENVIRONMENTS:
        return
    # Outside development/test, a token that is empty, a known development default,
    # or a .env.example placeholder is rejected at startup; a missing runtime token
    # is rejected too. Real strong tokens come from `run april setup tokens`.
    insecure: list[str] = []
    if not settings.api.token or settings.api.token in INSECURE_API_TOKENS:
        insecure.append("APRIL_API_TOKEN")
    if not settings.runtime.token or settings.runtime.token in INSECURE_RUNTIME_TOKENS:
        insecure.append("APRIL_RUNTIME_TOKEN")
    if insecure:
        raise ConfigError(
            "Known development tokens, placeholder tokens, or empty/missing tokens "
            "are not allowed outside development/test mode.",
            {"environment": settings.environment, "settings": insecure},
        )
