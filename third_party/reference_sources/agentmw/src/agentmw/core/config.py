"""Central configuration for agentmw.

Precedence (highest wins):
    1. explicit kwargs in code
    2. environment variables (AGENTMW_*)
    3. ~/.agentmw/config.toml  (or $AGENTMW_CONFIG)
    4. built-in defaults

Nothing in the library reads env vars or files directly. Code reads
`AgentmwConfig` instance attributes only. Tests pass synthetic configs.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:  # 3.11+
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass
class MonitorConfig:
    """Heuristic fallback monitor tuning. Used when the LLM monitor is off
    or the configured provider is unreachable."""

    loop_window: int = 6
    loop_threshold: int = 3
    second_guess_min_count: int = 2


@dataclass
class CompressionConfig:
    keep_recent: int = 2
    max_tool_result_chars: int = 240


@dataclass
class MemoryConfig:
    db_path: str | None = None  # None → ~/.agentmw/memory.db
    semantic_threshold: float = 0.55
    recall_limit: int = 3
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@dataclass
class ProviderConfig:
    """Which LLM provider powers the LLM-based monitors / judges.

    `name` is "auto" by default: agentmw picks the first available among
    ollama → anthropic → openai → openrouter based on reachability + keys.
    """

    name: str = "auto"  # auto | ollama | openai | anthropic | openrouter | none
    model: str | None = None  # provider default if None
    api_key: str | None = None  # for cloud providers
    base_url: str | None = None  # override (ollama host, openai-compat endpoint, …)
    timeout_seconds: float = 30.0
    max_retries: int = 2


@dataclass
class MonitorPipelineConfig:
    """Pipeline orchestration: LLM primary, heuristics fallback."""

    use_llm: bool = True
    use_heuristics_fallback: bool = True
    # If `use_llm` is True AND provider works, heuristics still run as a fast
    # prefilter (cheap, deterministic). LLM verdict overrides on conflict.
    heuristics_prefilter: bool = True


@dataclass
class ExtractorConfig:
    enabled: bool = True              # auto-extract patterns from completed sessions
    on_completion_only: bool = True   # only run when session looks "done"
    background: bool = True           # run extractor in a daemon thread
    dedup_threshold: float = 0.92


@dataclass
class RecorderConfig:
    enabled: bool = True              # auto-save every session to disk
    directory: str | None = None      # None → ~/.agentmw/sessions/


@dataclass
class BreakerConfig:
    enabled: bool = True
    failure_threshold: int = 3
    failure_window_seconds: float = 30.0
    cooldown_seconds: float = 60.0


@dataclass
class TelemetryConfig:
    enabled: bool = True
    flush_every_calls: int = 1   # safe default; raise for high-volume workloads


@dataclass
class AgentmwConfig:
    monitors: MonitorConfig = field(default_factory=MonitorConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    pipeline: MonitorPipelineConfig = field(default_factory=MonitorPipelineConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_env_and_file(cls, config_path: str | Path | None = None) -> "AgentmwConfig":
        cfg = cls()
        cfg = cfg._apply_file(config_path)
        cfg = cfg._apply_env()
        return cfg

    def _apply_file(self, config_path: str | Path | None) -> "AgentmwConfig":
        if tomllib is None:
            return self
        path = Path(config_path) if config_path else Path(
            os.environ.get("AGENTMW_CONFIG") or os.path.expanduser("~/.agentmw/config.toml")
        )
        if not path.is_file():
            return self
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, ValueError):
            return self
        return _merge_into(self, data)

    def _apply_env(self) -> "AgentmwConfig":
        # Monitor tuning
        self.monitors.loop_window = _env_int("AGENTMW_LOOP_WINDOW", self.monitors.loop_window)
        self.monitors.loop_threshold = _env_int("AGENTMW_LOOP_THRESHOLD", self.monitors.loop_threshold)
        self.monitors.second_guess_min_count = _env_int(
            "AGENTMW_SECOND_GUESS_MIN", self.monitors.second_guess_min_count
        )
        # Compression
        self.compression.keep_recent = _env_int("AGENTMW_COMPRESS_KEEP", self.compression.keep_recent)
        self.compression.max_tool_result_chars = _env_int(
            "AGENTMW_COMPRESS_MAX_CHARS", self.compression.max_tool_result_chars
        )
        # Memory
        self.memory.db_path = os.environ.get("AGENTMW_DB_PATH") or self.memory.db_path
        self.memory.semantic_threshold = _env_float(
            "AGENTMW_SEMANTIC_THRESHOLD", self.memory.semantic_threshold
        )
        self.memory.recall_limit = _env_int("AGENTMW_RECALL_LIMIT", self.memory.recall_limit)
        self.memory.embedding_model = os.environ.get(
            "AGENTMW_EMBEDDING_MODEL", self.memory.embedding_model
        )
        # Provider
        self.provider.name = os.environ.get("AGENTMW_PROVIDER", self.provider.name)
        self.provider.model = os.environ.get("AGENTMW_MODEL", self.provider.model)
        self.provider.api_key = (
            os.environ.get("AGENTMW_API_KEY") or self.provider.api_key
        )
        self.provider.base_url = os.environ.get("AGENTMW_BASE_URL", self.provider.base_url)
        self.provider.timeout_seconds = _env_float(
            "AGENTMW_TIMEOUT", self.provider.timeout_seconds
        )
        self.provider.max_retries = _env_int("AGENTMW_MAX_RETRIES", self.provider.max_retries)
        # Pipeline
        self.pipeline.use_llm = _env_bool("AGENTMW_USE_LLM", self.pipeline.use_llm)
        self.pipeline.use_heuristics_fallback = _env_bool(
            "AGENTMW_USE_HEURISTICS", self.pipeline.use_heuristics_fallback
        )
        self.pipeline.heuristics_prefilter = _env_bool(
            "AGENTMW_HEURISTICS_PREFILTER", self.pipeline.heuristics_prefilter
        )
        # Extractor / recorder / breaker / telemetry
        self.extractor.enabled = _env_bool("AGENTMW_EXTRACTOR", self.extractor.enabled)
        self.extractor.background = _env_bool("AGENTMW_EXTRACTOR_BG", self.extractor.background)
        self.recorder.enabled = _env_bool("AGENTMW_RECORDER", self.recorder.enabled)
        self.recorder.directory = os.environ.get("AGENTMW_RECORDER_DIR", self.recorder.directory)
        self.breaker.enabled = _env_bool("AGENTMW_BREAKER", self.breaker.enabled)
        self.breaker.failure_threshold = _env_int(
            "AGENTMW_BREAKER_THRESHOLD", self.breaker.failure_threshold
        )
        self.telemetry.enabled = _env_bool("AGENTMW_TELEMETRY", self.telemetry.enabled)
        return self


def _merge_into(cfg: AgentmwConfig, data: dict[str, Any]) -> AgentmwConfig:
    """Apply a TOML-parsed dict into an AgentmwConfig in place."""
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        target = getattr(cfg, section_name, None)
        if target is None or not dataclasses.is_dataclass(target):
            continue
        target_fields = {f.name for f in fields(target)}
        for key, value in section_data.items():
            if key in target_fields:
                setattr(target, key, value)
    return cfg


def default_config() -> AgentmwConfig:
    """Convenience: build a config from env + file, falling back to defaults."""
    return AgentmwConfig.from_env_and_file()
