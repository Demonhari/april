"""Redacted, local-only structural fingerprint of APRIL's configuration.

A *fingerprint* captures only the non-secret structural shape of the active
configuration — model ids/roles/backends/basenames/chat-format presence, the
runtime backend, the memory embedding provider/model, whether voice is enabled
and which voice artifact slots are configured, and the structural permission
policy. It deliberately excludes anything sensitive: tokens, absolute paths,
local usernames, environment variables, raw GGUF metadata, prompts, transcripts,
and patch contents.

The fingerprint is embedded in newly generated verification reports so readiness
can answer "is this report still valid for the current configuration?" — a report
whose embedded fingerprint differs from the current one is *stale* even if it is
recent. Computing the fingerprint never opens the microphone, loads a model,
reaches the network, or mutates anything; it only reads ``configs`` + settings.

This is intentionally not a security boundary or a Git replacement: it is a
cheap, deterministic, redacted signature that works with no Git installed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from april_common.errors import ConfigError
from april_common.settings import AprilSettings, load_settings
from services.april_runtime.model_registry import ModelRegistry

# Bump when the *structure* of the fingerprint changes so older reports are not
# silently compared against an incompatible shape.
FINGERPRINT_SCHEMA_VERSION = 1

# Stable order of voice artifact slots reported as "configured" (basename present
# in config), never the path itself.
_VOICE_ARTIFACT_SLOTS = (
    "whisper_binary",
    "whisper_model",
    "piper_binary",
    "piper_model",
    "wake_word_model",
)


class FingerprintModel(BaseModel):
    """One model's redacted structural identity (no absolute path)."""

    id: str
    role: str
    backend: str
    # Basename only — never the absolute path or directory.
    path_basename: str | None = None
    # The chat-format family if explicitly configured (generic/granite/qwen), or
    # ``null`` when it is resolved from GGUF/native metadata or name inference.
    chat_format: str | None = None


class ConfigFingerprint(BaseModel):
    """A redacted structural signature of the active configuration."""

    schema_version: int = FINGERPRINT_SCHEMA_VERSION
    # Short, stable digest of every structural field below. Reports embed this so
    # staleness can be detected by comparing digests, not by re-deriving state.
    digest: str
    runtime_backend: str
    embedding_provider: str
    embedding_model_id: str | None = None
    voice_enabled: bool = False
    # Sorted artifact slot names that are configured (basename present in config).
    voice_artifacts_configured: list[str] = Field(default_factory=list)
    external_actions_enabled: bool = False
    # A structural signature of the permission policy (not a real "version" field —
    # APRIL has none — so we derive a stable one from the bounded policy knobs).
    permission_policy_signature: str = ""
    models: list[FingerprintModel] = Field(default_factory=list)


def _fingerprint_models(registry: ModelRegistry | None) -> list[FingerprintModel]:
    if registry is None:
        return []
    return [
        FingerprintModel(
            id=model.id,
            role=model.role,
            backend=model.backend,
            path_basename=model.resolved_path(registry.root).name,
            chat_format=model.chat_format,
        )
        for model in sorted(registry.list(), key=lambda item: item.id)
    ]


def _digest_payload(fingerprint: ConfigFingerprint) -> dict[str, object]:
    """Plain, deterministic dict the digest is computed over (excludes the digest)."""
    return {
        "schema_version": fingerprint.schema_version,
        "runtime_backend": fingerprint.runtime_backend,
        "embedding_provider": fingerprint.embedding_provider,
        "embedding_model_id": fingerprint.embedding_model_id,
        "voice_enabled": fingerprint.voice_enabled,
        "voice_artifacts_configured": list(fingerprint.voice_artifacts_configured),
        "external_actions_enabled": fingerprint.external_actions_enabled,
        "permission_policy_signature": fingerprint.permission_policy_signature,
        "models": [
            {
                "id": model.id,
                "role": model.role,
                "backend": model.backend,
                "path_basename": model.path_basename,
                "chat_format": model.chat_format,
            }
            for model in fingerprint.models
        ],
    }


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_config_fingerprint(
    settings: AprilSettings, *, registry: ModelRegistry | None = None
) -> ConfigFingerprint:
    """Build the redacted fingerprint from already-loaded settings.

    ``registry`` is loaded from ``configs/models.yaml`` when not supplied; a model
    registry that fails to load yields a fingerprint with no models rather than
    raising, so readiness/staleness never crash on a partially broken config.
    """
    if registry is None:
        try:
            registry = ModelRegistry.from_file(
                settings.home / "configs" / "models.yaml", root=settings.home
            )
        except ConfigError:
            registry = None
    voice_configured = sorted(
        slot
        for slot in _VOICE_ARTIFACT_SLOTS
        if getattr(settings.voice, f"{slot}_path", None) is not None
    )
    permission_signature = (
        f"expiry={settings.permissions.approval_expiry_seconds}"
        f";iters={settings.permissions.maximum_agent_tool_iterations}"
        f";external={int(settings.permissions.external_actions_enabled)}"
    )
    fingerprint = ConfigFingerprint(
        digest="",
        runtime_backend=settings.runtime.backend,
        embedding_provider=settings.memory.embedding_provider,
        embedding_model_id=settings.memory.embedding_model_id,
        voice_enabled=settings.voice.enabled,
        voice_artifacts_configured=voice_configured,
        external_actions_enabled=settings.permissions.external_actions_enabled,
        permission_policy_signature=permission_signature,
        models=_fingerprint_models(registry),
    )
    return fingerprint.model_copy(update={"digest": _digest(_digest_payload(fingerprint))})


def config_fingerprint_for_home(home: Path) -> ConfigFingerprint | None:
    """Best-effort fingerprint for an APRIL home; ``None`` if settings cannot load.

    Never raises — a broken config simply yields ``None`` so callers fall back to
    timestamp-only freshness.
    """
    try:
        settings = load_settings(root=home.expanduser().resolve())
    except ConfigError:
        return None
    return build_config_fingerprint(settings)


def config_fingerprint_digest(home: Path) -> str | None:
    fingerprint = config_fingerprint_for_home(home)
    return fingerprint.digest if fingerprint is not None else None
