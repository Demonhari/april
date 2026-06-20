from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from april_common.settings import AprilSettings


class VoiceComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "disabled"]
    message: str


class VoiceHealth(BaseModel):
    status: Literal["ok", "degraded", "disabled"]
    components: list[VoiceComponentHealth] = Field(default_factory=list)


def _binary_health(name: str, path: Path | None) -> VoiceComponentHealth:
    if path is None:
        return VoiceComponentHealth(name=name, status="degraded", message="No path configured.")
    if not path.exists():
        return VoiceComponentHealth(name=name, status="degraded", message=f"Missing: {path}")
    return VoiceComponentHealth(name=name, status="ok", message=str(path))


def voice_health(settings: AprilSettings) -> VoiceHealth:
    if not settings.voice.enabled:
        return VoiceHealth(status="disabled", components=[])
    components = [
        _binary_health("whisper.cpp", settings.voice.whisper_binary_path),
        _binary_health("whisper model", settings.voice.whisper_model_path),
        _binary_health("piper", settings.voice.piper_binary_path),
        _binary_health("piper model", settings.voice.piper_model_path),
        _binary_health("wake word model", settings.voice.wake_word_model_path),
    ]
    status = "ok" if all(component.status == "ok" for component in components) else "degraded"
    return VoiceHealth(status=status, components=components)
