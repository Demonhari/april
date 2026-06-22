from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Literal

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


def _configured_path_health(
    name: str, settings: AprilSettings, path: Path | None
) -> VoiceComponentHealth:
    if path is None:
        return VoiceComponentHealth(name=name, status="degraded", message="No path configured.")
    resolved = settings.resolve_path(path)
    if not resolved.exists():
        return VoiceComponentHealth(name=name, status="degraded", message=f"Missing: {resolved}")
    return VoiceComponentHealth(name=name, status="ok", message=str(resolved))


def _audio_cache_health(settings: AprilSettings) -> VoiceComponentHealth:
    path = settings.audio_cache_path
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return VoiceComponentHealth(
            name="audio cache",
            status="degraded",
            message=f"Not writable: {exc}",
        )
    return VoiceComponentHealth(name="audio cache", status="ok", message=str(path))


def voice_health(settings: AprilSettings) -> VoiceHealth:
    if not settings.voice.enabled:
        return VoiceHealth(status="disabled", components=[])
    components = [
        _configured_path_health("whisper.cpp", settings, settings.voice.whisper_binary_path),
        _configured_path_health("whisper model", settings, settings.voice.whisper_model_path),
        _configured_path_health("piper", settings, settings.voice.piper_binary_path),
        _configured_path_health("piper model", settings, settings.voice.piper_model_path),
        _configured_path_health("wake word model", settings, settings.voice.wake_word_model_path),
    ]
    status = "ok" if all(component.status == "ok" for component in components) else "degraded"
    return VoiceHealth(status=status, components=components)


def query_audio_devices() -> dict[str, Any]:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        # ImportError: package not installed. OSError: the sounddevice package is
        # present but the PortAudio native library is missing/unloadable.
        return {
            "sounddevice_installed": False,
            "input_devices": [],
            "output_devices": [],
            "error": f"sounddevice/PortAudio unavailable: {exc}",
        }
    try:
        devices = sd.query_devices()
        if not isinstance(devices, list):
            devices = list(devices)
        input_devices = []
        output_devices = []
        for index, device in enumerate(devices):
            if not isinstance(device, dict):
                continue
            record = {"index": index, "name": device.get("name")}
            if int(device.get("max_input_channels", 0) or 0) > 0:
                input_devices.append(record)
            if int(device.get("max_output_channels", 0) or 0) > 0:
                output_devices.append(record)
    except Exception as exc:
        # PortAudioError (and any backend error) at query time: degrade to the same
        # shape rather than crashing the voice doctor/health report.
        return {
            "sounddevice_installed": False,
            "input_devices": [],
            "output_devices": [],
            "error": f"sounddevice/PortAudio unavailable: {exc}",
        }
    return {
        "sounddevice_installed": True,
        "input_devices": input_devices,
        "output_devices": output_devices,
    }


def voice_doctor(settings: AprilSettings) -> dict[str, Any]:
    devices = query_audio_devices()
    components = [
        VoiceComponentHealth(
            name="voice enabled",
            status="ok" if settings.voice.enabled else "disabled",
            message=str(settings.voice.enabled),
        ),
        VoiceComponentHealth(
            name="sounddevice import",
            status="ok" if devices["sounddevice_installed"] else "degraded",
            message="installed" if devices["sounddevice_installed"] else "missing",
        ),
        VoiceComponentHealth(
            name="input devices",
            status="ok" if devices["input_devices"] else "degraded",
            message=str(len(devices["input_devices"])),
        ),
        VoiceComponentHealth(
            name="output devices",
            status="ok" if devices["output_devices"] else "degraded",
            message=str(len(devices["output_devices"])),
        ),
        _configured_path_health("whisper binary", settings, settings.voice.whisper_binary_path),
        _configured_path_health("whisper model", settings, settings.voice.whisper_model_path),
        _configured_path_health("piper binary", settings, settings.voice.piper_binary_path),
        _configured_path_health("piper model", settings, settings.voice.piper_model_path),
        _configured_path_health("wake-word model", settings, settings.voice.wake_word_model_path),
        _audio_cache_health(settings),
    ]
    degraded = [component for component in components if component.status == "degraded"]
    return {
        "status": "degraded" if degraded else "ok",
        "voice_enabled": settings.voice.enabled,
        "sounddevice_installed": devices["sounddevice_installed"],
        "input_devices": devices["input_devices"],
        "output_devices": devices["output_devices"],
        "components": [component.model_dump() for component in components],
        "macos_microphone_permission_guidance": (
            "macOS: System Settings > Privacy & Security > Microphone. "
            "Allow the terminal app used to run APRIL."
            if platform.system() == "Darwin"
            else None
        ),
        "audio_cache_path": str(settings.audio_cache_path),
        "audio_cache_writable": os.access(settings.audio_cache_path, os.W_OK),
    }
