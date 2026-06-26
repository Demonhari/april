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
    """Enumerate audio devices, distinguishing a *missing dependency* from a
    *device/permission* failure.

    ``import_ok`` records whether ``sounddevice``/PortAudio could be imported at
    all; ``failure_kind`` then says *why* enumeration failed:

    * ``"missing_dependency"`` — the package or its PortAudio native library is
      not installed/loadable. The fix is to install it.
    * ``"device_error"`` — the package imported fine but querying devices failed
      (commonly a denied macOS microphone permission or no device present). The
      fix is a permission/hardware check, *not* an install.

    ``sounddevice_installed`` stays ``True`` only on a fully successful query, so
    existing callers keep their meaning; the new fields refine the diagnosis.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        # ImportError: package not installed. OSError: the sounddevice package is
        # present but the PortAudio native library is missing/unloadable.
        return {
            "sounddevice_installed": False,
            "import_ok": False,
            "failure_kind": "missing_dependency",
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
        # PortAudioError (and any backend error) at query time: the dependency is
        # present but the device/permission is not. Report it as such instead of
        # conflating it with a missing install.
        return {
            "sounddevice_installed": False,
            "import_ok": True,
            "failure_kind": "device_error",
            "input_devices": [],
            "output_devices": [],
            "error": f"sounddevice/PortAudio device error: {exc}",
        }
    return {
        "sounddevice_installed": True,
        "import_ok": True,
        "failure_kind": None,
        "input_devices": input_devices,
        "output_devices": output_devices,
    }


def microphone_access(devices: dict[str, Any]) -> dict[str, str]:
    """Classify microphone availability into a permission-vs-dependency verdict.

    Returns ``{"status": <verdict>, "message": <guidance>}`` where ``status`` is
    one of ``ok``, ``missing_dependency``, ``permission_or_device``, or
    ``no_input_device``. This is the field the doctor and Desktop use to tell a
    user *which* fix applies.
    """
    if not devices.get("import_ok", False):
        return {
            "status": "missing_dependency",
            "message": (
                "sounddevice/PortAudio is not installed. Install the optional voice "
                "extra (`pip install -e '.[voice]'`) and the PortAudio native library."
            ),
        }
    if devices.get("failure_kind") == "device_error":
        return {
            "status": "permission_or_device",
            "message": (
                "sounddevice is installed but querying audio devices failed. This is "
                "usually a denied microphone permission, not a missing dependency."
            ),
        }
    if not devices.get("input_devices"):
        return {
            "status": "no_input_device",
            "message": (
                "No input devices were reported. Check that a microphone is connected "
                "and that microphone permission is granted to the terminal app."
            ),
        }
    return {"status": "ok", "message": f"{len(devices['input_devices'])} input device(s)"}


def voice_doctor(settings: AprilSettings) -> dict[str, Any]:
    devices = query_audio_devices()
    mic = microphone_access(devices)
    # Push-to-talk is the always-available fallback: it needs a usable microphone
    # but no wake-word model. Wake-word listening is the only thing that requires
    # the openWakeWord model.
    push_to_talk_available = bool(devices["sounddevice_installed"] and devices["input_devices"])
    wake_word_configured = settings.voice.wake_word_model_path is not None
    components = [
        VoiceComponentHealth(
            name="voice enabled",
            status="ok" if settings.voice.enabled else "disabled",
            message=str(settings.voice.enabled),
        ),
        VoiceComponentHealth(
            name="sounddevice import",
            status="ok" if devices.get("import_ok") else "degraded",
            message="installed" if devices.get("import_ok") else "missing",
        ),
        VoiceComponentHealth(
            name="microphone access",
            status="ok" if mic["status"] == "ok" else "degraded",
            # Distinguishes permission failure from a missing dependency.
            message=f"{mic['status']}: {mic['message']}",
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
        VoiceComponentHealth(
            name="push-to-talk fallback",
            status="ok" if push_to_talk_available else "degraded",
            message=(
                "available (no wake-word model required)"
                if push_to_talk_available
                else "unavailable: needs a usable microphone"
            ),
        ),
        _audio_cache_health(settings),
    ]
    degraded = [component for component in components if component.status == "degraded"]
    return {
        "status": "degraded" if degraded else "ok",
        "voice_enabled": settings.voice.enabled,
        "sounddevice_installed": devices["sounddevice_installed"],
        "sounddevice_import_ok": bool(devices.get("import_ok")),
        "microphone_access": mic["status"],
        "microphone_access_message": mic["message"],
        "push_to_talk_available": push_to_talk_available,
        "wake_word_model_configured": wake_word_configured,
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
        "wake_word_guidance": (
            "Wake-word ('April') listening needs a custom local openWakeWord model "
            "configured at voice.wake_word_model_path; APRIL never downloads or trains one. "
            "Push-to-talk (run april voice ptt) works without any wake-word model."
        ),
    }
