from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from april_common.settings import AprilSettings


def openwakeword_available() -> bool:
    """Whether the optional ``openwakeword`` package is importable.

    Uses an :mod:`importlib` spec lookup so no import side effects run and no
    model is loaded. This distinguishes "the wake-word *engine* is unavailable"
    (a missing ``.[voice]`` install) from "the wake-word *model* file is missing"
    — two different fixes the doctor must not conflate.
    """
    try:
        return importlib.util.find_spec("openwakeword") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


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


def _openwakeword_health(available: bool, *, wake_word_configured: bool) -> VoiceComponentHealth:
    """Health of the optional openWakeWord *engine* (package), separate from the
    wake-word model file. Missing-but-unused is ``disabled`` (push-to-talk does
    not need it); missing-while-a-model-is-configured is ``degraded`` because the
    wake-word setup is genuinely broken."""
    if available:
        return VoiceComponentHealth(name="openWakeWord engine", status="ok", message="installed")
    if wake_word_configured:
        return VoiceComponentHealth(
            name="openWakeWord engine",
            status="degraded",
            message=(
                "not installed, but a wake-word model is configured (pip install -e '.[voice]')"
            ),
        )
    return VoiceComponentHealth(
        name="openWakeWord engine",
        status="disabled",
        message="not installed; wake-word listening unavailable (push-to-talk does not need it)",
    )


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


def _path_present(settings: AprilSettings, path: Path | None) -> bool:
    return path is not None and settings.resolve_path(path).exists()


def _voice_readiness(
    settings: AprilSettings,
    *,
    push_to_talk_available: bool,
    output_devices_present: bool,
    openwakeword: bool,
) -> dict[str, Any]:
    """Compute the three escalating voice-readiness rungs, with the *named*
    artifacts blocking each (never paths, so this stays redaction-safe).

    * ``push_to_talk_ready`` — a usable microphone plus local whisper.cpp (STT) and
      Piper (TTS). It deliberately does **not** require a wake-word model.
    * ``wake_word_ready`` — everything push-to-talk needs, plus the openWakeWord
      engine and a configured wake-word model file. A wake-word model is mandatory
      here.
    * ``full_voice_loop_ready`` — wake-word readiness plus an output device for the
      hands-free wake → listen → respond → speak loop.
    """
    whisper_ready = _path_present(settings, settings.voice.whisper_binary_path) and _path_present(
        settings, settings.voice.whisper_model_path
    )
    piper_ready = _path_present(settings, settings.voice.piper_binary_path) and _path_present(
        settings, settings.voice.piper_model_path
    )
    wake_model_present = _path_present(settings, settings.voice.wake_word_model_path)

    ptt_missing: list[str] = []
    if not push_to_talk_available:
        ptt_missing.append("microphone")
    if not _path_present(settings, settings.voice.whisper_binary_path):
        ptt_missing.append("whisper.cpp binary")
    if not _path_present(settings, settings.voice.whisper_model_path):
        ptt_missing.append("whisper model")
    if not _path_present(settings, settings.voice.piper_binary_path):
        ptt_missing.append("piper binary")
    if not _path_present(settings, settings.voice.piper_model_path):
        ptt_missing.append("piper voice model")
    push_to_talk_ready = not ptt_missing

    wake_missing = list(ptt_missing)
    if not openwakeword:
        wake_missing.append("openWakeWord package")
    if not wake_model_present:
        wake_missing.append("wake-word model")
    wake_word_ready = not wake_missing

    loop_missing = list(wake_missing)
    if not output_devices_present:
        loop_missing.append("output device")
    full_voice_loop_ready = not loop_missing

    return {
        "push_to_talk_ready": push_to_talk_ready,
        "push_to_talk_blocked_by": ptt_missing,
        "wake_word_ready": wake_word_ready,
        "wake_word_blocked_by": wake_missing,
        "full_voice_loop_ready": full_voice_loop_ready,
        "full_voice_loop_blocked_by": loop_missing,
        "whisper_ready": whisper_ready,
        "piper_ready": piper_ready,
        "wake_word_model_present": wake_model_present,
        "openwakeword_available": openwakeword,
    }


def offline_voice_milestone(*, enabled: bool, readiness: dict[str, Any]) -> str:
    """The highest *offline* voice milestone reached, as a single redacted enum.

    This is derived purely from configuration and local artifact presence — it
    never opens the microphone and never claims a *live* pass. The two live rungs
    (``live_verified`` / ``wake_live_verified``) are layered on top by callers
    that have read the redacted voice-live / wake-word-live verification reports.

    Returns one of: ``disabled``, ``not_configured``, ``push_to_talk_ready``,
    ``wake_word_ready``, ``full_voice_loop_ready``.
    """
    if not enabled:
        return "disabled"
    if readiness.get("full_voice_loop_ready"):
        return "full_voice_loop_ready"
    if readiness.get("wake_word_ready"):
        return "wake_word_ready"
    if readiness.get("push_to_talk_ready"):
        return "push_to_talk_ready"
    return "not_configured"


def voice_readiness_summary(settings: AprilSettings, devices: dict[str, Any]) -> dict[str, Any]:
    """Public, redaction-safe voice-readiness verdicts for API/desktop consumers.

    Reuses the same logic the doctor uses so the CLI doctor and the Core API
    ``/readiness`` voice block never disagree. ``devices`` is the result of
    :func:`query_audio_devices` so the caller can avoid re-enumerating hardware.
    """
    push_to_talk_available = bool(
        devices.get("sounddevice_installed") and devices.get("input_devices")
    )
    readiness = _voice_readiness(
        settings,
        push_to_talk_available=push_to_talk_available,
        output_devices_present=bool(devices.get("output_devices")),
        openwakeword=openwakeword_available(),
    )
    # The offline milestone is the single source of truth for the voice rung; the
    # live rungs are added by API callers that have read the live reports.
    readiness["voice_milestone"] = offline_voice_milestone(
        enabled=settings.voice.enabled, readiness=readiness
    )
    return readiness


def voice_doctor(settings: AprilSettings) -> dict[str, Any]:
    devices = query_audio_devices()
    mic = microphone_access(devices)
    # Push-to-talk is the always-available fallback: it needs a usable microphone
    # but no wake-word model. Wake-word listening is the only thing that requires
    # the openWakeWord engine and model.
    push_to_talk_available = bool(devices["sounddevice_installed"] and devices["input_devices"])
    wake_word_configured = settings.voice.wake_word_model_path is not None
    openwakeword = openwakeword_available()
    readiness = _voice_readiness(
        settings,
        push_to_talk_available=push_to_talk_available,
        output_devices_present=bool(devices["output_devices"]),
        openwakeword=openwakeword,
    )
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
        _openwakeword_health(openwakeword, wake_word_configured=wake_word_configured),
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
        "openwakeword_available": openwakeword,
        # Composite, escalating readiness verdicts. push_to_talk_ready never
        # requires a wake-word model; wake_word_ready always does.
        "push_to_talk_ready": readiness["push_to_talk_ready"],
        "wake_word_ready": readiness["wake_word_ready"],
        "full_voice_loop_ready": readiness["full_voice_loop_ready"],
        "voice_readiness": readiness,
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
