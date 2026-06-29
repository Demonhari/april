from __future__ import annotations

import builtins
import struct
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from april_common.errors import RuntimeUnavailableError
from services.voice.audio_player import FakeAudioPlayer, SoundDeviceAudioPlayer
from services.voice.conversation_loop import PushToTalkLoop, normalize_transcript
from services.voice.health import (
    microphone_access,
    offline_voice_milestone,
    openwakeword_available,
    query_audio_devices,
    voice_doctor,
    voice_health,
    voice_readiness_summary,
)
from services.voice.microphone import FakeMicrophone, SoundDeviceMicrophone
from services.voice.speech_to_text import FakeSpeechToText
from services.voice.text_to_speech import FakeTextToSpeech
from services.voice.vad import VoiceActivityDetector, pcm16le_rms
from services.voice.wake_word import OpenWakeWordDetector


class FakeApi:
    def __init__(self) -> None:
        self.payloads: list[dict[str, str]] = []

    async def post(self, path: str, payload: dict[str, str]) -> dict[str, object]:
        self.payloads.append(payload)
        return {"result": {"final_message": "voice answer"}}


class FakeWakeDetector:
    def __init__(self, *, available: bool) -> None:
        self._available = available
        self.detected = False

    def available(self) -> bool:
        return self._available

    def detect(self, frame: bytes) -> bool:
        if not self.detected:
            self.detected = True
            return True
        return False


def test_voice_degraded_without_dependencies(settings_tmp) -> None:
    enabled = settings_tmp.model_copy(
        update={"voice": settings_tmp.voice.model_copy(update={"enabled": True})}
    )
    assert voice_health(enabled).status == "degraded"


def test_voice_doctor_reports_devices(settings_tmp, monkeypatch) -> None:
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    devices = query_audio_devices()
    assert devices["input_devices"][0]["name"] == "Mic"
    report = voice_doctor(settings_tmp)
    assert report["sounddevice_installed"] is True
    assert report["input_devices"]
    assert report["output_devices"]


def test_query_audio_devices_degrades_when_portaudio_missing(settings_tmp, monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    devices = query_audio_devices()
    assert devices["sounddevice_installed"] is False
    assert devices["input_devices"] == []
    assert devices["output_devices"] == []
    assert "PortAudio" in devices["error"]

    report = voice_doctor(settings_tmp)
    assert report["sounddevice_installed"] is False
    assert report["input_devices"] == []
    assert report["output_devices"] == []
    assert report["status"] in {"ok", "degraded"}


def test_query_audio_devices_degrades_when_query_raises(settings_tmp, monkeypatch) -> None:
    class PortAudioError(Exception):
        pass

    def raise_query_devices():
        raise PortAudioError("Error querying device -1")

    fake_sounddevice = types.SimpleNamespace(query_devices=raise_query_devices)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    devices = query_audio_devices()
    assert devices["sounddevice_installed"] is False
    assert devices["input_devices"] == []
    assert devices["output_devices"] == []
    assert "PortAudio" in devices["error"]

    report = voice_doctor(settings_tmp)
    assert report["sounddevice_installed"] is False
    assert report["input_devices"] == []
    assert report["output_devices"] == []


def test_microphone_access_classifies_missing_dependency() -> None:
    verdict = microphone_access(
        {"import_ok": False, "failure_kind": "missing_dependency", "input_devices": []}
    )
    assert verdict["status"] == "missing_dependency"
    assert "install" in verdict["message"].lower()


def test_microphone_access_classifies_permission_or_device() -> None:
    verdict = microphone_access(
        {"import_ok": True, "failure_kind": "device_error", "input_devices": []}
    )
    assert verdict["status"] == "permission_or_device"
    assert "permission" in verdict["message"].lower()


def test_microphone_access_distinguishes_no_input_device() -> None:
    verdict = microphone_access({"import_ok": True, "failure_kind": None, "input_devices": []})
    assert verdict["status"] == "no_input_device"


def test_query_audio_devices_marks_import_ok_on_device_error(settings_tmp, monkeypatch) -> None:
    # sounddevice imports fine but querying raises: that is a device/permission
    # failure, not a missing dependency.
    def raise_query_devices():
        raise RuntimeError("Error querying device -1")

    fake_sounddevice = types.SimpleNamespace(query_devices=raise_query_devices)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    devices = query_audio_devices()
    assert devices["import_ok"] is True
    assert devices["failure_kind"] == "device_error"
    report = voice_doctor(settings_tmp)
    assert report["microphone_access"] == "permission_or_device"


def test_voice_doctor_reports_push_to_talk_fallback_available(settings_tmp, monkeypatch) -> None:
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    report = voice_doctor(settings_tmp)
    assert report["push_to_talk_available"] is True
    assert report["microphone_access"] == "ok"
    ptt = next(c for c in report["components"] if c["name"] == "push-to-talk fallback")
    assert ptt["status"] == "ok"


def test_voice_doctor_push_to_talk_unavailable_without_microphone(
    settings_tmp, monkeypatch
) -> None:
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    report = voice_doctor(settings_tmp)
    assert report["push_to_talk_available"] is False
    assert report["microphone_access"] == "no_input_device"


def test_voice_doctor_reports_missing_dependency_distinctly(settings_tmp, monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    report = voice_doctor(settings_tmp)
    assert report["microphone_access"] == "missing_dependency"
    assert report["push_to_talk_available"] is False
    assert report["sounddevice_import_ok"] is False


def _stub_mic_and_speaker(monkeypatch) -> None:
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)


def _voice_settings_with_artifacts(settings_tmp, tmp_path: Path, *, wake_word: bool):
    """A settings copy whose whisper/piper (and optionally wake-word) artifacts all
    exist on disk, so the doctor's readiness rungs evaluate against real files."""
    artifacts = tmp_path / "voice_artifacts"
    artifacts.mkdir(exist_ok=True)

    def make(name: str) -> Path:
        path = artifacts / name
        path.write_bytes(b"stub")
        return path

    update = {
        "enabled": True,
        "whisper_binary_path": make("whisper"),
        "whisper_model_path": make("ggml-base.en.bin"),
        "piper_binary_path": make("piper"),
        "piper_model_path": make("voice.onnx"),
    }
    if wake_word:
        update["wake_word_model_path"] = make("april.onnx")
    return settings_tmp.model_copy(update={"voice": settings_tmp.voice.model_copy(update=update)})


def test_voice_doctor_push_to_talk_ready_without_wake_word_model(
    settings_tmp, tmp_path: Path, monkeypatch
) -> None:
    _stub_mic_and_speaker(monkeypatch)
    settings = _voice_settings_with_artifacts(settings_tmp, tmp_path, wake_word=False)
    report = voice_doctor(settings)
    # Push-to-talk is ready with mic + whisper + piper, even with no wake-word model.
    assert report["push_to_talk_ready"] is True
    # Wake-word readiness must require a configured wake-word model.
    assert report["wake_word_ready"] is False
    assert "wake-word model" in report["voice_readiness"]["wake_word_blocked_by"]
    assert report["full_voice_loop_ready"] is False


def test_voice_doctor_wake_word_ready_with_engine_and_model(
    settings_tmp, tmp_path: Path, monkeypatch
) -> None:
    _stub_mic_and_speaker(monkeypatch)
    monkeypatch.setattr("services.voice.health.openwakeword_available", lambda: True)
    settings = _voice_settings_with_artifacts(settings_tmp, tmp_path, wake_word=True)
    report = voice_doctor(settings)
    assert report["push_to_talk_ready"] is True
    assert report["wake_word_ready"] is True
    assert report["full_voice_loop_ready"] is True
    assert report["voice_readiness"]["wake_word_blocked_by"] == []


def test_voice_doctor_distinguishes_missing_openwakeword_engine_from_model(
    settings_tmp, tmp_path: Path, monkeypatch
) -> None:
    _stub_mic_and_speaker(monkeypatch)
    # Engine missing but a wake-word model IS configured: wake-word setup is broken
    # by the missing package, not the model.
    monkeypatch.setattr("services.voice.health.openwakeword_available", lambda: False)
    settings = _voice_settings_with_artifacts(settings_tmp, tmp_path, wake_word=True)
    report = voice_doctor(settings)
    assert report["openwakeword_available"] is False
    assert report["wake_word_ready"] is False
    assert "openWakeWord package" in report["voice_readiness"]["wake_word_blocked_by"]
    assert "wake-word model" not in report["voice_readiness"]["wake_word_blocked_by"]
    engine = next(c for c in report["components"] if c["name"] == "openWakeWord engine")
    assert engine["status"] == "degraded"
    # Push-to-talk is unaffected by a missing wake-word engine.
    assert report["push_to_talk_ready"] is True


def test_openwakeword_available_is_a_spec_lookup() -> None:
    # Whatever the environment, the probe must return a bool and not raise.
    assert isinstance(openwakeword_available(), bool)


def test_offline_voice_milestone_enum_escalates() -> None:
    assert offline_voice_milestone(enabled=False, readiness={}) == "disabled"
    assert offline_voice_milestone(enabled=True, readiness={}) == "not_configured"
    assert (
        offline_voice_milestone(enabled=True, readiness={"push_to_talk_ready": True})
        == "push_to_talk_ready"
    )
    assert (
        offline_voice_milestone(
            enabled=True, readiness={"push_to_talk_ready": True, "wake_word_ready": True}
        )
        == "wake_word_ready"
    )
    assert (
        offline_voice_milestone(
            enabled=True,
            readiness={
                "push_to_talk_ready": True,
                "wake_word_ready": True,
                "full_voice_loop_ready": True,
            },
        )
        == "full_voice_loop_ready"
    )


def test_voice_readiness_summary_includes_milestone(
    settings_tmp, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("services.voice.health.openwakeword_available", lambda: True)
    settings = _voice_settings_with_artifacts(settings_tmp, tmp_path, wake_word=True)
    devices = {
        "sounddevice_installed": True,
        "input_devices": [{"name": "Mic"}],
        "output_devices": [{"name": "Speaker"}],
    }
    summary = voice_readiness_summary(settings, devices)
    assert summary["voice_milestone"] == "full_voice_loop_ready"


def test_voice_readiness_summary_milestone_disabled_when_voice_off(
    settings_tmp, tmp_path: Path
) -> None:
    summary = voice_readiness_summary(settings_tmp, {"sounddevice_installed": False})
    assert summary["voice_milestone"] == "disabled"


@pytest.mark.asyncio
async def test_fake_voice_conversation_loop(settings_tmp, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    api = FakeApi()
    loop = PushToTalkLoop(
        api_client=api,  # type: ignore[arg-type]
        microphone=FakeMicrophone(audio),
        stt=FakeSpeechToText("April, open the project"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        conversation_id="voice-conv-1",
    )
    assert await loop.run_once() == "voice answer"
    assert api.payloads == [{"message": "open the project", "conversation_id": "voice-conv-1"}]


def test_push_to_talk_accepts_explicit_seconds(settings_tmp) -> None:
    loop = PushToTalkLoop(api_client=FakeApi(), record_seconds=1.5)  # type: ignore[arg-type]
    assert loop.record_seconds == 1.5


@pytest.mark.asyncio
async def test_wake_word_loop_segments_fake_frames(settings_tmp, tmp_path: Path) -> None:
    from services.voice.conversation_loop import WakeWordConversationLoop

    loud = (b"\xff\x7f") * 160
    silent = (b"\x00\x00") * 160
    api = FakeApi()
    loop = WakeWordConversationLoop(
        api_client=api,  # type: ignore[arg-type]
        microphone=FakeMicrophone(
            tmp_path / "unused.wav",
            frames=[silent, loud, loud, loud, silent, silent, silent],
        ),
        stt=FakeSpeechToText("April, inspect this"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        detector=FakeWakeDetector(available=True),  # type: ignore[arg-type]
        conversation_id="listen-conv-1",
        record_seconds=2.0,
    )
    assert await loop.run_once() == "voice answer"
    assert api.payloads == [{"message": "inspect this", "conversation_id": "listen-conv-1"}]


@pytest.mark.asyncio
async def test_wake_word_loop_falls_back_to_push_to_talk(settings_tmp, tmp_path: Path) -> None:
    from services.voice.conversation_loop import WakeWordConversationLoop

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    api = FakeApi()
    loop = WakeWordConversationLoop(
        api_client=api,  # type: ignore[arg-type]
        microphone=FakeMicrophone(audio),
        stt=FakeSpeechToText("April, fallback"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
        detector=FakeWakeDetector(available=False),  # type: ignore[arg-type]
        conversation_id="fallback-conv-1",
    )
    assert await loop.run_once() == "voice answer"
    assert api.payloads == [{"message": "fallback", "conversation_id": "fallback-conv-1"}]


def test_transcript_normalization_preserves_paths_and_code() -> None:
    text = normalize_transcript("  April,   open /tmp/My Project/app.py  ", wake_word="april")
    assert text == "open /tmp/My Project/app.py"


@pytest.mark.asyncio
async def test_voice_loop_rejects_empty_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    loop = PushToTalkLoop(
        api_client=FakeApi(),  # type: ignore[arg-type]
        microphone=FakeMicrophone(audio),
        stt=FakeSpeechToText("   "),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    with pytest.raises(ValueError, match="empty"):
        await loop.run_once()


def test_energy_vad_requires_configured_speech_frames() -> None:
    detector = VoiceActivityDetector(energy_threshold=0.01, required_frames=2)
    loud_frame = (b"\xff\x7f") * 160
    assert detector.is_speech(loud_frame) is False
    assert detector.is_speech(loud_frame) is True
    assert detector.is_speech(b"\x00\x00" * 160) is False


def test_pcm16le_rms_known_vectors() -> None:
    assert pcm16le_rms(b"") == 0.0
    assert pcm16le_rms(struct.pack("<hhh", 0, 0, 0)) == 0.0
    assert pcm16le_rms(struct.pack("<hh", 3, 4)) == pytest.approx((12.5**0.5) / 32768.0)
    assert pcm16le_rms(struct.pack("<hh", -3, -4)) == pytest.approx((12.5**0.5) / 32768.0)


def test_pcm16le_rms_clipping_boundaries() -> None:
    assert pcm16le_rms(struct.pack("<h", 32767)) == pytest.approx(32767 / 32768.0)
    assert pcm16le_rms(struct.pack("<h", -32768)) == 1.0
    assert pcm16le_rms(struct.pack("<hh", -32768, 32767)) == pytest.approx(
        (((32768**2) + (32767**2)) / 2) ** 0.5 / 32768.0
    )


def test_pcm16le_rms_rejects_malformed_frame() -> None:
    detector = VoiceActivityDetector(energy_threshold=0.01, required_frames=1)
    with pytest.raises(ValueError, match="16-bit PCM"):
        pcm16le_rms(b"\x00")
    with pytest.raises(ValueError, match="16-bit PCM"):
        detector.is_speech(b"\x00")


def test_energy_vad_threshold_boundary_is_inclusive() -> None:
    detector = VoiceActivityDetector(energy_threshold=0.5, required_frames=1)
    assert detector.is_speech(struct.pack("<h", 16383)) is False
    assert detector.is_speech(struct.pack("<h", 16384)) is True


@pytest.mark.asyncio
async def test_sounddevice_microphone_rejects_unsafe_duration(tmp_path: Path) -> None:
    microphone = SoundDeviceMicrophone(max_seconds=0)
    with pytest.raises(RuntimeUnavailableError, match="duration"):
        await microphone.record_push_to_talk(tmp_path / "capture.wav")


@pytest.mark.asyncio
async def test_sounddevice_player_rejects_invalid_wav(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not a wav")
    player = SoundDeviceAudioPlayer()
    with pytest.raises(RuntimeUnavailableError, match="valid WAV"):
        await player.play(invalid)


def test_openwakeword_requires_configured_existing_model(tmp_path: Path) -> None:
    detector = OpenWakeWordDetector(tmp_path / "missing.onnx")
    with pytest.raises(RuntimeUnavailableError, match="missing"):
        detector.detect(b"\x00\x00" * 80)


def test_openwakeword_validates_pcm_frame_width(tmp_path: Path) -> None:
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"fake")
    detector = OpenWakeWordDetector(model)
    with pytest.raises(RuntimeUnavailableError, match="16-bit PCM"):
        detector.detect(b"\x00")


@pytest.mark.asyncio
async def test_sounddevice_player_plays_valid_wav(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "valid.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((np.ones(160, dtype=np.int16) * 100).tobytes())
    calls: list[tuple[object, int, object]] = []

    fake_sounddevice = types.SimpleNamespace(
        play=lambda data, samplerate, device=None: calls.append((data, samplerate, device)),
        wait=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    player = SoundDeviceAudioPlayer(device="speaker")
    await player.play(audio)
    assert calls[0][1:] == (16_000, "speaker")


@pytest.mark.asyncio
async def test_sounddevice_microphone_records_wav(tmp_path: Path, monkeypatch) -> None:
    def fake_rec(frames: int, *, samplerate: int, channels: int, dtype: str, device: object):
        assert (samplerate, channels, dtype, device) == (16_000, 1, "int16", "mic")
        return np.ones((frames, channels), dtype=np.int16)

    fake_sounddevice = types.SimpleNamespace(rec=fake_rec, wait=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    output = tmp_path / "capture.wav"
    microphone = SoundDeviceMicrophone(device="mic", max_seconds=0.01)
    captured = await microphone.record_push_to_talk(output)
    assert captured == output
    with wave.open(str(output), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)


def test_openwakeword_detects_with_fake_model(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"fake")

    class FakeModel:
        def __init__(self, *, wakeword_models: list[str]) -> None:
            self.wakeword_models = wakeword_models

        def predict(self, frame: bytes) -> dict[str, float]:
            return {"april": 0.9}

    fake_module = types.ModuleType("openwakeword.model")
    fake_module.Model = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", types.ModuleType("openwakeword"))
    monkeypatch.setitem(sys.modules, "openwakeword.model", fake_module)
    detector = OpenWakeWordDetector(model, threshold=0.5)
    # A full 80 ms window (1280 samples) is required before a prediction runs.
    assert detector.detect(b"\x00\x00" * 1280) is True
