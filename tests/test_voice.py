from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from april_common.errors import RuntimeUnavailableError
from services.voice.audio_player import FakeAudioPlayer, SoundDeviceAudioPlayer
from services.voice.conversation_loop import PushToTalkLoop, normalize_transcript
from services.voice.health import query_audio_devices, voice_doctor, voice_health
from services.voice.microphone import FakeMicrophone, SoundDeviceMicrophone
from services.voice.speech_to_text import FakeSpeechToText
from services.voice.text_to_speech import FakeTextToSpeech
from services.voice.vad import VoiceActivityDetector
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
    assert detector.detect(b"\x00\x00" * 80) is True
