from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.voice.microphone import write_pcm_wav
from services.wake.fakes import FakeSpeakerVerifier
from services.wake.sentinel import configured_speaker_verifier
from services.wake.speaker import MAX_SPEAKER_AUDIO_SAMPLES, OnnxSpeakerVerifier


def _pcm(value: int, *, samples: int = 8) -> bytes:
    return np.full(samples, value, dtype=np.int16).tobytes()


def _wav(path: Path, value: int) -> Path:
    return write_pcm_wav(path, [_pcm(value)])


def test_onnx_speaker_verifier_parses_wav_and_normalizes_pcm(tmp_path: Path) -> None:
    observed: list[np.ndarray] = []

    def inference(samples: np.ndarray) -> np.ndarray:
        observed.append(samples.copy())
        return np.array([1.0, 0.0], dtype=np.float32)

    enrollment = _wav(tmp_path / "enroll.wav", 16_384)
    verifier = OnnxSpeakerVerifier(tmp_path / "unused-model", inference=inference)

    assert verifier.score([enrollment], _pcm(8192)) == pytest.approx(1.0)
    assert observed[0].tolist() == pytest.approx([0.5] * 8)
    assert observed[1].tolist() == pytest.approx([0.25] * 8)


def test_onnx_speaker_verifier_cosine_mapping_and_clamp(tmp_path: Path) -> None:
    def inference(samples: np.ndarray) -> np.ndarray:
        if samples[0] > 0.0:
            return np.array([1.0, 0.0], dtype=np.float32)
        if samples[0] < 0.0:
            return np.array([-1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    verifier = OnnxSpeakerVerifier(tmp_path / "unused-model", inference=inference)
    enrollment = _wav(tmp_path / "enroll.wav", 1000)

    assert verifier.score([enrollment], _pcm(1000)) == pytest.approx(1.0)
    assert verifier.score([enrollment], _pcm(0)) == pytest.approx(0.5)
    assert verifier.score([enrollment], _pcm(-1000)) == pytest.approx(0.0)


def test_onnx_speaker_verifier_uses_mean_enrollment_embedding(tmp_path: Path) -> None:
    def inference(samples: np.ndarray) -> np.ndarray:
        if samples[0] == pytest.approx(1000 / 32768.0):
            return np.array([1.0, 0.0], dtype=np.float32)
        if samples[0] == pytest.approx(2000 / 32768.0):
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 1.0], dtype=np.float32)

    verifier = OnnxSpeakerVerifier(tmp_path / "unused-model", inference=inference)
    enrollment = [
        _wav(tmp_path / "enroll-1.wav", 1000),
        _wav(tmp_path / "enroll-2.wav", 2000),
    ]

    assert verifier.score(enrollment, _pcm(3000)) == pytest.approx(1.0)


def test_onnx_speaker_verifier_rejects_malformed_wav(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.wav"
    malformed.write_bytes(b"not a wav")
    verifier = OnnxSpeakerVerifier(
        tmp_path / "unused-model",
        inference=lambda samples: np.array([float(samples.size)], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Invalid WAV"):
        verifier.score([malformed], _pcm(1000))


def test_onnx_speaker_verifier_bounds_utterance_pcm(tmp_path: Path) -> None:
    lengths: list[int] = []

    def inference(samples: np.ndarray) -> np.ndarray:
        lengths.append(samples.size)
        return np.array([1.0], dtype=np.float32)

    verifier = OnnxSpeakerVerifier(tmp_path / "unused-model", inference=inference)
    enrollment = _wav(tmp_path / "enroll.wav", 1000)
    oversized = _pcm(1000, samples=MAX_SPEAKER_AUDIO_SAMPLES + 100)

    assert verifier.score([enrollment], oversized) == pytest.approx(1.0)
    assert lengths[-1] == MAX_SPEAKER_AUDIO_SAMPLES


def test_onnx_speaker_verifier_reports_optional_runtime_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_import(name: str) -> object:
        assert name == "onnxruntime"
        raise ImportError("not installed")

    monkeypatch.setattr("services.wake.speaker.importlib.import_module", fail_import)

    with pytest.raises(ImportError, match="onnxruntime is required"):
        OnnxSpeakerVerifier(tmp_path / "speaker-model")


def test_onnx_speaker_verifier_reports_session_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenRuntime:
        @staticmethod
        def InferenceSession(*args: object, **kwargs: object) -> object:
            raise ValueError("bad graph")

    monkeypatch.setattr(
        "services.wake.speaker.importlib.import_module",
        lambda name: BrokenRuntime,
    )

    with pytest.raises(RuntimeError, match="Unable to load local speaker-verifier ONNX"):
        OnnxSpeakerVerifier(tmp_path / "speaker-model")


def _speaker_settings(settings_tmp, *, gate: str, model_path: Path | None = None):
    return settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={
                    "speaker_gate": gate,
                    "speaker_verifier_model_path": model_path,
                }
            )
        }
    )


def test_configured_speaker_verifier_decision_matrix(settings_tmp) -> None:
    off = _speaker_settings(settings_tmp, gate="off")
    assert configured_speaker_verifier(off) == (None, None)

    soft_without_path = _speaker_settings(settings_tmp, gate="soft")
    assert configured_speaker_verifier(soft_without_path) == (
        None,
        "local_verifier_unavailable",
    )

    missing = _speaker_settings(settings_tmp, gate="soft", model_path=Path("missing.onnx"))
    assert configured_speaker_verifier(missing) == (None, "model_missing")

    model_path = settings_tmp.home / "speaker-model.stub"
    model_path.write_bytes(b"test stub")
    configured = _speaker_settings(settings_tmp, gate="soft", model_path=model_path)

    def unavailable_factory(path: Path) -> FakeSpeakerVerifier:
        assert path == model_path
        raise ImportError("onnxruntime unavailable")

    assert configured_speaker_verifier(configured, factory=unavailable_factory) == (
        None,
        "onnxruntime_unavailable",
    )

    def broken_model_factory(path: Path) -> FakeSpeakerVerifier:
        assert path == model_path
        raise RuntimeError("invalid model")

    assert configured_speaker_verifier(configured, factory=broken_model_factory) == (
        None,
        "model_load_failed",
    )

    verifier = FakeSpeakerVerifier(0.9)
    built, reason = configured_speaker_verifier(configured, factory=lambda path: verifier)
    assert built is verifier
    assert reason is None
