from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SPEAKER_MATCH_THRESHOLD = 0.5
SPEAKER_SAMPLE_RATE = 16_000
MAX_SPEAKER_AUDIO_SECONDS = 15
MAX_SPEAKER_AUDIO_SAMPLES = SPEAKER_SAMPLE_RATE * MAX_SPEAKER_AUDIO_SECONDS

SpeakerInference = Callable[[np.ndarray], object]


class SpeakerVerifier(Protocol):
    """Local convenience filter for an accepted wake, never authentication.

    Implementations compare operator-owned enrollment WAV files with bounded
    16-bit mono PCM from the wake ring buffer and return a score in ``[0, 1]``.
    The score may suppress an accidental wake, but it must never grant access,
    lower a permission level, or serve as an identity/security boundary.
    """

    def score(self, enrollment: Sequence[Path], utterance: bytes) -> float: ...


class OnnxSpeakerVerifier:
    """Local raw-waveform ONNX speaker-embedding verifier.

    This is a convenience filter only: its score must never be treated as
    authentication, identity proof, or permission evidence. The default
    inference adapter imports ONNX Runtime only when this class is constructed.
    Tests inject a deterministic callable and need neither ONNX Runtime nor a
    model file.
    """

    def __init__(
        self,
        model_path: Path,
        inference: SpeakerInference | None = None,
    ) -> None:
        self.model_path = model_path
        self.inference = inference or _onnx_inference(model_path)

    def score(self, enrollment: Sequence[Path], utterance: bytes) -> float:
        """Compare bounded enrollment audio with bounded utterance PCM."""
        from services.voice.microphone import read_pcm_wav

        if not enrollment:
            raise ValueError("Speaker verification requires at least one enrollment WAV.")
        enrollment_embeddings = [
            self._embed(
                read_pcm_wav(
                    path,
                    sample_rate=SPEAKER_SAMPLE_RATE,
                    channels=1,
                    max_frames=MAX_SPEAKER_AUDIO_SAMPLES,
                )
            )
            for path in enrollment
        ]
        utterance_embedding = self._embed(_bounded_pcm(utterance))
        dimensions = {embedding.shape for embedding in enrollment_embeddings}
        dimensions.add(utterance_embedding.shape)
        if len(dimensions) != 1:
            raise ValueError("Speaker verifier returned inconsistent embedding dimensions.")
        mean_enrollment = np.mean(np.stack(enrollment_embeddings), axis=0)
        denominator = float(np.linalg.norm(mean_enrollment) * np.linalg.norm(utterance_embedding))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("Speaker verifier returned a zero or invalid embedding.")
        cosine = float(np.dot(mean_enrollment, utterance_embedding) / denominator)
        if not np.isfinite(cosine):
            raise ValueError("Speaker verifier returned an invalid cosine score.")
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    def _embed(self, pcm: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        embedding = np.asarray(self.inference(samples), dtype=np.float32).reshape(-1)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise ValueError("Speaker verifier returned an empty or non-finite embedding.")
        return embedding


def onnxruntime_importable() -> bool:
    """Return whether the optional local ONNX Runtime dependency imports."""
    try:
        importlib.import_module("onnxruntime")
    except Exception:
        return False
    return True


def _bounded_pcm(pcm: bytes) -> bytes:
    if not pcm:
        raise ValueError("Speaker verification requires non-empty utterance PCM.")
    if len(pcm) % 2:
        raise ValueError("Speaker verification requires complete 16-bit PCM samples.")
    max_bytes = MAX_SPEAKER_AUDIO_SAMPLES * 2
    return pcm[-max_bytes:]


def _onnx_inference(model_path: Path) -> SpeakerInference:
    try:
        runtime = importlib.import_module("onnxruntime")
    except Exception as exc:
        raise ImportError(
            "onnxruntime is required for wake.speaker_gate=soft; install APRIL's voice extra."
        ) from exc
    try:
        session = runtime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        model_input = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
    except Exception as exc:
        raise RuntimeError(f"Unable to load local speaker-verifier ONNX: {model_path}") from exc

    def infer(samples: np.ndarray) -> object:
        input_shape: Any = getattr(model_input, "shape", None)
        rank = len(input_shape) if isinstance(input_shape, Sequence) else 2
        if rank == 1:
            model_audio = samples
        elif rank == 2:
            model_audio = samples[np.newaxis, :]
        else:
            raise RuntimeError(
                "Speaker-verifier ONNX must accept a rank-1 waveform or rank-2 batched waveform."
            )
        try:
            return session.run([output_name], {model_input.name: model_audio})[0]
        except Exception as exc:
            raise RuntimeError("Local speaker-verifier ONNX inference failed.") from exc

    return infer
