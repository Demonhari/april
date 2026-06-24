from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path

from apps.cli.client import AprilApiClient
from april_common.settings import get_settings
from services.voice.audio_player import AudioPlayer, SoundDeviceAudioPlayer
from services.voice.microphone import (
    Microphone,
    SoundDeviceMicrophone,
    aclose_frame_source,
    write_pcm_wav,
)
from services.voice.speech_to_text import SpeechToText, WhisperCppSpeechToText
from services.voice.text_to_speech import PiperTextToSpeech, TextToSpeech
from services.voice.vad import VoiceActivityDetector
from services.voice.wake_word import OpenWakeWordDetector


class VoiceTimeout(RuntimeError):
    """Raised when no wake word arrives within the wake-word waiting timeout."""


def normalize_transcript(text: str, *, wake_word: str | None = None) -> str:
    normalized = " ".join(text.split())
    if wake_word and normalized.lower().startswith(wake_word.lower()):
        normalized = normalized[len(wake_word) :].lstrip(" ,.:;")
    return normalized


class PushToTalkLoop:
    def __init__(
        self,
        *,
        api_client: AprilApiClient,
        microphone: Microphone | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        player: AudioPlayer | None = None,
        conversation_id: str | None = None,
        record_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        max_seconds = record_seconds or settings.voice.max_record_seconds
        self.api_client = api_client
        self.microphone = microphone or SoundDeviceMicrophone(
            device=settings.voice.input_device,
            max_seconds=max_seconds,
        )
        self.stt = stt or WhisperCppSpeechToText(
            settings.voice.whisper_binary_path,
            settings.voice.whisper_model_path,
        )
        self.tts = tts or PiperTextToSpeech(
            settings.voice.piper_binary_path,
            settings.voice.piper_model_path,
        )
        self.player = player or SoundDeviceAudioPlayer(device=settings.voice.output_device)
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.record_seconds = max_seconds
        self.vad = VoiceActivityDetector(
            energy_threshold=settings.voice.vad_energy_threshold,
            required_frames=settings.voice.vad_required_frames,
        )

    async def run_once(self) -> str:
        self.settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
        audio_path = self.settings.audio_cache_path / f"{uuid.uuid4()}.wav"
        spoken_path = await self.microphone.record_push_to_talk(audio_path)
        text = normalize_transcript(await self.stt.transcribe(spoken_path), wake_word="april")
        if not text:
            raise ValueError("Voice transcript was empty.")
        response = await self.api_client.post(
            "/voice/input",
            {"message": text, "conversation_id": self.conversation_id},
        )
        answer = response["result"]["final_message"]
        tts_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-reply.wav"
        output_path = await self.tts.synthesize(answer, tts_path)
        await self.player.play(output_path)
        if not self.settings.voice.retain_debug_audio:
            for path in (audio_path, tts_path):
                if Path(path).exists():
                    Path(path).unlink()
        return answer


class VoiceConversationLoop(PushToTalkLoop):
    async def run_forever(self) -> None:
        while True:
            await self.run_once()


class WakeWordConversationLoop(PushToTalkLoop):
    def __init__(
        self, *args: object, detector: OpenWakeWordDetector | None = None, **kwargs: object
    ):
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.detector = detector or OpenWakeWordDetector(
            self.settings.voice.wake_word_model_path,
            threshold=self.settings.voice.wake_word_threshold,
            cooldown_seconds=self.settings.voice.wake_word_cooldown_seconds,
        )

    async def run_once(self) -> str:
        if not self.detector.available():
            return await super().run_once()
        self.settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
        audio_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-utterance.wav"
        spoken_path = await self._capture_wake_utterance(audio_path)
        text = normalize_transcript(await self.stt.transcribe(spoken_path), wake_word="april")
        if not text:
            raise ValueError("Voice transcript was empty.")
        response = await self.api_client.post(
            "/voice/input",
            {"message": text, "conversation_id": self.conversation_id},
        )
        answer = response["result"]["final_message"]
        tts_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-reply.wav"
        output_path = await self.tts.synthesize(answer, tts_path)
        await self.player.play(output_path)
        if not self.settings.voice.retain_debug_audio:
            for path in (audio_path, tts_path):
                if Path(path).exists():
                    Path(path).unlink()
        return answer

    async def run_forever(self) -> None:
        try:
            while True:
                await self.run_once()
        except KeyboardInterrupt:
            return

    async def _capture_wake_utterance(
        self, output_path: Path, *, clock: Callable[[], float] = time.monotonic
    ) -> Path:
        voice = self.settings.voice
        # Reset detector and VAD at the conversation boundary so a prior
        # utterance cannot leak into this one.
        self.vad.reset()
        detector_reset = getattr(self.detector, "reset", None)
        if callable(detector_reset):
            detector_reset()

        pre_roll: deque[bytes] = deque(maxlen=max(0, voice.wake_pre_roll_frames))
        frames: list[bytes] = []
        wake_seen = False
        speech_seen = False
        silence_frames = 0
        # The wake-word waiting timeout runs from the start; the utterance timeout
        # only begins once the wake word (or push-to-talk) has activated.
        wake_deadline = clock() + voice.wake_wait_seconds
        utterance_deadline: float | None = None

        frame_source = self.microphone.frames()
        try:
            async for frame in frame_source:
                if not wake_seen:
                    pre_roll.append(frame)
                    if self.detector.detect(frame):
                        wake_seen = True
                        # Pre-roll recovers the audio captured while the wake word
                        # was being confirmed, so the onset is not discarded.
                        frames.extend(pre_roll)
                        pre_roll.clear()
                        self.vad.reset()
                        utterance_deadline = clock() + voice.utterance_max_seconds
                    elif clock() >= wake_deadline:
                        raise VoiceTimeout("No wake word detected before the wake timeout.")
                    continue
                # Capture every post-wake frame so the start of speech survives the
                # VAD onset confirmation; VAD only decides when speech has ended.
                frames.append(frame)
                if self.vad.is_speech(frame):
                    speech_seen = True
                    silence_frames = 0
                elif speech_seen:
                    silence_frames += 1
                    if silence_frames >= voice.vad_required_frames:
                        break
                if utterance_deadline is not None and clock() >= utterance_deadline:
                    break
        finally:
            # Release the microphone stream on every exit path (break, timeout,
            # cancellation, or shutdown).
            await aclose_frame_source(frame_source)
        if not frames:
            raise ValueError("Voice utterance was empty.")
        return write_pcm_wav(output_path, frames, sample_rate=16_000)
