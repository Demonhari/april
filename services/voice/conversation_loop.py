from __future__ import annotations

import uuid
from pathlib import Path

from apps.cli.client import AprilApiClient
from april_common.settings import get_settings
from services.voice.audio_player import AudioPlayer, SoundDeviceAudioPlayer
from services.voice.microphone import Microphone, SoundDeviceMicrophone
from services.voice.speech_to_text import SpeechToText, WhisperCppSpeechToText
from services.voice.text_to_speech import PiperTextToSpeech, TextToSpeech
from services.voice.vad import VoiceActivityDetector
from services.voice.wake_word import OpenWakeWordDetector


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
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.api_client = api_client
        self.microphone = microphone or SoundDeviceMicrophone(
            device=settings.voice.input_device,
            max_seconds=settings.voice.max_record_seconds,
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
    pass


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
        return await super().run_once()
