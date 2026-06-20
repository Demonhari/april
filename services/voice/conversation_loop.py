from __future__ import annotations

import uuid
from pathlib import Path

from apps.cli.client import AprilApiClient
from april_common.settings import get_settings
from services.voice.audio_player import AudioPlayer
from services.voice.microphone import Microphone
from services.voice.speech_to_text import SpeechToText, WhisperCppSpeechToText
from services.voice.text_to_speech import PiperTextToSpeech, TextToSpeech


class PushToTalkLoop:
    def __init__(
        self,
        *,
        api_client: AprilApiClient,
        microphone: Microphone | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.api_client = api_client
        self.microphone = microphone or Microphone()
        self.stt = stt or WhisperCppSpeechToText(
            settings.voice.whisper_binary_path,
            settings.voice.whisper_model_path,
        )
        self.tts = tts or PiperTextToSpeech(
            settings.voice.piper_binary_path,
            settings.voice.piper_model_path,
        )
        self.player = player or AudioPlayer()

    async def run_once(self) -> str:
        self.settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
        audio_path = self.settings.audio_cache_path / f"{uuid.uuid4()}.wav"
        spoken_path = await self.microphone.record_push_to_talk(audio_path)
        text = await self.stt.transcribe(spoken_path)
        response = await self.api_client.post("/voice/input", {"message": text})
        answer = response["result"]["final_message"]
        tts_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-reply.wav"
        output_path = await self.tts.synthesize(answer, tts_path)
        await self.player.play(output_path)
        if not self.settings.voice.retain_debug_audio:
            for path in (audio_path, tts_path):
                if Path(path).exists():
                    Path(path).unlink()
        return answer
