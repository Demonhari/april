from __future__ import annotations

from pathlib import Path

import pytest

from services.voice.audio_player import FakeAudioPlayer
from services.voice.conversation_loop import PushToTalkLoop
from services.voice.health import voice_health
from services.voice.microphone import FakeMicrophone
from services.voice.speech_to_text import FakeSpeechToText
from services.voice.text_to_speech import FakeTextToSpeech


class FakeApi:
    async def post(self, path: str, payload: dict[str, str]) -> dict[str, object]:
        return {"result": {"final_message": "voice answer"}}


def test_voice_degraded_without_dependencies(settings_tmp) -> None:
    enabled = settings_tmp.model_copy(
        update={"voice": settings_tmp.voice.model_copy(update={"enabled": True})}
    )
    assert voice_health(enabled).status == "degraded"


@pytest.mark.asyncio
async def test_fake_voice_conversation_loop(settings_tmp, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    loop = PushToTalkLoop(
        api_client=FakeApi(),  # type: ignore[arg-type]
        microphone=FakeMicrophone(audio),
        stt=FakeSpeechToText("hello"),
        tts=FakeTextToSpeech(),
        player=FakeAudioPlayer(),
    )
    assert await loop.run_once() == "voice answer"
