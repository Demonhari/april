from __future__ import annotations

import asyncio
from pathlib import Path

from april_common.errors import RuntimeUnavailableError


class SpeechToText:
    async def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


class WhisperCppSpeechToText(SpeechToText):
    def __init__(
        self, binary_path: Path | None, model_path: Path | None, *, timeout: float = 60.0
    ) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.timeout = timeout

    async def transcribe(self, audio_path: Path) -> str:
        if self.binary_path is None or self.model_path is None:
            raise RuntimeUnavailableError("whisper.cpp binary/model paths are not configured.")
        if not self.binary_path.exists() or not self.model_path.exists():
            raise RuntimeUnavailableError("whisper.cpp binary or model path is missing.")
        process = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            "-m",
            str(self.model_path),
            "-f",
            str(audio_path),
            "-nt",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeUnavailableError("whisper.cpp timed out.") from exc
        if process.returncode:
            raise RuntimeUnavailableError(
                "whisper.cpp failed.",
                {"stderr": stderr.decode("utf-8", errors="replace")[:1000]},
            )
        return stdout.decode("utf-8", errors="replace").strip()


class FakeSpeechToText(SpeechToText):
    def __init__(self, text: str = "April, plan my work today.") -> None:
        self.text = text

    async def transcribe(self, audio_path: Path) -> str:
        return self.text
