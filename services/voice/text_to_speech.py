from __future__ import annotations

import asyncio
from pathlib import Path

from april_common.errors import RuntimeUnavailableError


class TextToSpeech:
    async def synthesize(self, text: str, output_path: Path) -> Path:
        raise NotImplementedError


class PiperTextToSpeech(TextToSpeech):
    def __init__(
        self, binary_path: Path | None, model_path: Path | None, *, timeout: float = 60.0
    ) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.timeout = timeout

    async def synthesize(self, text: str, output_path: Path) -> Path:
        if self.binary_path is None or self.model_path is None:
            raise RuntimeUnavailableError("Piper binary/model paths are not configured.")
        if not self.binary_path.exists() or not self.model_path.exists():
            raise RuntimeUnavailableError("Piper binary or model path is missing.")
        process = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--output_file",
            str(output_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(text.encode("utf-8")),
                timeout=self.timeout,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeUnavailableError("Piper timed out.") from exc
        if process.returncode:
            raise RuntimeUnavailableError(
                "Piper failed.",
                {"stderr": stderr.decode("utf-8", errors="replace")[:1000]},
            )
        if not output_path.exists() or not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeUnavailableError("Piper did not create a non-empty WAV output.")
        return output_path


class FakeTextToSpeech(TextToSpeech):
    async def synthesize(self, text: str, output_path: Path) -> Path:
        output_path.write_text(text, encoding="utf-8")
        return output_path
