from __future__ import annotations

from pathlib import Path

from april_common.errors import RuntimeUnavailableError
from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)


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
        result = await run_restricted_process(
            [
                str(self.binary_path),
                "-m",
                str(self.model_path),
                "-f",
                str(audio_path),
                "-nt",
            ],
            cwd=self.binary_path.parent,
            category=ProcessCategory.SENTINEL_VOICE,
            timeout_seconds=self.timeout,
            max_stdout_bytes=100_000,
            max_stderr_bytes=100_000,
            resource_limit_profile=ResourceLimitProfile.MODEL_UTILITY,
        )
        if result.status is ProcessStatus.TIMED_OUT:
            raise RuntimeUnavailableError("whisper.cpp timed out.")
        if result.status is not ProcessStatus.COMPLETED or result.returncode:
            raise RuntimeUnavailableError(
                "whisper.cpp failed.",
                {"stderr": result.stderr[:1000], "failure_code": result.failure_code},
            )
        return result.stdout.strip()


class FakeSpeechToText(SpeechToText):
    def __init__(self, text: str = "April, plan my work today.") -> None:
        self.text = text

    async def transcribe(self, audio_path: Path) -> str:
        return self.text
