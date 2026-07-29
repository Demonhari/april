from __future__ import annotations

from pathlib import Path

from april_common.errors import RuntimeUnavailableError
from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)


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
        result = await run_restricted_process(
            [
                str(self.binary_path),
                "--model",
                str(self.model_path),
                "--output_file",
                str(output_path),
            ],
            cwd=self.binary_path.parent,
            category=ProcessCategory.SENTINEL_VOICE,
            timeout_seconds=self.timeout,
            max_stdout_bytes=10_000,
            max_stderr_bytes=100_000,
            resource_limit_profile=ResourceLimitProfile.MODEL_UTILITY,
            stdin_bytes=text.encode("utf-8"),
        )
        if result.status is ProcessStatus.TIMED_OUT:
            raise RuntimeUnavailableError("Piper timed out.")
        if result.status is not ProcessStatus.COMPLETED or result.returncode:
            raise RuntimeUnavailableError(
                "Piper failed.",
                {"stderr": result.stderr[:1000], "failure_code": result.failure_code},
            )
        if not output_path.exists() or not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeUnavailableError("Piper did not create a non-empty WAV output.")
        return output_path


class FakeTextToSpeech(TextToSpeech):
    async def synthesize(self, text: str, output_path: Path) -> Path:
        output_path.write_text(text, encoding="utf-8")
        return output_path
