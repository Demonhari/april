from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_registry import ModelDefinition


class FakeBackend(RuntimeBackend):
    supports_concurrent_generation = False

    def __init__(self, *, fail_stream: bool = False, fail_generate: bool = False) -> None:
        self.loaded_model: ModelDefinition | None = None
        self.fail_stream = fail_stream
        self.fail_generate = fail_generate

    async def load(self, model: ModelDefinition) -> None:
        await asyncio.sleep(0)
        self.loaded_model = model

    async def unload(self) -> None:
        await asyncio.sleep(0)
        self.loaded_model = None

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        if self.fail_generate:
            raise RuntimeError("fake generation failure")
        text = self._response_for_prompt(prompt)
        for sequence in stop or []:
            if sequence:
                text = text.split(sequence, maxsplit=1)[0]
        output_words = text.split()[:max_output_tokens]
        content = " ".join(output_words)
        return GenerationResult(
            text=content,
            input_tokens=len(await self.tokenize(prompt)),
            output_tokens=len(output_words),
        )

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        if self.fail_stream:
            raise RuntimeError("fake stream failure")
        result = await self.generate(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )
        words = result.text.split()
        for index, word in enumerate(words):
            await asyncio.sleep(0)
            suffix = " " if index < len(words) - 1 else ""
            yield word + suffix

    async def tokenize(self, text: str) -> list[int]:
        return [index for index, _ in enumerate(re.findall(r"\S+", text))]

    async def health(self) -> BackendHealth:
        return BackendHealth(ok=True, message="fake backend ready")

    def _response_for_prompt(self, prompt: str) -> str:
        lower = prompt.lower()
        if "route this request" in lower or '"intent"' in lower:
            if "animation" in lower or "repository" in lower or "code" in lower:
                return (
                    '{"intent":"coding_repo_analysis","agent":"coding_agent",'
                    '"model_id":"april-coding","tools_needed":["git_status","search_files"],'
                    '"memory_queries":[],"permission_level":1,"risk_level":"read_only",'
                    '"needs_confirmation":false,'
                    '"task_steps":["Inspect repository status","Search relevant files"],'
                    '"decision_summary":"Read-only repository investigation"}'
                )
            return (
                '{"intent":"planning","agent":"general_agent","model_id":"april-brain",'
                '"tools_needed":[],"memory_queries":[],"permission_level":0,'
                '"risk_level":"none","needs_confirmation":false,'
                '"task_steps":["Answer directly"],"decision_summary":"General response"}'
            )
        if "plan my work today" in lower:
            return (
                "Start with priorities, schedule focused blocks, "
                "then leave room for follow-up work."
            )
        if "animation" in lower:
            return (
                "I inspected the repository context and found likely "
                "animation-related files to review."
            )
        return "APRIL fake response."
