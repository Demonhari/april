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
        if "return exactly one json object with type final_answer" in lower:
            return self._structured_agent_response(prompt, lower)
        if (
            "route this request" in lower
            or "route the user request" in lower
            or '"intent"' in lower
        ):
            route_text = self._routing_user_text(prompt).lower()
            if "apply the fix" in route_text:
                return (
                    '{"intent":"code_modification","agent":"coding_agent",'
                    '"model_id":"april-coding","tools_needed":["patch_generator",'
                    '"patch_applier"],"memory_queries":[],"permission_level":3,'
                    '"risk_level":"code_write","needs_confirmation":true,'
                    '"task_steps":["Generate patch","Request exact patch approval"],'
                    '"decision_summary":"Code modification through structured loop"}'
                )
            if "animation" in route_text or "repository" in route_text or "code" in route_text:
                return (
                    '{"intent":"coding_repo_analysis","agent":"coding_agent",'
                    '"model_id":"april-coding","tools_needed":["git_status","search_files"],'
                    '"memory_queries":[],"permission_level":1,"risk_level":"read_only",'
                    '"needs_confirmation":false,'
                    '"task_steps":["Inspect repository status","Search relevant files"],'
                    '"decision_summary":"Read-only repository investigation"}'
                )
            if "summarize" in route_text or "readme" in route_text:
                return (
                    '{"intent":"document_reading","agent":"reading_agent",'
                    '"model_id":"april-reading","tools_needed":["read_file"],'
                    '"memory_queries":[],"permission_level":1,"risk_level":"read_only",'
                    '"needs_confirmation":false,"task_steps":["Read file"],'
                    '"decision_summary":"Read requested local document"}'
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

    def _routing_user_text(self, prompt: str) -> str:
        if "Current request:" in prompt:
            return prompt.rsplit("Current request:", maxsplit=1)[-1]
        if "USER:" in prompt:
            return prompt.rsplit("USER:", maxsplit=1)[-1]
        if "<|user|>" in prompt:
            return prompt.rsplit("<|user|>", maxsplit=1)[-1]
        if "<|im_start|>user" in prompt:
            return prompt.rsplit("<|im_start|>user", maxsplit=1)[-1]
        return prompt

    def _structured_agent_response(self, prompt: str, lower: str) -> str:
        if "approved tool result" in lower or "tool result" in lower:
            if '"tool": "patch_applier"' in lower or '"tool":"patch_applier"' in lower:
                return (
                    '{"type":"final_answer","message":"Applied the approved patch.",'
                    '"summary":"patch applied","citations":[]}'
                )
            if '"tool": "patch_generator"' in lower or '"tool":"patch_generator"' in lower:
                patch_path = self._extract_json_string(prompt, "patch_path") or "patch.patch"
                return (
                    '{"type":"tool_request","tool":"patch_applier","args":{'
                    f'"patch_path":{self._json_string(patch_path)}'
                    '},"reason":"Apply the generated patch after approval."}'
                )
            return (
                '{"type":"final_answer","message":"Completed the requested inspection.",'
                '"summary":"done","citations":[{"path":"README.md"}]}'
            )
        if "apply the fix" in lower:
            patch = (
                "diff --git a/README.md b/README.md\n"
                "--- a/README.md\n"
                "+++ b/README.md\n"
                "@@ -1,2 +1,3 @@\n"
                " # verify\n"
                " animation bug\n"
                "+fixed animation\n"
            )
            return (
                '{"type":"tool_request","tool":"patch_generator","args":{'
                f'"patch":{self._json_string(patch)}'
                '},"reason":"Create an immutable draft patch artifact."}'
            )
        if "read" in lower or "summarize" in lower:
            return (
                '{"type":"tool_request","tool":"read_file","args":{"path":"README.md"},'
                '"reason":"Read the requested local file."}'
            )
        if "animation" in lower or "repository" in lower:
            return (
                '{"type":"tool_request","tool":"search_files",'
                '"args":{"path":".","query":"animation","limit":20},'
                '"reason":"Find animation-related files."}'
            )
        return (
            '{"type":"final_answer","message":"APRIL fake structured response.",'
            '"summary":"fake","citations":[]}'
        )

    def _extract_json_string(self, text: str, key: str) -> str | None:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text)
        if not match:
            return None
        return match.group(1).encode("utf-8").decode("unicode_escape")

    def _json_string(self, value: str) -> str:
        import json

        return json.dumps(value)
