from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from april_common.settings import AprilSettings, load_settings, reset_settings_cache
from services.april_runtime.schemas import ChatMessage, ChatResponse, Usage


@pytest.fixture
def settings_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AprilSettings:
    monkeypatch.setenv("APRIL_HOME", str(tmp_path))
    monkeypatch.setenv("APRIL_RUNTIME_BACKEND", "fake")
    monkeypatch.setenv("APRIL_DATABASE_PATH", str(tmp_path / "data" / "april.db"))
    monkeypatch.setenv("APRIL_VECTOR_INDEX_PATH", str(tmp_path / "data" / "vector_index"))
    monkeypatch.setenv("APRIL_AUDIT_PATH", str(tmp_path / "logs" / "audit.jsonl"))
    monkeypatch.setenv("APRIL_ALLOWED_FILESYSTEM_ROOTS", str(tmp_path))
    reset_settings_cache()
    settings = load_settings(root=tmp_path)
    (tmp_path / "README.md").write_text("# test repo\nanimation bug\n", encoding="utf-8")
    yield settings
    reset_settings_cache()
    for key in list(os.environ):
        if key.startswith("APRIL_"):
            monkeypatch.delenv(key, raising=False)


class FakeRuntimeClient:
    def __init__(self) -> None:
        self.last_messages: list[ChatMessage] = []
        self.calls: list[list[ChatMessage]] = []
        self.stream_called = False

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: Any | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        snapshot = [message.model_copy() for message in messages]
        self.last_messages = snapshot
        self.calls.append(snapshot)
        joined = "\n".join(message.content for message in messages)
        lower = joined.lower()
        if "return exactly one json object with type final_answer" in lower:
            content = self._structured_response(joined, lower)
        elif "route this request" in lower or "route the user request" in lower:
            if "apply the fix" in lower:
                content = (
                    '{"intent":"code_modification","agent":"coding_agent","model_id":"april-coding",'
                    '"tools_needed":["patch_generator","patch_applier"],'
                    '"memory_queries":[],"permission_level":3,'
                    '"risk_level":"code_write","needs_confirmation":true,'
                    '"task_steps":["Generate patch","Request exact patch approval"],'
                    '"decision_summary":"Code write request"}'
                )
            elif "animation" in lower:
                content = (
                    '{"intent":"coding_repo_analysis","agent":"coding_agent","model_id":"april-coding",'
                    '"tools_needed":["search_files","read_file"],"memory_queries":[],"permission_level":1,'
                    '"risk_level":"read_only","needs_confirmation":false,'
                    '"task_steps":["Search files","Read relevant file"],'
                    '"decision_summary":"Read-only repo analysis"}'
                )
            elif "summarize" in lower or "readme" in lower:
                content = (
                    '{"intent":"document_reading","agent":"reading_agent",'
                    '"model_id":"april-reading","tools_needed":["read_file"],'
                    '"memory_queries":[],"permission_level":1,"risk_level":"read_only",'
                    '"needs_confirmation":false,"task_steps":["Read file"],'
                    '"decision_summary":"Read requested local document"}'
                )
            else:
                content = (
                    '{"intent":"planning","agent":"general_agent","model_id":"april-brain",'
                    '"tools_needed":[],"memory_queries":[],"permission_level":0,'
                    '"risk_level":"none","needs_confirmation":false,'
                    '"task_steps":["Answer directly"],"decision_summary":"General planning"}'
                )
        elif "unified diff patch only" in lower:
            content = (
                "diff --git a/README.md b/README.md\n"
                "--- a/README.md\n"
                "+++ b/README.md\n"
                "@@ -1,2 +1,3 @@\n"
                " # test repo\n"
                " animation bug\n"
                "+fixed animation\n"
            )
        else:
            content = "Start with the most important task, then schedule focused work blocks."
        return ChatResponse(
            request_id=request_id or "test-request",
            model_id=model_id,
            content=content,
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    def _structured_response(self, prompt: str, lower: str) -> str:
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
                    f'"patch_path":{json.dumps(patch_path)}'
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
                " # test repo\n"
                " animation bug\n"
                "+fixed animation\n"
            )
            return (
                '{"type":"tool_request","tool":"patch_generator","args":{'
                f'"patch":{json.dumps(patch)}'
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

    async def models(self) -> dict[str, Any]:
        return {"models": []}

    async def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        return {"status": "ok", "backend": "fake", "timeout": timeout}

    async def load(self, model_id: str, *, request_id: str | None = None) -> dict[str, Any]:
        return {
            "request_id": request_id or "test-request",
            "model_id": model_id,
            "state": "loaded",
            "message": "loaded",
        }

    async def unload(self, model_id: str, *, request_id: str | None = None) -> dict[str, Any]:
        return {
            "request_id": request_id or "test-request",
            "model_id": model_id,
            "state": "unloaded",
            "message": "unloaded",
        }

    async def stream(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: Any | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        self.stream_called = True
        self.last_messages = messages
        for token in ("streamed ", "answer"):
            yield json.dumps(
                {
                    "request_id": request_id or "test-request",
                    "event": "token",
                    "model_id": model_id,
                    "payload": {"text": token},
                }
            )
        yield json.dumps(
            {
                "request_id": request_id or "test-request",
                "event": "usage",
                "model_id": model_id,
                "payload": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            }
        )
        yield json.dumps(
            {
                "request_id": request_id or "test-request",
                "event": "done",
                "model_id": model_id,
                "payload": {"finish_reason": "stop"},
            }
        )
