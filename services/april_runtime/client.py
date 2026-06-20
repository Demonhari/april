from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.schemas import ChatMessage, ChatRequest, ChatResponse, GenerationOptions


class RuntimeClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        request = ChatRequest(
            model_id=model_id,
            messages=messages,
            options=options or GenerationOptions(),
            request_id=request_id,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/runtime/chat",
                    json=request.model_dump(),
                )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return ChatResponse.model_validate(response.json())

    async def models(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/runtime/models")
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
        if response.status_code >= 400:
            raise RuntimeUnavailableError("April Runtime returned an error.", response.json())
        return response.json()

    async def stream(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        request = ChatRequest(
            model_id=model_id,
            messages=messages,
            options=options or GenerationOptions(),
            request_id=request_id,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client, client.stream(
                "POST",
                f"{self.base_url}/runtime/stream",
                json=request.model_dump(),
            ) as response:
                if response.status_code >= 400:
                    raise RuntimeUnavailableError(
                        "April Runtime returned an error.",
                        {"status_code": response.status_code},
                    )
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "April Runtime is offline.", {"url": self.base_url}
            ) from exc
