"""A deliberately inert provider protocol for future local MCP adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from services.integrations.mcp.contracts import ExternalToolManifest


class LocalProvider(Protocol):
    async def discover(self) -> list[ExternalToolManifest]: ...

    async def invoke(
        self, tool: ExternalToolManifest, arguments: Mapping[str, object]
    ) -> object: ...


ProviderFactory = Callable[[ExternalToolManifest], Awaitable[LocalProvider]]
