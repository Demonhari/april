"""Trust-boundary types for local MCP and browser providers."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

IntegrationKind = Literal["mcp_stdio", "mcp_loopback", "browser_loopback"]
ManifestStatus = Literal["discovered", "candidate", "validated", "verified", "approved", "active"]
_STATUS_ORDER = {
    "discovered": 0,
    "candidate": 1,
    "validated": 2,
    "verified": 3,
    "approved": 4,
    "active": 5,
}


class IntegrationEndpoint(BaseModel):
    kind: IntegrationKind
    command: tuple[str, ...] = ()
    url: str | None = None

    @model_validator(mode="after")
    def local_only(self) -> IntegrationEndpoint:
        if self.kind == "mcp_stdio":
            if not self.command or any(not item or "\x00" in item for item in self.command):
                raise ValueError("stdio integrations require a bounded command")
            if self.url is not None:
                raise ValueError("stdio integrations do not use a URL")
            return self
        if not self.url or not _loopback_url(self.url):
            raise ValueError("integration endpoints must be local stdio or loopback")
        if self.command:
            raise ValueError("URL integrations do not use a command")
        return self


class ExternalToolManifest(BaseModel):
    integration_id: str
    server_identity: str
    tool_name: str
    input_schema: dict[str, object]
    origin: str = "local"
    browser_profile: str | None = None
    declared_risk: int = Field(ge=0, le=4)
    requested_network_domains: tuple[str, ...] = ()
    declared_side_effects: tuple[str, ...] = ()
    endpoint: IntegrationEndpoint
    status: ManifestStatus = "discovered"
    specification_digest: str
    artifact_digest: str | None = None
    approval_id: str | None = None
    exact_arguments_digest: str | None = None

    @property
    def schema_fingerprint(self) -> str:
        encoded = json.dumps(self.input_schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def with_status(
        self, status: ManifestStatus, *, approval_id: str | None = None
    ) -> ExternalToolManifest:
        if _STATUS_ORDER[status] != _STATUS_ORDER[self.status] + 1:
            raise ValueError("tool manifest lifecycle transition is not allowed")
        if status == "active" and not approval_id and not self.approval_id:
            raise ValueError("activation requires an APRIL approval identity")
        return self.model_copy(
            update={"status": status, "approval_id": approval_id or self.approval_id}
        )

    def schema_changed_from(self, previous: ExternalToolManifest) -> bool:
        return self.schema_fingerprint != previous.schema_fingerprint

    def activate_against(
        self, previous: ExternalToolManifest, *, approval_id: str
    ) -> ExternalToolManifest:
        if self.schema_changed_from(previous):
            raise ValueError("tool schema drift invalidates the previous approval")
        if previous.status != "approved" or self.status != "approved":
            raise ValueError("only an approved manifest can become active")
        return self.with_status("active", approval_id=approval_id)


def _loopback_url(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
