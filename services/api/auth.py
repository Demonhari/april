from __future__ import annotations

import secrets

from fastapi import Header

from april_common.errors import PermissionDeniedError
from april_common.settings import AprilSettings


async def require_bearer_token(
    settings: AprilSettings,
    authorization: str | None = Header(default=None),
) -> None:
    configured_token = settings.api.token.strip()
    if not configured_token:
        raise PermissionDeniedError("API bearer token is not configured.")
    if not authorization or not authorization.startswith("Bearer "):
        raise PermissionDeniedError("Bearer token is required.")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, configured_token):
        raise PermissionDeniedError("Invalid bearer token.")
