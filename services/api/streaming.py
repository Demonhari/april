from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, request_id: str, payload: dict[str, Any]) -> str:
    body = {"request_id": request_id, "event": event, "payload": payload}
    return f"event: {event}\ndata: {json.dumps(body)}\n\n"
