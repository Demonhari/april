from __future__ import annotations

import json
from typing import Literal

from april_common.time import utc_now_iso
from services.april_runtime.schemas import StreamEvent


def stream_event(
    *,
    event: Literal["meta", "token", "usage", "done", "error"],
    request_id: str,
    model_id: str,
    payload: dict[str, object],
) -> str:
    envelope = StreamEvent(
        request_id=request_id,
        event=event,
        timestamp=utc_now_iso(),
        model_id=model_id,
        payload=payload,
    )
    return f"event: {event}\ndata: {json.dumps(envelope.model_dump())}\n\n"
