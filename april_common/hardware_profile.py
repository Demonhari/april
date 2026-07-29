from __future__ import annotations

import hashlib
import json
import os
import platform
from typing import Any


def safe_hardware_profile() -> dict[str, Any]:
    profile = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor_family": platform.processor().split()[0][:64] if platform.processor() else None,
        "cpu_count": os.cpu_count(),
    }
    digest = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**profile, "id": digest}
