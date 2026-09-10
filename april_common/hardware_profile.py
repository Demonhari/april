from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from april_common.process_environment import ProcessCategory
from april_common.process_runner import ProcessStatus, run_restricted_process_sync


@lru_cache(maxsize=1)
def physical_cpu_count() -> int | None:  # pragma: no cover - host-specific probes
    """Return measured physical cores, or None when the platform cannot say."""
    if sys.platform == "darwin":
        try:
            result = run_restricted_process_sync(
                ["sysctl", "-n", "hw.physicalcpu"],
                cwd=Path("/"),
                category=ProcessCategory.DAEMON,
                timeout_seconds=2.0,
                max_stdout_bytes=128,
                max_stderr_bytes=128,
            )
            if result.status == ProcessStatus.COMPLETED and result.returncode == 0:
                value = int(result.stdout.strip())
                return value if value > 0 else None
        except (OSError, RuntimeError, ValueError):
            return None
        return None
    if sys.platform.startswith("linux"):
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except OSError:
            return None
        pairs: set[tuple[str, str]] = set()
        physical: str | None = None
        core: str | None = None
        for line in [*text.splitlines(), ""]:
            key, _, value = line.partition(":")
            if key.strip() == "physical id":
                physical = value.strip()
            elif key.strip() == "core id":
                core = value.strip()
            elif not line.strip():
                if physical is not None and core is not None:
                    pairs.add((physical, core))
                physical = core = None
        return len(pairs) or None
    return None


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
