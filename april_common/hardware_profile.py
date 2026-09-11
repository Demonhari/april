from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

_physical_cpu_cache: int | None = None
_cpu_brand_cache: str | None = None


def _darwin_sysctl_uint(name: str) -> int | None:  # pragma: no cover - host-specific probe
    try:
        libc = ctypes.CDLL(None)
        sysctlbyname = libc.sysctlbyname
        sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        sysctlbyname.restype = ctypes.c_int
        value = ctypes.c_uint32()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        result = sysctlbyname(
            name.encode("ascii"), ctypes.byref(value), ctypes.byref(size), None, 0
        )
        return int(value.value) if result == 0 and value.value > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _darwin_sysctl_string(name: str) -> str | None:  # pragma: no cover - host-specific probe
    try:
        libc = ctypes.CDLL(None)
        sysctlbyname = libc.sysctlbyname
        sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        sysctlbyname.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(256)
        size = ctypes.c_size_t(len(buffer))
        result = sysctlbyname(name.encode("ascii"), buffer, ctypes.byref(size), None, 0)
        if result != 0:
            return None
        value = buffer.value.decode("utf-8", errors="replace").strip()
        return value or None
    except (AttributeError, OSError, TypeError, ValueError, UnicodeError):
        return None


def physical_cpu_count() -> int | None:
    """Return measured physical cores, caching successful values only."""
    global _physical_cpu_cache
    if _physical_cpu_cache is not None:
        return _physical_cpu_cache
    if sys.platform == "darwin":
        value = _darwin_sysctl_uint("hw.physicalcpu")
    elif sys.platform.startswith("linux"):
        value = _linux_physical_cpu_count()
    else:
        value = None
    if value is not None:
        _physical_cpu_cache = value
    return value


def _linux_physical_cpu_count() -> int | None:  # pragma: no cover - host-specific probe
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


def cpu_brand() -> str | None:
    """Return a stable CPU brand string, or None when unavailable."""
    global _cpu_brand_cache
    if _cpu_brand_cache is not None:
        return _cpu_brand_cache
    if sys.platform == "darwin":
        value = _darwin_sysctl_string("machdep.cpu.brand_string")
    elif sys.platform.startswith("linux"):
        value = _linux_cpu_brand()
    else:
        value = None
    if value is not None:
        _cpu_brand_cache = value
    return value


def _linux_cpu_brand() -> str | None:  # pragma: no cover - host-specific probe
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware", "Processor"}:
                result = value.strip()
                if result:
                    return result
    except OSError:
        pass
    return None


def safe_hardware_profile() -> dict[str, Any]:
    profile = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor_family": (cpu_brand() or "").split()[0][:64] or None,
        "cpu_count": os.cpu_count(),
    }
    digest = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**profile, "id": digest}
