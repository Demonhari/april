#!/usr/bin/env python3
"""Check that the checked-in pip compatibility pins match uv.lock."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s;#]+)")


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _constraints() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (ROOT / "constraints-dev.txt").read_text(encoding="utf-8").splitlines():
        match = _REQUIREMENT.match(line)
        if match:
            result[_normalize(match.group(1))] = match.group(2)
    return result


def _locked() -> dict[str, str]:
    data = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {
        _normalize(str(package["name"])): str(package["version"])
        for package in data.get("package", [])
        if "name" in package and "version" in package
    }


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_names = {
        _normalize(str(requirement).split("[", 1)[0].split(">", 1)[0].split("=", 1)[0])
        for requirement in pyproject.get("build-system", {}).get("requires", [])
    }
    constraints = _constraints()
    locked = _locked()
    mismatches: list[str] = []
    for name, pinned in sorted(constraints.items()):
        if name in build_names:
            continue
        locked_version = locked.get(name)
        if locked_version is None:
            mismatches.append(f"{name}: missing from uv.lock")
        elif pinned != locked_version:
            mismatches.append(f"{name}: constraints={pinned}, uv.lock={locked_version}")
    if mismatches:
        print("Dependency lock drift detected:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  {mismatch}", file=sys.stderr)
        return 1
    print("constraints-dev.txt matches uv.lock for all shared dependency pins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
