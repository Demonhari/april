from __future__ import annotations

import re
import tomllib
from pathlib import Path

SUPPORTED_RANGE = ">=3.11,<3.14"
SUPPORTED_MINORS = {"3.11", "3.12", "3.13"}


def test_python_support_range_matches_project_lock_docs_and_ci() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert project["project"]["requires-python"] == SUPPORTED_RANGE
    assert lock["requires-python"].replace(" ", "") == SUPPORTED_RANGE
    assert "APRIL supports Python 3.11 through 3.13" in readme

    matrix_versions = set(re.findall(r'python-version:\s*"(3\.\d+)"', workflow))
    assert matrix_versions >= SUPPORTED_MINORS
    assert not ({"3.10", "3.14"} & matrix_versions)
