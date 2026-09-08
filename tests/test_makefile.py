from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def test_desktop_js_propagates_an_earlier_node_failure(tmp_path: Path) -> None:
    """The desktop target must not hide a failed check behind later successes."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state = tmp_path / "node-called"
    fake_node = fake_bin / "node"
    fake_node.write_text(
        "#!/bin/sh\n"
        'if [ ! -e "$NODE_FAILURE_STATE" ]; then\n'
        '  : > "$NODE_FAILURE_STATE"\n'
        "  exit 17\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_node.chmod(0o755)

    make = shutil.which("make")
    assert make is not None
    environment = os.environ.copy()
    environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
    environment["NODE_FAILURE_STATE"] = str(state)
    result = subprocess.run(
        [make, "desktop-js"],
        cwd=Path(__file__).parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert state.exists()
