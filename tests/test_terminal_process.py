from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_terminal_process_path_round_trips_input_and_output(tmp_path: Path) -> None:
    """Exercise a real nested subprocess with inherited streams."""
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from april_common.process_environment import ProcessCategory
            from april_common.process_runner import run_terminal_process_sync

            raise SystemExit(run_terminal_process_sync(
                [
                    sys.executable,
                    "-c",
                    "print('PROMPT', flush=True); print('ECHO:' + input(), flush=True)",
                ],
                cwd=Path(sys.argv[1]),
                category=ProcessCategory.CLI,
                april_home=Path(sys.argv[1]),
            ))
            """
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(driver), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        text=True,
    )
    output, _ = process.communicate("terminal input\n", timeout=10)
    assert process.returncode == 0
    assert "PROMPT" in output
    assert "ECHO:terminal input" in output
