from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path

from april_common.process_environment import ProcessCategory, build_process_environment


class JobWorkerProcessManager:
    """Own a separate Job Worker when no external supervisor is configured."""

    def __init__(self, *, april_home: Path) -> None:
        self.april_home = april_home.expanduser().resolve(strict=True)
        self.runtime_directory = self.april_home / "data" / "runtime" / "job-worker"
        self.status_path = self.runtime_directory / "status.json"
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> bool:
        self.runtime_directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.runtime_directory, 0o700)
        if os.environ.get("APRIL_JOB_WORKER_EXTERNAL") == "1":
            return self.status_path.exists()
        self.status_path.unlink(missing_ok=True)
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "services.jobs.worker",
            "--april-home",
            str(self.april_home),
            "--status-file",
            str(self.status_path),
            cwd=str(self.april_home),
            env=build_process_environment(
                ProcessCategory.JOB_WORKER,
                april_home=self.april_home,
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(100):
            if self.process.returncode is not None:
                return False
            if self.status_path.exists():
                return True
            await asyncio.sleep(0.02)
        await self.stop()
        return False

    async def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
