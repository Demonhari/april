"""``run april setup checklist`` — read-only first-run onboarding checklist.

Prints the recommended setup order from a fresh checkout to a hardened
daily-driver, and detects what is already done. It is strictly read-only: it
never installs, downloads, mutates configuration, starts a service, loads a
model, or opens the microphone. Each step's state is derived from the same
redacted, offline signals the daily-driver doctor uses, so the two never
disagree.

Per step it reports one of: ``done``, ``warning`` (done with a caveat, or an
optional step left unconfigured), ``blocker`` (a hard prerequisite is missing),
or ``next`` (a pending recommended action). The single headline ``next_command``
points at the first thing worth doing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.daily_driver import DailyDriverReport, build_daily_driver_report
from april_common.time import utc_now_iso

ChecklistStatus = Literal["done", "warning", "blocker", "next"]

# Map the daily-driver status enum onto the checklist's onboarding vocabulary.
_STATUS_MAP: dict[str, ChecklistStatus] = {
    "ready": "done",
    "warning": "warning",
    "blocker": "blocker",
    "not_run": "next",
}

# (step number, title, daily-driver check name or None, exact command).
_STEPS: tuple[tuple[int, str, str | None, str], ...] = (
    (1, "install dependencies", "llama-cpp-python", "pip install -e '.[runtime]'"),
    (2, "setup tokens", "token hardening", "run april setup tokens"),
    (
        3,
        "setup models",
        "configured GGUF presence",
        "run april setup models --brain /absolute/path/brain.gguf "
        "--coding /absolute/path/coding.gguf --reading /absolute/path/reading.gguf --apply",
    ),
    (4, "config validate", "config validation", "run april config validate"),
    (
        5,
        "verify all configured models",
        "latest real-model verification",
        "run april verify --all-configured-models --require-real-model "
        "--report data/verification/mac-readiness.json",
    ),
    (
        6,
        "verify workflow real",
        "latest workflow-real verification",
        "run april verify --workflow --real-model --report data/verification/workflow-real.json",
    ),
    (7, "go-live", "latest go-live", "run april go-live --write-report --start-services"),
    (
        8,
        "setup embeddings (optional, recommended)",
        "embedding provider",
        "run april setup embeddings --model /absolute/path/to/embedding.gguf "
        "--id april-embedding --apply",
    ),
    (9, "memory reindex", "vector index compatibility", "run april memory reindex"),
    (
        10,
        "optional voice setup",
        "voice milestone",
        "run april setup voice --whisper-binary /path/to/whisper.cpp/main "
        "--whisper-model /path/to/ggml-base.en.bin --piper-binary /path/to/piper "
        "--piper-model /path/to/voice.onnx --dry-run",
    ),
    (11, "optional desktop app stub", None, "run april setup app-stub"),
)


class ChecklistStep(BaseModel):
    number: int
    title: str
    status: ChecklistStatus
    detail: str
    command: str


class SetupChecklist(BaseModel):
    schema_version: int = 1
    report_type: Literal["setup_checklist"] = "setup_checklist"
    generated_at: str
    steps: list[ChecklistStep] = Field(default_factory=list)
    next_command: str | None = None


def _desktop_stub_step(home: Path, command: str) -> ChecklistStep:
    # The unsigned local app stub is optional; detect it without creating one.
    stub = home / "dist" / "APRIL.app"
    if stub.exists():
        return ChecklistStep(
            number=11,
            title="optional desktop app stub",
            status="done",
            detail="dist/APRIL.app present",
            command=command,
        )
    return ChecklistStep(
        number=11,
        title="optional desktop app stub",
        status="next",
        detail="optional: no app stub created (read-only Desktop works via `run april desktop`)",
        command=command,
    )


def build_setup_checklist(home: Path, *, daily: DailyDriverReport | None = None) -> SetupChecklist:
    """Assemble the read-only onboarding checklist from daily-driver signals."""
    home = home.expanduser().resolve()
    report = daily or build_daily_driver_report(home)
    checks = {check.name: check for check in report.checks}

    steps: list[ChecklistStep] = []
    for number, title, check_name, command in _STEPS:
        if check_name is None:
            steps.append(_desktop_stub_step(home, command))
            continue
        check = checks.get(check_name)
        if check is None:
            steps.append(
                ChecklistStep(
                    number=number,
                    title=title,
                    status="next",
                    detail="not detected",
                    command=command,
                )
            )
            continue
        status = _STATUS_MAP.get(check.status, "next")
        steps.append(
            ChecklistStep(
                number=number,
                title=title,
                status=status,
                detail=check.detail,
                command=command,
            )
        )

    return SetupChecklist(
        generated_at=utc_now_iso(),
        steps=steps,
        next_command=_headline_next(steps),
    )


def _headline_next(steps: list[ChecklistStep]) -> str | None:
    # First hard blocker, then first pending step, then first caveat — in order.
    for wanted in ("blocker", "next", "warning"):
        for step in steps:
            if step.status == wanted:
                return step.command
    return None
