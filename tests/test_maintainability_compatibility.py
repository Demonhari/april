from __future__ import annotations

import hashlib
from pathlib import Path

from typer.main import get_command

from apps.cli.main import app as legacy_cli_app
from apps.runner.readiness import (
    ReadinessCheck,
    ReadinessModel,
    ReadinessReport,
    VoiceArtifact,
    build_readiness_report,
)
from services.evolution.rollouts import (
    CanaryContext,
    CanarySelection,
    PromotionReadiness,
    RealPromptShadowEvaluator,
    RolloutBlocked,
    RolloutError,
    RolloutRecord,
    RolloutService,
    ShadowMetrics,
    inspect_rollout_state,
    reviewed_dataset_hash,
)
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import (
    GenerationValidationResult,
    VectorMemory,
)

EXPECTED_READINESS_FIELD_COUNT = 98
EXPECTED_READINESS_SCHEMA_SHA256 = (
    "ea2bf2c3425d3ac5e0ddabe572ba3c551a3837d99fdd686c5a16c560bc888335"
)

EXPECTED_LEGACY_CLI_COMMANDS = {
    line
    for line in """
agent
agent pool
agent run
approvals
approve
ask
bad
briefing
chat
conversation
conversation delete
daemon
daemon install
daemon start
daemon status
daemon stop
daemon uninstall
deny
doc
doc add
doc list
doc search
evolve
evolve adapter
evolve adapter activate
evolve adapter list
evolve adapter rollback
evolve approve
evolve dataset
evolve dataset export
evolve diff
evolve evals
evolve evals pending
evolve evals promote
evolve evals reject
evolve evals show
evolve history
evolve off
evolve on
evolve pending
evolve report
evolve rollback
evolve status
evolve versions
good
health
jobs
jobs cancel
jobs list
jobs retry
jobs show
jobs submit
memory
memory delete
memory export
memory inspect
memory list
memory reindex
memory repair-index
memory search
model
model load
model unload
models
mute
playbook
playbook adopt
playbook list
playbook mine
playbook run
project
project add
project index
projects
reminder
reminder create
reminder delete
reminder list
sessions
task
task list
voice
voice devices
voice doctor
voice enroll
voice health
voice listen
voice ptt
voice test-record
voice test-stt
voice test-tts
""".splitlines()
    if line
}


def _cli_inventory(command: object, prefix: str = "") -> set[str]:
    inventory: set[str] = set()
    for name, child in getattr(command, "commands", {}).items():
        path = f"{prefix} {name}".strip()
        inventory.add(path)
        inventory.update(_cli_inventory(child, path))
    return inventory


def test_readiness_public_imports_and_schema_are_stable() -> None:
    assert ReadinessCheck.__name__ == "ReadinessCheck"
    assert VoiceArtifact.__name__ == "VoiceArtifact"
    assert ReadinessModel.__name__ == "ReadinessModel"
    assert callable(build_readiness_report)
    fields = "\n".join(sorted(ReadinessReport.model_fields))
    assert len(ReadinessReport.model_fields) == EXPECTED_READINESS_FIELD_COUNT
    assert hashlib.sha256(fields.encode()).hexdigest() == EXPECTED_READINESS_SCHEMA_SHA256


def test_memory_and_rollout_facade_imports_are_stable() -> None:
    assert SqliteMemory.__name__ == "SqliteMemory"
    assert VectorMemory.__name__ == "VectorMemory"
    assert GenerationValidationResult.__name__ == "GenerationValidationResult"
    for public_object in (
        RolloutService,
        RolloutRecord,
        RolloutError,
        RolloutBlocked,
        ShadowMetrics,
        RealPromptShadowEvaluator,
        CanaryContext,
        CanarySelection,
        PromotionReadiness,
        inspect_rollout_state,
        reviewed_dataset_hash,
    ):
        assert public_object is not None


def test_legacy_cli_inventory_is_stable() -> None:
    assert _cli_inventory(get_command(legacy_cli_app)) == EXPECTED_LEGACY_CLI_COMMANDS


def test_refactored_implementation_modules_stay_below_800_lines() -> None:
    modules = [
        *Path("apps/runner").glob("readiness*.py"),
        Path("apps/cli/main.py"),
        *Path("apps/cli/commands").glob("*.py"),
        *Path("services/memory").glob("sqlite_*.py"),
        *Path("services/memory").glob("vector_*.py"),
        *Path("services/evolution").glob("rollout*.py"),
    ]
    oversized = {
        str(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in modules
        if len(path.read_text(encoding="utf-8").splitlines()) > 800
    }
    assert oversized == {}
