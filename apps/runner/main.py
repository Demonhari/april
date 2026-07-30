from __future__ import annotations

from typing import Any

from apps.runner.acceptance import run_acceptance as run_acceptance
from apps.runner.audit_commands import audit_app
from apps.runner.commands.common import (
    DesktopTokenBridge as DesktopTokenBridge,
)
from apps.runner.commands.common import (
    _delegate as _delegate,
)
from apps.runner.commands.common import (
    _desktop_base_url as _desktop_base_url,
)
from apps.runner.commands.common import (
    _doctor as _doctor,
)
from apps.runner.commands.common import (
    _effective_fake as _effective_fake,
)
from apps.runner.commands.common import (
    _effective_oneshot as _effective_oneshot,
)
from apps.runner.commands.common import (
    _ensure_services as _ensure_services,
)
from apps.runner.commands.common import (
    _manager as _manager,
)
from apps.runner.commands.common import (
    _open_desktop_browser as _open_desktop_browser,
)
from apps.runner.commands.common import (
    _open_desktop_native as _open_desktop_native,
)
from apps.runner.commands.common import (
    _print_benchmark as _print_benchmark,
)
from apps.runner.commands.common import (
    _print_brain_eval as _print_brain_eval,
)
from apps.runner.commands.common import (
    _print_model_doctor as _print_model_doctor,
)
from apps.runner.commands.common import (
    _print_model_recommendation as _print_model_recommendation,
)
from apps.runner.commands.common import (
    _print_status as _print_status,
)
from apps.runner.commands.common import (
    _print_verification_table as _print_verification_table,
)
from apps.runner.commands.common import (
    _run_april_cli as _run_april_cli,
)
from apps.runner.commands.common import (
    _same_file as _same_file,
)
from apps.runner.commands.common import (
    _status_payload as _status_payload,
)
from apps.runner.commands.finetune import finetune_app
from apps.runner.commands.model_compare import register_model_compare
from apps.runner.commands.model_import import register_model_import_commands
from apps.runner.commands.packaging import package_app
from apps.runner.commands.registry import (
    agent_app,
    app,
    april_app,
    config_app,
    conversation_app,
    eval_app,
    evolve_app,
    jobs_app,
    memory_app,
    model_app,
    profile_app,
    project_app,
    reminder_app,
    reports_app,
    rollout_app,
    setup_app,
    task_app,
    user_profile_app,
    voice_app,
)
from apps.runner.commands.speaker import register_speaker_commands
from apps.runner.database_commands import database_app
from apps.runner.model_downloads import run_model_downloads as run_model_downloads
from apps.runner.preflight import build_preflight_report as build_preflight_report
from apps.runner.readiness import build_readiness_report as build_readiness_report
from apps.runner.security_commands import security_app
from apps.runner.soak import run_fake_soak as run_fake_soak
from apps.runner.verify import TargetMacValidator as TargetMacValidator
from apps.runner.verify import (
    run_all_configured_models_verification as run_all_configured_models_verification,
)
from apps.runner.verify import run_fake_verification as run_fake_verification
from apps.runner.verify import run_real_model_verification as run_real_model_verification
from apps.runner.verify import run_workflow_verification as run_workflow_verification
from apps.runner.voice_conversation_live import (
    run_voice_conversation_live_verification as run_voice_conversation_live_verification,
)
from apps.runner.voice_live import (
    run_voice_live_verification as run_voice_live_verification,
)
from apps.runner.wake_live import (
    run_sentinel_live_verification as run_sentinel_live_verification,
)
from april_common.config_validation import validate_configuration as validate_configuration
from services.voice.health import voice_doctor

collect_voice_doctor = voice_doctor

run_wake_word_live_verification = run_sentinel_live_verification

app.add_typer(april_app, name="april")
april_app.add_typer(model_app, name="model")
model_app.add_typer(profile_app, name="profile")
april_app.add_typer(project_app, name="project")
april_app.add_typer(memory_app, name="memory")
april_app.add_typer(conversation_app, name="conversation")
april_app.add_typer(config_app, name="config")
april_app.add_typer(agent_app, name="agent")
april_app.add_typer(voice_app, name="voice")
april_app.add_typer(reminder_app, name="reminder")
april_app.add_typer(task_app, name="task")
april_app.add_typer(eval_app, name="eval")
april_app.add_typer(setup_app, name="setup")
april_app.add_typer(user_profile_app, name="profile")
april_app.add_typer(reports_app, name="reports")
april_app.add_typer(jobs_app, name="jobs")
april_app.add_typer(evolve_app, name="evolve")
evolve_app.add_typer(rollout_app, name="rollout")
april_app.add_typer(security_app, name="security")
april_app.add_typer(audit_app, name="audit")
april_app.add_typer(database_app, name="database")
april_app.add_typer(finetune_app, name="finetune")
april_app.add_typer(package_app, name="package")
register_model_compare(model_app)
register_model_import_commands(model_app)
register_speaker_commands(voice_app)

from apps.runner.commands import evolve_rollout as _evolve_rollout  # noqa: E402
from apps.runner.commands import runner_acceptance as _runner_acceptance  # noqa: E402
from apps.runner.commands import (  # noqa: E402
    runner_configuration as _runner_configuration,
)
from apps.runner.commands import runner_core as _runner_core  # noqa: E402
from apps.runner.commands import runner_jobs as _runner_jobs  # noqa: E402
from apps.runner.commands import runner_memory as _runner_memory  # noqa: E402
from apps.runner.commands import runner_models as _runner_models  # noqa: E402
from apps.runner.commands import (  # noqa: E402
    runner_productivity as _runner_productivity,
)
from apps.runner.commands import runner_reports as _runner_reports  # noqa: E402
from apps.runner.commands import runner_services as _runner_services  # noqa: E402
from apps.runner.commands import runner_setup as _runner_setup  # noqa: E402
from apps.runner.commands import (  # noqa: E402
    runner_verification as _runner_verification,
)
from apps.runner.commands import runner_voice as _runner_voice  # noqa: E402

_COMMAND_MODULES = (
    _runner_jobs,
    _runner_core,
    _runner_models,
    _runner_memory,
    _runner_productivity,
    _runner_voice,
    _runner_configuration,
    _runner_setup,
    _runner_verification,
    _runner_acceptance,
    _runner_services,
    _runner_reports,
    _evolve_rollout,
)


def __getattr__(name: str) -> Any:
    for module in _COMMAND_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


if __name__ == "__main__":
    app()
