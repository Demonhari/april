"""Typer command groups shared by the legacy CLI composition modules."""

from __future__ import annotations

import typer

app = typer.Typer(help="APRIL local assistant CLI.")
model_app = typer.Typer(help="Model operations.")
project_app = typer.Typer(help="Project operations.")
memory_app = typer.Typer(help="Memory operations.")
voice_app = typer.Typer(help="Voice operations.")
conversation_app = typer.Typer(help="Conversation operations.")
agent_app = typer.Typer(help="Direct specialist agent operations.")
reminder_app = typer.Typer(help="Reminder operations.")
task_app = typer.Typer(help="Task inspection operations.")
doc_app = typer.Typer(help="Document operations.")
daemon_app = typer.Typer(help="Daemon operations.")
playbook_app = typer.Typer(help="Playbook operations.")
evolve_app = typer.Typer(help="Evolution operations.")
jobs_app = typer.Typer(help="Durable background-job operations.")
