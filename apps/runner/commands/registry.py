"""Typed Typer command groups shared by the CLI composition root."""

import typer

app = typer.Typer(help="Global command dispatcher.")
april_app = typer.Typer(help="Run APRIL from any folder.", invoke_without_command=True)
model_app = typer.Typer(help="Model operations.")
profile_app = typer.Typer(help="Model profile operations.")
project_app = typer.Typer(help="Project operations.")
memory_app = typer.Typer(help="Memory operations.")
conversation_app = typer.Typer(help="Conversation operations.")
config_app = typer.Typer(help="Configuration operations.")
agent_app = typer.Typer(help="Direct specialist agent operations.")
voice_app = typer.Typer(help="Voice operations.")
reminder_app = typer.Typer(help="Reminder operations.")
task_app = typer.Typer(help="Task inspection operations.")
eval_app = typer.Typer(help="Local evaluation operations.")
setup_app = typer.Typer(help="Local setup utilities.")
user_profile_app = typer.Typer(help="Local user-profile operations.")
reports_app = typer.Typer(help="Browse local verification reports.")
jobs_app = typer.Typer(help="Durable background-job operations.")
