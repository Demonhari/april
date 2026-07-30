from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from apps.cli.groups import evolve_app, playbook_app
from apps.cli.render import console, print_jsonish


def client() -> Any:
    from apps.cli import main as cli_main

    return cli_main.client()


def run(coro: Any) -> Any:
    from apps.cli import main as cli_main

    return cli_main.run(coro)


@playbook_app.command("list")
def playbook_list() -> None:
    print_jsonish(run(client().get("/playbooks")))


@playbook_app.command("run")
def playbook_run(
    playbook_id: str,
    project_id: str | None = typer.Option(None, "--project-id"),
    conversation_id: str | None = typer.Option(None, "--conversation-id"),
) -> None:
    payload = {"project_id": project_id, "conversation_id": conversation_id}
    print_jsonish(run(client().post(f"/playbooks/{playbook_id}/run", payload)))


@playbook_app.command("mine")
def playbook_mine(
    support_threshold: int = typer.Option(3, "--support", min=2),
    lookback_days: int = typer.Option(14, "--lookback-days", min=1),
) -> None:
    path = f"/playbooks/mine?support_threshold={support_threshold}&lookback_days={lookback_days}"
    print_jsonish(run(client().post(path, {})))


@playbook_app.command("adopt")
def playbook_adopt(path: Path) -> None:
    import json

    import yaml

    from skills.playbooks import PlaybookDefinition

    resolved = path.expanduser().resolve()
    if resolved.suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    playbook = PlaybookDefinition.model_validate(payload)
    print_jsonish(run(client().post("/playbooks/adopt", playbook.model_dump())))


@evolve_app.command("versions")
def evolve_versions(agent: str | None = typer.Option(None, "--agent")) -> None:
    params = {"agent": agent} if agent else None
    print_jsonish(run(client().get("/evolution/versions", params=params)))


@evolve_app.command("rollback")
def evolve_rollback(agent: str, version: int) -> None:
    print_jsonish(run(client().post("/evolution/rollback", {"agent": agent, "version": version})))


@evolve_app.command("report")
def evolve_report() -> None:
    print_jsonish(run(client().get("/evolution/report/latest")))


@evolve_app.command("status")
def evolve_status() -> None:
    """Show evolution enablement, kill switch, last run, and overlay counts."""
    print_jsonish(run(client().get("/evolution/status")))


@evolve_app.command("history")
def evolve_history(limit: int = typer.Option(20, "--limit", min=1, max=200)) -> None:
    """List past Dreamer runs, newest first."""
    print_jsonish(run(client().get("/evolution/history", params={"limit": limit})))


@evolve_app.command("diff")
def evolve_diff(
    agent: str,
    from_version: int | None = typer.Option(None, "--from", min=1),
    to_version: int | None = typer.Option(None, "--to", min=1),
) -> None:
    """Unified diff between two prompt-overlay versions of one agent."""
    params: dict[str, Any] = {"agent": agent}
    if from_version is not None:
        params["from_version"] = from_version
    if to_version is not None:
        params["to_version"] = to_version
    data = run(client().get("/evolution/diff", params=params))
    if data.get("diff"):
        console.print(data["diff"])
    else:
        print_jsonish(data)


@evolve_app.command("off")
def evolve_off() -> None:
    """Set the local kill switch: the Dreamer never runs while it is active."""
    data = run(client().post("/evolution/off", {}))
    print_jsonish(data)
    console.print("Evolution is now hard-disabled. Re-enable with: april evolve on")


@evolve_app.command("on")
def evolve_on() -> None:
    """Clear the local kill switch (config evolution.enabled still applies)."""
    print_jsonish(run(client().post("/evolution/on", {})))


@evolve_app.command("pending")
def evolve_pending() -> None:
    """List write-capable agent overlays awaiting explicit approval."""
    print_jsonish(run(client().get("/evolution/overlays/pending")))


@evolve_app.command("approve")
def evolve_approve(agent: str, content_hash: str) -> None:
    """Approve one pending overlay by agent and exact SHA-256 content hash."""
    data = run(
        client().post(
            "/evolution/overlays/approve",
            {"agent": agent, "content_hash": content_hash},
        )
    )
    print_jsonish(data)


evolve_evals_app = typer.Typer(help="Review staged feedback eval cases.")
evolve_app.add_typer(evolve_evals_app, name="evals")


@evolve_evals_app.command("pending")
def evolve_evals_pending() -> None:
    """List staged eval cases awaiting human review."""
    print_jsonish(run(client().get("/evolution/evals/pending")))


@evolve_evals_app.command("show")
def evolve_evals_show(case_id: str) -> None:
    """Show one pending eval case in full for local review."""
    print_jsonish(run(client().get(f"/evolution/evals/pending/{case_id}")))


@evolve_evals_app.command("promote")
def evolve_evals_promote(
    case_id: str,
    expected: str = typer.Option(
        ...,
        "--expected",
        help="Human-reviewed expected behaviour for this case (required).",
    ),
) -> None:
    """Promote a pending case into an active reviewed eval case."""
    data = run(
        client().post(
            "/evolution/evals/promote",
            {"case_id": case_id, "expected_behavior": expected},
        )
    )
    print_jsonish(data)


@evolve_evals_app.command("reject")
def evolve_evals_reject(
    case_id: str,
    reason: str = typer.Option(
        ...,
        "--reason",
        help="Why this case should not become an eval (required).",
    ),
) -> None:
    """Reject a pending eval case with a human-supplied reason."""
    data = run(
        client().post(
            "/evolution/evals/reject",
            {"case_id": case_id, "reason": reason},
        )
    )
    print_jsonish(data)


evolve_dataset_app = typer.Typer(help="Fine-tuning dataset operations (export only).")
evolve_app.add_typer(evolve_dataset_app, name="dataset")

evolve_adapter_app = typer.Typer(help="LoRA adapter lifecycle operations.")
evolve_app.add_typer(evolve_adapter_app, name="adapter")


@evolve_dataset_app.command("export")
def evolve_dataset_export(
    name: str | None = typer.Option(None, "--name", help="Dataset basename."),
) -> None:
    """Export the reviewable JSONL fine-tune dataset under data/evolution/datasets."""
    data = run(client().post("/evolution/dataset/export", {"name": name}))
    print_jsonish(data)


@evolve_adapter_app.command("list")
def evolve_adapter_list(
    model_id: str | None = typer.Option(None, "--model-id", help="Limit to one model id."),
) -> None:
    """List versioned LoRA adapter pointers and DB history."""
    params = {"model_id": model_id} if model_id else None
    print_jsonish(run(client().get("/evolution/adapters", params=params)))


@evolve_adapter_app.command("activate")
def evolve_adapter_activate(
    model_id: str,
    adapter_path: Path,
    evidence_path: Path | None = typer.Option(
        None,
        "--evidence",
        help="Perplexity evidence JSON from scripts/finetune.",
    ),
    verification_report_path: Path | None = typer.Option(
        None,
        "--verification-report",
        help="Fresh real-model verification report required in production.",
    ),
) -> None:
    """Activate a LoRA adapter after deterministic evidence gates pass."""
    payload = {
        "model_id": model_id,
        "adapter_path": str(adapter_path),
        "evidence_path": str(evidence_path) if evidence_path else None,
        "verification_report_path": (
            str(verification_report_path) if verification_report_path else None
        ),
    }
    print_jsonish(run(client().post("/evolution/adapters/activate", payload)))


@evolve_adapter_app.command("rollback")
def evolve_adapter_rollback(
    model_id: str,
    version: int | None = typer.Option(
        None,
        "--version",
        min=1,
        help="Target version; defaults to the previous active version.",
    ),
) -> None:
    """Flip the active adapter pointer back to a previous version."""
    print_jsonish(
        run(
            client().post(
                "/evolution/adapters/rollback",
                {"model_id": model_id, "version": version},
            )
        )
    )
