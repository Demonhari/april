from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

import yaml

from apps.runner.setup_checklist import build_setup_checklist
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.time import utc_now


def _copy_configs(home: Path) -> None:
    shutil.copytree(Path.cwd() / "configs", home / "configs")


def _create_gguf_files(home: Path) -> None:
    models = yaml.safe_load((home / "configs" / "models.yaml").read_text(encoding="utf-8"))
    for model in models["models"].values():
        if model.get("backend") != "llama_cpp":
            continue
        path = home / model["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF stub")


def _ready_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    _create_gguf_files(home)
    return home


def _write_report(home: Path, basename: str, payload: dict) -> None:
    reports = home / "data" / "verification"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / basename).write_text(json.dumps(payload), encoding="utf-8")


def test_checklist_has_expected_ordered_steps(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    checklist = build_setup_checklist(home)
    titles = [step.title for step in checklist.steps]
    assert [step.number for step in checklist.steps] == list(range(1, 12))
    assert "install dependencies" in titles[0]
    assert "setup tokens" in titles[1]
    assert "setup models" in titles[2]
    assert "config validate" in titles[3]
    assert "go-live" in titles[6]
    assert "desktop app stub" in titles[10]


def test_checklist_is_redacted(tmp_path: Path) -> None:
    home = _ready_home(tmp_path)
    blob = json.dumps(build_setup_checklist(home).model_dump())
    assert str(home) not in blob
    assert "local-dev-token" not in blob


def test_checklist_marks_done_and_next(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    checklist = build_setup_checklist(home)
    by_title = {step.title: step for step in checklist.steps}
    # Models are present and config validates in a ready home.
    assert by_title["setup models"].status == "done"
    assert by_title["config validate"].status == "done"
    # No go-live report yet → pending "next".
    assert by_title["go-live"].status == "next"
    # The headline next command points at the first actionable step.
    assert checklist.next_command is not None


def test_checklist_blocker_when_models_missing(tmp_path: Path, llama_cpp_available: None) -> None:
    # With the runtime extra present, the first hard blocker on a fresh checkout is
    # the missing model files — not install-dependencies.
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)  # no GGUF files
    checklist = build_setup_checklist(home)
    by_title = {step.title: step for step in checklist.steps}
    assert by_title["install dependencies"].status == "done"
    assert by_title["setup models"].status == "blocker"
    # A blocker is surfaced as the headline next command.
    assert checklist.next_command == by_title["setup models"].command


def test_checklist_all_done_when_fresh_reports_present(
    tmp_path: Path, llama_cpp_available: None
) -> None:
    home = _ready_home(tmp_path)
    fingerprint = config_fingerprint_digest(home)
    generated = (utc_now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Harden tokens + embeddings so those steps read done.
    config = home / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data.setdefault("api", {})["token"] = "a-real-rotated-api-token"
    data.setdefault("runtime", {})["token"] = "a-real-rotated-runtime-token"
    data.setdefault("memory", {})["embedding_provider"] = "runtime-local"
    data["memory"]["embedding_model_id"] = "april-embedding"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    # Register an embedding model so runtime-local validates structurally.
    models_path = home / "configs" / "models.yaml"
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    embed = home / "models" / "embed.gguf"
    embed.write_bytes(b"GGUF stub")
    models["models"]["embedding"] = {
        "id": "april-embedding",
        "name": "embed",
        "path": "models/embed.gguf",
        "backend": "llama_cpp",
        "role": "embedding",
        "threads": 1,
        "context_size": 512,
        "temperature": 0.0,
        "max_output_tokens": 1,
        "keep_loaded": False,
    }
    models_path.write_text(yaml.safe_dump(models), encoding="utf-8")
    fingerprint = config_fingerprint_digest(home)
    base = {"generated_at": generated, "config_fingerprint": fingerprint}
    _write_report(
        home,
        "mac-readiness.json",
        {
            **base,
            "report_type": "multi_model",
            "summary": "pass",
            "real_model_verified": True,
            "verification_level": "all",
            "models": [],
        },
    )
    _write_report(
        home,
        "workflow-real.json",
        {**base, "report_type": "workflow", "summary": "pass", "real_model_verified": True},
    )
    _write_report(
        home,
        "go-live.json",
        {**base, "report_type": "go_live", "final_status": "pass", "hardened_go_live_ready": True},
    )
    checklist = build_setup_checklist(home)
    by_title = {step.title: step for step in checklist.steps}
    assert by_title["verify all configured models"].status == "done"
    assert by_title["verify workflow real"].status == "done"
    assert by_title["go-live"].status == "done"
