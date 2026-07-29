from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typer.testing import CliRunner

from apps.runner.main import app
from april_common.audit import AuditLogger, MemoryAuditAnchor


def _process_audit_writer(path: str, count: int) -> None:
    logger = AuditLogger(Path(path))
    for index in range(count):
        logger.write({"event_type": "process_event", "index": index})


def test_valid_genesis_multi_event_and_secret_redaction(tmp_path: Path) -> None:
    anchor = MemoryAuditAnchor()
    logger = AuditLogger(tmp_path / "audit.jsonl", anchor=anchor)
    logger.write(
        {
            "event_type": "started",
            "api_token": "secret-value",
            "prompt": "private prompt",
            "safe": "visible",
        }
    )
    logger.write({"event_type": "completed", "count": 2})
    result = logger.verify()
    assert result.status == "valid"
    assert result.record_count == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["sequence"] == 1
    assert records[1]["previous_hash"] == records[0]["record_hash"]
    text = json.dumps(records)
    assert "secret-value" not in text
    assert "private prompt" not in text
    assert records[0]["payload"]["safe"] == "visible"


def test_concurrent_thread_and_process_writers(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: logger.write({"event_type": "thread_event", "index": index}),
                range(40),
            )
        )
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_process_audit_writer, args=(str(path), 10)) for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    result = logger.verify()
    assert result.status == "valid"
    assert result.record_count == 60


def test_mutation_reorder_and_middle_deletion_are_detected(tmp_path: Path) -> None:
    for mutation, expected in (
        ("mutate", "incorrect_event_hash"),
        ("reorder", "reordered_record"),
        ("delete", "sequence_gap"),
    ):
        path = tmp_path / f"{mutation}.jsonl"
        logger = AuditLogger(path)
        for index in range(4):
            logger.write({"event_type": "event", "index": index})
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if mutation == "mutate":
            records[1]["payload"]["index"] = 99
        elif mutation == "reorder":
            records[1], records[2] = records[2], records[1]
        else:
            records.pop(1)
        path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        result = logger.verify()
        assert result.corrupt
        assert expected in {issue.code for issue in result.issues}

    genesis_path = tmp_path / "genesis.jsonl"
    genesis = AuditLogger(genesis_path)
    genesis.write({"event_type": "one"})
    genesis.write({"event_type": "two"})
    genesis_path.write_text(
        genesis_path.read_text(encoding="utf-8").splitlines()[1] + "\n",
        encoding="utf-8",
    )
    assert "missing_genesis" in {issue.code for issue in genesis.verify().issues}


def test_terminal_truncation_malformed_line_and_anchor_lag(tmp_path: Path) -> None:
    anchor = MemoryAuditAnchor()
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path, anchor=anchor)
    logger.write({"event_type": "one"})
    logger.write({"event_type": "two"})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n", encoding="utf-8")
    assert "terminal_truncation" in {issue.code for issue in logger.verify().issues}

    malformed_path = tmp_path / "malformed.jsonl"
    malformed = AuditLogger(malformed_path)
    malformed.write({"event_type": "one"})
    with malformed_path.open("ab") as handle:
        handle.write(b'{"partial":')
    assert "malformed_json" in {issue.code for issue in malformed.verify().issues}
    malformed.write({"event_type": "two"})
    assert malformed.verify().status == "valid"

    lag_anchor = MemoryAuditAnchor()
    lag_logger = AuditLogger(tmp_path / "lag.jsonl", anchor=lag_anchor)
    lag_logger.write({"event_type": "one"})
    terminal = json.loads((tmp_path / "lag.jsonl").read_text(encoding="utf-8").splitlines()[0])
    lag_anchor.value = None
    lag = lag_logger.verify()
    assert lag.status == "anchor_lagged"
    assert lag.anchor_lagged
    assert terminal["sequence"] == 1


def test_audit_verify_cli_exit_codes_and_json(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("APRIL_HOME", str(tmp_path))  # type: ignore[attr-defined]
    monkeypatch.setenv("APRIL_ENV", "test")  # type: ignore[attr-defined]
    logger = AuditLogger(tmp_path / "logs" / "audit.jsonl")
    logger.write({"event_type": "valid"})
    runner = CliRunner()
    valid = runner.invoke(app, ["april", "audit", "verify", "--json"])
    assert valid.exit_code == 0
    assert json.loads(valid.stdout)["status"] == "valid"
    path = tmp_path / "logs" / "audit.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    corrupt = runner.invoke(app, ["april", "audit", "verify", "--json"])
    assert corrupt.exit_code == 1
    assert json.loads(corrupt.stdout)["status"] == "corrupt"
