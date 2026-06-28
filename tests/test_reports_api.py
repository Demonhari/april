from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
from fastapi.testclient import TestClient

from services.api.server import create_app
from tests.test_core_api import auth, make_container


def _write(directory: Path, name: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def _acceptance(generated_at: str) -> dict[str, Any]:
    return {
        "report_type": "acceptance",
        "generated_at": generated_at,
        "final_status": "warning",
        "acceptance_level": "fake_sanity",
        "runtime_backend": "llama_cpp",
        "services": {"requested": True, "startup_status": "ok", "shutdown_status": "stopped"},
        "next_actions": ["run april acceptance --require-real-models"],
        "api_token": "tok-super-secret",
    }


def _activation(generated_at: str) -> dict[str, Any]:
    return {
        "report_type": "mac_activation",
        "generated_at": generated_at,
        "final_status": "applied",
        "next_actions": ["fix model at /Users/secret/models/brain.gguf"],
    }


def test_reports_endpoints_require_auth_and_shape(settings_tmp) -> None:
    directory = settings_tmp.home / "data" / "verification"
    _write(directory, "acc.json", _acceptance("2026-06-28T00:00:00Z"))
    _write(directory, "act.json", _activation("2026-06-27T00:00:00Z"))
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        headers = auth(settings_tmp)
        for path in ("/reports", "/reports/latest", "/reports/latest/acceptance"):
            assert client.get(path).status_code in (401, 403), path

        index = client.get("/reports", headers=headers).json()
        assert index["count"] == 2
        assert index["reports"][0]["basename"] == "acc.json"  # newest first

        latest = client.get("/reports/latest", headers=headers).json()
        assert latest["status"] == "ok"
        assert latest["report"]["report_type"] == "acceptance"

        activation = client.get("/reports/latest/mac_activation", headers=headers).json()
        assert activation["report"]["report_type"] == "mac_activation"
        assert activation["report"]["status"] == "applied"

        missing = client.get("/reports/latest/voice_live", headers=headers).json()
        assert missing["status"] == "not_found"

        unknown = client.get("/reports/latest/bogus", headers=headers)
        assert unknown.status_code == 404


def test_reports_endpoints_redact_secrets_and_paths(settings_tmp) -> None:
    directory = settings_tmp.home / "data" / "verification"
    _write(directory, "acc.json", _acceptance("2026-06-28T00:00:00Z"))
    _write(directory, "act.json", _activation("2026-06-27T00:00:00Z"))
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        headers = auth(settings_tmp)
        body = json.dumps(client.get("/reports", headers=headers).json())
        # The browser projection drops non-allowlisted secret fields entirely.
        assert "tok-super-secret" not in body
        # Absolute paths in next actions are reduced to basenames.
        assert "/Users/secret/models" not in body
        assert "brain.gguf" in body
        assert str(settings_tmp.home) not in body
