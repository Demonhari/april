from __future__ import annotations

import subprocess

import pytest

from april_common.errors import PermissionDeniedError
from services.permissions.engine import PermissionEngine
from skills.apps.open_app import open_app
from skills.apps.open_url import open_url
from skills.registry import default_registry


def engine() -> PermissionEngine:
    return PermissionEngine(default_registry())


def test_level_1_executes_without_approval(settings_tmp) -> None:
    decision = engine().evaluate(
        tool="read_file", args={"path": str(settings_tmp.home / "README.md")}, agent="coding_agent"
    )
    assert decision.permission_level == 1
    assert decision.confirmation_required is False


def test_level_2_executes_under_policy(settings_tmp) -> None:
    decision = engine().evaluate(tool="create_note", args={"title": "x"}, agent="creative_agent")
    assert decision.permission_level == 2
    assert decision.confirmation_required is False


def test_level_3_blocked_pending_approval(settings_tmp) -> None:
    decision = engine().evaluate(
        tool="write_file", args={"path": str(settings_tmp.home / "x.py")}, agent="coding_agent"
    )
    assert decision.permission_level == 3
    assert decision.confirmation_required is True


def test_level_4_blocked_pending_approval() -> None:
    decision = engine().evaluate(
        tool="open_app", args={"name": "Safari"}, agent="system_action_agent"
    )
    assert decision.permission_level == 4
    assert decision.confirmation_required is True


def test_unknown_tool_denied() -> None:
    with pytest.raises(PermissionDeniedError):
        engine().evaluate(tool="unknown", args={}, agent="coding_agent")


def test_blocked_tool_denied_for_agent(settings_tmp) -> None:
    with pytest.raises(PermissionDeniedError):
        engine().evaluate(
            tool="write_file", args={"path": str(settings_tmp.home / "x")}, agent="reading_agent"
        )


def test_model_cannot_lower_permission_level(settings_tmp) -> None:
    decision = engine().evaluate(
        tool="write_file",
        args={"path": str(settings_tmp.home / "x")},
        agent="coding_agent",
        model_permission_level=0,
        model_risk_level="none",
    )
    assert decision.permission_level == 3


@pytest.mark.asyncio
async def test_open_app_requires_configured_allowlist(settings_tmp) -> None:
    result = await open_app({"name": "Safari"})
    assert result.ok is False
    assert "allowlist" in result.stderr


@pytest.mark.asyncio
async def test_open_app_uses_macos_open_with_argv(settings_tmp, tmp_path, monkeypatch) -> None:
    configs = settings_tmp.home / "configs"
    configs.mkdir()
    (configs / "tools.yaml").write_text(
        "tools:\n"
        "  command_allowlist: []\n"
        "  open_app_allowlist: [TextEdit]\n"
        "  open_url_allowed_schemes: [https]\n",
        encoding="utf-8",
    )
    open_binary = tmp_path / "open"
    open_binary.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr("skills.apps.open_app.sys.platform", "darwin")
    monkeypatch.setattr("skills.apps.open_app.OPEN_BINARY", open_binary)
    monkeypatch.setattr("skills.apps.open_app.subprocess.run", fake_run)
    result = await open_app({"name": "TextEdit"})
    assert result.ok is True
    assert calls == [[str(open_binary), "-a", "TextEdit"]]


@pytest.mark.asyncio
async def test_open_url_rejects_non_http_schemes(settings_tmp) -> None:
    result = await open_url({"url": "file:///tmp/secret"})
    assert result.ok is False
    assert "http or https" in result.stderr


@pytest.mark.asyncio
async def test_open_url_normalizes_and_uses_argv(settings_tmp, tmp_path, monkeypatch) -> None:
    configs = settings_tmp.home / "configs"
    configs.mkdir()
    (configs / "tools.yaml").write_text(
        "tools:\n"
        "  command_allowlist: []\n"
        "  open_app_allowlist: []\n"
        "  open_url_allowed_schemes: [https]\n",
        encoding="utf-8",
    )
    open_binary = tmp_path / "open"
    open_binary.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr("skills.apps.open_url.sys.platform", "darwin")
    monkeypatch.setattr("skills.apps.open_url.OPEN_BINARY", open_binary)
    monkeypatch.setattr("skills.apps.open_url.subprocess.run", fake_run)
    result = await open_url({"url": "HTTPS://Example.COM?q=1"})
    assert result.ok is True
    assert result.data == {"url": "https://example.com/?q=1"}
    assert calls == [[str(open_binary), "https://example.com/?q=1"]]
