from __future__ import annotations

import os
import platform
import stat
from pathlib import Path

import pytest

from april_common.credentials import (
    CommandResult,
    CredentialKey,
    CredentialStoreError,
    FileCredentialStore,
    InMemoryCredentialStore,
    MacOSKeychainCredentialStore,
    select_credential_store,
)
from april_common.errors import ConfigError
from april_common.process_environment import ProcessCategory, build_process_environment
from april_common.token_setup import (
    migrate_legacy_credentials,
    rotate_credentials,
)


def test_keychain_commands_are_argv_only_minimal_and_redacted(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, str], float]] = []
    values: dict[str, str] = {}

    def runner(argv: object, environment: object, timeout: float) -> CommandResult:
        args = tuple(argv)  # type: ignore[arg-type]
        env = dict(environment)  # type: ignore[arg-type]
        calls.append((args, env, timeout))
        if args[1] == "add-generic-password":
            values[args[args.index("-a") + 1]] = args[-1]
            return CommandResult(0, "", "")
        if args[1] == "find-generic-password":
            account = args[args.index("-a") + 1]
            if account not in values:
                return CommandResult(44, "", "not found")
            return CommandResult(0, values[account] + "\n", "")
        return CommandResult(0, "", "")

    store = MacOSKeychainCredentialStore(
        runner=runner,
        executable=Path("/bin/true"),
        environment={"HOME": "/safe-home", "LANG": "C"},
    )
    secret = "never-print-this-token"
    store.set(CredentialKey.API_TOKEN, secret)
    assert store.get(CredentialKey.API_TOKEN) == secret
    assert calls[0][0][0] == "/bin/true"
    assert calls[0][0][1] == "add-generic-password"
    assert calls[0][1] == {"HOME": "/safe-home", "LANG": "C"}
    assert all("shell" not in item for call in calls for item in call[0])

    def failed(_argv: object, _environment: object, _timeout: float) -> CommandResult:
        return CommandResult(1, secret, secret)

    failing = MacOSKeychainCredentialStore(
        runner=failed,
        executable=Path("/bin/true"),
    )
    with pytest.raises(CredentialStoreError) as error:
        failing.set(CredentialKey.API_TOKEN, secret)
    assert secret not in str(error.value)


def test_legacy_migration_success_and_idempotency(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APRIL_API_TOKEN=legacy-api\nAPRIL_RUNTIME_TOKEN=legacy-runtime\nOTHER=value\n",
        encoding="utf-8",
    )
    store = InMemoryCredentialStore()
    anchor = tmp_path / "logs" / "audit.jsonl.anchor"
    anchor.parent.mkdir()
    anchor.write_text('{"sequence":1,"record_hash":"safe"}\n', encoding="utf-8")
    os.chmod(anchor, 0o600)
    result = migrate_legacy_credentials(
        home=tmp_path,
        store=store,
        legacy_audit_anchor_file=anchor,
    )
    assert result.status == "migrated"
    assert store.get(CredentialKey.API_TOKEN) == "legacy-api"
    assert store.get(CredentialKey.RUNTIME_TOKEN) == "legacy-runtime"
    assert store.get(CredentialKey.AUDIT_ANCHOR) is not None
    assert not anchor.exists()
    sanitized = env_file.read_text(encoding="utf-8")
    assert "legacy-api" not in sanitized
    assert "legacy-runtime" not in sanitized
    assert "OTHER=value" in sanitized
    assert migrate_legacy_credentials(home=tmp_path, store=store).status == "already_migrated"


def test_failed_migration_leaves_legacy_file_unchanged(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    original = "APRIL_API_TOKEN=legacy-api\nAPRIL_RUNTIME_TOKEN=legacy-runtime\n"
    env_file.write_text(original, encoding="utf-8")

    class FailingStore(InMemoryCredentialStore):
        def set(self, key: CredentialKey, value: str) -> None:
            if key is CredentialKey.RUNTIME_TOKEN:
                raise CredentialStoreError("injected")
            super().set(key, value)

    with pytest.raises(CredentialStoreError):
        migrate_legacy_credentials(home=tmp_path, store=FailingStore())
    assert env_file.read_text(encoding="utf-8") == original


def test_rotation_rolls_back_partial_backend_failure() -> None:
    class FailingStore(InMemoryCredentialStore):
        def rotate(self, key: CredentialKey, value: str) -> str | None:
            if key is CredentialKey.RUNTIME_TOKEN:
                raise CredentialStoreError("injected")
            return super().rotate(key, value)

    store = FailingStore(
        {
            CredentialKey.API_TOKEN: "old-api",
            CredentialKey.RUNTIME_TOKEN: "old-runtime",
        }
    )
    with pytest.raises(CredentialStoreError):
        rotate_credentials(
            store=store,
            rotate_api=True,
            rotate_runtime=True,
        )
    assert store.get(CredentialKey.API_TOKEN) == "old-api"
    assert store.get(CredentialKey.RUNTIME_TOKEN) == "old-runtime"


def test_file_store_rejects_repository_and_insecure_permissions(tmp_path: Path) -> None:
    with pytest.raises(CredentialStoreError):
        FileCredentialStore(tmp_path / "credentials.json", repository_root=tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-credentials.json"
    external.write_text(
        '{"format_version":1,"credentials":{"core-api-token":"secret"}}\n',
        encoding="utf-8",
    )
    os.chmod(external, 0o644)
    try:
        store = FileCredentialStore(external, repository_root=tmp_path)
        with pytest.raises(CredentialStoreError):
            store.get(CredentialKey.API_TOKEN)
    finally:
        external.unlink(missing_ok=True)


def test_file_store_writes_owner_only(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-secure.json"
    try:
        store = FileCredentialStore(external, repository_root=tmp_path)
        store.set(CredentialKey.API_TOKEN, "secret")
        assert stat.S_IMODE(external.stat().st_mode) == 0o600
    finally:
        external.unlink(missing_ok=True)


def test_production_never_selects_file_or_memory_fallback(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        select_credential_store(
            backend="file",
            environment="production",
            repository_root=tmp_path,
            file_path=tmp_path.parent / "credentials.json",
        )
    with pytest.raises(ConfigError):
        select_credential_store(
            backend="memory",
            environment="production",
            repository_root=tmp_path,
        )


def test_macos_production_fails_closed_when_keychain_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    def unavailable(
        _argv: object,
        _environment: object,
        _timeout: float,
    ) -> CommandResult:
        return CommandResult(1, "secret output", "secret error")

    with pytest.raises(CredentialStoreError) as error:
        select_credential_store(
            backend="auto",
            environment="production",
            repository_root=tmp_path,
            keychain_runner=unavailable,
        )
    assert "secret output" not in str(error.value)
    assert "secret error" not in str(error.value)


def test_runtime_credential_is_absent_from_frontend_sources() -> None:
    web = Path.cwd() / "apps" / "desktop" / "web"
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in web.iterdir())
    assert "APRIL_RUNTIME_TOKEN" not in rendered
    assert "runtime-auth-token" not in rendered


def test_child_service_and_tool_environments_do_not_inherit_raw_tokens() -> None:
    source = {
        "PATH": "/usr/bin",
        "APRIL_API_TOKEN": "api-secret",
        "APRIL_RUNTIME_TOKEN": "runtime-secret",
        "APRIL_CREDENTIAL_STORE": "keychain",
        "AWS_SECRET_ACCESS_KEY": "cloud-secret",
        "SSH_AUTH_SOCK": "/private/socket",
        "HTTPS_PROXY": "http://proxy",
    }
    for category in (
        ProcessCategory.CORE_API,
        ProcessCategory.RUNTIME,
        ProcessCategory.JOB_WORKER,
        ProcessCategory.TEST_RUNNER,
        ProcessCategory.GIT,
    ):
        environment = build_process_environment(category, source=source)
        assert "APRIL_API_TOKEN" not in environment
        assert "APRIL_RUNTIME_TOKEN" not in environment
        assert "AWS_SECRET_ACCESS_KEY" not in environment
        assert "SSH_AUTH_SOCK" not in environment
    assert (
        build_process_environment(ProcessCategory.RUNTIME, source=source)["APRIL_CREDENTIAL_STORE"]
        == "keychain"
    )
