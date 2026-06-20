from __future__ import annotations

from pathlib import Path

import pytest

from april_common.errors import PermissionDeniedError
from skills.filesystem.read_file import read_file


@pytest.mark.asyncio
async def test_normal_in_root_read(settings_tmp) -> None:
    result = await read_file({"path": str(settings_tmp.home / "README.md")})
    assert result.ok
    assert "animation bug" in result.stdout


@pytest.mark.asyncio
async def test_traversal_rejected(settings_tmp, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    with pytest.raises(PermissionDeniedError):
        await read_file({"path": str(settings_tmp.home / ".." / outside.name)})


@pytest.mark.asyncio
async def test_symlink_escape_rejected(settings_tmp, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = settings_tmp.home / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(PermissionDeniedError):
        await read_file({"path": str(link)})


@pytest.mark.asyncio
async def test_sensitive_path_rejected(settings_tmp) -> None:
    ssh = settings_tmp.home / ".ssh"
    ssh.mkdir()
    secret = ssh / "id_rsa"
    secret.write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionDeniedError):
        await read_file({"path": str(secret)})


@pytest.mark.asyncio
async def test_oversized_read_rejected(settings_tmp) -> None:
    big = settings_tmp.home / "big.txt"
    big.write_text("x" * (settings_tmp.paths.max_file_read_bytes + 1), encoding="utf-8")
    with pytest.raises(PermissionDeniedError):
        await read_file({"path": str(big)})


@pytest.mark.asyncio
async def test_binary_read_rejected(settings_tmp) -> None:
    binary = settings_tmp.home / "bin.dat"
    binary.write_bytes(b"\x00\x01")
    with pytest.raises(PermissionDeniedError):
        await read_file({"path": str(binary)})
