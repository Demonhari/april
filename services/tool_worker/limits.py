from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


class UnsafeToolWorkerSocket(RuntimeError):
    pass


def default_tool_worker_runtime_directory(april_home: Path) -> Path:
    home = april_home.expanduser().resolve(strict=True)
    preferred = home / "data" / "runtime" / "tool-worker"
    if len(os.fsencode(preferred / "worker.sock")) <= 96:
        return preferred
    digest = hashlib.sha256(str(home).encode("utf-8")).hexdigest()[:16]
    return Path("/tmp").resolve() / f"april-tool-worker-{os.getuid()}-{digest}"


def prepare_runtime_directory(path: Path, *, april_home: Path) -> Path:
    home = april_home.expanduser().resolve(strict=True)
    requested = path.expanduser()
    if requested.is_symlink():
        raise UnsafeToolWorkerSocket("runtime_directory_is_symlink")
    resolved = requested.resolve(strict=False)
    expected_temporary = default_tool_worker_runtime_directory(home)
    if not _is_relative_to(resolved, home) and resolved != expected_temporary:
        raise UnsafeToolWorkerSocket("runtime_directory_outside_april_home")
    resolved.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = resolved.stat()
    if info.st_uid != os.getuid():
        raise UnsafeToolWorkerSocket("runtime_directory_wrong_owner")
    if stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(resolved, 0o700)
    return resolved


def prepare_socket_path(path: Path, *, runtime_directory: Path) -> Path:
    runtime = runtime_directory.resolve(strict=True)
    if path.parent.resolve(strict=True) != runtime:
        raise UnsafeToolWorkerSocket("socket_outside_runtime_directory")
    if path.is_symlink():
        raise UnsafeToolWorkerSocket("socket_path_is_symlink")
    if path.exists():
        info = path.lstat()
        if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):
            raise UnsafeToolWorkerSocket("unsafe_existing_socket")
        path.unlink()
    return path


def validate_live_socket(path: Path, *, runtime_directory: Path) -> str:
    if path.is_symlink():
        raise UnsafeToolWorkerSocket("socket_path_is_symlink")
    if path.parent.resolve(strict=True) != runtime_directory.resolve(strict=True):
        raise UnsafeToolWorkerSocket("socket_outside_runtime_directory")
    info = path.stat()
    if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):
        raise UnsafeToolWorkerSocket("unsafe_socket")
    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o600:
        raise UnsafeToolWorkerSocket("unsafe_socket_mode")
    return f"{mode:04o}"


def write_capability_file(path: Path, capability: str, *, runtime_directory: Path) -> None:
    if path.parent.resolve(strict=True) != runtime_directory.resolve(strict=True):
        raise UnsafeToolWorkerSocket("capability_outside_runtime_directory")
    if path.is_symlink():
        raise UnsafeToolWorkerSocket("capability_path_is_symlink")
    if path.exists():
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise UnsafeToolWorkerSocket("unsafe_capability_file")
        path.unlink()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, capability.encode("ascii"))
    finally:
        os.close(descriptor)


def read_capability_file(path: Path, *, runtime_directory: Path) -> str:
    if path.is_symlink() or path.parent.resolve(strict=True) != runtime_directory.resolve(
        strict=True
    ):
        raise UnsafeToolWorkerSocket("unsafe_capability_path")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise UnsafeToolWorkerSocket("unsafe_capability_file")
    value = path.read_text(encoding="ascii")
    if not 32 <= len(value) <= 256:
        raise UnsafeToolWorkerSocket("invalid_capability")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
