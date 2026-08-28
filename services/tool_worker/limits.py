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
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeToolWorkerSocket("runtime_directory_not_directory")
    if info.st_uid != os.getuid():
        raise UnsafeToolWorkerSocket("runtime_directory_wrong_owner")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise UnsafeToolWorkerSocket("unsafe_runtime_directory_mode")
    return resolved


def prepare_socket_path(path: Path, *, runtime_directory: Path) -> Path:
    runtime = runtime_directory.resolve(strict=True)
    if path.parent.resolve(strict=True) != runtime:
        raise UnsafeToolWorkerSocket("socket_outside_runtime_directory")
    if os.path.lexists(path):
        if path.is_symlink():
            raise UnsafeToolWorkerSocket("socket_path_is_symlink")
        info = path.lstat()
        if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):
            raise UnsafeToolWorkerSocket("unsafe_existing_socket")
        raise UnsafeToolWorkerSocket("socket_already_exists")
    if path.is_symlink():
        raise UnsafeToolWorkerSocket("socket_path_is_symlink")
    return path


def validate_live_socket(path: Path, *, runtime_directory: Path) -> str:
    if path.is_symlink():
        raise UnsafeToolWorkerSocket("socket_path_is_symlink")
    if not os.path.lexists(path):
        raise FileNotFoundError(path)
    if path.parent.resolve(strict=True) != runtime_directory.resolve(strict=True):
        raise UnsafeToolWorkerSocket("socket_outside_runtime_directory")
    info = path.lstat()
    if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):
        raise UnsafeToolWorkerSocket("unsafe_socket")
    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o600:
        raise UnsafeToolWorkerSocket("unsafe_socket_mode")
    return f"{mode:04o}"


def write_capability_file(path: Path, capability: str, *, runtime_directory: Path) -> None:
    if path.parent.resolve(strict=True) != runtime_directory.resolve(strict=True):
        raise UnsafeToolWorkerSocket("capability_outside_runtime_directory")
    if os.path.lexists(path) and path.is_symlink():
        raise UnsafeToolWorkerSocket("capability_path_is_symlink")
    if path.exists():
        info = path.stat()
        if (
            info.st_uid != os.getuid()
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
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
    if (
        info.st_uid != os.getuid()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise UnsafeToolWorkerSocket("unsafe_capability_file")
    value = path.read_text(encoding="ascii")
    if not 32 <= len(value) <= 256:
        raise UnsafeToolWorkerSocket("invalid_capability")
    return value


def socket_identity(path: Path, *, runtime_directory: Path) -> tuple[int, int]:
    """Return the validated device/inode identity of an owner-only socket."""
    validate_live_socket(path, runtime_directory=runtime_directory)
    info = path.lstat()
    return info.st_dev, info.st_ino


def remove_owned_socket(
    path: Path,
    *,
    runtime_directory: Path,
    identity: tuple[int, int],
) -> bool:
    """Unlink only the exact validated socket created by this manager/server."""
    if not os.path.lexists(path) or path.is_symlink():
        return False
    current = socket_identity(path, runtime_directory=runtime_directory)
    if current != identity:
        return False
    path.unlink()
    return True


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
