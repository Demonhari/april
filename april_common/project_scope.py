from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from april_common.errors import PermissionDeniedError
from april_common.path_security import MODEL_SUFFIXES, deny_sensitive_path

GIT_TIMEOUT_SECONDS = 15.0
MAX_GIT_CAPTURE_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class PatchArtifact:
    patch_sha256: str
    patch_byte_length: int
    affected_paths: list[str]
    repo_root: str
    repo_head: str | None
    repo_state_digest: str | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_project_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PermissionDeniedError("Project root must be an existing directory.")
    deny_sensitive_path(root)
    return root


def normalize_project_child(
    path: str | Path,
    *,
    project_root: str | Path,
    must_exist: bool,
    allow_absolute: bool = False,
) -> Path:
    root = normalize_project_root(project_root)
    requested = Path(path).expanduser()
    if requested.is_absolute():
        if not allow_absolute:
            raise PermissionDeniedError("Absolute paths are not accepted for this project tool.")
        target = requested
    else:
        if ".." in requested.parts:
            raise PermissionDeniedError("Path traversal outside the selected project is denied.")
        target = root / requested
    resolved = target.resolve(strict=True) if must_exist else _resolve_new_project_path(target)
    _ensure_under_root(resolved, root)
    deny_sensitive_path(resolved)
    return resolved


def validate_patch_text(patch: str, project_root: str | Path) -> list[str]:
    if not patch.strip():
        raise PermissionDeniedError("Patch proposal is empty.")
    root = normalize_project_root(project_root)
    affected: set[str] = set()
    for raw_path in _extract_patch_paths(patch):
        relative = _patch_relative_path(raw_path)
        if relative is None:
            continue
        _validate_relative_project_path(relative, root)
        affected.add(relative.as_posix())
    if not affected:
        raise PermissionDeniedError("Patch does not declare any affected project files.")
    return sorted(affected)


async def inspect_patch_file(*, patch_path: str | Path, repo_root: str | Path) -> PatchArtifact:
    root = normalize_project_root(repo_root)
    patch = normalize_project_child(
        patch_path,
        project_root=root,
        must_exist=True,
        allow_absolute=True,
    )
    text = patch.read_text(encoding="utf-8", errors="replace")
    affected_paths = validate_patch_text(text, root)
    return PatchArtifact(
        patch_sha256=sha256_file(patch),
        patch_byte_length=patch.stat().st_size,
        affected_paths=affected_paths,
        repo_root=str(root),
        repo_head=await git_head(root),
        repo_state_digest=await git_worktree_digest(root),
    )


async def inspect_patch_bytes(*, patch_bytes: bytes, repo_root: str | Path) -> PatchArtifact:
    root = normalize_project_root(repo_root)
    text = patch_bytes.decode("utf-8", errors="replace")
    affected_paths = validate_patch_text(text, root)
    return PatchArtifact(
        patch_sha256=sha256_bytes(patch_bytes),
        patch_byte_length=len(patch_bytes),
        affected_paths=affected_paths,
        repo_root=str(root),
        repo_head=await git_head(root),
        repo_state_digest=await git_worktree_digest(root),
    )


async def git_head(repo_root: str | Path) -> str | None:
    root = normalize_project_root(repo_root)
    if not (root / ".git").exists():
        return None
    code, stdout, _stderr = await _run_git(root, ["rev-parse", "HEAD"])
    if code != 0:
        return None
    return stdout.strip() or None


async def git_worktree_digest(repo_root: str | Path) -> str | None:
    root = normalize_project_root(repo_root)
    if not (root / ".git").exists():
        return None
    chunks: list[bytes] = []
    for args in (
        ["status", "--porcelain=v1", "-z", "--untracked-files=no"],
        ["diff", "--binary"],
        ["diff", "--cached", "--binary"],
    ):
        code, stdout, stderr = await _run_git_bytes(root, args)
        if code != 0:
            raise PermissionDeniedError(
                "Unable to calculate Git repository state.",
                {"stderr": stderr},
            )
        chunks.append(b"\0".join([b" ".join(arg.encode() for arg in args), stdout]))
    return sha256_bytes(b"\0".join(chunks))


async def git_staged_digest(repo_root: str | Path) -> str:
    root = normalize_project_root(repo_root)
    if not (root / ".git").exists():
        raise PermissionDeniedError("Git commit approval requires a Git repository.")
    code, stdout, stderr = await _run_git_bytes(root, ["diff", "--cached", "--binary"])
    if code != 0:
        raise PermissionDeniedError("Unable to calculate staged Git digest.", {"stderr": stderr})
    return sha256_bytes(stdout)


async def git_staged_tree_id(repo_root: str | Path) -> str:
    root = normalize_project_root(repo_root)
    if not (root / ".git").exists():
        raise PermissionDeniedError("Git commit approval requires a Git repository.")
    code, stdout, stderr = await _run_git(root, ["write-tree"])
    if code != 0:
        raise PermissionDeniedError("Unable to calculate staged Git tree.", {"stderr": stderr})
    return stdout.strip()


async def git_apply_check(repo_root: str | Path, patch_path: str | Path) -> tuple[bool, str, str]:
    root = normalize_project_root(repo_root)
    patch = normalize_project_child(
        patch_path,
        project_root=root,
        must_exist=True,
        allow_absolute=True,
    )
    code, stdout, stderr = await _run_git(root, ["apply", "--check", "--", str(patch)])
    return code == 0, stdout, stderr


async def git_apply_check_bytes(repo_root: str | Path, patch_bytes: bytes) -> tuple[bool, str, str]:
    root = normalize_project_root(repo_root)
    code, stdout, stderr = await _run_git_bytes_with_input(
        root,
        ["apply", "--check", "-"],
        patch_bytes,
    )
    return (
        code == 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def git_apply_bytes(repo_root: str | Path, patch_bytes: bytes) -> tuple[int, str, str]:
    root = normalize_project_root(repo_root)
    code, stdout, stderr = await _run_git_bytes_with_input(root, ["apply", "-"], patch_bytes)
    return (
        code,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def _run_git(repo_root: Path, args: list[str]) -> tuple[int, str, str]:
    code, stdout, stderr = await _run_git_bytes(repo_root, args)
    return (
        code,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def _run_git_bytes(repo_root: Path, args: list[str]) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repo_root),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, b"", b"Git command timed out."
    return (
        process.returncode or 0,
        stdout[:MAX_GIT_CAPTURE_BYTES],
        stderr[:MAX_GIT_CAPTURE_BYTES],
    )


async def _run_git_bytes_with_input(
    repo_root: Path,
    args: list[str],
    stdin: bytes,
) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repo_root),
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin),
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, b"", b"Git command timed out."
    return (
        process.returncode or 0,
        stdout[:MAX_GIT_CAPTURE_BYTES],
        stderr[:MAX_GIT_CAPTURE_BYTES],
    )


def _extract_patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.extend([parts[2], parts[3]])
        elif line.startswith("--- ") or line.startswith("+++ "):
            paths.append(line[4:].strip().split("\t", maxsplit=1)[0])
    return paths


def _patch_relative_path(raw_path: str) -> Path | None:
    cleaned = raw_path.strip().strip('"')
    if cleaned == "/dev/null":
        return None
    if cleaned.startswith("a/") or cleaned.startswith("b/"):
        cleaned = cleaned[2:]
    relative = Path(cleaned)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionDeniedError("Patch targets a path outside the selected project.")
    return relative


def _validate_relative_project_path(relative: Path, root: Path) -> None:
    casefold_parts = {part.casefold() for part in relative.parts}
    if ".git" in casefold_parts:
        raise PermissionDeniedError("Patch targets Git internals.")
    if relative.suffix.lower() in MODEL_SUFFIXES:
        raise PermissionDeniedError("Patch targets a model or binary artifact.")
    target = root / relative
    resolved = target.resolve(strict=True) if target.exists() else _resolve_new_project_path(target)
    _ensure_under_root(resolved, root)
    deny_sensitive_path(resolved)


def _resolve_new_project_path(target: Path) -> Path:
    current = target
    while not current.exists():
        if current.parent == current:
            break
        current = current.parent
    resolved_parent = current.resolve(strict=True)
    return (resolved_parent / target.relative_to(current)).resolve(strict=False)


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionDeniedError("Path escapes the selected project.") from exc
    path_text = str(path).casefold()
    root_text = str(root).casefold().rstrip("/")
    if path_text != root_text and not path_text.startswith(root_text + "/"):
        raise PermissionDeniedError("Case-insensitive path escape is denied.")
