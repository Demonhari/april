from __future__ import annotations

import argparse
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from april_common.settings import project_root

APRIL_WRAPPER_MARKER = "# APRIL_RUN_WRAPPER=1"


@dataclass(frozen=True)
class InstallResult:
    installed: list[Path]
    skipped: list[Path]


def wrapper_content(*, repo_root: Path) -> str:
    root = repo_root.expanduser().resolve()
    python_path = root / ".venv" / "bin" / "python"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            APRIL_WRAPPER_MARKER,
            "set -euo pipefail",
            f'export APRIL_HOME="{root}"',
            f'exec "{python_path}" -m apps.runner.main "$@"',
            "",
        ]
    )


def is_april_wrapper(path: Path) -> bool:
    try:
        return APRIL_WRAPPER_MARKER in path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False


def install_wrappers(*, repo_root: Path, bin_dir: Path, force: bool = False) -> InstallResult:
    bin_dir = bin_dir.expanduser().resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    skipped: list[Path] = []
    content = wrapper_content(repo_root=repo_root)
    for name in ("run", "april-run"):
        target = bin_dir / name
        if target.exists() and not is_april_wrapper(target) and not force:
            raise FileExistsError(
                f"{target} already exists and is not APRIL-owned. "
                "Re-run with --force to replace it."
            )
        if target.exists() and is_april_wrapper(target) and target.read_text() == content:
            skipped.append(target)
            continue
        target.write_text(content, encoding="utf-8")
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        installed.append(target)
    return InstallResult(installed=installed, skipped=skipped)


def uninstall_wrappers(*, bin_dir: Path) -> InstallResult:
    bin_dir = bin_dir.expanduser().resolve()
    removed: list[Path] = []
    skipped: list[Path] = []
    for name in ("run", "april-run"):
        target = bin_dir / name
        if is_april_wrapper(target):
            target.unlink()
            removed.append(target)
        else:
            skipped.append(target)
    return InstallResult(installed=removed, skipped=skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install or remove APRIL global wrappers.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--install", action="store_true")
    action.add_argument("--uninstall", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=project_root())
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local" / "bin")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.install:
        result = install_wrappers(repo_root=args.repo_root, bin_dir=args.bin_dir, force=args.force)
        for path in result.installed:
            print(f"Installed {path}")
        for path in result.skipped:
            print(f"Already up to date: {path}")
        local_bin = str(args.bin_dir.expanduser().resolve())
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if local_bin not in path_parts:
            print(f'Add this to your shell PATH: export PATH="{local_bin}:$PATH"')
        return 0

    result = uninstall_wrappers(bin_dir=args.bin_dir)
    for path in result.installed:
        print(f"Removed {path}")
    for path in result.skipped:
        print(f"Skipped non-APRIL wrapper or missing file: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
