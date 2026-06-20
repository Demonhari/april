from __future__ import annotations

import argparse
import os
import stat
from dataclasses import dataclass
from pathlib import Path

APRIL_WRAPPER_MARKER = "# APRIL_RUN_WRAPPER=1"
APRIL_RUN_COMMAND_MARKER = "# APRIL_RUN_COMMAND=python -m apps.runner.main"
PATH_BLOCK_START = "# >>> APRIL launcher PATH >>>"
PATH_BLOCK_END = "# <<< APRIL launcher PATH <<<"
PATH_EXPORT_LINE = 'export PATH="$HOME/.local/bin:$PATH"'
WRAPPER_NAMES = ("run", "april-run")


@dataclass(frozen=True)
class InstallResult:
    installed: list[Path]
    skipped: list[Path]


def wrapper_content(*, repo_root: Path) -> str:
    root = repo_root.expanduser().resolve()
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            APRIL_WRAPPER_MARKER,
            APRIL_RUN_COMMAND_MARKER,
            "set -euo pipefail",
            f'export APRIL_HOME="{root}"',
            'export PYTHONPATH="$APRIL_HOME${PYTHONPATH:+:$PYTHONPATH}"',
            'APRIL_PYTHON="${APRIL_PYTHON:-}"',
            'if [[ -z "$APRIL_PYTHON" && -x "$APRIL_HOME/.venv/bin/python" ]]; then',
            '  APRIL_PYTHON="$APRIL_HOME/.venv/bin/python"',
            "fi",
            'if [[ -z "$APRIL_PYTHON" ]]; then',
            '  APRIL_PYTHON="python3.11"',
            "fi",
            'if ! command -v "$APRIL_PYTHON" >/dev/null 2>&1; then',
            '  echo "APRIL launcher could not find a Python interpreter: $APRIL_PYTHON" >&2',
            '  echo "Create .venv with python3.11 -m venv .venv and install APRIL." >&2',
            '  echo "Or set APRIL_PYTHON to a provisioned interpreter." >&2',
            "  exit 127",
            "fi",
            'if ! "$APRIL_PYTHON" -c "import apps.runner.main" >/dev/null 2>&1; then',
            '  echo "APRIL is not importable with interpreter: $APRIL_PYTHON" >&2',
            "  echo \"Run: python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'\" >&2",
            '  echo "Or set APRIL_PYTHON to an interpreter with APRIL dependencies." >&2',
            "  exit 1",
            "fi",
            'exec "$APRIL_PYTHON" -m apps.runner.main "$@"',
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
    for name in WRAPPER_NAMES:
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
    for name in WRAPPER_NAMES:
        target = bin_dir / name
        if is_april_wrapper(target):
            target.unlink()
            removed.append(target)
        else:
            skipped.append(target)
    return InstallResult(installed=removed, skipped=skipped)


def verify_wrappers(*, repo_root: Path, bin_dir: Path) -> list[str]:
    root = repo_root.expanduser().resolve()
    errors: list[str] = []
    for name in WRAPPER_NAMES:
        target = bin_dir.expanduser().resolve() / name
        if not target.exists():
            errors.append(f"Missing wrapper: {target}")
            continue
        content = target.read_text(encoding="utf-8", errors="replace")
        required = [
            APRIL_WRAPPER_MARKER,
            str(root),
            "APRIL_PYTHON",
            "-m apps.runner.main",
        ]
        for needle in required:
            if needle not in content:
                errors.append(f"{target} does not contain required text: {needle}")
        if not os.access(target, os.X_OK):
            errors.append(f"{target} is not executable")
    return errors


def path_contains_dir(bin_dir: Path, *, path_value: str | None = None) -> bool:
    resolved_bin = bin_dir.expanduser().resolve()
    path_value = os.environ.get("PATH", "") if path_value is None else path_value
    for raw in path_value.split(os.pathsep):
        if not raw:
            continue
        try:
            if Path(raw).expanduser().resolve() == resolved_bin:
                return True
        except OSError:
            continue
    return False


def shell_config_path(*, shell: str | None = None, home: Path | None = None) -> Path:
    shell_name = Path(shell or os.environ.get("SHELL", "")).name
    user_home = (home or Path.home()).expanduser()
    if shell_name == "zsh":
        return user_home / ".zshrc"
    if shell_name == "bash":
        return user_home / ".bashrc"
    raise ValueError("Only zsh and bash PATH updates are supported automatically.")


def add_path_block(*, shell: str | None = None, home: Path | None = None) -> tuple[Path, bool]:
    config_path = shell_config_path(shell=shell, home=home)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = ""
    if config_path.exists():
        content = config_path.read_text(encoding="utf-8")
    if PATH_BLOCK_START in content and PATH_BLOCK_END in content:
        return config_path, False
    block = f"{PATH_BLOCK_START}\n{PATH_EXPORT_LINE}\n{PATH_BLOCK_END}\n"
    prefix = "" if not content or content.endswith("\n") else "\n"
    with config_path.open("a", encoding="utf-8") as file:
        file.write(prefix + block)
    return config_path, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install or remove APRIL global wrappers.")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--install", action="store_true")
    action_group.add_argument("--uninstall", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local" / "bin")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--add-to-path", action="store_true")
    args = parser.parse_args(argv)

    if args.install:
        result = install_wrappers(repo_root=args.repo_root, bin_dir=args.bin_dir, force=args.force)
        print("APRIL global launcher wrappers:")
        for path in result.installed:
            print(f"Installed {path}")
        for path in result.skipped:
            print(f"Already up to date: {path}")
        errors = verify_wrappers(repo_root=args.repo_root, bin_dir=args.bin_dir)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        for name in WRAPPER_NAMES:
            print(f"Verified {args.bin_dir.expanduser().resolve() / name}")
        local_bin = str(args.bin_dir.expanduser().resolve())
        in_path = path_contains_dir(args.bin_dir)
        print(f"{local_bin} in current PATH: {'yes' if in_path else 'no'}")
        if args.add_to_path:
            try:
                config_path, changed = add_path_block()
            except ValueError as exc:
                print(f"ERROR: {exc}")
                return 1
            action_text = "Updated" if changed else "Already configured"
            print(f"{action_text}: {config_path}")
            print(f"Reload your shell with: source {config_path}")
        if not in_path:
            print('Add this to your current shell now: export PATH="$HOME/.local/bin:$PATH"')
        print("Next command: run april --fake")
        print(f'Fallback command: "{Path(local_bin) / "run"}" april --fake')
        return 0

    result = uninstall_wrappers(bin_dir=args.bin_dir)
    for path in result.installed:
        print(f"Removed {path}")
    for path in result.skipped:
        print(f"Skipped non-APRIL wrapper or missing file: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
