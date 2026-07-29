from __future__ import annotations

import os
import plistlib
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

APP_IDENTIFIER = "local.april.assistant"
APP_NAME = "APRIL"
MINIMUM_MACOS = "13.0"
APPLE_TOOL_TIMEOUT_SECONDS = 900.0
MAX_TOOL_OUTPUT_BYTES = 100_000

_BANNED_PARTS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".april_tmp",
    "adapters",
    "credentials",
    "voice_recordings",
}
_BANNED_SUFFIXES = {
    ".gguf",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".wav",
    ".mp3",
    ".flac",
    ".pem",
    ".key",
    ".onnx",
    ".safetensors",
    ".tflite",
}
_SECRET_NAME = re.compile(
    r"(?i)(?:^|[._-])(tokens?|secrets?|credentials?|private[-_]?keys?)(?:[._-]|$)"
)


class ReleaseValidationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AppleToolResult:
    returncode: int
    output: str


def validate_release_zip(path: Path) -> tuple[str, ...]:
    """Inspect a ZIP without extracting it and reject local/sensitive artifacts."""
    resolved = path.expanduser().resolve(strict=True)
    if not zipfile.is_zipfile(resolved):
        raise ReleaseValidationError("Input is not a valid ZIP archive.")
    accepted: list[str] = []
    with zipfile.ZipFile(resolved) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ReleaseValidationError("Archive contains an unsafe member path.")
            lowered_parts = {part.casefold() for part in member.parts}
            suffix = member.suffix.casefold()
            basename = member.name.casefold()
            if lowered_parts & _BANNED_PARTS:
                raise ReleaseValidationError(
                    f"Archive contains forbidden category: {member.name or 'directory'}."
                )
            if suffix in _BANNED_SUFFIXES or _SECRET_NAME.search(basename):
                raise ReleaseValidationError(
                    f"Archive contains forbidden file category: {member.name}."
                )
            if "data/verification" in member.as_posix().casefold():
                raise ReleaseValidationError("Archive contains generated verification reports.")
            accepted.append(member.as_posix())
    return tuple(accepted)


def build_production_app(
    output: Path,
    *,
    version: str,
    icon: Path | None = None,
) -> Path:
    destination = output.expanduser().resolve(strict=False)
    if destination.suffix != ".app":
        raise ValueError("Production bundle output must end in .app.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".april-app-", dir=str(destination.parent)))
    staged = temporary_root / destination.name
    contents = staged / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, mode=0o755)
    resources.mkdir(mode=0o755)
    try:
        icon_name: str | None = None
        if icon is not None:
            resolved_icon = icon.expanduser().resolve(strict=True)
            if not resolved_icon.is_file() or resolved_icon.suffix.casefold() != ".icns":
                raise ValueError("Application icon must be a regular .icns file.")
            icon_name = "APRIL.icns"
            shutil.copyfile(resolved_icon, resources / icon_name)
            os.chmod(resources / icon_name, 0o644)
        info: dict[str, object] = {
            "CFBundleDevelopmentRegion": "en",
            "CFBundleDisplayName": APP_NAME,
            "CFBundleExecutable": APP_NAME,
            "CFBundleIdentifier": APP_IDENTIFIER,
            "CFBundleInfoDictionaryVersion": "6.0",
            "CFBundleName": APP_NAME,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": version,
            "CFBundleVersion": version,
            "LSMinimumSystemVersion": MINIMUM_MACOS,
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": (
                "APRIL uses the microphone only for operator-enabled local voice features."
            ),
        }
        if icon_name is not None:
            info["CFBundleIconFile"] = icon_name
        with (contents / "Info.plist").open("wb") as handle:
            plistlib.dump(info, handle, sort_keys=True)
        os.chmod(contents / "Info.plist", 0o644)
        launcher = macos / APP_NAME
        launcher.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            'if command -v run >/dev/null 2>&1; then exec run april desktop "$@"; fi\n'
            'exec python3 -m apps.runner.main april desktop "$@"\n',
            encoding="utf-8",
        )
        os.chmod(launcher, 0o755)
        entitlements = contents / "APRIL.entitlements"
        with entitlements.open("wb") as handle:
            plistlib.dump(
                {
                    "com.apple.security.cs.allow-jit": False,
                    "com.apple.security.cs.disable-library-validation": False,
                },
                handle,
                sort_keys=True,
            )
        os.chmod(entitlements, 0o644)
        validate_app_bundle(staged)
        if destination.exists():
            raise FileExistsError("Refusing to overwrite an existing application bundle.")
        os.replace(staged, destination)
        _fsync_directory(destination.parent)
        return destination
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def validate_app_bundle(app_path: Path) -> None:
    app = app_path.expanduser().resolve(strict=True)
    info_path = app / "Contents" / "Info.plist"
    launcher = app / "Contents" / "MacOS" / APP_NAME
    entitlements = app / "Contents" / "APRIL.entitlements"
    if not info_path.is_file() or not launcher.is_file() or not entitlements.is_file():
        raise ReleaseValidationError("Application bundle structure is incomplete.")
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)
    required = {
        "CFBundleIdentifier": APP_IDENTIFIER,
        "CFBundleExecutable": APP_NAME,
        "LSMinimumSystemVersion": MINIMUM_MACOS,
    }
    if any(info.get(key) != value for key, value in required.items()):
        raise ReleaseValidationError("Application bundle metadata is invalid.")
    if not info.get("NSMicrophoneUsageDescription"):
        raise ReleaseValidationError("Application bundle lacks microphone disclosure.")
    if stat.S_IMODE(launcher.stat().st_mode) != 0o755:
        raise ReleaseValidationError("Application launcher permissions are invalid.")
    for path in app.rglob("*"):
        if path.is_file() and (
            path.suffix.casefold() in _BANNED_SUFFIXES or _SECRET_NAME.search(path.name.casefold())
        ):
            raise ReleaseValidationError("Application bundle contains forbidden local data.")


def run_apple_tool(
    argv: Sequence[str],
    *,
    extra_environment: Mapping[str, str] | None = None,
    timeout_seconds: float = APPLE_TOOL_TIMEOUT_SECONDS,
) -> AppleToolResult:
    if not argv or not Path(argv[0]).is_absolute():
        raise ValueError("Apple tool executable must use an absolute path.")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin",
        "LANG": "C",
        "LC_ALL": "C",
    }
    if home := os.environ.get("HOME"):
        environment["HOME"] = home
    if extra_environment:
        environment.update(extra_environment)
    completed = subprocess.run(
        list(argv),
        cwd="/",
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=max(1.0, min(timeout_seconds, APPLE_TOOL_TIMEOUT_SECONDS)),
        check=False,
        shell=False,
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return AppleToolResult(
        completed.returncode,
        _redact_tool_output(combined[:MAX_TOOL_OUTPUT_BYTES]),
    )


def write_launch_agent(app_path: Path, destination: Path) -> Path:
    app = app_path.expanduser().resolve(strict=True)
    validate_app_bundle(app)
    target = destination.expanduser().resolve(strict=False)
    expected_parent = Path.home() / "Library" / "LaunchAgents"
    if target.parent != expected_parent or target.name != f"{APP_IDENTIFIER}.plist":
        raise ValueError("LaunchAgent destination is not the APRIL owner LaunchAgents path.")
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = {
        "Label": APP_IDENTIFIER,
        "ProgramArguments": [str(app / "Contents" / "MacOS" / APP_NAME)],
        "RunAtLoad": True,
        "KeepAlive": False,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".april-launchagent-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            plistlib.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _redact_tool_output(value: str) -> str:
    value = re.sub(r"(?i)(password|token|secret)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    return re.sub(r"(?i)(apple-id|apple_id)\s+\S+", r"\1 [REDACTED]", value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
