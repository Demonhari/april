from __future__ import annotations

import os
from pathlib import Path

import typer

from apps.cli.render import console
from apps.runner.release_tools import (
    APP_IDENTIFIER,
    build_production_app,
    run_apple_tool,
    validate_app_bundle,
    validate_release_zip,
    write_launch_agent,
)

package_app = typer.Typer(help="Build and validate production macOS artifacts.")
launch_agent_app = typer.Typer(help="Manage the optional owner LaunchAgent.")
package_app.add_typer(launch_agent_app, name="launch-agent")


def _run_or_exit(argv: list[str]) -> None:
    result = run_apple_tool(argv)
    if result.output:
        console.print(result.output)
    if result.returncode:
        raise typer.Exit(result.returncode)


@package_app.command("build")
def build(
    output: Path = typer.Option(Path("dist/APRIL.app"), "--output"),
    version: str = typer.Option("0.1.0", "--version"),
    icon: Path | None = typer.Option(None, "--icon"),
) -> None:
    app = build_production_app(output, version=version, icon=icon)
    console.print(f"Built unsigned production bundle: {app}")
    console.print("No models, credentials, adapters, or user data were bundled.")


@package_app.command("validate")
def validate(path: Path) -> None:
    validate_app_bundle(path)
    console.print("Application bundle structure and local-data exclusions are valid.")


@package_app.command("validate-release-zip")
def validate_zip(path: Path) -> None:
    members = validate_release_zip(path)
    console.print(f"Release ZIP is valid ({len(members)} entries inspected).")


@package_app.command("sign")
def sign(path: Path, identity: str = typer.Option(..., "--identity")) -> None:
    app = path.expanduser().resolve(strict=True)
    validate_app_bundle(app)
    _run_or_exit(
        [
            "/usr/bin/codesign",
            "--force",
            "--deep",
            "--options",
            "runtime",
            "--timestamp",
            "--entitlements",
            str(app / "Contents" / "APRIL.entitlements"),
            "--sign",
            identity,
            str(app),
        ]
    )
    console.print("Developer ID signing completed successfully.")


@package_app.command("verify-signature")
def verify_signature(path: Path) -> None:
    _run_or_exit(
        [
            "/usr/bin/codesign",
            "--verify",
            "--deep",
            "--strict",
            "--verbose=2",
            str(path.expanduser().resolve(strict=True)),
        ]
    )
    console.print("Signature verification succeeded.")


@package_app.command("archive")
def archive(
    path: Path,
    output: Path = typer.Option(Path("dist/APRIL.zip"), "--output"),
) -> None:
    app = path.expanduser().resolve(strict=True)
    target = output.expanduser().resolve(strict=False)
    validate_app_bundle(app)
    if target.exists():
        raise ValueError("Refusing to overwrite an existing release archive.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run_or_exit(
        [
            "/usr/bin/ditto",
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(app),
            str(target),
        ]
    )
    validate_release_zip(target)
    console.print("Signed application archive created and exclusion-validated.")


@package_app.command("notarize-submit")
def notarize_submit(
    path: Path,
    keychain_profile: str = typer.Option(..., "--keychain-profile"),
) -> None:
    _run_or_exit(
        [
            "/usr/bin/xcrun",
            "notarytool",
            "submit",
            str(path.expanduser().resolve(strict=True)),
            "--keychain-profile",
            keychain_profile,
            "--output-format",
            "json",
        ]
    )


@package_app.command("notarize-status")
def notarize_status(
    submission_id: str,
    keychain_profile: str = typer.Option(..., "--keychain-profile"),
) -> None:
    _run_or_exit(
        [
            "/usr/bin/xcrun",
            "notarytool",
            "info",
            submission_id,
            "--keychain-profile",
            keychain_profile,
            "--output-format",
            "json",
        ]
    )


@package_app.command("staple")
def staple(path: Path) -> None:
    _run_or_exit(
        [
            "/usr/bin/xcrun",
            "stapler",
            "staple",
            str(path.expanduser().resolve(strict=True)),
        ]
    )


@package_app.command("gatekeeper")
def gatekeeper(path: Path) -> None:
    _run_or_exit(
        [
            "/usr/sbin/spctl",
            "--assess",
            "--type",
            "execute",
            "--verbose=2",
            str(path.expanduser().resolve(strict=True)),
        ]
    )
    console.print("Gatekeeper assessment succeeded.")


@launch_agent_app.command("install")
def launch_agent_install(path: Path) -> None:
    destination = Path.home() / "Library" / "LaunchAgents" / f"{APP_IDENTIFIER}.plist"
    written = write_launch_agent(path, destination)
    _run_or_exit(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(written)])
    console.print("APRIL owner LaunchAgent installed.")


@launch_agent_app.command("remove")
def launch_agent_remove() -> None:
    destination = Path.home() / "Library" / "LaunchAgents" / f"{APP_IDENTIFIER}.plist"
    if destination.exists():
        _run_or_exit(["/bin/launchctl", "bootout", f"gui/{os.getuid()}", str(destination)])
        destination.unlink()
    console.print("APRIL owner LaunchAgent removed.")
