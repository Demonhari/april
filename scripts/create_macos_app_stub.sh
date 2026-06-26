#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_path="${APRIL_APP_STUB_OUTPUT:-"$repo_root/dist/APRIL.app"}"
contents="$app_path/Contents"
macos="$contents/MacOS"
resources="$contents/Resources"
launcher="$macos/APRIL"

mkdir -p "$macos" "$resources"

cat > "$contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>APRIL</string>
  <key>CFBundleDisplayName</key>
  <string>APRIL</string>
  <key>CFBundleIdentifier</key>
  <string>local.april.dev</string>
  <key>CFBundleVersion</key>
  <string>0.1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleExecutable</key>
  <string>APRIL</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHumanReadableCopyright</key>
  <string>Unsigned local development launcher. No models, tokens, or secrets are bundled.</string>
</dict>
</plist>
PLIST

cat > "$launcher" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail

cd "$repo_root"
if command -v run >/dev/null 2>&1; then
  exec run april desktop "\$@"
elif [ -x ".venv/bin/python" ]; then
  exec ".venv/bin/python" -m apps.runner.main april desktop "\$@"
else
  exec python -m apps.runner.main april desktop "\$@"
fi
LAUNCHER

chmod 755 "$launcher"

printf 'Created unsigned APRIL development launcher: %s\n' "$app_path"
printf 'This bundle contains no models, tokens, secrets, signing, or notarization.\n'
