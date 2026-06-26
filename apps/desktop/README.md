# APRIL Desktop

A local, single-page Desktop UI for APRIL. It is plain static HTML/CSS/JS (no
Node, no npm, no build step) served by the Core API over authenticated loopback
HTTP. The UI never imports model bindings, runtime internals, tool executors, or
SQLite repositories — it only talks to the existing Core API endpoints.

## Cockpit dashboard

The default screen is a wide **cockpit dashboard** — a dark cyan/terminal layout
optimised for a landscape MacBook screen, with a narrow/stacked fallback. It is
built with CSS grid and pure HTML/CSS (no canvas, no SVG namespace, no external
assets):

- **Top system rail** — brand, overall systems status, backend + simulated/real
  badge, active project, conversation short id, and a `127.0.0.1 · offline-capable`
  indicator.
- **Left systems stack** — Core API, April Runtime, backend, database, vector
  index, voice, and scheduler status, each with a severity-coloured dot.
- **Centre** — a router/orbit visualisation of the six specialists
  (`GEN/COD/RDG/CRE/RSN/SYS`) that highlights the agent last surfaced by a chat
  stream; an active-specialist card; a Level 0–5 permission ladder; a runtime
  telemetry strip (tokens/sec, first-token latency, context size, process RSS,
  loaded models, active requests, generation errors); and the runtime models
  card with load/unload controls.
- **Right** — pending approvals (exact-ID approve/deny) and a reminders/tasks +
  briefing summary card.
- **Bottom** — the redacted activity feed (terminal style) beside a command
  console, with a wide command/chat input.

Layout breakpoints: full three-column cockpit at `>= 1180px`, two columns at
`800–1179px`, and a single stacked column below `800px`. All glow and orbit
motion is disabled under `prefers-reduced-motion`.

Operational values are reported honestly: when an endpoint does not provide a
value the UI shows `unknown`/`not available` rather than a fabricated number (and
`0` is preserved as a real value). A **simulated** runtime is always badged so it
can never be mistaken for a verified real model. The dashboard is data-driven by
polling the existing authenticated endpoints (health ~8s, approvals ~8s, activity
~10s, models ~13s, reminders/tasks/briefing ~45s); a failed poll keeps the last
known data and flips the rail to a degraded/offline state instead of blanking.

The cockpit is primary; every previous screen plus **Readiness** (Chat, Projects,
Approvals, Memory, Reminders, Readiness, Status & Models, Activity) is still
reachable from the compact top nav as a detail screen.

## Readiness screen

Readiness is an authenticated, local-only setup detail screen backed by
`GET /readiness`, `GET /verification/report/latest`, and
`GET /verification/reports`. It renders only sanitized fields:

- Core readiness: API health, Runtime health, fake/simulated vs real backend,
  database, vector index, and scheduler state.
- Model readiness: registered model id/name/role/backend/state/keep_loaded,
  missing-path and simulated badges, `llama-cpp-python` availability, and model
  path basenames only.
- Verification guidance: exact `run april verify --all-configured-models
  --require-real-model --report data/verification/mac-readiness.json` and
  single-model target-Mac commands, plus warnings that fake verification is not
  real model verification and generated reports/app stubs are ignored.
- Voice readiness: voice enabled/disabled, macOS microphone guidance,
  sounddevice availability/counts when safely queryable, configured/missing
  whisper.cpp/Piper/wake-word artifacts, and push-to-talk availability without
  wake word.
- Security readiness: allowed-root labels, token configured/missing status only,
  localhost binding state, CORS state, and development-token warnings.
- Latest report: generated timestamp, report type, pass/degraded/fail summary,
  verification level (`none` / `partial` / `core` / `all`), real-model counts,
  skipped checks, threshold failures, and model basenames.
- Report history: sanitized summaries from `data/verification/*.json`, sorted
  newest first, with report type, summary, verification level, generated
  timestamp, skipped count, and threshold failure count.
- Guided setup examples: `run april setup models`, `run april setup voice`, and
  `run april setup app-stub`.
- CI/local gates: the Python quality gates and Node/static Desktop checks run
  locally and in CI.

If no report exists, it shows `not verified yet`. The report endpoints read only
APRIL's own `data/verification/*.json` directory. The basename endpoint rejects
path traversal, slashes, backslashes, absolute paths, symlinks, non-JSON files,
and arbitrary query paths. Command copy buttons are shown only when
`navigator.clipboard.writeText` is available; otherwise the command remains
visible as text. The screen never starts verification, loads a model, records
audio, starts wake-word listening, or executes commands.

## Launch

```bash
run april desktop            # ensure services + open the browser to the UI
run april desktop --fake     # same, with the deterministic fake runtime
run april desktop --native   # optional native window (needs the [desktop] extra)
run april desktop --no-open   # resolve services + print the URL without opening
```

`run april desktop`:

- ensures April Runtime and the Core API are running (honoring `--fake`),
- never starts voice, wake-word, or the microphone,
- resolves the API token from the same settings/.env source the CLI uses,
- opens `http://127.0.0.1:<api_port>/desktop#token=<TOKEN>`.

The token is passed in the URL **fragment** (after `#`), which browsers never
send to the server. On load the SPA reads the token into an in-memory variable
and immediately calls `history.replaceState` to strip it from the address bar.
The token is never written to `localStorage`, `sessionStorage`, or cookies.

### Optional native window

`pip install -e '.[desktop]'` adds [pywebview]. With `--native`, APRIL opens a
native window and the SPA fetches the token asynchronously through the minimal JS
bridge (`window.pywebview.api.get_token()`, awaited after the `pywebviewready`
event), so it never appears in any URL and is never assumed to be a synchronously
injected global. If pywebview is not installed, `--native` prints a message and
falls back to the browser path. The default path needs zero extra runtime
dependencies.

### Unsigned app stub vs. a real signed/notarized app

For a local app bundle wrapper around the same command:

```bash
scripts/create_macos_app_stub.sh
run april setup app-stub
```

| | Unsigned app stub (today) | Signed/notarized app (future work) |
|---|---|---|
| What it is | `dist/APRIL.app`, a thin launcher that runs `run april desktop` | A distributable, Gatekeeper-trusted `.app` |
| Code signing | None | Developer ID signature required |
| Notarization | None | Apple notarization + stapling required |
| Gatekeeper | First launch needs right-click → Open (or a quarantine prompt) | Launches with no warning |
| Bundles models/voice/tokens | No — none of these are ever bundled | No — still local-only |
| Launch-at-login | No | Possible future option |
| Git | Ignored (`dist/` is not committed) | n/a |

Both commands create `dist/APRIL.app` as a **development launcher only**. They
perform no signing/notarization, install nothing, and bundle no models, voice
binaries, tokens, or secrets. The generated bundle is ignored by Git. Signed
packaging, notarization, and launch-at-login are future work; until then the
Desktop is a local SPA reached through `run april desktop` or this unsigned stub.

No authenticated request is issued until token acquisition succeeds. If it fails,
the SPA shows a safe local "locked" screen and starts no polling; the dashboard,
which fetches on mount, only mounts after the token is in memory.

## How it talks to the Core API

- Authenticated loopback HTTP only. Every data request sends
  `Authorization: Bearer <token>`. The static assets under `/desktop` ship no
  secrets and are the only unauthenticated surface besides the redacted
  `GET /health`.
- Chat streams from `POST /chat/stream`, consumed with `fetch()` +
  `ReadableStream` (not `EventSource`, which cannot POST or set headers). It
  renders `meta`/`token`/`approval_required`/`usage`/`done`/`error` events and
  reuses one `conversation_id` per session.
- Approvals use `GET /approvals`, `POST /tools/approve`, and `POST /tools/deny`
  by exact approval ID, showing the bound details (tool, paths, digest, expiry).
  A chat "yes" is never approval; when an `approval_required` event arrives the
  UI routes the user to the Approvals screen instead of acting.
- Projects (`GET`/`POST /projects`), Memory (`GET /memory/search`,
  `GET /memory/export`, `POST /memory`, `DELETE /memory/{id}`), Reminders &
  Tasks (`/reminders`, `/tasks`, `/scheduler/briefing/preview`), Status & Models
  (`/health`, `/diagnostics`, `/runtime/models` + load/unload), report history
  (`/verification/report/latest`, `/verification/reports`), and an Activity/Logs
  feed (`GET /diagnostics/activity`).

Memory is never created automatically from chat; the UI shows exactly what is
stored. The Activity feed is sourced from the sanitized audit log through a
strict allowlist projection — it shows event types, timestamps, reference IDs,
and risk levels, and never prompt content, file contents, tool arguments, tokens,
or secrets. The dashboard projects the same allowlist a **second** time on the
client (`redactActivityEvent`) as defence in depth, and the chat command console
shows only structural stream chips (agent, route method, tool name, status, risk)
— never streamed token text, the final message, or routing decision summaries.

## Files

- `web/index.html` — markup and the compact cockpit nav shell.
- `web/styles.css` — dark APRIL cockpit theme (deep navy, cyan structure, green
  ok, orange approval, red deny), monospace accents, CSS-variable design system,
  responsive grid, `prefers-reduced-motion` support, and no external assets/CDNs.
- `web/dashboard_helpers.js` — pure, DOM-free formatting/redaction helpers
  (honest `unknown` values, activity allowlist projection, content-free stream
  chips, agent/permission/telemetry view models). Exported CommonJS-style and
  unit tested under Node (`tests/js/desktop_dashboard.test.cjs`).
- `web/app.js` — token bootstrap, the authenticated + silent-polling API clients,
  the SSE chat reader, the cockpit dashboard, and all detail screens.

[pywebview]: https://pywebview.flowrl.com/
