# APRIL Desktop

A local, single-page Desktop UI for APRIL. It is plain static HTML/CSS/JS (no
Node, no npm, no build step) served by the Core API over authenticated loopback
HTTP. The UI never imports model bindings, runtime internals, tool executors, or
SQLite repositories — it only talks to the existing Core API endpoints.

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
native window and injects the token through the JS bridge (`window.__APRIL_TOKEN__`),
so it never appears in any URL. If pywebview is not installed, `--native` prints
a message and falls back to the browser path. The default path needs zero extra
runtime dependencies.

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
  (`/health`, `/diagnostics`, `/runtime/models` + load/unload), and an
  Activity/Logs feed (`GET /diagnostics/activity`).

Memory is never created automatically from chat; the UI shows exactly what is
stored. The Activity feed is sourced from the sanitized audit log through a
strict allowlist projection — it shows event types, timestamps, reference IDs,
and risk levels, and never prompt content, file contents, tool arguments, tokens,
or secrets.

## Files

- `web/index.html` — markup and left-nav shell.
- `web/styles.css` — system-font UI with monospace accents, light/dark via
  `prefers-color-scheme`, no external assets or CDNs.
- `web/app.js` — token bootstrap, the authenticated API client, the SSE chat
  reader, and all screens.

[pywebview]: https://pywebview.flowrl.com/
