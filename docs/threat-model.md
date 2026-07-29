# Threat Model

Main boundaries:

- user input is untrusted
- model output is untrusted
- retrieved local files are untrusted
- repository content is untrusted
- command output is untrusted

Controls:

- model output cannot execute tools directly
- specialist model output is parsed as strict structured JSON and may only
  request tools through APRIL application code
- deterministic permission engine is authoritative
- configured allowed filesystem roots are enforced after symlink resolution
- sensitive paths are denied
- sensitive file names are checked case-insensitively, including `.env`,
  `.env.*`, `.netrc`, `.npmrc`, SSH private keys, PEM/private key material,
  credential/token files, browser credential stores, system keychains, and
  direct tool access to `data/april.db`
- shell execution uses argv arrays and `shell=False`
- Level 3+ operations require exact-action one-time approvals
- structured approvals bind to a persisted suspended run and resume only that
  run after the exact approved tool executes once
- approved tools are revalidated against current policy before execution
- patch and commit approvals bind immutable digests and repository state, then
  recalculate those digests immediately before execution
- log/cache cleanup uses a two-stage, immutable boundary modeled on the patch
  flow: `plan_log_cleanup` (Level 1, read-only) enumerates only ordinary files
  under an APRIL-owned root (`logs` or the audio cache) into a content-addressed
  manifest and deletes nothing; `apply_log_cleanup` (Level 4) requires
  exact-action approval bound to that manifest, revalidates root containment and
  each file's identity (size + SHA-256) before deletion, never follows symlinks,
  never deletes directories, cannot broaden the candidate set, fails closed on a
  tampered manifest, and is marked one-time-use to prevent replay
- there is no generic, recursive, or caller-rooted delete tool; cleanup roots are
  derived from settings, never from caller-supplied paths
- risky approved tools audit a start record before running and consume approvals after success or failure
- repository operations require explicit project selection and allowed-root validation
- retrieved memory and indexed repository chunks are marked as context, not instructions
- external actions are disabled by default
- approved external actions are rechecked against the current
  `external_actions_enabled` setting before execution
- `open_app` is restricted to configured plain application names and macOS
  `/usr/bin/open -a` argv execution
- `open_url` is restricted to normalized `http`/`https` URLs without embedded
  credentials and macOS `/usr/bin/open` argv execution
- API/Runtime credentials and the protected audit anchor live behind a typed
  credential-store interface; macOS production uses Keychain and fails closed
  when it is unavailable
- child-service environments are allowlisted and carry credential-store
  identifiers rather than raw API/Runtime tokens; test runners, repository
  tools, proxy variables, SSH variables, and unrelated cloud secrets are
  excluded
- bearer tokens, prompts, transcripts, audio, secret environment values, and
  credential-like values are redacted before audit hashing and persistence
- audit records form a cross-process-serialized SHA-256 chain with a protected
  terminal anchor, so modification, reordering, middle deletion, and terminal
  truncation are detectable
- database backups use SQLite's backup API under the existing write fence and
  restores require manifest/hash/integrity/schema validation plus automatic
  rollback

Phase 4B does not defend against a compromised logged-in user session, malware
that can use an unlocked Keychain, an unlocked stolen machine, or deletion of
both the audit file and its Keychain anchor. It does not add disk encryption,
native signing/notarization, remote attestation, or off-device backup.
