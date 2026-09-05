# Local Security and Data Integrity

Phase 4B moves APRIL's service credentials out of plaintext configuration and
adds tamper-evident audit records plus validated SQLite maintenance.

## Credentials

On macOS, production-style APRIL uses generic-password items in the user's
Keychain. The stable service name is
`com.april.local-assistant.credentials`; APRIL uses separate accounts for the
Core API token, Runtime token, and audit terminal anchor. Only the dedicated
credential adapter invokes `/usr/bin/security`. It uses argv execution,
`shell=False`, a minimal environment, captured output, and a bounded timeout.
Errors never include command output or credential values.

Plaintext `APRIL_API_TOKEN` and `APRIL_RUNTIME_TOKEN` entries in `.env` or
`configs/april.yaml` are legacy inputs. Migrate them explicitly:

```bash
run april security credentials migrate
```

APRIL writes both secrets to the selected store, reads them back, and only then
atomically removes their plaintext values. The rewritten `.env` contains
non-secret backend and credential identifiers. A mode-0600 rollback copy exists
only while the migration transaction is in progress. Re-running the command is
safe and reports `already_migrated`.

Rotate credentials without revealing the generated values:

```bash
run april security credentials rotate --api
run april security credentials rotate --runtime
run april security credentials rotate --all
run april security credentials rotate --all --restart-services
```

API rotation requires Core API restart. Runtime rotation requires both April
Runtime and Core API restart. Multi-credential rotation retains prior values
until every selected write and read-back check succeeds; a partial failure rolls
back. Rotation audit events contain credential identifiers and outcomes, never
values.

Non-macOS development must explicitly select a file store and an absolute path
outside the repository:

```bash
run april setup tokens --store file \
  --credential-file /absolute/private/path/april-credentials.json
```

The file and its directory are owner-only, writes are atomic and fsynced, and an
existing group/world-readable file is rejected. File and memory stores are
rejected in production. The memory store exists only for injected tests.

## Audit verification

Audit records are canonical JSON Lines with schema version, sequence, UUID,
UTC timestamp, event type, redacted payload, previous hash, and current SHA-256.
The current hash excludes only its own field. A dedicated lock serializes
threads and processes; appends and anchor updates are fsynced.

```bash
run april audit verify
run april audit verify --json
```

Verification detects malformed records, schema errors, gaps, duplicates,
reordering, changed hashes, missing genesis state, and final-record removal
compared with the protected anchor. A single event durably appended before a
crash but not yet reflected in the anchor is reported as `anchor_lagged`, not
corruption. Actual corruption exits non-zero. An unterminated final write is
reported by verification and is deterministically discarded before the next
append only when every preceding complete record is valid.

## Database integrity, backup, and restore

The bounded normal check does not run the expensive full integrity scan:

```bash
run april database check
run april database check --json
run april database check --full
```

It reports availability, `quick_check`, foreign keys, migration state, WAL,
synchronous mode, busy timeout, WAL/SHM diagnostics, and last successful backup.
`--full` additionally runs `integrity_check`.

Create a backup package:

```bash
run april database backup --output /private/backups/april-2026-07-29.april
```

The output is an owner-only directory containing `database.sqlite3` and
`manifest.json`. APRIL acquires its existing cross-process database write fence,
uses SQLite's backup API for a WAL-consistent snapshot, validates it, fsyncs it,
and atomically publishes the completed directory. Cancellation removes staging
files and never publishes an incomplete destination.

Restore only from a validated package:

```bash
run april database restore --input /private/backups/april-2026-07-29.april
run april database restore --input /private/backups/april-2026-07-29.april \
  --stop-services
```

Restore refuses to proceed while services run unless `--stop-services` safely
stops them. It validates manifest format, SHA-256, size, quick/full integrity,
foreign keys, and schema compatibility before acquiring the write fence. It
creates a rollback package, atomically replaces the database, removes obsolete
WAL/SHM files, reopens and revalidates, and automatically restores the rollback
if post-replacement validation fails. There is no force/bypass option. Services
remain stopped for operator review.

## Troubleshooting safely

Use status words and error classes, not copied Keychain output:

```bash
run april verify
run april readiness --json
run april audit verify --json
run april database check --json
```

Do not paste `.env`, Keychain output, Authorization headers, audit payloads, or
the credential file into bug reports. Readiness and verification intentionally
report only credential availability and backend selection.

## Audit verification and recovery

Inspect the chain without changing it:

```bash
run april audit verify --json
run april audit recover --json
```

Recovery is dry-run by default. Applying recovery requires an exact, approved,
unexpired `audit_recovery` operation in production, bound to the precise
`reason` and `apply=true` action. APRIL locks and snapshots the audit log and
protected anchor, quarantines the original bytes under
`data/backups/audit-quarantine/` with checksums and owner-only permissions, and
refuses to claim success if the log, anchor, approval, or audit write changes
unexpectedly. Do not delete the original log or anchor to make readiness pass.
If no approved operation exists, the production CLI refuses recovery; it does
not create or accept an operator-supplied arbitrary approval string.

## Threat-model boundary

This phase protects against casual plaintext-secret disclosure, inheritance of
unrelated parent secrets by child services, undetected audit mutation or
terminal truncation, torn live-database copies, corrupt backup restoration, and
failed restore replacement. It does not protect a fully compromised logged-in
macOS account, malware able to use the operator's unlocked Keychain, physical
access to an unlocked machine, deliberate deletion of both audit data and its
Keychain anchor, or loss of the disk without an independently retained backup.
It adds no signing, notarization, cloud backup, remote attestation, or encryption
beyond the operating system's Keychain and filesystem protections.
