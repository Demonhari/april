# Durable jobs and isolated tool execution

APRIL schema 19 stores local durable work in `background_jobs` and bounded
progress history in `background_job_events`. Jobs move through
`queued → running → cancelling → cancelled` or
`queued → running → succeeded|failed|interrupted`.

Only the current lease owner may heartbeat or complete a running job. Claims
and state changes are conditional SQLite transactions using APRIL's existing
cross-process write coordination. An expired lease is requeued only when its
allowlisted definition is both idempotent and restart-safe and still has an
attempt available. Mutation jobs are never automatically retried.

The separate Job Worker claims one job at a time by default. Fully integrated
job types are `repository_index`, `memory_reindex`, `document_index`,
`configured_test`, and `self_check`. Configured tests require an exact existing
`test_runner` approval. Model verification, benchmarking, and Dream Cycle
definitions are visible but deliberately unavailable until their phase-specific
permission and hardware workflows are implemented. Fine-tuning is not
registered.

Authenticated API routes are `POST /jobs`, `GET /jobs`,
`GET /jobs/{job_id}`, `POST /jobs/{job_id}/cancel`, and
`POST /jobs/{job_id}/retry`. Matching commands are
`run april jobs submit|list|show|cancel|retry`. `--wait` polls with a bounded
timeout; Ctrl-C stops only the wait and does not cancel the durable job.

Risky command, test, patch, and Git-commit execution is outside Core in the
Tool Worker. It listens only on an owner-controlled Unix socket, uses a 0600
socket and separate 0600 capability file, and accepts a versioned,
length-bounded JSON protocol. It independently validates the project root,
command allowlist, output/timeout bounds, and approval-bound patch or staged-Git
metadata. There is no in-Core fallback.

Child environments come from explicit category allowlists in
`april_common.process_environment`. Tool, test, Git, repository, and document
children receive no API token, Runtime token, cloud credential, proxy setting,
or SSH agent socket. Runtime receives only its Runtime credential; Core
receives the API and Runtime credentials it needs. Diagnostics never report
environment values.

Both workers are enabled by default. Development-only read-only operation can
explicitly set `APRIL_TOOL_WORKER_ENABLED=false` and
`APRIL_JOB_WORKER_ENABLED=false`; risky tools still fail closed and durable jobs
remain unavailable. When enabled, either worker being unavailable is a
readiness failure.

Restricted children use isolated process groups. Timeout and cancellation send
SIGTERM to the group, wait for a bounded grace period, then send SIGKILL and
reap the child. Captured output is bounded and has explicit truncation flags.
CPU time, open files, process count, file size, and address-space profiles are
applied where supported; unsupported limits are reported rather than claimed.
