# Threat Model

Main boundaries:

- user input is untrusted
- model output is untrusted
- retrieved local files are untrusted
- repository content is untrusted
- command output is untrusted

Controls:

- model output cannot execute tools directly
- deterministic permission engine is authoritative
- configured allowed filesystem roots are enforced after symlink resolution
- sensitive paths are denied
- shell execution uses argv arrays and `shell=False`
- Level 3+ operations require exact-action one-time approvals
- approved tools are revalidated against current policy before execution
- risky approved tools audit a start record before running and consume approvals after success or failure
- repository operations require explicit project selection and allowed-root validation
- retrieved memory and indexed repository chunks are marked as context, not instructions
- external actions are disabled by default
- bearer tokens and credential-like values are redacted from audit logs
