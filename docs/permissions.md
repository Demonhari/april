# Permissions

Permission levels:

- Level 0: no tools
- Level 1: read-only filesystem and Git inspection
- Level 2: safe local writes such as notes, reminders, draft patch files, and local indexes
- Level 3: code writes, patch application, tests, formatting, and commits
- Level 4: system actions
- Level 5: external actions

APRIL requires explicit approval for Level 3 and above. The model cannot lower permission level or risk. The deterministic permission engine combines:

- model-declared risk
- tool policy risk
- argument-sensitive risk

Unknown tools are denied. Tools not allowed for the selected agent are denied.

Approval execution is one-time and exact-action:

- APRIL reloads the approval record before execution.
- The canonical hash of tool name, normalized arguments, and immutable approval
  metadata must still match.
- Current tool policy is re-evaluated for the scoped agent.
- Patch approvals bind the patch SHA-256, normalized affected paths, repository
  root, Git HEAD when available, working-tree/index state digest when available,
  expected side effects, expiry, and approval ID.
- Immediately before patch application, APRIL re-reads the patch, recalculates
  SHA-256, parses every target path again, rejects traversal, absolute patch
  paths, symlink escapes, `.git` internals, sensitive paths, and out-of-project
  targets, then runs `git apply --check`.
- Git commit approvals bind the exact staged diff digest. APRIL recalculates the
  staged diff immediately before `git commit` and rejects changed staged content.
- Level 3+ execution writes an audit start record before running; if this fails, the action does not run.
- Tool calls are recorded in SQLite.
- Failed executions consume the approval into a terminal state so replay is denied.

Project-scoped tool arguments from models are advisory. When a trusted
`project_id` is selected, APRIL derives repository roots from that project and
does not allow model-provided `repo_path`, `project_path`, `root`, or tool
`path` values to escape it.

The command policy permits only exact executable/subcommand patterns. `python
-m` is restricted to explicitly allowlisted modules, and package installers,
shell interpreters, shell metacharacters, pipes, redirects, command
substitution, environment-prefix execution, and executable paths are denied.
