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

Every tool execution is scoped by `ToolExecutionContext`. For project-scoped
tools, APRIL overwrites or derives roots, working directories, and repository
arguments from the selected project. Direct API calls cannot point repository
tools at arbitrary unregistered paths. Recorded tool-call rows use the
authoritative permission decision from the registry, not executor-reported
metadata.

Specialist agents never execute tools directly. They emit structured
`tool_request` JSON, and APRIL application code normalizes the args, evaluates
policy, and either executes a low-risk tool or creates an exact approval.

Approval execution is one-time and exact-action:

- APRIL reloads the approval record before execution.
- The canonical hash of tool name, normalized arguments, and immutable approval
  metadata must still match.
- Structured-agent approvals also bind the approval to the suspended run:
  agent run ID, conversation ID, project ID, agent ID, model ID, tool name,
  canonical args hash, request ID, and iteration.
- Current tool policy is re-evaluated for the scoped agent.
- Patch approvals bind an APRIL-owned immutable artifact ID, patch SHA-256,
  exact byte length, normalized affected paths, selected project ID, repository
  root, Git HEAD when available, working-tree/index state digest when available,
  expected side effects, expiry, and approval ID.
- The patch artifact is stored under `data/artifacts/patches/` and may live
  outside the selected repository. Patch target paths may not.
- Immediately before patch application, APRIL loads the approved artifact bytes,
  recalculates SHA-256, parses every target path again, rejects traversal,
  absolute patch paths, symlink escapes, `.git` internals, sensitive paths, and
  out-of-project targets, then runs `git -C REPO apply --check -` and
  `git -C REPO apply -` against those same bytes.
- Git commit approvals bind the exact staged diff digest, staged tree ID, commit
  message, and repository identity. APRIL recalculates the staged state
  immediately before `git commit` and rejects changed staged content.
- Level 3+ execution writes an audit start record before running; if this fails, the action does not run.
- Tool calls are recorded in SQLite.
- Failed executions consume the approval into a terminal state so replay is denied.

For structured specialist approvals, `/tools/approve` first validates that the
conversation and project still exist. It then executes the exact approved tool
once, appends only the sanitized result to the saved loop messages, marks the
suspended state resumed, and continues the same agent run until final answer,
structured error, or another approval. `/tools/deny` marks the suspended run
denied and does not execute the tool. Expired, consumed, replayed, missing-scope,
or tampered approvals do not resume.

Project-scoped tool arguments from models are advisory. When a trusted
`project_id` is selected, APRIL derives repository roots from that project and
does not allow model-provided `repo_path`, `project_path`, `root`, or tool
`path` values to escape it.

The command policy permits only exact executable/subcommand patterns. `python
-m` is restricted to explicitly allowlisted modules, and package installers,
shell interpreters, shell metacharacters, pipes, redirects, command
substitution, environment-prefix execution, and executable paths are denied.
`run_command` always runs with the selected project as cwd.

`configs/tools.yaml` is the active source for command allowlists and local
opener allowlists. YAML can add safe command entries only within permanent hard
constraints; it cannot enable shells, package managers, arbitrary executable
paths, shell syntax, or arbitrary `python -m` modules.

`open_app` is a Level 4 system action. It accepts only configured plain
application names and, after exact approval, executes `/usr/bin/open -a APP`
with an argv list on macOS. `open_url` is a Level 5 external action. It requires
`permissions.yaml` `external_actions_enabled: true`, accepts only normalized
`http`/`https` URLs without credentials, and executes `/usr/bin/open URL` with
an argv list after exact approval. External actions are checked both before
creating approval requests and again before approved execution.
