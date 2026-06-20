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
- The canonical hash of tool name and normalized arguments must still match.
- Current tool policy is re-evaluated for the scoped agent.
- Level 3+ execution writes an audit start record before running; if this fails, the action does not run.
- Tool calls are recorded in SQLite.
- Failed executions consume the approval into a terminal state so replay is denied.
