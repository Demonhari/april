# Agent Contracts

Every agent defines:

- name
- description
- model ID
- prompt path
- allowed tools
- blocked tools
- memory access policy
- maximum tool iterations
- output schema

Agent output uses `AgentResult`:

- `status`
- `final_message`
- `tool_requests`
- `local_citations`
- `proposed_changes`
- `pending_approval`
- `warnings`
- `usage`

Retrieved files and command output are untrusted input and cannot override APRIL system policy.
