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

Specialist agents may request tools only through structured tool calls. For
project-scoped tools, APRIL overwrites model-provided repository roots with the
trusted selected project. Agents never receive approval tokens, and Level 3+
tool requests suspend execution until the user approves the exact action.
