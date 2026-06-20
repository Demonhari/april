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

Specialist agents use `StructuredAgentLoop` by default. `/chat` can still use
the Brain to choose an agent, but once a specialist is selected the execution
path is the structured loop. `/agents/run` requires an explicit agent and also
uses the structured loop. General Agent simple chat remains a direct model
response.

Specialist agents may request tools only through structured tool calls. For
project-scoped tools, APRIL derives repository roots from the trusted selected
project and rejects model-supplied absolute overrides. Agents never receive
approval tokens, and Level 3+ tool requests suspend execution until the user
approves the exact action.

Structured specialist loop iterations must return exactly one strict JSON
object:

- `final_answer`
- `tool_request`
- `approval_required`
- `structured_error`

APRIL permits one repair attempt for malformed structured output. Tool output
fed back to a model is sanitized and truncated. Reasoning agents without an
explicit configured model return `unavailable` instead of silently using another
model.

Suspension stores sanitized loop messages and the exact pending tool request.
After approval, APRIL appends a sanitized tool result and resumes the same run
from the next iteration. A second Level 3+ tool request can suspend the same run
again with a new approval ID.
