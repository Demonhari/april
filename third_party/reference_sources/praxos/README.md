# Praxos

**Experience OS for AI employees.**

Praxos is not a generic memory layer. It is the flight recorder and learning system for AI agents that do real work.

When an agent completes a task, gets corrected, violates a policy, or produces a bad outcome, Praxos turns that episode into reusable experience: lessons, action checks, policies, and regression evidence.

> Humans gain experience. Agents should too.

## Why This Exists

Companies are moving from chatbots to AI employees that answer customers, update CRMs, triage tickets, write code, and operate internal tools. These agents still repeat the same mistakes because their experience is not captured as an operating asset.

Praxos records:

- what the agent tried to do
- what context and action it used
- what happened
- what a human corrected
- what should happen next time
- which future actions should be warned or blocked

The result is an experience layer that makes agents safer and better over time.

## YC Wedge

Start with **customer-facing AI agents for B2B SaaS**.

The first painful workflow:

> "Before this support or success agent replies to an enterprise customer, check whether the answer contradicts past commitments, escalations, product decisions, or human corrections."

Praxos can warn or block risky actions before they touch customers.

## Core Objects

```text
Episode
  A real task attempt: task, action, outcome, feedback, sources.

EvidenceReceipt
  Proof for a memory: source URI, snippet, observed_at, confidence, metadata.

Lesson
  Reusable experience generated from episodes and corrections.

Policy
  A company rule that can warn or block future agent actions.

Business Context
  Account, Customer, Commitment, Escalation, Decision.

ActionCheck
  A pre-flight decision: allow, warn, or block, with evidence.

ReviewItem
  Human approval queue for newly compiled lessons.
```

## Quick Start

```bash
pip install -e .
praxos demo --story
```

By default Praxos stores its ledger in `~/.praxos/praxos.db`. In locked-down
environments where the home directory is read-only, it falls back to
`.praxos/praxos.db` in the current project. You can also set `PRAXOS_DATA_DIR`
or pass `--db` explicitly.

The demo writes to an isolated demo workspace each run, so repeated demos stay
clean instead of duplicating policy matches.

The story demo prints the investor-facing moment:

```text
Praxos story demo

1. A support agent is about to promise Enterprise A that Feature X ships by Friday.
2. A human previously corrected this exact failure.
3. Praxos compiled that failure into a lesson, evidence receipts, and a review item.
4. Praxos checks the future action before it reaches the customer.

Decision: BLOCK
```

Record an agent failure and turn it into a lesson:

```bash
praxos record \
  --agent support-agent \
  --task "Reply to Enterprise A asking when Feature X ships" \
  --action "Tell them Feature X will ship by Friday" \
  --outcome failure \
  --feedback "Never promise delivery dates. Product moved Feature X to Q3 in the April 12 escalation."
```

Add a hard policy:

```bash
praxos policy add \
  --name "No delivery promises" \
  --trigger "ship by friday delivery date promise eta" \
  --severity block \
  --instruction "Do not promise delivery dates unless there is an approved product source."
```

Check a future action before the agent sends it:

```bash
praxos check \
  --task "Reply to Enterprise A about Feature X" \
  --action "Say we can deliver Feature X by Friday"
```

## MCP With mcp-use

Praxos uses **mcp-use** to expose its real MCP server. We use `mcp-use` because it provides a full-stack MCP framework for Python, including server tools via the `MCPServer` API, while staying compatible with MCP clients.

Reference: <https://docs.mcp-use.com/python/server>

Start Praxos as an MCP server over stdio:

```bash
praxos mcp
# or
praxos-mcp
```

Start Praxos as an MCP server over streamable HTTP:

```bash
praxos mcp --transport streamable-http --host 127.0.0.1 --port 8766
```

MCP tools exposed through `mcp-use`:

```text
get_experience
check_action
record_outcome
learn_from_feedback
```

Example Claude/Cursor-style local MCP config:

```json
{
  "mcpServers": {
    "praxos": {
      "command": "praxos-mcp",
      "args": ["--db", ".praxos/praxos.db"]
    }
  }
}
```

## Local JSON Tool Server

For debugging without an MCP client, Praxos also ships a small local JSON tool server:

```bash
praxos server --host 127.0.0.1 --port 8765
```

JSON tool endpoints:

```text
GET  /tools
POST /tools/get_experience
POST /tools/check_action
POST /tools/record_outcome
POST /tools/learn_from_feedback
```

Example:

```bash
curl -X POST http://127.0.0.1:8765/tools/check_action \
  -H "Content-Type: application/json" \
  -d '{"task":"Reply to Enterprise A about Feature X","action":"Promise Friday delivery"}'
```

Example result:

```json
{
  "decision": "block",
  "reasons": [
    "Policy matched: No delivery promises",
    "Relevant lesson: Reply to Enterprise A asking when Feature X ships"
  ]
}
```

## SDK

```python
from praxos import ExperienceLedger

ledger = ExperienceLedger(".praxos/praxos.db")

episode = ledger.record_episode(
    agent_id="support-agent",
    task="Reply to Enterprise A asking when Feature X ships",
    action="Tell them Feature X will ship by Friday",
    outcome="failure",
    human_feedback="Never promise delivery dates. Product moved Feature X to Q3.",
)

check = ledger.check_action(
    task="Reply to Enterprise A about Feature X",
    action="Say we can deliver Feature X by Friday",
)

print(check.decision)
print(check.reasons)
```

## Business Context

For the first wedge, Praxos models B2B SaaS customer-facing context:

```bash
praxos business account --name "Enterprise A" --external-ref crm://enterprise-a
praxos business commitment --account acct_... \
  --description "Do not promise Friday delivery for Feature X." \
  --source-uri crm://enterprise-a/commitments/feature-x
praxos business escalation --account acct_... \
  --summary "Feature X timing caused a prior customer escalation." \
  --severity high
praxos business decision --account acct_... \
  --decision "Product moved Feature X to Q3; avoid near-term delivery promises."
```

When an agent calls `check_action`, matching commitments, escalations, and decisions can warn or block future action.

## Human Review

Every automatically compiled lesson enters a review queue.

```bash
praxos review list
praxos review approve rev_...
praxos review reject rev_...
```

Rejecting a lesson archives it. Approved lessons remain active and auditable.

## Matching

By default Praxos uses dependency-free hybrid matching:

- token overlap
- character trigram similarity
- small domain synonym expansion
- phrase matching

For semantic matching, plug in any reranker:

```bash
export PRAXOS_RERANK_URL=http://localhost:9000/rerank
export PRAXOS_RERANK_TOKEN=optional-token
```

The reranker endpoint receives `{"query":"...","documents":["..."]}` and returns `{"scores":[0.0,0.9]}`.

## What Makes It Different

Praxos is not trying to remember everything.

It is trying to make AI workers learn from work:

- every correction becomes future behavior
- every policy has enforcement
- every check has evidence
- every failure can become a regression case
- every agent gets better without retraining a model

## Current Status

This repo is a clean MVP:

- SQLite experience ledger
- CLI
- Python SDK
- MCP server built with `mcp-use`
- dependency-free local HTTP tool server
- evidence receipts with source snippets and confidence
- automatic lesson compiler from failed/corrected episodes
- policy-based action checks
- B2B SaaS objects: accounts, customers, commitments, escalations, decisions
- human review queue
- unit tests

Next product steps:

- integrations for Slack, Zendesk, Linear, Salesforce, Gmail
- temporal commitments and contradiction detection
- hosted team workspace
- eval suite for repeated agent failures
