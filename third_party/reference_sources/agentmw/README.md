# agentmw

Open-source middleware that catches mid-run failures, compresses stale
context, and grows a private reasoning library across every agent run.
Works with any model, any framework. Apache-2.0.

```bash
pip install -e '.[all]'
agentmw demo
```

## What it does

| Layer | What it does | Default |
|---|---|---|
| **LLM monitor** | One call to your provider (Ollama / OpenAI / Anthropic / OpenRouter) classifies the latest turns for `loop`, `redundant_tool_call`, `contradiction`, `abandonment`, `hallucination`. | primary |
| **Heuristic monitor** | Deterministic regex checks (same categories, conservative). Runs as prefilter and as fallback when the LLM is unreachable. | fallback |
| **Compression** | Truncates stale `tool_result` blocks while keeping the most recent intact. | on |
| **Reasoning library** | SQLite-backed memory of patterns from past runs. Semantic recall via `fastembed` (BGE-small ONNX, 30 MB, local). Keyword fallback if embeddings unavailable. | on |
| **Time-travel CLI** | Walks a saved trace, names the **point of no return**, quantifies wasted tokens, and (with `--counterfactual`) asks the provider to simulate the divergent branch. | — |
| **Auto-record + auto-extract** | Every wrapped session is saved to disk; on completion, a background extractor distills 1–3 reusable patterns and adds them to the reasoning library. The loop closes itself. | on |
| **Circuit breaker** | Provider calls are short-circuited after 3 failures in 30s, cooldown 60s. Inner client is never slowed by a sick monitor. | on |
| **Telemetry** | Counters persisted to `~/.agentmw/telemetry.json`, flushed each call and at process exit. Inspect with `agentmw stats`. | on |
| **Async support** | `wrap_async()` for `AsyncAnthropic` / `AsyncOpenAI` clients. | — |

Nothing is hardcoded. All knobs live in `AgentmwConfig` and resolve
`defaults < ~/.agentmw/config.toml < env vars < explicit kwargs`.

## Install

```bash
pip install -e .                  # core only (heuristic + keyword recall)
pip install -e '.[semantic]'      # + semantic recall via fastembed
pip install -e '.[mcp]'           # + MCP server
pip install -e '.[all]'           # everything
```

## CLI

```bash
agentmw demo                                 # mock run; shows monitors + compression + memory
agentmw config show                          # print resolved config (env + file + defaults)
agentmw memory save --task "..." --pattern "..."
agentmw memory recall --task "..."
agentmw record --file run.json               # save a trace as a session
agentmw record < run.json                    # or read from stdin
agentmw sessions list
agentmw timeline run.json --judge            # time-travel through a raw trace
agentmw replay <session-id> --counterfactual # full replay + ghost branch
agentmw extract <session-id>                 # mine reusable patterns from a session
agentmw stats                                # telemetry counters
agentmw serve                                # MCP server (stdio)
```

Provider flags (work on `timeline`, `replay`, `config show`):

```bash
--provider {auto,ollama,openai,anthropic,openrouter,none}
--model    <model-name>
--no-llm   # heuristics only
```

## Config

`~/.agentmw/config.toml` (any subset; env vars override):

```toml
[monitors]
loop_threshold = 3

[compression]
keep_recent = 2

[memory]
semantic_threshold = 0.55
embedding_model = "BAAI/bge-small-en-v1.5"

[provider]
name = "anthropic"
model = "claude-haiku-4-5-20251001"
api_key = "..."   # or set AGENTMW_API_KEY

[pipeline]
use_llm = true
heuristics_prefilter = true
use_heuristics_fallback = true
```

Env vars: `AGENTMW_PROVIDER`, `AGENTMW_MODEL`, `AGENTMW_API_KEY`,
`AGENTMW_BASE_URL`, `AGENTMW_LOOP_THRESHOLD`, `AGENTMW_SEMANTIC_THRESHOLD`,
`AGENTMW_USE_LLM`, `AGENTMW_DB_PATH`, `AGENTMW_HOME`, …

## Library usage

```python
import anthropic
from agentmw import wrap
from agentmw.core.config import default_config

client = wrap(anthropic.Anthropic(), agentmw_config=default_config())
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    messages=[{"role": "user", "content": "Find the auth bug."}],
)

trace = client.config.traces[-1]
print(trace.monitors.triggered)
print(trace.compression.ratio, trace.recalled_patterns)

# Async clients:
import anthropic
from agentmw import wrap_async
aclient = wrap_async(anthropic.AsyncAnthropic())
# await aclient.messages.create(...)
```

## MCP

```bash
agentmw serve   # stdio; add this to your Claude Desktop / Cursor mcp config
```

Tools exposed: `agentmw_recall`, `agentmw_save`, `agentmw_check_trace`,
`agentmw_stats`.

## License

Apache-2.0.
