# Costzera for Python

Find and fix what makes your AI agents slow and expensive, without a tracing
tool and without sending us your prompts.

```bash
pip install "costzera[openai]"      # or [anthropic], [langchain], [llama-index], ...
export COSTZERA_API_KEY=czk_...      # Settings → API keys → Send traces
```

```python
import costzera
costzera.init(agent="support-agent")
```

That is the whole setup for agents built on OpenAI, Anthropic, LangChain,
LlamaIndex, LiteLLM, CrewAI, Bedrock, the OpenAI Agents SDK or DSPy: their model,
tool and retrieval calls are traced automatically.

## Hand-written agents

```python
@costzera.tool
def fetch_order(order_id: str) -> dict: ...

with costzera.run("handle_ticket"):      # one trace per run of the agent
    with costzera.step("plan"):           # a named stage within it
        ...
```

Group each run of your agent in `costzera.run()`. Without it, every model call
is its own trace, and Costzera cannot see loops, retries or repeated tool calls
across a run.

## What leaves your process

Costzera receives the model, token counts, timings, errors, tool names, and the
*sizes* and *fingerprints* of inputs and outputs. It never receives prompts,
completions, tool arguments, tool results or retrieved documents. User and
session ids are sent as fingerprints.

This is enforced by allowlist in [`costzera/_wire.py`](https://github.com/costzera-ai/costzera-sdk/blob/main/python/src/costzera/_wire.py):
a span is converted into what is sent, and only the attributes named there are
copied. Anything else, including anything a framework adds in a future
release, is never read into the payload.

Exception messages are not sent by default, because they often quote the
input that caused them. Only the exception's type is. Pass
`error_messages=True` to send them, cut to 500 characters.

## Already using OpenTelemetry?

`init()` adds Costzera to your existing tracer provider instead of replacing
it, so your current exporters keep working. To send the same traces to
Langfuse, Phoenix or anything else, add their exporter as you normally would.

## Options

| `init(...)` | Default | |
|---|---|---|
| `agent` | `COSTZERA_AGENT` | The name the agent appears under in Costzera |
| `api_key` | `COSTZERA_API_KEY` | A key with the Send traces scope |
| `sample_rate` | `1.0` | Share of successful traces kept. Traces with an error are always kept |
| `instrument` | `True` | Trace installed frameworks automatically |
| `error_messages` | `False` | Also send exception messages |

Without a key, `init()` logs once and does nothing.

## Behaviour you can rely on

- Only whole traces are sent, once their root span finishes.
- Tracing never raises into your code. Export runs on a background thread with
  bounded memory; if Costzera is unreachable, traces are dropped, not queued
  forever.
- Traces are batched, at most one request every 15 seconds per process under
  steady load.
- `costzera.flush()` sends what has finished, for short scripts and serverless
  functions. It also runs at exit.

## Licence

MIT
