# Costzera SDK

Find and fix what makes your AI agents slow and expensive, without a tracing
tool and without sending us your prompts.

| Language | Package | |
|---|---|---|
| Python | [`costzera`](python/) | `pip install costzera` |
| TypeScript | coming | |

The SDK traces your agent with OpenTelemetry and the open-source
[OpenInference](https://github.com/Arize-ai/openinference) instrumentors, then
sends Costzera the shape of each run: models, token counts, timings, errors,
tool names, and the sizes and fingerprints of inputs and outputs. Prompts,
completions, tool arguments and results never leave your process. What is sent
is decided by an allowlist you can read in
[`python/src/costzera/_wire.py`](python/src/costzera/_wire.py).

Docs: [costzera.com/docs](https://costzera.com/docs#sdk)

## Licence

MIT
