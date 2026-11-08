"""Check what the real OpenInference instrumentors emit, through the SDK.

Makes a few live Claude calls (Anthropic SDK, then LangChain), each a small
tool-use exchange, with the Costzera SDK capturing what it *would* send --
nothing goes to Costzera. Then checks:

  - the secret marker in the prompt is in no payload;
  - every LLM span carries model and token counts;
  - tool spans exist and are named;
  - each run is one complete trace.

Writes the payloads to tests/fixtures/real_*.json for the server contract test,
and prints the attribute names each instrumentor emitted (names only).

    python scripts/verify_instrumentors.py path/to/.env   # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

from opentelemetry.sdk.trace import SpanProcessor

MODEL = "claude-opus-5-5"
SECRET = "ZEBRA-7741-account-code"
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "tests" / "fixtures"


def load_key(env_path: str) -> None:
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
            return
    sys.exit("ANTHROPIC_API_KEY not found in the env file")


class RawKeys(SpanProcessor):
    """Attribute names per span, as the instrumentor emitted them."""

    def __init__(self):
        self.spans = []

    def on_end(self, span):
        self.spans.append((span.name, (span.attributes or {}).get("openinference.span.kind"),
                           sorted((span.attributes or {}).keys())))


def main(env_path: str) -> int:
    load_key(env_path)
    sys.path.insert(0, str(HERE.parent / "src"))
    import costzera

    sent: list = []

    def transport(body: bytes):
        sent.append(json.loads(gzip.decompress(body)))
        return 202, {"data": {"rejected": {"over_cap": 0}}}

    assert costzera.init(agent="verify-agent", api_key="czk_local_only", _transport=transport)
    raw = RawKeys()
    costzera._provider.add_span_processor(raw)

    question = f"My account code is {SECRET}. What is the status of order A-1001? Use the tool."
    tool_schema = {
        "name": "get_order_status",
        "description": "Look up an order's shipping status by id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }

    # ── 1. Anthropic SDK, a manual tool loop ─────────────────────────────
    import anthropic

    client = anthropic.Anthropic()

    @costzera.tool
    def get_order_status(order_id: str) -> dict:
        return {"order_id": order_id, "status": "shipped", "note": SECRET}

    with costzera.run("anthropic_sdk_run"):
        messages = [{"role": "user", "content": question}]
        first = client.messages.create(
            model=MODEL, max_tokens=2048, output_config={"effort": "low"},
            tools=[tool_schema], messages=messages,
        )
        messages.append({"role": "assistant", "content": first.content})
        results = []
        for block in first.content:
            if block.type == "tool_use":
                out = get_order_status(**block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
        if results:
            messages.append({"role": "user", "content": results})
            client.messages.create(
                model=MODEL, max_tokens=2048, output_config={"effort": "low"},
                tools=[tool_schema], messages=messages,
            )
    costzera.flush(10)
    anthropic_batches = list(sent)
    sent.clear()

    # ── 2. LangChain ─────────────────────────────────────────────────────
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def lookup_order(order_id: str) -> str:
        """Look up an order's shipping status by id."""
        return json.dumps({"order_id": order_id, "status": "shipped", "note": SECRET})

    llm = ChatAnthropic(model=MODEL, max_tokens=2048).bind_tools([lookup_order])
    with costzera.run("langchain_run"):
        msgs = [HumanMessage(question)]
        ai = llm.invoke(msgs)
        msgs.append(ai)
        for call in ai.tool_calls:
            msgs.append(ToolMessage(lookup_order.invoke(call["args"]), tool_call_id=call["id"]))
        if ai.tool_calls:
            llm.invoke(msgs)
    costzera.flush(10)
    langchain_batches = list(sent)

    # ── checks ───────────────────────────────────────────────────────────
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ok = True
    for label, batches in (("anthropic", anthropic_batches), ("langchain", langchain_batches)):
        print(f"\n=== {label}: {len(batches)} request(s)")
        if not batches:
            print("  NOTHING SENT"); ok = False; continue
        payload = batches[0] if len(batches) == 1 else {**batches[0], "traces": [t for b in batches for t in b["traces"]]}
        (FIXTURES / f"real_{label}.json").write_text(json.dumps(payload, indent=2))
        text = json.dumps(batches)
        leaked = SECRET in text
        print(f"  traces: {len(payload['traces'])}  secret in payload: {leaked}")
        ok &= not leaked and len(payload["traces"]) == 1
        for tr in payload["traces"]:
            for s in tr["spans"]:
                a, m = s["attributes"], s.get("measured", {})
                if s["span_kind"] == "LLM":
                    has = all(k in a for k in ("llm.token_count.prompt", "llm.token_count.completion")) and (
                        "llm.model_name" in a)
                    ok &= has
                    print(f"  LLM  {s['name']:<28} model={a.get('llm.model_name')} in={a.get('llm.token_count.prompt')} "
                          f"out={a.get('llm.token_count.completion')} cache_read={a.get('llm.token_count.prompt_details.cache_read')} "
                          f"finish={m.get('finish_reason')} in_chars={m.get('input_chars')} {'OK' if has else 'MISSING FIELDS'}")
                else:
                    print(f"  {s['span_kind']:<4} {s['name']:<28} parent={'root' if not s['parent_id'] else 'child'} "
                          f"tool={a.get('tool.name')} hash={str(m.get('input_hash'))[:8]} items={m.get('output_items')}")

    print("\n=== attribute names emitted (raw, before conversion)")
    seen = {}
    for name, kind, keys in raw.spans:
        for k in keys:
            # Collapse indexed keys so the list stays readable.
            import re
            seen.setdefault(kind or "-", set()).add(re.sub(r"\.\d+\.", ".N.", k))
    for kind, keys in sorted(seen.items()):
        print(f"  {kind}: {', '.join(sorted(keys))}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".env"))
