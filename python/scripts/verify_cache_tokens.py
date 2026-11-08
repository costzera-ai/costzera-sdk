"""Does the Anthropic instrumentor report cached tokens? Two calls sharing a
cached ~6k-token system prompt; the second should read from the cache."""
import gzip, json, os, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.startswith("ANTHROPIC_API_KEY="):
        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import costzera, anthropic
sent = []
costzera.init(agent="cache-check", api_key="czk_local_only",
              _transport=lambda b: (sent.append(json.loads(gzip.decompress(b))), (202, {}))[1])
system = [{"type": "text", "text": "Reference notes. " + " ".join(f"Fact {i}: item {i} ships in {i % 7 + 1} days." for i in range(900)),
           "cache_control": {"type": "ephemeral"}}]
client = anthropic.Anthropic()
usage = []
with costzera.run("cache_run"):
    for q in ("How long does item 5 take?", "How long does item 12 take?"):
        r = client.messages.create(model="claude-opus-5-5", max_tokens=1024, output_config={"effort": "low"},
                                   system=system, messages=[{"role": "user", "content": q}])
        usage.append({k: getattr(r.usage, k, None) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")})
costzera.flush(10)
print("API usage:", usage)
for s in sent[0]["traces"][0]["spans"]:
    if s["span_kind"] == "LLM":
        print("span tokens:", {k: v for k, v in s["attributes"].items() if k.startswith("llm.token_count")})
