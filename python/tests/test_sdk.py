"""The SDK's promises, tested against a fake transport.

The four that matter most: content never leaves the process; only whole
traces are sent; nothing ever raises into the host; sampling never drops a
failure.
"""

import asyncio
import gzip
import json
import time
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode

import costzera
from costzera._pipeline import Settings, TraceExporter

SECRET = "TOP-SECRET-customer-4111-1111-1111-1111"


class FakeTransport:
    def __init__(self, status=202, reply=None, raises=False):
        self.status, self.reply, self.raises = status, reply or {"data": {}}, raises
        self.bodies = []

    def __call__(self, body: bytes):
        self.bodies.append(body)
        if self.raises:
            raise ConnectionError("down")
        return self.status, self.reply

    @property
    def payloads(self):
        return [json.loads(gzip.decompress(b)) for b in self.bodies]

    @property
    def traces(self):
        return [t for p in self.payloads for t in p["traces"]]

    def raw(self) -> str:
        return "".join(gzip.decompress(b).decode() for b in self.bodies)


def make(transport, **overrides):
    options = {"grace_s": 0.02, "flush_interval_s": 0.05, "retries": 0, **overrides}
    settings = Settings(
        api_key="czk_test", agent="test-agent", endpoint="http://unused", sdk_version="test",
        **options,
    )
    exporter = TraceExporter(settings, transport=transport)
    provider = TracerProvider()
    provider.add_span_processor(exporter)
    return exporter, provider


@pytest.fixture
def sdk():
    """Point the public helpers at a private provider, then put things back."""
    transport = FakeTransport()
    exporter, provider = make(transport)
    old = costzera._provider, costzera._exporter
    costzera._provider, costzera._exporter = provider, exporter
    yield transport, exporter, provider
    costzera._provider, costzera._exporter = old
    exporter.shutdown()


def settle(exporter):
    assert exporter.force_flush(2000)
    time.sleep(0.05)


# ── whole traces ─────────────────────────────────────────────────────────

def test_a_trace_is_sent_once_its_root_ends(sdk):
    transport, exporter, _ = sdk

    @costzera.tool
    def lookup(x):
        return {"x": x}

    with costzera.run("handle"):
        with costzera.step("plan"):
            lookup(1)
            time.sleep(0.1)
            assert transport.bodies == [], "sent before the root ended"
    settle(exporter)

    assert len(transport.traces) == 1
    spans = transport.traces[0]["spans"]
    kinds = {s["name"]: s["span_kind"] for s in spans}
    assert kinds == {"lookup": "TOOL", "plan": "CHAIN", "handle": "AGENT"}
    roots = [s for s in spans if s["parent_id"] is None]
    assert [r["name"] for r in roots] == ["handle"]
    assert transport.payloads[0]["agent"] == "test-agent"


def test_a_trace_whose_root_never_ends_is_never_sent():
    transport = FakeTransport()
    exporter, provider = make(transport, stale_after_s=0.1)
    tracer = provider.get_tracer("t")
    root = tracer.start_span("root")
    from opentelemetry import trace as t
    with tracer.start_as_current_span("child", context=t.set_span_in_context(root)):
        pass
    time.sleep(0.3)
    exporter.force_flush(200)
    assert transport.bodies == []
    assert exporter._open == {}, "a stale trace should be dropped, not held forever"
    exporter.shutdown()


# ── content stays in the process ─────────────────────────────────────────

def test_no_content_is_ever_sent(sdk):
    transport, exporter, provider = sdk
    tracer = provider.get_tracer("t")

    @costzera.tool
    def search(query, limit=3):
        return [{"doc": SECRET}]

    with costzera.run("handle"):
        search(SECRET, limit=2)
        # What an LLM instrumentor records, content included.
        with tracer.start_as_current_span("ChatAnthropic", attributes={
            "openinference.span.kind": "LLM",
            "llm.model_name": "claude-haiku-4-5",
            "llm.token_count.prompt": 1200,
            "llm.token_count.completion": 80,
            "input.value": json.dumps({"messages": [{"role": "user", "content": SECRET}]}),
            "output.value": json.dumps({"content": SECRET, "stop_reason": "end_turn"}),
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": SECRET,
            "llm.output_messages.0.message.content": SECRET,
            "llm.invocation_parameters": json.dumps({"system": SECRET}),
            "llm.prompt_template.variables": json.dumps({"name": SECRET}),
            "retrieval.documents.0.document.content": SECRET,
            "llm.tools.0.tool.json_schema": SECRET,
            "metadata": json.dumps({"pipeline_step": "answer", "note": SECRET}),
            "user.id": "jane@example.com",
            "session.id": SECRET,
            "something.new.in.v2": SECRET,
        }):
            pass
        try:
            with costzera.step("risky"):
                raise ValueError(f"bad input {SECRET}")
        except ValueError:
            pass
    settle(exporter)

    raw = transport.raw()
    assert transport.traces, "nothing was sent"
    assert SECRET not in raw
    assert "jane@example.com" not in raw

    llm = next(s for s in transport.traces[0]["spans"] if s["span_kind"] == "LLM")
    assert llm["attributes"]["llm.model_name"] == "claude-haiku-4-5"
    assert llm["attributes"]["llm.token_count.prompt"] == 1200
    assert llm["attributes"]["user.id"].startswith("h_")
    assert json.loads(llm["attributes"]["metadata"]) == {"pipeline_step": "answer"}
    assert llm["measured"]["finish_reason"] == "end_turn"
    assert llm["measured"]["input_chars"] > 0

    risky = next(s for s in transport.traces[0]["spans"] if s["name"] == "risky")
    assert risky["status_code"] == "ERROR"
    assert risky["status_message"] == "ValueError"


def test_measurements_are_taken_before_content_is_left_behind(sdk):
    transport, exporter, _ = sdk

    @costzera.tool
    def fetch(**kw):
        return [1, 2, 3]

    with costzera.run():
        fetch(a=1, b=2)
        fetch(b=2, a=1)   # same call, keys in another order
        fetch(a=1, b=3)
    settle(exporter)

    tools = [s for s in transport.traces[0]["spans"] if s["span_kind"] == "TOOL"]
    hashes = [t["measured"]["input_hash"] for t in tools]
    assert hashes[0] == hashes[1] != hashes[2]
    assert all(t["measured"]["output_items"] == 3 for t in tools)
    assert all(t["measured"]["input_chars"] > 0 for t in tools)


def test_async_tools_are_traced(sdk):
    transport, exporter, _ = sdk

    @costzera.tool(name="lookup_async")
    async def lookup(x):
        await asyncio.sleep(0)
        return x

    async def main():
        with costzera.run():
            assert await lookup(5) == 5

    asyncio.run(main())
    settle(exporter)
    assert [s["name"] for s in transport.traces[0]["spans"]][0] == "lookup_async"


# ── never hurts the host ─────────────────────────────────────────────────

def test_an_unreachable_server_never_raises_into_the_app():
    transport = FakeTransport(raises=True)
    exporter, provider = make(transport)
    tracer = provider.get_tracer("t")
    for _ in range(3):
        with tracer.start_as_current_span("root"):
            pass
    exporter.force_flush(500)
    exporter.shutdown()
    assert transport.bodies, "it should have tried"


def test_errors_inside_traced_code_still_reach_the_caller(sdk):
    @costzera.tool
    def boom():
        raise KeyError("x")

    with pytest.raises(KeyError):
        with costzera.run():
            boom()


def test_a_refused_key_stops_sending_rather_than_retrying_forever():
    transport = FakeTransport(status=403, reply={"error": "agent cap"})
    exporter, provider = make(transport)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("first"):
        pass
    settle(exporter)
    with tracer.start_as_current_span("second"):
        pass
    settle(exporter)
    assert len(transport.bodies) == 1
    exporter.shutdown()


def test_init_without_a_key_does_nothing(monkeypatch):
    monkeypatch.delenv("COSTZERA_API_KEY", raising=False)
    old = costzera._exporter
    costzera._exporter = None
    try:
        assert costzera.init(agent="x") is False
        assert costzera._exporter is None
    finally:
        costzera._exporter = old


# ── sampling and batching ────────────────────────────────────────────────

def test_sampling_drops_successes_but_never_failures():
    transport = FakeTransport()
    exporter, provider = make(transport, sample_rate=0.0)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("ok"):
        pass
    with tracer.start_as_current_span("failed") as span:
        span.set_status(Status(StatusCode.ERROR))
    settle(exporter)
    assert [t["spans"][0]["name"] for t in transport.traces] == ["failed"]
    exporter.shutdown()


def test_traces_are_batched_within_the_request_limits():
    transport = FakeTransport()
    exporter, provider = make(transport, max_traces_per_request=10, flush_interval_s=5.0)
    tracer = provider.get_tracer("t")
    for i in range(25):
        with tracer.start_as_current_span(f"t{i}"):
            pass
    settle(exporter)
    sizes = [len(p["traces"]) for p in transport.payloads]
    assert sum(sizes) == 25
    assert max(sizes) <= 10
    assert len(sizes) <= 4, f"{len(sizes)} requests for 25 traces"
    exporter.shutdown()


def test_write_a_sample_batch_for_the_server_contract_test(sdk):
    """Writes the payload the server's parser is checked against."""
    transport, exporter, provider = sdk
    tracer = provider.get_tracer("t")

    @costzera.tool
    def fetch_order(order_id):
        return {"status": "shipped"}

    with costzera.run("handle_ticket"):
        with costzera.step("triage"):
            with tracer.start_as_current_span("ChatOpenAI", attributes={
                "openinference.span.kind": "LLM", "llm.model_name": "gpt-4o-mini",
                "llm.token_count.prompt": 900, "llm.token_count.completion": 60,
                "input.value": "hello", "output.value": json.dumps({"choices": [{"finish_reason": "stop"}]}),
            }):
                pass
        fetch_order("A1")
        fetch_order("A1")
    settle(exporter)

    out = Path(__file__).parent / "fixtures" / "sample_batch.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(transport.payloads[0], indent=2))
    assert SECRET not in out.read_text()


# ── one model call, recorded once ────────────────────────────────────────

def test_a_framework_wrapping_a_client_is_not_counted_twice(sdk):
    """LangChain's ChatAnthropic over the Anthropic SDK records each call twice."""
    transport, exporter, provider = sdk
    tracer = provider.get_tracer("t")
    llm = {"openinference.span.kind": "LLM", "llm.model_name": "m",
           "llm.token_count.prompt": 100, "llm.token_count.completion": 10}
    with costzera.run("r"):
        with tracer.start_as_current_span("ChatAnthropic", attributes={
            **llm, "metadata": json.dumps({"pipeline_step": "answer"})}):
            with tracer.start_as_current_span("messages.create", attributes=llm):
                pass
    settle(exporter)

    spans = transport.traces[0]["spans"]
    llms = [s for s in spans if s["span_kind"] == "LLM"]
    assert [s["name"] for s in llms] == ["messages.create"]
    root = next(s for s in spans if s["parent_id"] is None)
    assert llms[0]["parent_id"] == root["span_id"], "the inner call moves up to the wrapper's parent"
    assert json.loads(llms[0]["attributes"]["metadata"]) == {"pipeline_step": "answer"}


def test_a_wrapper_recorded_beside_its_client_call_is_not_counted_twice(sdk):
    """What LangChain's instrumentor actually produces: siblings, not nested."""
    transport, exporter, provider = sdk
    tracer = provider.get_tracer("t")
    from opentelemetry import trace as t
    llm = {"openinference.span.kind": "LLM", "llm.model_name": "m",
           "llm.token_count.prompt": 409, "llm.token_count.completion": 54}
    with costzera.run("r") as _:
        here = t.get_current_span()
        outer = tracer.start_span("ChatAnthropic", attributes=llm)   # not made current
        with tracer.start_as_current_span("messages.create", attributes=llm):
            time.sleep(0.002)
        outer.end()
        # A genuinely separate call with other tokens stays.
        with tracer.start_as_current_span("messages.create", attributes={**llm, "llm.token_count.prompt": 507}):
            pass
    settle(exporter)
    names = [s["name"] for s in transport.traces[0]["spans"] if s["span_kind"] == "LLM"]
    assert names == ["messages.create", "messages.create"]
    assert here is not None
