"""Collect finished spans into whole traces and send them, off the hot path.

Two rules shape this module:

1. **Only whole traces are sent.** Costzera builds each trace's totals from the
   spans in one request, so a trace split across two requests would be stored
   with partial totals. Spans are held per trace until its root ends, plus a
   short grace period for children that finish just after it.

2. **The host application is never affected.** Nothing here raises into the
   caller. Export runs on one daemon thread; memory is bounded (oldest traces
   are dropped first); every request has a timeout; failures are logged once,
   not repeatedly.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from ._wire import convert

logger = logging.getLogger("costzera")

# A transport takes a gzipped JSON body and returns (status, parsed body).
Transport = Callable[[bytes], "tuple[int, dict]"]


@dataclass
class Settings:
    api_key: str
    agent: str
    endpoint: str
    sdk_version: str
    sample_rate: float = 1.0
    error_messages: bool = False
    # Batching and limits.
    # A trace waits at most this long for others to share its request. Each
    # request is a function invocation on our side, so batching is what keeps
    # a busy agent from costing a request per trace.
    flush_interval_s: float = 15.0
    grace_s: float = 1.0
    max_traces_per_request: int = 200
    max_request_bytes: int = 1_000_000
    max_spans_per_trace: int = 2000
    max_open_traces: int = 10_000
    max_queued_traces: int = 2_000
    stale_after_s: float = 600.0
    timeout_s: float = 5.0
    retries: int = 3


@dataclass
class _Open:
    spans: list = field(default_factory=list)
    has_error: bool = False
    dropped_spans: int = 0
    started: float = field(default_factory=time.monotonic)
    root_ended_at: Optional[float] = None


def http_transport(settings: Settings) -> Transport:
    def send(body: bytes) -> "tuple[int, dict]":
        req = urllib.request.Request(
            settings.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "User-Agent": f"costzera-python/{settings.sdk_version}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=settings.timeout_s) as res:
                return res.status, _json(res.read())
        except urllib.error.HTTPError as e:
            return e.code, _json(e.read())
    return send


def collapse_nested_llm(spans: list) -> list:
    """One model call, recorded once.

    Instrumenting a framework and the client it wraps -- LangChain's
    ChatAnthropic over the Anthropic SDK, say -- records each call twice: a
    framework LLM span and a client LLM span with the same tokens, the client's
    inside the framework's in time. Counted as is, cost and call counts double.

    The pair is not always parent and child: LangChain's instrumentor does not
    make its span current, so the client's span lands beside it, under the same
    parent. Either way the signal is the same -- an LLM span whose time window
    holds another LLM span with identical token counts. The inner one is the
    request that went over the wire, so it is kept; the wrapper is dropped, its
    children move to its parent, and its step name passes to the inner span
    when that has none.
    """
    def tokens(s):
        a = s["attributes"]
        return (a.get("llm.token_count.prompt"), a.get("llm.token_count.completion"))

    # In the order spans ended. A call ends before the wrapper around it, so the
    # inner span is always the earlier one -- which also settles ties, where
    # both fall in the same millisecond and each looks like it holds the other.
    llms = [s for s in spans if s["span_kind"] == "LLM" and s.get("end_time")]
    inner_of: dict = {}
    for i, outer in enumerate(llms):
        if tokens(outer) == (None, None):
            continue
        for inner in llms[:i]:
            if inner["span_id"] in inner_of or any(v is inner for v in inner_of.values()):
                continue
            related = (inner.get("parent_id") == outer["span_id"]
                       or inner.get("parent_id") == outer.get("parent_id"))
            within = (outer["start_time"] <= inner["start_time"]
                      and inner["end_time"] <= outer["end_time"])
            if related and within and tokens(inner) == tokens(outer):
                inner_of[outer["span_id"]] = inner
                break
    if not inner_of:
        return spans

    by_id = {s["span_id"]: s for s in spans}
    out = []
    for s in spans:
        if s["span_id"] in inner_of:
            continue
        parent = s.get("parent_id")
        if parent in inner_of:
            s = {**s, "parent_id": by_id[parent].get("parent_id")}
        # The kept span inherits the wrapper's step name.
        wrapper = next((by_id[w] for w, i in inner_of.items() if i is by_id.get(s["span_id"])), None)
        if wrapper is not None:
            meta = wrapper["attributes"].get("metadata")
            if meta and "metadata" not in s["attributes"]:
                s = {**s, "attributes": {**s["attributes"], "metadata": meta}}
        out.append(s)
    return out


def _json(raw: bytes) -> dict:
    try:
        parsed = json.loads(raw or b"{}")
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


class TraceExporter(SpanProcessor):
    """A SpanProcessor that buffers by trace and exports complete traces."""

    def __init__(self, settings: Settings, transport: Optional[Transport] = None):
        self.settings = settings
        self._transport = transport or http_transport(settings)
        self._lock = threading.Condition()
        self._open: "OrderedDict[int, _Open]" = OrderedDict()
        self._ready: list = []          # complete traces waiting to be sent
        self._ready_since: Optional[float] = None
        self._flush_now = False
        self._disabled: Optional[str] = None
        self._warned: set = set()
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="costzera-export", daemon=True)
        self._thread.start()

    # ── SpanProcessor ─────────────────────────────────────────────────────

    def on_start(self, span, parent_context=None) -> None:  # noqa: D401
        return None

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._on_end(span)
        except Exception:  # never into the host
            self._warn_once("convert", "Costzera could not record a span; it was skipped.")

    def shutdown(self) -> None:
        self.force_flush(timeout_millis=3000)
        with self._lock:
            self._stopping = True
            self._lock.notify_all()

    def force_flush(self, timeout_millis: int = 3000) -> bool:
        deadline = time.monotonic() + timeout_millis / 1000
        with self._lock:
            # Anything whose root has ended is due now, grace or not.
            for trace_id, t in list(self._open.items()):
                if t.root_ended_at is not None:
                    self._promote(trace_id)
            self._flush_now = True
            self._lock.notify_all()
        while time.monotonic() < deadline:
            with self._lock:
                if not self._ready:
                    return True
            time.sleep(0.05)
        return False

    # ── buffering ────────────────────────────────────────────────────────

    def _on_end(self, span: ReadableSpan) -> None:
        if self._disabled:
            return
        ctx = span.get_span_context()
        trace_id = ctx.trace_id
        wire = convert(span, error_messages=self.settings.error_messages)
        # A root is a span with no parent, or one whose parent lives in another
        # process: that is where this process's part of the trace ends.
        is_root = span.parent is None or span.parent.is_remote

        with self._lock:
            t = self._open.get(trace_id)
            if t is None:
                if len(self._open) >= self.settings.max_open_traces:
                    self._open.popitem(last=False)  # drop the oldest
                    self._warn_once("open", "Costzera dropped traces: too many open at once.")
                t = self._open[trace_id] = _Open()
            if len(t.spans) < self.settings.max_spans_per_trace:
                t.spans.append(wire)
            else:
                t.dropped_spans += 1
            t.has_error = t.has_error or wire["status_code"] == "ERROR"
            if is_root:
                t.root_ended_at = time.monotonic()
                self._lock.notify_all()

    def _promote(self, trace_id: int) -> None:
        """Move one finished trace to the send queue (lock held)."""
        t = self._open.pop(trace_id, None)
        if t is None:
            return
        # Tail sampling, decided with the whole trace in hand: a trace with any
        # failure is always kept.
        if not t.has_error and self.settings.sample_rate < 1.0:
            if random.random() >= self.settings.sample_rate:
                return
        if len(self._ready) >= self.settings.max_queued_traces:
            self._ready.pop(0)
            self._warn_once("queue", "Costzera dropped traces: the send queue is full.")
        if not self._ready:
            self._ready_since = time.monotonic()
        self._ready.append({"trace_id": format(trace_id, "032x"), "spans": collapse_nested_llm(t.spans)})

    def _sweep(self) -> None:
        """Promote traces past their grace period; drop ones that never ended."""
        now = time.monotonic()
        for trace_id, t in list(self._open.items()):
            if t.root_ended_at is not None and now - t.root_ended_at >= self.settings.grace_s:
                self._promote(trace_id)
            elif t.root_ended_at is None and now - t.started >= self.settings.stale_after_s:
                del self._open[trace_id]

    # ── sending ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            with self._lock:
                self._lock.wait(timeout=min(self.settings.grace_s, self.settings.flush_interval_s))
                self._sweep()
                if self._stopping and not self._ready:
                    return
                due = (
                    self._flush_now
                    or self._stopping
                    or len(self._ready) >= self.settings.max_traces_per_request
                    or (self._ready_since is not None
                        and time.monotonic() - self._ready_since >= self.settings.flush_interval_s)
                )
                batch = self._take_batch() if due and self._ready else []
                if not self._ready:
                    self._flush_now = False
                    self._ready_since = None
                elif batch:
                    # What is left has waited as long as the batch just taken.
                    self._ready_since = time.monotonic()
            if batch:
                self._send(batch)

    def _take_batch(self) -> list:
        """Up to the request limits, from the front of the queue (lock held)."""
        batch, size = [], 0
        while self._ready and len(batch) < self.settings.max_traces_per_request:
            nxt = self._ready[0]
            cost = len(json.dumps(nxt, separators=(",", ":")))
            if batch and size + cost > self.settings.max_request_bytes:
                break
            batch.append(self._ready.pop(0))
            size += cost
        return batch

    def _send(self, traces: list) -> None:
        payload = {
            "sdk": {"name": "costzera-python", "version": self.settings.sdk_version},
            "agent": self.settings.agent,
            "traces": traces,
        }
        body = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        delay = 1.0
        for attempt in range(self.settings.retries + 1):
            try:
                status, reply = self._transport(body)
            except Exception:
                status, reply = 0, {}
            if 200 <= status < 300:
                over = ((reply.get("data") or {}).get("rejected") or {}).get("over_cap", 0)
                if over:
                    self._warn_once("cap", f"Costzera refused {over} trace(s): this month's plan limit is reached.")
                return
            if status in (401, 403, 409):
                # Not something a retry fixes: a bad key, a key that cannot
                # ingest, the plan's agent limit, or a name taken by a synced
                # agent. Stop sending rather than retry forever.
                self._disabled = (reply.get("error") or f"HTTP {status}")
                logger.warning("Costzera stopped sending traces: %s", self._disabled)
                return
            if 400 <= status < 500 and status != 429:
                self._warn_once(f"http{status}", f"Costzera refused a batch ({status}): {reply.get('error')}")
                return
            if attempt < self.settings.retries:
                time.sleep(delay)
                delay *= 2
        self._warn_once("unreachable", "Costzera could not be reached; some traces were not sent.")

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)
