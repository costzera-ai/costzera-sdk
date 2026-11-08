"""Turn a finished OpenTelemetry span into what is sent to Costzera.

This module is the privacy boundary, so it works by allowlist. A span is
*converted*, never edited: only the attributes named below are copied, and
everything else -- prompts, completions, tool arguments and results, retrieved
documents, invocation parameters, anything an instrumentor adds in a future
release -- is simply never read into the payload.

Content is still *measured* here, in the customer's process, before it is left
behind: character counts, a SHA-256 fingerprint of the input (duplicate
detection only needs fingerprints to agree within one trace), the number of
items returned, the finish reason. Those facts are what the optimizer needs.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

# Copied as they are. Model, provider, token counts and tool names are not
# content, and the server prices calls from them.
EXACT_KEYS = frozenset({
    "llm.model_name",
    "llm.request.model_name",
    "llm.response.model_name",
    "llm.provider",
    "llm.system",
    "tool.name",
    "agent.name",
    "graph.node.name",
    "gen_ai.system",
    "gen_ai.request.model",
    "gen_ai.response.model",
})
PREFIX_KEYS = ("llm.token_count.",)

# Identifiers that can be personal data (a user id is often an email). Sent as
# a stable fingerprint: per-user and per-session patterns still show, the
# person does not.
HASHED_KEYS = frozenset({"user.id", "session.id"})

# The keys of `metadata` that describe a step rather than its content.
METADATA_KEYS = frozenset({"pipeline_step", "step", "task_type", "ls_run_name"})

KINDS = frozenset({
    "LLM", "TOOL", "CHAIN", "AGENT", "RETRIEVER", "EMBEDDING", "RERANKER",
    "GUARDRAIL", "EVALUATOR", "PROMPT", "DECISION", "UNKNOWN",
})

_MESSAGE_CONTENT = re.compile(r"^llm\.(input|output)_messages\.\d+\.message\.content$")
_PROMPTS = re.compile(r"^llm\.prompts\.\d+$")
_DOCUMENT = re.compile(r"^retrieval\.documents\.(\d+)\.")


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _canonical(value: Any) -> str:
    """JSON with sorted keys, so {"a":1,"b":2} and {"b":2,"a":1} hash alike."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def _iso(ns: Optional[int]) -> Optional[str]:
    if ns is None:
        return None
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _sizes(attrs: Mapping[str, Any], direction: str) -> Optional[int]:
    value = attrs.get(f"{direction}.value")
    if isinstance(value, str):
        return len(value)
    total, seen = 0, False
    for key, v in attrs.items():
        m = _MESSAGE_CONTENT.match(key)
        if m and m.group(1) == direction and isinstance(v, str):
            total += len(v)
            seen = True
        elif direction == "input" and _PROMPTS.match(key) and isinstance(v, str):
            total += len(v)
            seen = True
    return total if seen else None


def _input_fingerprint(attrs: Mapping[str, Any]) -> Optional[str]:
    value = attrs.get("input.value")
    if value is not None:
        return fingerprint(_canonical(value))
    parts = [
        f"{k}={v}" for k, v in sorted(attrs.items())
        if isinstance(v, str) and k.startswith("llm.input_messages.")
    ]
    return fingerprint("\n".join(parts)) if parts else None


def _output_items(attrs: Mapping[str, Any]) -> Optional[int]:
    docs = {m.group(1) for k in attrs if (m := _DOCUMENT.match(k))}
    if docs:
        return len(docs)
    value = attrs.get("output.value")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, list):
            return len(parsed)
    return None


def _finish_reason(attrs: Mapping[str, Any]) -> Optional[str]:
    explicit = attrs.get("llm.finish_reason")
    if isinstance(explicit, str):
        return explicit[:40]
    reasons = attrs.get("gen_ai.response.finish_reasons")
    if isinstance(reasons, (list, tuple)) and reasons:
        return str(reasons[0])[:40]
    value = attrs.get("output.value")
    if isinstance(value, str) and value.startswith("{"):
        try:
            out = json.loads(value)
        except ValueError:
            return None
        if isinstance(out, dict):
            if isinstance(out.get("stop_reason"), str):
                return out["stop_reason"][:40]
            choices = out.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                reason = choices[0].get("finish_reason")
                if isinstance(reason, str):
                    return reason[:40]
    return None


def _metadata(attrs: Mapping[str, Any]) -> Optional[str]:
    raw = attrs.get("metadata")
    kept: dict[str, Any] = {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            kept = {
                k: v for k, v in parsed.items()
                if k in METADATA_KEYS and isinstance(v, (str, int, float, bool))
            }
    # A LangGraph node is a step; name it as one when nothing else did.
    node = attrs.get("graph.node.name")
    if "pipeline_step" not in kept and isinstance(node, str):
        kept["pipeline_step"] = node[:120]
    return json.dumps(kept, separators=(",", ":")) if kept else None


def span_kind(attrs: Mapping[str, Any]) -> str:
    kind = str(attrs.get("openinference.span.kind") or "UNKNOWN").upper()
    return kind if kind in KINDS else "UNKNOWN"


def _hex(value: int, width: int) -> str:
    return format(value, f"0{width}x")


def convert(span: Any, *, error_messages: bool = False) -> dict:
    """One `ReadableSpan` as a wire span. Never raises for odd attributes."""
    attrs: Mapping[str, Any] = dict(span.attributes or {})
    ctx = span.get_span_context()
    parent = span.parent

    out_attrs: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in EXACT_KEYS or key.startswith(PREFIX_KEYS):
            if isinstance(value, (str, int, float, bool)):
                out_attrs[key] = value[:200] if isinstance(value, str) else value
        elif key in HASHED_KEYS and value is not None:
            out_attrs[key] = "h_" + fingerprint(str(value))[:16]

    # The OTel GenAI conventions name tokens differently; the server reads the
    # OpenInference names.
    if "llm.token_count.prompt" not in out_attrs and "gen_ai.usage.input_tokens" in attrs:
        out_attrs["llm.token_count.prompt"] = attrs["gen_ai.usage.input_tokens"]
    if "llm.token_count.completion" not in out_attrs and "gen_ai.usage.output_tokens" in attrs:
        out_attrs["llm.token_count.completion"] = attrs["gen_ai.usage.output_tokens"]

    meta = _metadata(attrs)
    if meta:
        out_attrs["metadata"] = meta

    measured: dict[str, Any] = {}
    for name, value in (
        ("input_chars", _sizes(attrs, "input")),
        ("output_chars", _sizes(attrs, "output")),
        ("input_hash", _input_fingerprint(attrs)),
        ("output_items", _output_items(attrs)),
        ("finish_reason", _finish_reason(attrs)),
    ):
        if value is not None:
            measured[name] = value

    is_error = span.status is not None and span.status.status_code.name == "ERROR"
    status_message = None
    if is_error:
        # The exception's type by default. Its message often quotes the input
        # that caused it, which is content.
        exc_type = next(
            (e.attributes.get("exception.type") for e in (span.events or [])
             if e.name == "exception" and e.attributes),
            None,
        )
        status_message = str(exc_type) if exc_type else "error"
        if error_messages and span.status.description:
            status_message = f"{status_message}: {span.status.description}"[:500]

    wire = {
        "span_id": _hex(ctx.span_id, 16),
        "parent_id": _hex(parent.span_id, 16) if parent is not None else None,
        "name": (span.name or "")[:200],
        "span_kind": span_kind(attrs),
        "start_time": _iso(span.start_time),
        "end_time": _iso(span.end_time),
        "status_code": "ERROR" if is_error else "OK",
        "attributes": out_attrs,
    }
    if status_message:
        wire["status_message"] = status_message
    if measured:
        wire["measured"] = measured
    return wire
