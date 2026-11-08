"""Costzera: find and fix what makes AI agents slow and expensive.

    import costzera
    costzera.init(agent="support-agent")      # reads COSTZERA_API_KEY

That is the whole setup for agents built on a supported framework: their model,
tool and retrieval calls are traced automatically. For hand-written agents:

    @costzera.tool
    def fetch_order(order_id): ...

    with costzera.run("handle_ticket"):     # one trace per run of the agent
        with costzera.step("plan"):          # a named stage within it
            ...

Prompts, completions, tool arguments and results never leave your process.
Costzera receives their sizes and fingerprints, the model, token counts,
timings and errors -- what it needs to find waste -- and nothing else. See
`costzera._wire` for exactly what is sent.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode

from ._instrument import instrument_installed
from ._pipeline import Settings, TraceExporter, Transport

__version__ = "0.1.0"
__all__ = ["init", "run", "step", "tool", "flush", "shutdown", "__version__"]

DEFAULT_ENDPOINT = "https://costzera.com/api/v1/traces"

logger = logging.getLogger("costzera")
F = TypeVar("F", bound=Callable[..., Any])

_lock = threading.Lock()
_exporter: Optional[TraceExporter] = None
_provider: Optional[TracerProvider] = None


def init(
    agent: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
    instrument: bool = True,
    error_messages: bool = False,
    _transport: Optional[Transport] = None,
) -> bool:
    """Start sending traces. Safe to call more than once; later calls do nothing.

    agent:          the name this agent appears under in Costzera. Defaults to
                    COSTZERA_AGENT.
    api_key:        a Costzera key with the Send traces scope. Defaults to
                    COSTZERA_API_KEY.
    sample_rate:    the share of successful traces to keep, 0-1. Traces with an
                    error are always kept.
    instrument:     trace installed frameworks automatically.
    error_messages: also send exception messages, cut to 500 characters. Off by
                    default, because a message often quotes the input that
                    caused it.

    Returns True when tracing started. Without a key it logs once and does
    nothing, so an unconfigured environment behaves exactly as before.
    """
    global _exporter, _provider
    with _lock:
        if _exporter is not None:
            return True
        key = api_key or os.environ.get("COSTZERA_API_KEY")
        name = (agent or os.environ.get("COSTZERA_AGENT") or "").strip()
        if not key:
            logger.warning("Costzera is not tracing: set COSTZERA_API_KEY or pass api_key.")
            return False
        if not name:
            logger.warning("Costzera is not tracing: pass agent= or set COSTZERA_AGENT.")
            return False

        settings = Settings(
            api_key=key,
            agent=name[:100],
            endpoint=endpoint or os.environ.get("COSTZERA_ENDPOINT") or DEFAULT_ENDPOINT,
            sdk_version=__version__,
            sample_rate=min(1.0, max(0.0, float(sample_rate))),
            error_messages=error_messages,
        )
        _exporter = TraceExporter(settings, transport=_transport)

        # Join an OpenTelemetry setup the application already has rather than
        # replacing it: its own exporters keep working, and ours is added.
        current = otel_trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            _provider = current
        else:
            _provider = TracerProvider()
            otel_trace.set_tracer_provider(_provider)
        _provider.add_span_processor(_exporter)

        if instrument:
            instrument_installed(_provider)
        atexit.register(shutdown)
        return True


def _tracer():
    return (_provider or otel_trace.get_tracer_provider()).get_tracer("costzera", __version__)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return repr(value)


@contextmanager
def _span(name: str, kind: str, attributes: Optional[dict] = None) -> Iterator[Any]:
    attrs = {"openinference.span.kind": kind, **(attributes or {})}
    with _tracer().start_as_current_span(name, attributes=attrs, record_exception=False) as span:
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


@contextmanager
def run(name: str = "run") -> Iterator[None]:
    """One run of the agent: everything inside is one trace."""
    with _span(name, "AGENT"):
        yield


@contextmanager
def step(name: str) -> Iterator[None]:
    """A named stage of a run, so cost and latency can be attributed to it."""
    with _span(name, "CHAIN", {"metadata": json.dumps({"pipeline_step": name})}):
        yield


def tool(fn: Optional[F] = None, *, name: Optional[str] = None) -> Any:
    """Trace a function as a tool call. Works on sync and async functions.

    Arguments and the return value are recorded only so their size and
    fingerprint can be measured; they are not sent.
    """
    def decorate(func: F) -> F:
        tool_name = name or func.__name__

        def attrs(args: tuple, kwargs: dict) -> dict:
            return {"tool.name": tool_name, "input.value": _safe_json({"args": args, "kwargs": kwargs})}

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _span(tool_name, "TOOL", attrs(args, kwargs)) as span:
                    result = await func(*args, **kwargs)
                    span.set_attribute("output.value", _safe_json(result))
                    return result
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _span(tool_name, "TOOL", attrs(args, kwargs)) as span:
                result = func(*args, **kwargs)
                span.set_attribute("output.value", _safe_json(result))
                return result
        return wrapper  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate


def flush(timeout_s: float = 3.0) -> bool:
    """Send everything finished so far. For short scripts and serverless."""
    return _exporter.force_flush(int(timeout_s * 1000)) if _exporter else True


def shutdown() -> None:
    """Flush and stop. Registered at exit; call it yourself only to stop early."""
    global _exporter
    with _lock:
        exporter, _exporter = _exporter, None
    if exporter is not None:
        try:
            exporter.shutdown()
        except Exception:
            pass
