"""Turn on the OpenInference instrumentor for each supported framework that is
installed. Each is optional and independent: a missing package is skipped, and
an instrumentor that fails to start is logged and skipped too, never raised.

Install the ones you use with extras, e.g. `pip install "costzera[openai]"`.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("costzera")

# (module path, class name). Kept to instrumentors whose spans Costzera's
# server mapping is tested against, plus those built on the same conventions.
INSTRUMENTORS = [
    ("openinference.instrumentation.openai", "OpenAIInstrumentor"),
    ("openinference.instrumentation.anthropic", "AnthropicInstrumentor"),
    ("openinference.instrumentation.langchain", "LangChainInstrumentor"),
    ("openinference.instrumentation.llama_index", "LlamaIndexInstrumentor"),
    ("openinference.instrumentation.litellm", "LiteLLMInstrumentor"),
    ("openinference.instrumentation.crewai", "CrewAIInstrumentor"),
    ("openinference.instrumentation.bedrock", "BedrockInstrumentor"),
    ("openinference.instrumentation.openai_agents", "OpenAIAgentsInstrumentor"),
    ("openinference.instrumentation.dspy", "DSPyInstrumentor"),
]


def instrument_installed(provider) -> list:
    """Instrument everything available; return the names that were enabled."""
    enabled = []
    for module_path, class_name in INSTRUMENTORS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        try:
            instrumentor = getattr(module, class_name)()
            if not getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
                instrumentor.instrument(tracer_provider=provider)
            enabled.append(class_name)
        except Exception as exc:  # a broken instrumentor must not break the app
            logger.warning("Costzera skipped %s: %s", class_name, exc)
    if enabled:
        logger.info("Costzera is tracing: %s", ", ".join(enabled))
    return enabled
