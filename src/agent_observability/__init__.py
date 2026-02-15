"""
agent-observability: Framework-agnostic agent observability SDK built on OpenTelemetry.

Quick start (one-line auto-instrumentation)::

    from agent_observability import auto_instrument
    auto_instrument()
    # All installed frameworks now emit spans automatically.

Manual setup::

    from agent_observability import AgentObserver, init_telemetry
    init_telemetry(service_name="my-agent-service")
    observer = AgentObserver()
"""

from agent_observability.auto import (
    auto_instrument,
    available_frameworks,
    instrument_anthropic,
    instrument_autogen,
    instrument_bedrock,
    instrument_crewai,
    instrument_google_adk,
    instrument_haystack,
    instrument_langchain,
    instrument_langgraph,
    instrument_llamaindex,
    instrument_openai_agents,
    instrument_phidata,
    instrument_pydantic_ai,
    instrument_semantic_kernel,
    instrument_smolagents,
    uninstrument,
)
from agent_observability.events import (
    AgentEvent,
    EventName,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agent_observability.observer import AgentObserver
from agent_observability.otel_setup import (
    ExporterType,
    init_telemetry,
    shutdown_telemetry,
)
from agent_observability.redaction import PayloadPolicy

__all__ = [
    # One-line auto-instrumentation
    "auto_instrument",
    "uninstrument",
    "available_frameworks",
    # Per-framework instrumentation
    "instrument_anthropic",
    "instrument_autogen",
    "instrument_bedrock",
    "instrument_crewai",
    "instrument_google_adk",
    "instrument_haystack",
    "instrument_langchain",
    "instrument_langgraph",
    "instrument_llamaindex",
    "instrument_openai_agents",
    "instrument_phidata",
    "instrument_pydantic_ai",
    "instrument_semantic_kernel",
    "instrument_smolagents",
    # Core SDK
    "AgentEvent",
    "AgentObserver",
    "EventName",
    "ExporterType",
    "PayloadPolicy",
    "init_telemetry",
    "new_llm_call_id",
    "new_run_id",
    "new_step_id",
    "new_tool_call_id",
    "shutdown_telemetry",
]

__version__ = "0.1.0"
