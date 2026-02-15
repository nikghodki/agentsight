"""
agent-observability: Framework-agnostic agent observability SDK built on OpenTelemetry.

Quick start:
    from agent_observability import AgentObserver, AgentEvent, EventName, init_telemetry

    init_telemetry(service_name="my-agent-service")
    observer = AgentObserver()
    observer.emit(AgentEvent(
        name=EventName.LIFECYCLE_START,
        agent_id="my-agent",
        run_id=new_run_id(),
    ))
"""

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
