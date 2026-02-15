"""
Agent Event Protocol — the stable contract for agent observability.

Every framework adapter emits AgentEvent instances. The observer consumes them
and maps them to OpenTelemetry spans, logs, and metrics.

Design principles:
  - Frozen dataclass: events are immutable once created.
  - Correlation IDs: run_id, step_id, tool_call_id enable proper span parenting
    without relying on OTel's implicit "current span" context.
  - attributes: carries framework-specific payloads (model, token counts, etc.).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventName(str, Enum):
    """Canonical event names. Adapters MUST use only these values."""

    # Lifecycle: one per agent invocation
    LIFECYCLE_START = "agent.lifecycle.start"
    LIFECYCLE_END = "agent.lifecycle.end"

    # Steps: reasoning iterations inside a run
    STEP_START = "agent.step.start"
    STEP_END = "agent.step.end"

    # Tool calls: individual tool invocations inside a step
    TOOL_CALL_START = "agent.tool.call.start"
    TOOL_CALL_END = "agent.tool.call.end"

    # LLM calls: model invocations (prompt -> completion)
    LLM_CALL_START = "agent.llm.call.start"
    LLM_CALL_END = "agent.llm.call.end"

    # Memory operations
    MEMORY_READ = "agent.memory.read"
    MEMORY_WRITE = "agent.memory.write"

    # Errors
    ERROR = "agent.error"


def _now_ns() -> int:
    return time.time_ns()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class AgentEvent:
    """
    A single observability event from an agent framework.

    Correlation IDs:
      - run_id:       Groups all events for one agent invocation.
      - step_id:      Groups events within a reasoning step.
      - tool_call_id: Identifies a specific tool call (start/end pair).
      - llm_call_id:  Identifies a specific LLM call (start/end pair).

    Adapters MUST set run_id. step_id, tool_call_id, and llm_call_id
    should be set when the corresponding scope is active.
    """

    name: EventName
    agent_id: str
    run_id: str

    # Correlation
    step_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    llm_call_id: Optional[str] = None

    # Distributed tracing bridge
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Tool metadata
    tool_name: Optional[str] = None

    # LLM metadata
    model_name: Optional[str] = None

    # Outcome
    ok: Optional[bool] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Extensible payload
    attributes: dict[str, Any] = field(default_factory=dict)

    # Auto-populated
    ts_ns: int = field(default_factory=_now_ns)
    event_id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id is required")
        if not self.run_id:
            raise ValueError("run_id is required")


def new_run_id() -> str:
    """Generate a new run correlation ID."""
    return _new_id()


def new_step_id() -> str:
    """Generate a new step correlation ID."""
    return _new_id()


def new_tool_call_id() -> str:
    """Generate a new tool call correlation ID."""
    return _new_id()


def new_llm_call_id() -> str:
    """Generate a new LLM call correlation ID."""
    return _new_id()
