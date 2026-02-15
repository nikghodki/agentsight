"""
Core observability SDK: converts AgentEvents into OpenTelemetry spans and metrics.

Key design decisions:
  1. Span handles are tracked by correlation ID (run_id, step_id, tool_call_id, llm_call_id),
     NOT by OTel's implicit "current span" context. This makes the system safe for:
       - Concurrent tool calls within a step
       - Out-of-order events (DAG execution)
       - Async frameworks

  2. Parent-child relationships are explicit:
       agent.run (run_id)
         └─ agent.step (step_id, parent=run_id)
              ├─ agent.tool (tool_call_id, parent=step_id)
              └─ agent.llm (llm_call_id, parent=step_id)

  3. Thread safety: span tracking uses a lock. OTel's own context is thread-local.

  4. Payload hygiene: all attributes pass through the PayloadPolicy before
     being set on spans.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode

from agentsight.events import AgentEvent, EventName
from agentsight.redaction import PayloadPolicy, sanitize_attributes

logger = logging.getLogger(__name__)


class _SpanHandle:
    """Internal bookkeeping for an open span."""

    __slots__ = ("span", "context_token", "otel_context")

    def __init__(self, span: Span, context_token: object, otel_ctx: Context) -> None:
        self.span = span
        self.context_token = context_token
        self.otel_context = otel_ctx


class AgentObserver:
    """
    Consumes AgentEvent instances and produces OpenTelemetry spans + metrics.

    Usage:
        observer = AgentObserver()
        observer.emit(event)

    Thread-safe. Async-safe (emit is synchronous and non-blocking;
    OTel SDK batches exports in background threads).
    """

    def __init__(
        self,
        tracer_name: str = "agentsight",
        meter_name: str = "agentsight",
        payload_policy: Optional[PayloadPolicy] = None,
    ) -> None:
        self._tracer = trace.get_tracer(tracer_name)
        self._meter = metrics.get_meter(meter_name)
        self._policy = payload_policy or PayloadPolicy()
        self._lock = threading.Lock()

        # Span tracking: keyed by correlation ID
        self._run_spans: dict[str, _SpanHandle] = {}
        self._step_spans: dict[str, _SpanHandle] = {}
        self._tool_spans: dict[str, _SpanHandle] = {}
        self._llm_spans: dict[str, _SpanHandle] = {}

        # --- Metrics ---
        self._run_counter = self._meter.create_counter(
            name="agent.runs.total",
            description="Total agent run invocations",
            unit="1",
        )
        self._step_counter = self._meter.create_counter(
            name="agent.steps.total",
            description="Total agent reasoning steps",
            unit="1",
        )
        self._tool_counter = self._meter.create_counter(
            name="agent.tool_calls.total",
            description="Total tool call invocations",
            unit="1",
        )
        self._llm_counter = self._meter.create_counter(
            name="agent.llm_calls.total",
            description="Total LLM call invocations",
            unit="1",
        )
        self._error_counter = self._meter.create_counter(
            name="agent.errors.total",
            description="Total agent errors",
            unit="1",
        )
        self._run_duration = self._meter.create_histogram(
            name="agent.run.duration_ms",
            description="Agent run duration in milliseconds",
            unit="ms",
        )
        self._step_duration = self._meter.create_histogram(
            name="agent.step.duration_ms",
            description="Agent step duration in milliseconds",
            unit="ms",
        )
        self._tool_duration = self._meter.create_histogram(
            name="agent.tool_call.duration_ms",
            description="Tool call duration in milliseconds",
            unit="ms",
        )
        self._llm_duration = self._meter.create_histogram(
            name="agent.llm_call.duration_ms",
            description="LLM call duration in milliseconds",
            unit="ms",
        )

    def emit(self, event: AgentEvent) -> None:
        """
        Process an agent event. Routes to the appropriate handler
        based on event name.

        This method is synchronous and non-blocking. Safe to call
        from any thread or async context.
        """
        try:
            handler = self._HANDLERS.get(event.name)
            if handler:
                handler(self, event)
            else:
                logger.warning("Unhandled event type: %s", event.name)
        except Exception:
            logger.exception("Error processing event %s (id=%s)", event.name, event.event_id)

    # --- Internal handlers ---

    def _on_lifecycle_start(self, event: AgentEvent) -> None:
        self._run_counter.add(1, {"agent.id": event.agent_id})
        attrs = self._build_attrs(event)
        span = self._tracer.start_span(name="agent.run", attributes=attrs)
        ctx = trace.set_span_in_context(span)
        token = otel_context.attach(ctx)
        with self._lock:
            self._run_spans[event.run_id] = _SpanHandle(span, token, ctx)

    def _on_lifecycle_end(self, event: AgentEvent) -> None:
        with self._lock:
            handle = self._run_spans.pop(event.run_id, None)
        if not handle:
            logger.warning("lifecycle.end for unknown run_id=%s", event.run_id)
            return
        self._finish_span(handle, event)
        if event.ts_ns and hasattr(handle.span, "start_time"):
            start = getattr(handle.span, "start_time", None)
            if start:
                duration_ms = (event.ts_ns - start) / 1_000_000
                self._run_duration.record(duration_ms, {"agent.id": event.agent_id})
        otel_context.detach(handle.context_token)  # type: ignore[arg-type]

    def _on_step_start(self, event: AgentEvent) -> None:
        if not event.step_id:
            logger.warning("step.start without step_id, agent=%s", event.agent_id)
            return
        self._step_counter.add(1, {"agent.id": event.agent_id})
        parent_ctx = self._get_parent_context(self._run_spans, event.run_id)
        attrs = self._build_attrs(event)
        span = self._tracer.start_span(name="agent.step", attributes=attrs, context=parent_ctx)
        ctx = trace.set_span_in_context(span, parent_ctx)
        token = otel_context.attach(ctx)
        with self._lock:
            self._step_spans[event.step_id] = _SpanHandle(span, token, ctx)

    def _on_step_end(self, event: AgentEvent) -> None:
        if not event.step_id:
            return
        with self._lock:
            handle = self._step_spans.pop(event.step_id, None)
        if not handle:
            logger.warning("step.end for unknown step_id=%s", event.step_id)
            return
        self._finish_span(handle, event)
        if event.ts_ns and hasattr(handle.span, "start_time"):
            start = getattr(handle.span, "start_time", None)
            if start:
                duration_ms = (event.ts_ns - start) / 1_000_000
                self._step_duration.record(duration_ms, {"agent.id": event.agent_id})
        otel_context.detach(handle.context_token)  # type: ignore[arg-type]

    def _on_tool_start(self, event: AgentEvent) -> None:
        if not event.tool_call_id:
            logger.warning("tool.call.start without tool_call_id, agent=%s", event.agent_id)
            return
        self._tool_counter.add(
            1,
            {"agent.id": event.agent_id, "tool.name": event.tool_name or "unknown"},
        )
        # Parent is step if available, else run
        parent_ctx = self._get_parent_context(self._step_spans, event.step_id) or self._get_parent_context(
            self._run_spans, event.run_id
        )
        attrs = self._build_attrs(event)
        span = self._tracer.start_span(name="agent.tool", attributes=attrs, context=parent_ctx)
        ctx = trace.set_span_in_context(span, parent_ctx)
        token = otel_context.attach(ctx)
        with self._lock:
            self._tool_spans[event.tool_call_id] = _SpanHandle(span, token, ctx)

    def _on_tool_end(self, event: AgentEvent) -> None:
        if not event.tool_call_id:
            return
        with self._lock:
            handle = self._tool_spans.pop(event.tool_call_id, None)
        if not handle:
            logger.warning("tool.call.end for unknown tool_call_id=%s", event.tool_call_id)
            return
        self._finish_span(handle, event)
        if event.ts_ns and hasattr(handle.span, "start_time"):
            start = getattr(handle.span, "start_time", None)
            if start:
                duration_ms = (event.ts_ns - start) / 1_000_000
                self._tool_duration.record(
                    duration_ms,
                    {"agent.id": event.agent_id, "tool.name": event.tool_name or "unknown"},
                )
        otel_context.detach(handle.context_token)  # type: ignore[arg-type]

    def _on_llm_start(self, event: AgentEvent) -> None:
        if not event.llm_call_id:
            logger.warning("llm.call.start without llm_call_id, agent=%s", event.agent_id)
            return
        self._llm_counter.add(
            1,
            {"agent.id": event.agent_id, "model.name": event.model_name or "unknown"},
        )
        parent_ctx = self._get_parent_context(self._step_spans, event.step_id) or self._get_parent_context(
            self._run_spans, event.run_id
        )
        attrs = self._build_attrs(event)
        span = self._tracer.start_span(name="agent.llm", attributes=attrs, context=parent_ctx)
        ctx = trace.set_span_in_context(span, parent_ctx)
        token = otel_context.attach(ctx)
        with self._lock:
            self._llm_spans[event.llm_call_id] = _SpanHandle(span, token, ctx)

    def _on_llm_end(self, event: AgentEvent) -> None:
        if not event.llm_call_id:
            return
        with self._lock:
            handle = self._llm_spans.pop(event.llm_call_id, None)
        if not handle:
            logger.warning("llm.call.end for unknown llm_call_id=%s", event.llm_call_id)
            return
        self._finish_span(handle, event)
        if event.ts_ns and hasattr(handle.span, "start_time"):
            start = getattr(handle.span, "start_time", None)
            if start:
                duration_ms = (event.ts_ns - start) / 1_000_000
                self._llm_duration.record(
                    duration_ms,
                    {"agent.id": event.agent_id, "model.name": event.model_name or "unknown"},
                )
        otel_context.detach(handle.context_token)  # type: ignore[arg-type]

    def _on_memory(self, event: AgentEvent) -> None:
        # Memory operations are span events on the current step or run span
        parent_ctx = self._get_parent_context(self._step_spans, event.step_id) or self._get_parent_context(
            self._run_spans, event.run_id
        )
        if parent_ctx:
            span = trace.get_current_span(parent_ctx)
            if span and span.is_recording():
                span.add_event(event.name.value, attributes=self._build_attrs(event))

    def _on_error(self, event: AgentEvent) -> None:
        self._error_counter.add(
            1,
            {
                "agent.id": event.agent_id,
                "error.type": event.error_type or "Error",
            },
        )
        # Try to attach error to the most specific active span
        span = self._find_active_span(event)
        if span and span.is_recording():
            span.set_status(Status(StatusCode.ERROR, event.error_message or ""))
            span.set_attribute("error.type", event.error_type or "Error")
            if event.error_message:
                span.set_attribute("error.message", event.error_message)
            span.add_event(
                "exception",
                attributes={
                    "exception.type": event.error_type or "Error",
                    "exception.message": event.error_message or "",
                },
            )

    # --- Handler dispatch table ---

    _HANDLERS: dict[EventName, Any] = {
        EventName.LIFECYCLE_START: _on_lifecycle_start,
        EventName.LIFECYCLE_END: _on_lifecycle_end,
        EventName.STEP_START: _on_step_start,
        EventName.STEP_END: _on_step_end,
        EventName.TOOL_CALL_START: _on_tool_start,
        EventName.TOOL_CALL_END: _on_tool_end,
        EventName.LLM_CALL_START: _on_llm_start,
        EventName.LLM_CALL_END: _on_llm_end,
        EventName.MEMORY_READ: _on_memory,
        EventName.MEMORY_WRITE: _on_memory,
        EventName.ERROR: _on_error,
    }

    # --- Helpers ---

    def _build_attrs(self, event: AgentEvent) -> dict[str, Any]:
        """Build and sanitize OTel attributes from an event."""
        attrs: dict[str, Any] = {
            "agent.id": event.agent_id,
            "agent.run_id": event.run_id,
            "agent.event": event.name.value,
        }
        if event.step_id:
            attrs["agent.step_id"] = event.step_id
        if event.tool_call_id:
            attrs["agent.tool_call_id"] = event.tool_call_id
        if event.llm_call_id:
            attrs["agent.llm_call_id"] = event.llm_call_id
        if event.tool_name:
            attrs["agent.tool.name"] = event.tool_name
        if event.model_name:
            attrs["agent.model.name"] = event.model_name

        # Sanitize user-provided attributes BEFORE prefixing so key-based
        # redaction (e.g. "password") matches the original key names.
        if event.attributes:
            sanitized_user = sanitize_attributes(event.attributes, self._policy)
            for k, v in sanitized_user.items():
                attrs[f"agent.attr.{k}"] = v

        return attrs

    def _get_parent_context(
        self,
        span_map: dict[str, _SpanHandle],
        key: Optional[str],
    ) -> Optional[Context]:
        """Look up a parent span's OTel context by correlation ID."""
        if not key:
            return None
        with self._lock:
            handle = span_map.get(key)
        return handle.otel_context if handle else None

    def _find_active_span(self, event: AgentEvent) -> Optional[Span]:
        """Find the most specific active span for an event."""
        # Try tool -> step -> run (most specific first)
        if event.tool_call_id:
            with self._lock:
                h = self._tool_spans.get(event.tool_call_id)
            if h:
                return h.span
        if event.llm_call_id:
            with self._lock:
                h = self._llm_spans.get(event.llm_call_id)
            if h:
                return h.span
        if event.step_id:
            with self._lock:
                h = self._step_spans.get(event.step_id)
            if h:
                return h.span
        with self._lock:
            h = self._run_spans.get(event.run_id)
        return h.span if h else None

    def _finish_span(self, handle: _SpanHandle, event: AgentEvent) -> None:
        """End a span, setting status and final attributes."""
        span = handle.span
        if not span.is_recording():
            return

        ok = event.ok if event.ok is not None else True
        span.set_attribute("agent.ok", ok)

        if event.attributes:
            sanitized = sanitize_attributes(event.attributes, self._policy)
            for k, v in sanitized.items():
                span.set_attribute(f"agent.attr.{k}", v)

        if not ok:
            span.set_status(Status(StatusCode.ERROR, event.error_message or ""))
        else:
            span.set_status(Status(StatusCode.OK))

        span.end()

    @property
    def open_span_count(self) -> int:
        """Number of currently open spans. Useful for testing and debugging."""
        with self._lock:
            return (
                len(self._run_spans)
                + len(self._step_spans)
                + len(self._tool_spans)
                + len(self._llm_spans)
            )
