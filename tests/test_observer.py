"""Tests for the AgentObserver (core OTel bridge)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import InMemorySpanExporter

from agentsight.events import (
    AgentEvent,
    EventName,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agentsight.observer import AgentObserver


class TestLifecycle:
    def test_start_and_end_creates_span(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "agent.run"
        assert spans[0].attributes["agent.id"] == "a1"
        assert spans[0].attributes["agent.ok"] is True

    def test_lifecycle_failure_sets_error_status(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id="a1",
                run_id=run_id,
                ok=False,
                error_message="timeout",
            )
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"

    def test_open_span_count(self, observer: AgentObserver):
        run_id = new_run_id()
        assert observer.open_span_count == 0
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        assert observer.open_span_count == 1
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))
        assert observer.open_span_count == 0


class TestSteps:
    def test_step_creates_child_span(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        step_id = new_step_id()

        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(AgentEvent(name=EventName.STEP_START, agent_id="a1", run_id=run_id, step_id=step_id))
        observer.emit(
            AgentEvent(name=EventName.STEP_END, agent_id="a1", run_id=run_id, step_id=step_id, ok=True)
        )
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        spans = span_exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "agent.step" in span_names
        assert "agent.run" in span_names

        # Step should be child of run
        step_span = next(s for s in spans if s.name == "agent.step")
        run_span = next(s for s in spans if s.name == "agent.run")
        assert step_span.parent is not None
        assert step_span.parent.span_id == run_span.context.span_id

    def test_multiple_steps(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))

        for _ in range(3):
            sid = new_step_id()
            observer.emit(AgentEvent(name=EventName.STEP_START, agent_id="a1", run_id=run_id, step_id=sid))
            observer.emit(AgentEvent(name=EventName.STEP_END, agent_id="a1", run_id=run_id, step_id=sid, ok=True))

        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "agent.step"]
        assert len(step_spans) == 3


class TestToolCalls:
    def test_tool_call_creates_span(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        step_id = new_step_id()
        tc_id = new_tool_call_id()

        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(AgentEvent(name=EventName.STEP_START, agent_id="a1", run_id=run_id, step_id=step_id))
        observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tc_id,
                tool_name="search",
                attributes={"query": "test"},
            )
        )
        observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tc_id,
                tool_name="search",
                ok=True,
                attributes={"results": "5"},
            )
        )
        observer.emit(
            AgentEvent(name=EventName.STEP_END, agent_id="a1", run_id=run_id, step_id=step_id, ok=True)
        )
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "agent.tool")
        step_span = next(s for s in spans if s.name == "agent.step")

        assert tool_span.attributes["agent.tool.name"] == "search"
        # Tool should be child of step
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == step_span.context.span_id

    def test_concurrent_tool_calls(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        """Multiple tool calls can be open simultaneously within a step."""
        run_id = new_run_id()
        step_id = new_step_id()
        tc1 = new_tool_call_id()
        tc2 = new_tool_call_id()

        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(AgentEvent(name=EventName.STEP_START, agent_id="a1", run_id=run_id, step_id=step_id))

        # Start both tools
        observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tc1,
                tool_name="search",
            )
        )
        observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tc2,
                tool_name="calculator",
            )
        )

        assert observer.open_span_count == 4  # run + step + 2 tools

        # End in different order
        observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tc2,
                ok=True,
            )
        )
        observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tc1,
                ok=True,
            )
        )

        observer.emit(
            AgentEvent(name=EventName.STEP_END, agent_id="a1", run_id=run_id, step_id=step_id, ok=True)
        )
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        tool_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.tool"]
        assert len(tool_spans) == 2


class TestLLMCalls:
    def test_llm_call_creates_span(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        step_id = new_step_id()
        llm_id = new_llm_call_id()

        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(AgentEvent(name=EventName.STEP_START, agent_id="a1", run_id=run_id, step_id=step_id))
        observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                llm_call_id=llm_id,
                model_name="gpt-4",
            )
        )
        observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id="a1",
                run_id=run_id,
                step_id=step_id,
                llm_call_id=llm_id,
                model_name="gpt-4",
                ok=True,
                attributes={"prompt_tokens": "100", "completion_tokens": "50"},
            )
        )
        observer.emit(
            AgentEvent(name=EventName.STEP_END, agent_id="a1", run_id=run_id, step_id=step_id, ok=True)
        )
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        spans = span_exporter.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "agent.llm")
        assert llm_span.attributes["agent.model.name"] == "gpt-4"


class TestErrors:
    def test_error_event_increments_counter(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        run_id = new_run_id()
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a1", run_id=run_id))
        observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id="a1",
                run_id=run_id,
                error_type="ValueError",
                error_message="bad input",
            )
        )
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=False))

        # Span should have error status
        spans = span_exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.status.status_code.name == "ERROR"


class TestPayloadSanitization:
    def test_attributes_are_sanitized(self, span_exporter: InMemorySpanExporter):
        from agentsight.redaction import PayloadPolicy

        observer = AgentObserver(payload_policy=PayloadPolicy())
        run_id = new_run_id()

        observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id="a1",
                run_id=run_id,
                attributes={"password": "secret123", "task": "normal value"},
            )
        )
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, agent_id="a1", run_id=run_id, ok=True))

        spans = span_exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.attributes["agent.attr.password"] == "[REDACTED]"
        assert run_span.attributes["agent.attr.task"] == "normal value"
