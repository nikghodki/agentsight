"""Tests for the generic adapter (context-manager API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import InMemorySpanExporter

from agentsight.adapters.generic import GenericAgentAdapter
from agentsight.observer import AgentObserver


class TestGenericAdapter:
    def test_full_run_lifecycle(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        agent = GenericAgentAdapter(observer, agent_id="test-agent")

        with agent.run(task="book flight") as run:
            with run.step(reason="search flights") as step:
                with step.tool_call("flight_search", input={"from": "SFO", "to": "JFK"}) as tc:
                    tc.set_output({"flights": 10})
                with step.llm_call(model="gpt-4") as llm:
                    llm.set_output("I found 10 flights", tokens={"prompt": 100, "completion": 50})

        spans = span_exporter.get_finished_spans()
        span_names = sorted(s.name for s in spans)
        assert span_names == ["agent.llm", "agent.run", "agent.step", "agent.tool"]

        # All spans should be properly ended (no open spans)
        assert observer.open_span_count == 0

    def test_error_in_tool_propagates(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        agent = GenericAgentAdapter(observer, agent_id="test-agent")

        with pytest.raises(RuntimeError, match="tool broke"):
            with agent.run(task="test") as run:
                with run.step(reason="do stuff") as step:
                    with step.tool_call("broken_tool") as tc:
                        raise RuntimeError("tool broke")

        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "agent.tool")
        assert tool_span.attributes["agent.ok"] is False

        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.attributes["agent.ok"] is False

    def test_multiple_steps(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        agent = GenericAgentAdapter(observer, agent_id="test-agent")

        with agent.run(task="multi-step task") as run:
            with run.step(reason="step 1"):
                pass
            with run.step(reason="step 2"):
                pass
            with run.step(reason="step 3"):
                pass

        step_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.step"]
        assert len(step_spans) == 3

    def test_concurrent_tools_in_step(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        """This tests the scenario where tools would be started/ended out of stack order,
        which the correlation-ID-based approach handles correctly."""
        agent = GenericAgentAdapter(observer, agent_id="test-agent")

        with agent.run(task="parallel tools") as run:
            with run.step(reason="search") as step:
                with step.tool_call("tool_a") as tc:
                    tc.set_output("result_a")
                with step.tool_call("tool_b") as tc:
                    tc.set_output("result_b")

        tool_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.tool"]
        assert len(tool_spans) == 2
        assert observer.open_span_count == 0

    def test_output_attributes_recorded(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        agent = GenericAgentAdapter(observer, agent_id="test-agent")

        with agent.run(task="test") as run:
            with run.step(reason="do stuff") as step:
                with step.tool_call("my_tool", input="hello") as tc:
                    tc.set_output("world", extra_key="extra_value")

        tool_span = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.tool")
        assert tool_span.attributes.get("agent.attr.output") == "world"
        assert tool_span.attributes.get("agent.attr.extra_key") == "extra_value"
