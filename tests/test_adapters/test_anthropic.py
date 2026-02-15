"""Tests for the Anthropic Claude agent adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from conftest import InMemorySpanExporter

from agentsight.adapters.anthropic_agents import (
    AgenticLoopAdapter,
    AnthropicMessageHooksAdapter,
)
from agentsight.observer import AgentObserver


class TestAgenticLoopAdapter:
    def test_full_run_with_tool(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")

        # Simulate: run > turn > LLM response > tool call > tool result
        with adapter.run(task="Research AI") as run:
            with run.turn() as turn:
                # Simulate LLM response
                response = MagicMock()
                response.model = "claude-sonnet-4-20250514"
                response.usage = MagicMock(input_tokens=100, output_tokens=50)
                response.usage.cache_read_input_tokens = None
                response.usage.cache_creation_input_tokens = None
                response.stop_reason = "tool_use"
                turn.record_llm_response(response)

                with turn.tool_call("search", {"query": "AI safety"}, "tu_123") as tc:
                    tc.set_result("Found 10 papers")

        spans = span_exporter.get_finished_spans()
        span_names = sorted(s.name for s in spans)
        assert span_names == ["agent.llm", "agent.run", "agent.step", "agent.tool"]
        assert observer.open_span_count == 0

    def test_error_propagation(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")

        with pytest.raises(ValueError, match="API error"):
            with adapter.run(task="test") as run:
                with run.turn() as turn:
                    with turn.tool_call("broken", {}) as tc:
                        raise ValueError("API error")

        spans = span_exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.attributes["agent.ok"] is False

    def test_multiple_turns(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")

        with adapter.run(task="multi-turn") as run:
            for i in range(3):
                with run.turn():
                    pass

        step_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.step"]
        assert len(step_spans) == 3


class TestAnthropicMessageHooksAdapter:
    def test_event_driven_flow(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AnthropicMessageHooksAdapter(observer, agent_id="claude")

        run_id = adapter.on_run_start(task="test")
        adapter.on_message_start()

        # Simulate tool_use
        tc_id = adapter.on_tool_use("search", {"q": "AI"}, tool_use_id="tu_1")
        adapter.on_tool_result("search", "results", tool_use_id="tu_1")

        # Simulate response
        response = MagicMock()
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=200, output_tokens=100)
        adapter.on_message_end(response=response, model="claude-sonnet-4-20250514")

        adapter.on_run_end(ok=True)

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        assert any(s.name == "agent.step" for s in spans)
        assert any(s.name == "agent.tool" for s in spans)
        assert any(s.name == "agent.llm" for s in spans)
