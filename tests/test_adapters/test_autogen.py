"""Tests for the AutoGen adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import InMemorySpanExporter

from agentsight.adapters.autogen import AutoGenAdapter
from agentsight.observer import AgentObserver


class TestAutoGenAdapter:
    def test_group_chat(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AutoGenAdapter(observer)

        with adapter.group_chat("research-team", task="Find papers") as chat:
            chat.on_agent_message("researcher", "user", "Find AI safety papers")
            tc_id = chat.on_tool_call("researcher", "arxiv_search", {"query": "AI safety"})
            chat.on_tool_result("researcher", "arxiv_search", "5 papers found", tool_call_id=tc_id)
            chat.on_agent_message("researcher", "critic", "Here are the papers")
            chat.on_agent_message("critic", "researcher", "Good work")

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        step_spans = [s for s in spans if s.name == "agent.step"]
        assert len(step_spans) >= 3  # 3 messages = 3 steps
        assert any(s.name == "agent.tool" for s in spans)

    def test_two_agent_chat(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AutoGenAdapter(observer)

        with adapter.two_agent_chat("alice", "bob", task="Discuss") as chat:
            chat.on_agent_message("alice", "bob", "Hello!")
            chat.on_agent_message("bob", "alice", "Hi there!")

        spans = span_exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.attributes["agent.id"] == "alice-bob"

    def test_llm_tracking(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AutoGenAdapter(observer)

        with adapter.group_chat("test") as chat:
            chat.on_agent_message("agent", "user", "thinking...")
            llm_id = chat.on_llm_call("agent", model="gpt-4")
            chat.on_llm_response("agent", tokens={"prompt_tokens": 100})

        llm_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.llm"]
        assert len(llm_spans) == 1

    def test_error_in_group_chat(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        adapter = AutoGenAdapter(observer)

        with pytest.raises(RuntimeError):
            with adapter.group_chat("test") as chat:
                chat.on_agent_message("agent", "user", "working...")
                raise RuntimeError("chat failed")

        spans = span_exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.attributes["agent.ok"] is False
