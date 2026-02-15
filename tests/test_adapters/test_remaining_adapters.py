"""Tests for Google ADK, Bedrock Agents, Smolagents, Haystack, PydanticAI, and Phidata adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from conftest import InMemorySpanExporter

from agent_observability.observer import AgentObserver


class TestGoogleADKAdapter:
    def test_full_run(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.google_adk import GoogleADKAdapter

        adapter = GoogleADKAdapter(observer)

        with adapter.run(agent_name="travel-agent", task="Book flight") as run:
            run.on_agent_turn_start()
            run.on_model_call(model="gemini-2.0-flash")
            run.on_model_response(tokens={"prompt_tokens": 100, "completion_tokens": 50})
            tc_id = run.on_tool_call(tool_name="search_flights", tool_input={"from": "SFO"})
            run.on_tool_result(tool_name="search_flights", result="Found 5 flights", tool_call_id=tc_id)
            run.on_agent_turn_end()

        spans = span_exporter.get_finished_spans()
        span_names = sorted(s.name for s in spans)
        assert "agent.llm" in span_names
        assert "agent.run" in span_names
        assert "agent.step" in span_names
        assert "agent.tool" in span_names

    def test_sub_agent_handoff(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.google_adk import GoogleADKAdapter

        adapter = GoogleADKAdapter(observer)

        with adapter.run(agent_name="main") as run:
            run.on_agent_turn_start("main")
            run.on_sub_agent_handoff("main", "sub-agent")
            run.on_agent_turn_end("sub-agent")

        step_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.step"]
        assert len(step_spans) == 2


class TestBedrockAgentsAdapter:
    def test_orchestration_trace(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.bedrock_agents import BedrockAgentsAdapter

        adapter = BedrockAgentsAdapter(observer, agent_id="bedrock-agent")

        with adapter.invocation(task="What is AI?") as inv:
            # Model invocation
            inv.process_event({
                "trace": {"trace": {"orchestrationTrace": {
                    "modelInvocationInput": {"type": "ORCHESTRATION"}
                }}}
            })
            inv.process_event({
                "trace": {"trace": {"orchestrationTrace": {
                    "modelInvocationOutput": {"metadata": {"usage": {
                        "inputTokens": 100, "outputTokens": 50
                    }}}
                }}}
            })
            # Rationale
            inv.process_event({
                "trace": {"trace": {"orchestrationTrace": {
                    "rationale": {"text": "I need to search for information"}
                }}}
            })
            # Tool call
            inv.process_event({
                "trace": {"trace": {"orchestrationTrace": {
                    "invocationInput": {"actionGroupInvocationInput": {
                        "function": "search",
                        "actionGroupName": "tools",
                        "parameters": {"query": "AI"}
                    }}
                }}}
            })
            # Tool result
            inv.process_event({
                "trace": {"trace": {"orchestrationTrace": {
                    "observation": {"actionGroupInvocationOutput": {
                        "text": "AI is artificial intelligence"
                    }}
                }}}
            })

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        assert any(s.name == "agent.llm" for s in spans)
        assert any(s.name == "agent.tool" for s in spans)

    def test_failure_trace(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.bedrock_agents import BedrockAgentsAdapter

        adapter = BedrockAgentsAdapter(observer)

        with adapter.invocation(task="test") as inv:
            inv.process_event({
                "trace": {"trace": {"failureTrace": {
                    "failureReason": "Access denied",
                    "traceId": "trace-123"
                }}}
            })

        # Should have error on the run span
        spans = span_exporter.get_finished_spans()
        run_span = next(s for s in spans if s.name == "agent.run")
        assert run_span.attributes.get("agent.ok") is True  # run itself didn't fail


class TestSmolagentsAdapter:
    def test_context_manager_run(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.smolagents import SmolagentsAdapter

        adapter = SmolagentsAdapter(observer, agent_id="code-agent")

        with adapter.run(task="Analyze data") as run:
            step_id = run.on_step_start(step_number=1)
            llm_id = run.on_llm_call(model_name="qwen-2.5-72b")
            run.on_llm_response(llm_id, tokens={"prompt": 200, "completion": 100})
            tc_id = run.on_tool_call("python_executor", "print('hello')", step_id=step_id)
            run.on_tool_result(tc_id, "python_executor", "hello", step_id=step_id)
            run.on_step_end(step_id)

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        assert any(s.name == "agent.step" for s in spans)
        assert any(s.name == "agent.llm" for s in spans)
        assert any(s.name == "agent.tool" for s in spans)
        assert observer.open_span_count == 0

    def test_monitor_callback(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.smolagents import SmolagentsAdapter

        adapter = SmolagentsAdapter(observer, agent_id="code-agent")
        monitor = adapter.create_monitor()

        # Simulate step_log
        step_log = MagicMock()
        step_log.tool_calls = [MagicMock(name="search", arguments={"q": "test"})]
        step_log.tool_calls[0].name = "search"
        step_log.observation = "Found results"
        step_log.error = None

        monitor(step_log)

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.step" for s in spans)
        assert any(s.name == "agent.tool" for s in spans)


class TestHaystackAdapter:
    def test_pipeline_run(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.haystack import HaystackAdapter

        adapter = HaystackAdapter(observer, agent_id="rag-pipeline")

        with adapter.pipeline_run(pipeline_name="qa") as run:
            adapter.on_component_start("retriever", "BM25Retriever", {"query": "AI"})
            adapter.on_component_end("retriever", {"documents": [1, 2, 3]})
            adapter.on_component_start("generator", "OpenAIGenerator")
            adapter.on_component_end("generator", {"meta": {"usage": {"prompt_tokens": 100}}})
            adapter.on_component_start("formatter", "PromptBuilder")
            adapter.on_component_end("formatter")

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        # retriever -> tool, generator -> llm, formatter -> step
        assert any(s.name == "agent.tool" for s in spans)
        assert any(s.name == "agent.llm" for s in spans)
        assert any(s.name == "agent.step" for s in spans)

    def test_error_in_pipeline(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.haystack import HaystackAdapter

        adapter = HaystackAdapter(observer)

        with pytest.raises(RuntimeError):
            with adapter.pipeline_run() as run:
                raise RuntimeError("pipeline broken")

        run_span = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.run")
        assert run_span.attributes["agent.ok"] is False


class TestPydanticAIAdapter:
    def test_full_run(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.pydantic_ai import PydanticAIAdapter

        adapter = PydanticAIAdapter(observer, agent_id="pydantic-agent")

        with adapter.run(task="Analyze data", model="openai:gpt-4") as run:
            llm_id = run.on_model_request(model="gpt-4", messages=["user msg"])
            run.on_model_response(llm_id, tool_calls=["search", "calculate"])
            tc1 = run.on_tool_call("search", {"query": "data"})
            run.on_tool_result(tc1, "search", "results")
            tc2 = run.on_tool_call("calculate", {"expr": "2+2"})
            run.on_tool_result(tc2, "calculate", "4")
            run.on_step_end()

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        assert any(s.name == "agent.llm" for s in spans)
        tool_spans = [s for s in spans if s.name == "agent.tool"]
        assert len(tool_spans) == 2

    def test_validation_error(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.pydantic_ai import PydanticAIAdapter

        adapter = PydanticAIAdapter(observer, agent_id="test")

        with adapter.run(task="test") as run:
            run.on_validation_error("field 'name' required", retry=True)

        # Should not fail; error is recorded as event
        assert observer.open_span_count == 0


class TestPhidataAdapter:
    def test_full_run(self, observer: AgentObserver, span_exporter: InMemorySpanExporter):
        from agent_observability.adapters.phidata import PhidataAdapter

        adapter = PhidataAdapter(observer, agent_id="research-agent")

        with adapter.run(task="Research trends") as run:
            llm_id = run.on_model_call(model="gpt-4", messages=["msg1"])
            run.on_model_response(llm_id, tokens={"prompt_tokens": 100})
            tc_id = run.on_tool_call("web_search", {"query": "AI trends 2025"})
            run.on_tool_result(tc_id, "web_search", "Results...")
            run.on_knowledge_search("AI trends", results=[1, 2, 3])
            run.on_memory_write("chat_history", message_count=5)

        spans = span_exporter.get_finished_spans()
        assert any(s.name == "agent.run" for s in spans)
        assert any(s.name == "agent.llm" for s in spans)
        # web_search + knowledge_base = 2 tool spans
        tool_spans = [s for s in spans if s.name == "agent.tool"]
        assert len(tool_spans) >= 2
