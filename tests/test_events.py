"""Tests for the AgentEvent protocol."""

import pytest

from agentsight.events import (
    AgentEvent,
    EventName,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)


class TestAgentEvent:
    def test_create_minimal_event(self):
        event = AgentEvent(
            name=EventName.LIFECYCLE_START,
            agent_id="test-agent",
            run_id="run-123",
        )
        assert event.name == EventName.LIFECYCLE_START
        assert event.agent_id == "test-agent"
        assert event.run_id == "run-123"
        assert event.ts_ns > 0
        assert len(event.event_id) == 32  # uuid4 hex

    def test_create_full_event(self):
        event = AgentEvent(
            name=EventName.TOOL_CALL_START,
            agent_id="test-agent",
            run_id="run-1",
            step_id="step-1",
            tool_call_id="tc-1",
            tool_name="search",
            attributes={"query": "test"},
        )
        assert event.tool_name == "search"
        assert event.attributes["query"] == "test"

    def test_event_is_frozen(self):
        event = AgentEvent(
            name=EventName.LIFECYCLE_START,
            agent_id="test",
            run_id="run-1",
        )
        with pytest.raises(AttributeError):
            event.agent_id = "changed"  # type: ignore[misc]

    def test_requires_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id="",
                run_id="run-1",
            )

    def test_requires_run_id(self):
        with pytest.raises(ValueError, match="run_id"):
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id="test",
                run_id="",
            )

    def test_event_ids_are_unique(self):
        e1 = AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a", run_id="r1")
        e2 = AgentEvent(name=EventName.LIFECYCLE_START, agent_id="a", run_id="r1")
        assert e1.event_id != e2.event_id

    def test_all_event_names_are_strings(self):
        for name in EventName:
            assert isinstance(name.value, str)
            assert name.value.startswith("agent.")


class TestIdGenerators:
    def test_run_id_unique(self):
        ids = {new_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_step_id_unique(self):
        ids = {new_step_id() for _ in range(100)}
        assert len(ids) == 100

    def test_tool_call_id_unique(self):
        ids = {new_tool_call_id() for _ in range(100)}
        assert len(ids) == 100

    def test_llm_call_id_unique(self):
        ids = {new_llm_call_id() for _ in range(100)}
        assert len(ids) == 100
