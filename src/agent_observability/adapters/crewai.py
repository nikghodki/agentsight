"""
CrewAI adapter: hooks into CrewAI's step/task/tool callbacks to emit AgentEvents.

Usage:
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.crewai import CrewAIAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = CrewAIAdapter(observer=observer)

    # Wrap a crew execution
    with adapter.observe_crew(crew_name="research-crew"):
        crew.kickoff()

    # Or use as step/task callbacks manually
    adapter.on_task_start(agent_role="researcher", task_description="Find papers")
    adapter.on_tool_use(tool_name="search", tool_input={"query": "AI safety"})
    adapter.on_tool_result(tool_name="search", result="...", success=True)
    adapter.on_task_end(agent_role="researcher", success=True)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from agent_observability.events import (
    AgentEvent,
    EventName,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)


class CrewAIAdapter:
    """
    Translates CrewAI execution lifecycle into AgentEvent emissions.

    CrewAI structure mapping:
      crew.kickoff() -> agent.lifecycle.start / end
      task execution -> agent.step.start / end
      tool use       -> agent.tool.call.start / end

    Since CrewAI doesn't have a formal callback handler interface,
    this adapter provides explicit methods + a context manager.
    """

    def __init__(self, observer: AgentObserver) -> None:
        self._observer = observer
        self._run_id: Optional[str] = None
        self._current_step_id: Optional[str] = None
        self._current_agent_id: str = "crewai"

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @contextmanager
    def observe_crew(
        self,
        crew_name: str = "crew",
        attributes: Optional[dict[str, Any]] = None,
    ) -> Generator[CrewAIAdapter, None, None]:
        """
        Context manager that wraps a full crew execution.

        Usage:
            with adapter.observe_crew(crew_name="my-crew"):
                crew.kickoff()
        """
        self._run_id = new_run_id()
        self._current_agent_id = crew_name

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=self._current_agent_id,
                run_id=self._run_id,
                attributes=attributes or {},
            )
        )
        ok = True
        error: Optional[BaseException] = None
        try:
            yield self
        except BaseException as e:
            ok = False
            error = e
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=self._current_agent_id,
                    run_id=self._run_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._current_agent_id,
                    run_id=self._run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                )
            )
            self._run_id = None

    def on_task_start(
        self,
        agent_role: str,
        task_description: str = "",
        attributes: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Call when a CrewAI task begins execution.

        Returns the step_id for correlation with tool calls.
        """
        if not self._run_id:
            self._run_id = new_run_id()
            self._current_agent_id = agent_role

        step_id = new_step_id()
        self._current_step_id = step_id
        self._current_agent_id = agent_role

        attrs = {"task.description": task_description}
        if attributes:
            attrs.update(attributes)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._current_agent_id,
                run_id=self._run_id,
                step_id=step_id,
                attributes=attrs,
            )
        )
        return step_id

    def on_task_end(
        self,
        agent_role: str,
        success: bool = True,
        output: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Call when a CrewAI task finishes."""
        if not self._current_step_id or not self._run_id:
            return

        attrs: dict[str, Any] = {}
        if output:
            attrs["output"] = output
        if attributes:
            attrs.update(attributes)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=agent_role,
                run_id=self._run_id,
                step_id=self._current_step_id,
                ok=success,
                attributes=attrs,
            )
        )
        self._current_step_id = None

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Call when a CrewAI agent invokes a tool.

        Returns the tool_call_id for correlation.
        """
        if not self._run_id:
            self._run_id = new_run_id()

        tool_id = new_tool_call_id()
        attrs: dict[str, Any] = {}
        if tool_input is not None:
            attrs["input"] = str(tool_input)
        if attributes:
            attrs.update(attributes)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._current_agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                tool_call_id=tool_id,
                tool_name=tool_name,
                attributes=attrs,
            )
        )
        return tool_id

    def on_tool_result(
        self,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        result: Any = None,
        success: bool = True,
        error: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Call when a tool invocation completes."""
        if not self._run_id:
            return

        # If no tool_call_id provided, generate one (less ideal but still works)
        tid = tool_call_id or new_tool_call_id()

        attrs: dict[str, Any] = {}
        if result is not None:
            attrs["output"] = str(result)
        if attributes:
            attrs.update(attributes)

        if not success and error:
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=self._current_agent_id,
                    run_id=self._run_id,
                    tool_call_id=tid,
                    tool_name=tool_name,
                    error_type="ToolError",
                    error_message=error,
                )
            )

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=self._current_agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                tool_call_id=tid,
                tool_name=tool_name,
                ok=success,
                error_message=error,
                attributes=attrs,
            )
        )
