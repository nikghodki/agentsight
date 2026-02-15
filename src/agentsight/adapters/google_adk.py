"""
Google Agent Development Kit (ADK) adapter.

Targets: google-adk >= 0.1

Google ADK provides a framework for building agents with:
  - Agent class with instructions, tools, and sub-agents
  - Runner that executes agent turns in a session
  - Events: agent turn start/end, tool call, model call
  - Session management with state

Mapping:
  runner.run()        -> agent.lifecycle.start / end
  agent turn          -> agent.step.start / end
  model generation    -> agent.llm.call.start / end
  tool execution      -> agent.tool.call.start / end
  sub-agent handoff   -> step end + step start (new agent)
  session state       -> agent.memory.write

Usage:
    from google.adk import Agent, Runner
    from agentsight import AgentObserver, init_telemetry
    from agentsight.adapters.google_adk import GoogleADKAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = GoogleADKAdapter(observer)

    # Context-manager approach:
    with adapter.run(agent_name="travel-agent", task="Book a flight") as run:
        async for event in runner.run_async(user_id="u1", session_id="s1",
                                             new_message=message):
            run.on_event(event)

    # Or use explicit methods:
    adapter.on_agent_turn_start("travel-agent")
    adapter.on_model_call("travel-agent", model="gemini-2.0-flash")
    adapter.on_model_response("travel-agent", response, tokens={...})
    adapter.on_tool_call("travel-agent", "search_flights", {"from": "SFO"})
    adapter.on_tool_result("travel-agent", "search_flights", result)
    adapter.on_agent_turn_end("travel-agent")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from agentsight.events import (
    AgentEvent,
    EventName,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)


class _ADKRunContext:
    """Context for a running ADK agent session."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_name: str,
        run_id: str,
    ) -> None:
        self._observer = observer
        self._agent_name = agent_name
        self._run_id = run_id
        self._active_steps: dict[str, str] = {}    # agent_name -> step_id
        self._active_tools: dict[str, str] = {}    # key -> tool_call_id
        self._active_llms: dict[str, str] = {}     # agent_name -> llm_call_id
        self._turn_count: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def on_event(self, event: Any) -> None:
        """
        Process a Google ADK event from the runner's event stream.

        Dispatches based on event type attributes.
        """
        event_type = type(event).__name__

        # ADK events have different structures; extract what we can
        content = getattr(event, "content", None)
        actions = getattr(event, "actions", None)
        author = getattr(event, "author", self._agent_name)

        if hasattr(event, "is_final_response") and event.is_final_response:
            # Final response from agent
            step_id = self._active_steps.pop(str(author), None)
            if step_id:
                self._observer.emit(
                    AgentEvent(
                        name=EventName.STEP_END,
                        agent_id=str(author),
                        run_id=self._run_id,
                        step_id=step_id,
                        ok=True,
                        attributes={"is_final": True},
                    )
                )
            return

        # Tool call events
        if actions:
            function_calls = getattr(actions, "function_calls", None)
            if function_calls:
                for fc in function_calls:
                    fc_name = getattr(fc, "name", "unknown")
                    fc_args = getattr(fc, "args", {})
                    fc_id = getattr(fc, "id", "")
                    self.on_tool_call(
                        str(author), str(fc_name), fc_args, call_id=fc_id
                    )

        # Tool result events
        function_responses = getattr(event, "function_responses", None)
        if function_responses:
            for fr in function_responses:
                fr_name = getattr(fr, "name", "unknown")
                fr_response = getattr(fr, "response", "")
                self.on_tool_result(str(author), str(fr_name), fr_response)

    def on_agent_turn_start(
        self, agent_name: Optional[str] = None, **attrs: Any
    ) -> str:
        """Record an agent turn starting. Returns step_id."""
        name = agent_name or self._agent_name
        self._turn_count += 1

        step_id = new_step_id()
        self._active_steps[name] = step_id

        step_attrs: dict[str, Any] = {
            "framework": "google-adk",
            "agent.name": name,
            "turn_number": self._turn_count,
        }
        step_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=name,
                run_id=self._run_id,
                step_id=step_id,
                attributes=step_attrs,
            )
        )
        return step_id

    def on_agent_turn_end(
        self,
        agent_name: Optional[str] = None,
        ok: bool = True,
        **attrs: Any,
    ) -> None:
        """Record an agent turn ending."""
        name = agent_name or self._agent_name
        step_id = self._active_steps.pop(name, None)
        if not step_id:
            return

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=name,
                run_id=self._run_id,
                step_id=step_id,
                ok=ok,
                attributes=attrs,
            )
        )

    def on_model_call(
        self,
        agent_name: Optional[str] = None,
        model: str = "unknown",
        **attrs: Any,
    ) -> str:
        """Record an LLM call. Returns llm_call_id."""
        name = agent_name or self._agent_name
        llm_id = new_llm_call_id()
        self._active_llms[name] = llm_id
        step_id = self._active_steps.get(name)

        call_attrs: dict[str, Any] = {"framework": "google-adk"}
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=name,
                run_id=self._run_id,
                step_id=step_id,
                llm_call_id=llm_id,
                model_name=model,
                attributes=call_attrs,
            )
        )
        return llm_id

    def on_model_response(
        self,
        agent_name: Optional[str] = None,
        response: Any = None,
        tokens: Optional[dict[str, int]] = None,
        ok: bool = True,
        **attrs: Any,
    ) -> None:
        """Record an LLM response."""
        name = agent_name or self._agent_name
        llm_id = self._active_llms.pop(name, None)
        if not llm_id:
            return

        resp_attrs: dict[str, Any] = {}
        if tokens:
            resp_attrs.update(tokens)
        # Extract from ADK response if available
        if response and hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            resp_attrs["prompt_tokens"] = getattr(meta, "prompt_token_count", 0)
            resp_attrs["completion_tokens"] = getattr(meta, "candidates_token_count", 0)
        resp_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id=name,
                run_id=self._run_id,
                step_id=self._active_steps.get(name),
                llm_call_id=llm_id,
                ok=ok,
                attributes=resp_attrs,
            )
        )

    def on_tool_call(
        self,
        agent_name: Optional[str] = None,
        tool_name: str = "unknown",
        tool_input: Any = None,
        call_id: str = "",
        **attrs: Any,
    ) -> str:
        """Record a tool call. Returns tool_call_id."""
        name = agent_name or self._agent_name
        tc_id = new_tool_call_id()
        key = f"{name}:{tool_name}:{tc_id}"
        self._active_tools[key] = tc_id

        call_attrs: dict[str, Any] = {"framework": "google-adk"}
        if tool_input is not None:
            call_attrs["input"] = str(tool_input)[:4096]
        if call_id:
            call_attrs["adk.call_id"] = call_id
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=name,
                run_id=self._run_id,
                step_id=self._active_steps.get(name),
                tool_call_id=tc_id,
                tool_name=tool_name,
                attributes=call_attrs,
            )
        )
        return tc_id

    def on_tool_result(
        self,
        agent_name: Optional[str] = None,
        tool_name: str = "unknown",
        result: Any = None,
        tool_call_id: Optional[str] = None,
        ok: bool = True,
        error: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        """Record a tool result."""
        name = agent_name or self._agent_name

        tc_id = tool_call_id
        if not tc_id:
            for key, tid in reversed(list(self._active_tools.items())):
                if key.startswith(f"{name}:{tool_name}:"):
                    tc_id = tid
                    self._active_tools.pop(key)
                    break
        if not tc_id:
            tc_id = new_tool_call_id()

        res_attrs: dict[str, Any] = {}
        if result is not None:
            res_attrs["output"] = str(result)[:4096]
        res_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=name,
                run_id=self._run_id,
                step_id=self._active_steps.get(name),
                tool_call_id=tc_id,
                tool_name=tool_name,
                ok=ok,
                error_message=error,
                attributes=res_attrs,
            )
        )

    def on_sub_agent_handoff(
        self, from_agent: str, to_agent: str
    ) -> None:
        """Record a handoff to a sub-agent."""
        # End current agent step
        self.on_agent_turn_end(from_agent, ok=True, handoff_to=to_agent)
        # Start new agent step
        self.on_agent_turn_start(to_agent, handoff_from=from_agent)

    def on_state_update(
        self, key: str, value: Any, agent_name: Optional[str] = None
    ) -> None:
        """Record a session state update."""
        name = agent_name or self._agent_name
        self._observer.emit(
            AgentEvent(
                name=EventName.MEMORY_WRITE,
                agent_id=name,
                run_id=self._run_id,
                step_id=self._active_steps.get(name),
                attributes={
                    "state.key": key,
                    "state.value": str(value)[:1000],
                },
            )
        )


class GoogleADKAdapter:
    """Google Agent Development Kit observability adapter."""

    def __init__(self, observer: AgentObserver) -> None:
        self._observer = observer

    @contextmanager
    def run(
        self,
        agent_name: str = "google-adk-agent",
        task: str = "",
        session_id: Optional[str] = None,
        **attrs: Any,
    ) -> Generator[_ADKRunContext, None, None]:
        """Wrap a full ADK agent session."""
        run_id = new_run_id()

        start_attrs: dict[str, Any] = {"framework": "google-adk"}
        if task:
            start_attrs["task"] = task
        if session_id:
            start_attrs["session_id"] = session_id
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=agent_name,
                run_id=run_id,
                attributes=start_attrs,
            )
        )

        ctx = _ADKRunContext(self._observer, agent_name, run_id)
        ok = True
        error: Optional[BaseException] = None
        try:
            yield ctx
        except BaseException as e:
            ok = False
            error = e
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=agent_name,
                    run_id=run_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            # Close any remaining open steps
            for name, step_id in list(ctx._active_steps.items()):
                self._observer.emit(
                    AgentEvent(
                        name=EventName.STEP_END,
                        agent_id=name,
                        run_id=run_id,
                        step_id=step_id,
                        ok=ok,
                    )
                )
            ctx._active_steps.clear()

            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=agent_name,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={"turn_count": ctx._turn_count},
                )
            )
