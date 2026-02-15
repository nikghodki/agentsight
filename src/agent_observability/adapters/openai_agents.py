"""
OpenAI Agents SDK adapter: implements RunHooks and AgentHooks to emit AgentEvents.

Targets: openai-agents >= 0.1 (the OpenAI Agents SDK, not the base openai package)

The OpenAI Agents SDK has two hook levels:
  - RunHooks:   Attached to Runner.run(), spans the entire execution.
  - AgentHooks: Attached per Agent, fires for that agent only.

This adapter provides both:
  - OpenAIRunHooksAdapter  -> RunHooks (recommended: captures full run)
  - OpenAIAgentHooksAdapter -> AgentHooks (per-agent granularity)

Mapping:
  RunHooks.on_agent_start     -> agent.step.start   (each agent turn = step)
  RunHooks.on_agent_end       -> agent.step.end
  RunHooks.on_tool_start      -> agent.tool.call.start
  RunHooks.on_tool_end        -> agent.tool.call.end
  RunHooks.on_handoff         -> agent.step.end + agent.step.start (new agent)

  AgentHooks.on_start         -> agent.lifecycle.start
  AgentHooks.on_end           -> agent.lifecycle.end
  AgentHooks.on_tool_start    -> agent.tool.call.start
  AgentHooks.on_tool_end      -> agent.tool.call.end
  AgentHooks.on_handoff       -> agent.step.end (handoff span event)

Usage:
    from agents import Agent, Runner
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.openai_agents import OpenAIRunHooksAdapter

    init_telemetry()
    observer = AgentObserver()
    hooks = OpenAIRunHooksAdapter(observer)

    agent = Agent(name="assistant", instructions="You are helpful.")
    result = await Runner.run(agent, "Hello!", run_hooks=hooks)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.events import (
    AgentEvent,
    EventName,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

try:
    from agents.lifecycle import AgentHooks, RunHooks
    from agents import Agent, Tool

    _OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    _OPENAI_AGENTS_AVAILABLE = False

    class RunHooks:  # type: ignore[no-redef]
        pass

    class AgentHooks:  # type: ignore[no-redef]
        pass


def _require_sdk() -> None:
    if not _OPENAI_AGENTS_AVAILABLE:
        raise ImportError(
            "OpenAI Agents SDK adapter requires 'openai-agents'. "
            "Install with: pip install agent-observability[openai-agents]"
        )


class OpenAIRunHooksAdapter(RunHooks):  # type: ignore[misc]
    """
    Wraps Runner.run() lifecycle. Captures the full multi-agent execution.

    Usage:
        hooks = OpenAIRunHooksAdapter(observer)
        result = await Runner.run(agent, input, run_hooks=hooks)
    """

    def __init__(self, observer: AgentObserver) -> None:
        _require_sdk()
        self._observer = observer
        self._run_id: str = new_run_id()
        self._agent_steps: dict[str, str] = {}      # agent_name -> step_id
        self._tool_calls: dict[str, str] = {}        # tool_name:agent -> tool_call_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def start_run(self) -> None:
        """Call manually before Runner.run() if you want an explicit run span."""
        self._run_id = new_run_id()
        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id="openai-agents-runner",
                run_id=self._run_id,
            )
        )

    def end_run(self, ok: bool = True, error: Optional[str] = None) -> None:
        """Call manually after Runner.run() completes."""
        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id="openai-agents-runner",
                run_id=self._run_id,
                ok=ok,
                error_message=error,
            )
        )

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        agent_name = getattr(agent, "name", "unknown")
        step_id = new_step_id()
        self._agent_steps[agent_name] = step_id

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=agent_name,
                run_id=self._run_id,
                step_id=step_id,
                attributes={
                    "framework": "openai-agents",
                    "agent.name": agent_name,
                    "agent.instructions": str(getattr(agent, "instructions", ""))[:500],
                },
            )
        )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        agent_name = getattr(agent, "name", "unknown")
        step_id = self._agent_steps.pop(agent_name, None)
        if not step_id:
            return

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=agent_name,
                run_id=self._run_id,
                step_id=step_id,
                ok=True,
                attributes={"output": str(output)[:2000]},
            )
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        agent_name = getattr(agent, "name", "unknown")
        tool_name = getattr(tool, "name", str(tool))
        tc_id = new_tool_call_id()
        key = f"{tool_name}:{agent_name}:{tc_id}"
        self._tool_calls[key] = tc_id

        step_id = self._agent_steps.get(agent_name)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=agent_name,
                run_id=self._run_id,
                step_id=step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                attributes={"framework": "openai-agents"},
            )
        )

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        agent_name = getattr(agent, "name", "unknown")
        tool_name = getattr(tool, "name", str(tool))

        # Find the most recent tool call for this tool+agent
        tc_id: Optional[str] = None
        remove_key: Optional[str] = None
        for key, tid in reversed(list(self._tool_calls.items())):
            if key.startswith(f"{tool_name}:{agent_name}:"):
                tc_id = tid
                remove_key = key
                break

        if remove_key:
            self._tool_calls.pop(remove_key, None)

        if tc_id:
            step_id = self._agent_steps.get(agent_name)
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=agent_name,
                    run_id=self._run_id,
                    step_id=step_id,
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    ok=True,
                    attributes={"output": str(result)[:2000]},
                )
            )

    async def on_handoff(
        self, context: Any, from_agent: Any, to_agent: Any
    ) -> None:
        from_name = getattr(from_agent, "name", "unknown")
        to_name = getattr(to_agent, "name", "unknown")

        # End current agent's step
        step_id = self._agent_steps.pop(from_name, None)
        if step_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=from_name,
                    run_id=self._run_id,
                    step_id=step_id,
                    ok=True,
                    attributes={"handoff_to": to_name},
                )
            )

        # Start new agent's step
        new_sid = new_step_id()
        self._agent_steps[to_name] = new_sid
        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=to_name,
                run_id=self._run_id,
                step_id=new_sid,
                attributes={
                    "framework": "openai-agents",
                    "handoff_from": from_name,
                },
            )
        )


class OpenAIAgentHooksAdapter(AgentHooks):  # type: ignore[misc]
    """
    Per-agent hooks. Attach to a single Agent for agent-scoped observability.

    Usage:
        hooks = OpenAIAgentHooksAdapter(observer, agent_id="my-agent")
        agent = Agent(name="my-agent", hooks=hooks)
    """

    def __init__(self, observer: AgentObserver, agent_id: str) -> None:
        _require_sdk()
        self._observer = observer
        self._agent_id = agent_id
        self._run_id: str = new_run_id()
        self._step_id: Optional[str] = None
        self._tool_calls: dict[str, str] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    async def on_start(self, context: Any, agent: Any) -> None:
        self._run_id = new_run_id()
        self._step_id = new_step_id()

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                attributes={
                    "framework": "openai-agents",
                    "agent.name": getattr(agent, "name", self._agent_id),
                },
            )
        )
        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
            )
        )

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        if self._step_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    ok=True,
                    attributes={"output": str(output)[:2000]},
                )
            )
            self._step_id = None

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                ok=True,
            )
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        tool_name = getattr(tool, "name", str(tool))
        tc_id = new_tool_call_id()
        self._tool_calls[f"{tool_name}:{tc_id}"] = tc_id

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
            )
        )

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        tool_name = getattr(tool, "name", str(tool))

        tc_id: Optional[str] = None
        remove_key: Optional[str] = None
        for key, tid in reversed(list(self._tool_calls.items())):
            if key.startswith(f"{tool_name}:"):
                tc_id = tid
                remove_key = key
                break

        if remove_key:
            self._tool_calls.pop(remove_key, None)

        if tc_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    ok=True,
                    attributes={"output": str(result)[:2000]},
                )
            )

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        source_name = getattr(source, "name", "unknown")

        if self._step_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    ok=True,
                    attributes={"handoff_from": source_name},
                )
            )
            self._step_id = None
