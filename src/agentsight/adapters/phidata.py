"""
Phidata (Agno) adapter: hooks into Phidata's agent monitoring system.

Targets: phidata >= 2.0 / agno >= 1.0

Phidata (rebranded as Agno) provides:
  - Agent class with model, tools, knowledge bases, and memory
  - Built-in monitoring via phidata.monitoring
  - Run logging with session tracking

Phidata agent execution:
  1. Agent receives message
  2. Model generates response (may include tool calls)
  3. Tools execute, results returned
  4. Repeat until final response
  5. Optional: memory storage, knowledge base lookup

Mapping:
  agent.run() / agent.print_response()  -> agent.lifecycle.start / end
  model call                            -> agent.llm.call.start / end
  tool execution                        -> agent.tool.call.start / end
  knowledge base search                 -> agent.tool.call.start / end (retrieval)
  memory read/write                     -> agent.memory.read / write

Usage:
    from phi.agent import Agent
    from agentsight import AgentObserver, init_telemetry
    from agentsight.adapters.phidata import PhidataAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = PhidataAdapter(observer, agent_id="research-agent")

    agent = Agent(model=OpenAIChat(id="gpt-4"), tools=[...])

    with adapter.run(task="Research AI trends") as run:
        response = agent.run("Research the latest AI trends")
        run.record_response(response)
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


class _RunContext:
    """Context for an active Phidata agent run."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._step_count: int = 0
        self._response: Optional[Any] = None

    @property
    def run_id(self) -> str:
        return self._run_id

    def record_response(self, response: Any) -> None:
        """Record the agent response."""
        self._response = response

        # Try to extract metrics from Phidata's RunResponse
        metrics = getattr(response, "metrics", None)
        if metrics:
            attrs: dict[str, Any] = {}
            if isinstance(metrics, dict):
                attrs.update(metrics)
            elif hasattr(metrics, "dict"):
                attrs.update(metrics.dict())
            if attrs:
                self._observer.emit(
                    AgentEvent(
                        name=EventName.MEMORY_WRITE,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        attributes={"response_metrics": str(attrs)[:2000]},
                    )
                )

    def on_model_call(
        self,
        model: str = "unknown",
        messages: Optional[list[Any]] = None,
        **attrs: Any,
    ) -> str:
        """Record an LLM call. Returns llm_call_id."""
        self._step_count += 1
        step_id = new_step_id()

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                attributes={
                    "framework": "phidata",
                    "step_number": self._step_count,
                },
            )
        )

        llm_id = new_llm_call_id()
        call_attrs: dict[str, Any] = {"framework": "phidata"}
        if messages:
            call_attrs["message_count"] = len(messages)
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
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
        llm_call_id: str,
        ok: bool = True,
        tokens: Optional[dict[str, int]] = None,
        **attrs: Any,
    ) -> None:
        """Record an LLM response."""
        resp_attrs: dict[str, Any] = {}
        if tokens:
            resp_attrs.update(tokens)
        resp_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                llm_call_id=llm_call_id,
                ok=ok,
                attributes=resp_attrs,
            )
        )

    def on_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        **attrs: Any,
    ) -> str:
        """Record a tool call. Returns tool_call_id."""
        tc_id = new_tool_call_id()
        call_attrs: dict[str, Any] = {"framework": "phidata"}
        if tool_input is not None:
            call_attrs["input"] = str(tool_input)[:4096]
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                attributes=call_attrs,
            )
        )
        return tc_id

    def on_tool_result(
        self,
        tool_call_id: str,
        tool_name: str = "",
        result: Any = None,
        ok: bool = True,
        **attrs: Any,
    ) -> None:
        """Record a tool result."""
        res_attrs: dict[str, Any] = {}
        if result is not None:
            res_attrs["output"] = str(result)[:4096]
        res_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                ok=ok,
                attributes=res_attrs,
            )
        )

    def on_knowledge_search(
        self, query: str, results: Optional[list[Any]] = None
    ) -> None:
        """Record a knowledge base search (start + end)."""
        tc_id = new_tool_call_id()
        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                tool_call_id=tc_id,
                tool_name="knowledge_base",
                attributes={"query": query[:2000]},
            )
        )
        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                tool_call_id=tc_id,
                tool_name="knowledge_base",
                ok=True,
                attributes={
                    "result_count": len(results) if results else 0,
                },
            )
        )

    def on_memory_read(self, memory_type: str = "chat_history") -> None:
        """Record a memory read operation."""
        self._observer.emit(
            AgentEvent(
                name=EventName.MEMORY_READ,
                agent_id=self._agent_id,
                run_id=self._run_id,
                attributes={"memory_type": memory_type},
            )
        )

    def on_memory_write(
        self, memory_type: str = "chat_history", **attrs: Any
    ) -> None:
        """Record a memory write operation."""
        write_attrs: dict[str, Any] = {"memory_type": memory_type}
        write_attrs.update(attrs)
        self._observer.emit(
            AgentEvent(
                name=EventName.MEMORY_WRITE,
                agent_id=self._agent_id,
                run_id=self._run_id,
                attributes=write_attrs,
            )
        )


class PhidataAdapter:
    """Phidata (Agno) agent observability adapter."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "phidata-agent",
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def run(
        self,
        task: str = "",
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        **attrs: Any,
    ) -> Generator[_RunContext, None, None]:
        """Wrap a full agent.run() or agent.print_response() execution."""
        run_id = new_run_id()

        start_attrs: dict[str, Any] = {"framework": "phidata"}
        if task:
            start_attrs["task"] = task
        if model:
            start_attrs["model"] = model
        if session_id:
            start_attrs["session_id"] = session_id
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=self._agent_id,
                run_id=run_id,
                attributes=start_attrs,
            )
        )

        ctx = _RunContext(self._observer, self._agent_id, run_id)
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
                    agent_id=self._agent_id,
                    run_id=run_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={"step_count": ctx._step_count},
                )
            )
