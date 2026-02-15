"""
HuggingFace smolagents adapter: hooks into smolagents' agent execution.

Targets: smolagents >= 1.0

smolagents is HuggingFace's lightweight agent framework. It provides:
  - CodeAgent: writes and executes Python code
  - ToolCallingAgent: uses structured tool calls
  - Both inherit from MultiStepAgent with step-based execution

The agent loop:
  1. Agent receives task
  2. For each step: LLM generates action -> action is executed -> observe result
  3. Agent returns final answer

Mapping:
  agent.run()        -> agent.lifecycle.start / end
  agent step         -> agent.step.start / end
  LLM generation     -> agent.llm.call.start / end
  tool execution     -> agent.tool.call.start / end
  code execution     -> agent.tool.call.start / end (tool_name="code_execution")
  memory operations  -> agent.memory.read / write

Usage:
    from smolagents import CodeAgent, HfApiModel
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.smolagents import SmolagentsAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = SmolagentsAdapter(observer, agent_id="code-agent")

    # Context-manager approach
    model = HfApiModel()
    agent = CodeAgent(tools=[...], model=model)
    with adapter.run(task="Analyze this dataset") as run:
        result = agent.run("Analyze this dataset")
        run.set_result(result)

    # Or use the monitor callback:
    monitor = adapter.create_monitor()
    agent = CodeAgent(tools=[...], model=model, step_callbacks=[monitor])
    result = agent.run("Analyze this dataset")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

from agent_observability.events import (
    AgentEvent,
    EventName,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)


class _RunContext:
    """Context for an active agent run."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._result: Optional[Any] = None
        self._step_count: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def set_result(self, result: Any) -> None:
        self._result = result

    def on_step_start(
        self,
        step_number: Optional[int] = None,
        **attrs: Any,
    ) -> str:
        """Record the start of an agent step. Returns step_id."""
        self._step_count += 1
        step_id = new_step_id()

        step_attrs: dict[str, Any] = {
            "framework": "smolagents",
            "step_number": step_number or self._step_count,
        }
        step_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                attributes=step_attrs,
            )
        )
        return step_id

    def on_step_end(
        self,
        step_id: str,
        ok: bool = True,
        observation: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        """Record the end of an agent step."""
        end_attrs: dict[str, Any] = {}
        if observation:
            end_attrs["observation"] = observation[:2000]
        end_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                ok=ok,
                attributes=end_attrs,
            )
        )

    def on_llm_call(
        self,
        model_name: str = "unknown",
        step_id: Optional[str] = None,
        **attrs: Any,
    ) -> str:
        """Record an LLM call. Returns llm_call_id."""
        llm_id = new_llm_call_id()

        call_attrs: dict[str, Any] = {"framework": "smolagents"}
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                llm_call_id=llm_id,
                model_name=model_name,
                attributes=call_attrs,
            )
        )
        return llm_id

    def on_llm_response(
        self,
        llm_call_id: str,
        step_id: Optional[str] = None,
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
                step_id=step_id,
                llm_call_id=llm_call_id,
                ok=ok,
                attributes=resp_attrs,
            )
        )

    def on_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        step_id: Optional[str] = None,
        **attrs: Any,
    ) -> str:
        """Record a tool call. Returns tool_call_id."""
        tc_id = new_tool_call_id()

        call_attrs: dict[str, Any] = {"framework": "smolagents"}
        if tool_input is not None:
            call_attrs["input"] = str(tool_input)[:4096]
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
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
        step_id: Optional[str] = None,
        ok: bool = True,
        error: Optional[str] = None,
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
                step_id=step_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                ok=ok,
                error_message=error,
                attributes=res_attrs,
            )
        )

    def on_code_execution(
        self,
        code: str,
        step_id: Optional[str] = None,
    ) -> str:
        """Record code execution (for CodeAgent). Returns tool_call_id."""
        return self.on_tool_call(
            tool_name="code_execution",
            tool_input=code[:4096],
            step_id=step_id,
            code_length=len(code),
        )


class SmolagentsAdapter:
    """
    HuggingFace smolagents observability adapter.
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "smolagents-agent",
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def run(
        self,
        task: str = "",
        **attrs: Any,
    ) -> Generator[_RunContext, None, None]:
        """Wrap a full agent.run() execution."""
        run_id = new_run_id()

        start_attrs: dict[str, Any] = {"framework": "smolagents"}
        if task:
            start_attrs["task"] = task
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
            end_attrs: dict[str, Any] = {"step_count": ctx._step_count}
            if ctx._result is not None:
                end_attrs["result_preview"] = str(ctx._result)[:500]

            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes=end_attrs,
                )
            )

    def create_monitor(self) -> Callable[..., Any]:
        """
        Create a step callback function compatible with smolagents'
        step_callbacks parameter.

        The callback receives the agent's step log after each step.

        Usage:
            monitor = adapter.create_monitor()
            agent = CodeAgent(tools=[...], model=model, step_callbacks=[monitor])
        """
        run_id = new_run_id()
        agent_id = self._agent_id
        observer = self._observer
        step_count = 0

        # Emit lifecycle start
        observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=agent_id,
                run_id=run_id,
                attributes={"framework": "smolagents", "mode": "monitor"},
            )
        )

        def _callback(step_log: Any) -> None:
            nonlocal step_count
            step_count += 1
            step_id = new_step_id()

            # Extract info from step_log (ActionStep or similar)
            step_attrs: dict[str, Any] = {
                "framework": "smolagents",
                "step_number": step_count,
            }

            # smolagents step_log has: .tool_calls, .observation, .error, etc.
            tool_calls = getattr(step_log, "tool_calls", None)
            observation = getattr(step_log, "observation", None)
            error = getattr(step_log, "error", None)

            if observation:
                step_attrs["observation_preview"] = str(observation)[:500]

            observer.emit(
                AgentEvent(
                    name=EventName.STEP_START,
                    agent_id=agent_id,
                    run_id=run_id,
                    step_id=step_id,
                    attributes=step_attrs,
                )
            )

            # Record tool calls if present
            if tool_calls:
                for tc in tool_calls:
                    tc_name = getattr(tc, "name", str(tc))
                    tc_args = getattr(tc, "arguments", getattr(tc, "args", ""))
                    tc_id = new_tool_call_id()

                    observer.emit(
                        AgentEvent(
                            name=EventName.TOOL_CALL_START,
                            agent_id=agent_id,
                            run_id=run_id,
                            step_id=step_id,
                            tool_call_id=tc_id,
                            tool_name=str(tc_name),
                            attributes={"input": str(tc_args)[:2000]},
                        )
                    )
                    observer.emit(
                        AgentEvent(
                            name=EventName.TOOL_CALL_END,
                            agent_id=agent_id,
                            run_id=run_id,
                            step_id=step_id,
                            tool_call_id=tc_id,
                            tool_name=str(tc_name),
                            ok=error is None,
                        )
                    )

            observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=agent_id,
                    run_id=run_id,
                    step_id=step_id,
                    ok=error is None,
                    error_message=str(error) if error else None,
                )
            )

        return _callback
