"""
Generic adapter: a convenience wrapper for custom/homegrown agent frameworks.

Provides a clean context-manager API so you don't have to construct
AgentEvent instances manually.

Usage:
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.generic import GenericAgentAdapter

    init_telemetry()
    observer = AgentObserver()
    agent = GenericAgentAdapter(observer, agent_id="my-agent")

    with agent.run(task="book a flight") as run:
        with run.step(reason="search for flights") as step:
            with step.tool_call("flight_search", input={"from": "SFO", "to": "JFK"}) as tc:
                results = do_search(...)
                tc.set_output(results)
            with step.llm_call(model="gpt-4") as llm:
                response = call_llm(...)
                llm.set_output(response, tokens={"prompt": 100, "completion": 50})
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

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


class _ToolCallContext:
    """Context for an active tool call."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
        step_id: Optional[str],
        tool_call_id: str,
        tool_name: str,
    ):
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._step_id = step_id
        self._tool_call_id = tool_call_id
        self._tool_name = tool_name
        self._output_attrs: dict[str, Any] = {}

    def set_output(self, output: Any, **extra_attrs: Any) -> None:
        self._output_attrs["output"] = str(output)
        self._output_attrs.update(extra_attrs)


class _LLMCallContext:
    """Context for an active LLM call."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
        step_id: Optional[str],
        llm_call_id: str,
        model_name: str,
    ):
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._step_id = step_id
        self._llm_call_id = llm_call_id
        self._model_name = model_name
        self._output_attrs: dict[str, Any] = {}

    def set_output(
        self,
        output: Any,
        tokens: Optional[dict[str, int]] = None,
        **extra_attrs: Any,
    ) -> None:
        self._output_attrs["output"] = str(output)
        if tokens:
            self._output_attrs.update(tokens)
        self._output_attrs.update(extra_attrs)


class _StepContext:
    """Context for an active reasoning step."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
        step_id: str,
    ):
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._step_id = step_id

    @contextmanager
    def tool_call(
        self,
        tool_name: str,
        input: Any = None,
        **attrs: Any,
    ) -> Generator[_ToolCallContext, None, None]:
        tool_id = new_tool_call_id()
        start_attrs: dict[str, Any] = {}
        if input is not None:
            start_attrs["input"] = str(input)
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                tool_call_id=tool_id,
                tool_name=tool_name,
                attributes=start_attrs,
            )
        )

        ctx = _ToolCallContext(
            self._observer,
            self._agent_id,
            self._run_id,
            self._step_id,
            tool_id,
            tool_name,
        )
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
                    run_id=self._run_id,
                    step_id=self._step_id,
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes=ctx._output_attrs,
                )
            )

    @contextmanager
    def llm_call(
        self,
        model: str = "unknown",
        **attrs: Any,
    ) -> Generator[_LLMCallContext, None, None]:
        llm_id = new_llm_call_id()

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                llm_call_id=llm_id,
                model_name=model,
                attributes=attrs,
            )
        )

        ctx = _LLMCallContext(
            self._observer,
            self._agent_id,
            self._run_id,
            self._step_id,
            llm_id,
            model,
        )
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
                    run_id=self._run_id,
                    step_id=self._step_id,
                    llm_call_id=llm_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    llm_call_id=llm_id,
                    model_name=model,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes=ctx._output_attrs,
                )
            )


class _RunContext:
    """Context for an active agent run."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
    ):
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @contextmanager
    def step(
        self,
        reason: str = "",
        **attrs: Any,
    ) -> Generator[_StepContext, None, None]:
        step_id = new_step_id()
        start_attrs: dict[str, Any] = {}
        if reason:
            start_attrs["reason"] = reason
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                attributes=start_attrs,
            )
        )

        ctx = _StepContext(self._observer, self._agent_id, self._run_id, step_id)
        ok = True
        error: Optional[BaseException] = None
        try:
            yield ctx
        except BaseException as e:
            ok = False
            error = e
            raise
        finally:
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=step_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                )
            )


class GenericAgentAdapter:
    """
    Convenience wrapper for custom agent frameworks.

    Provides nested context managers that automatically handle
    event emission and correlation ID management.
    """

    def __init__(self, observer: AgentObserver, agent_id: str) -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def run(
        self,
        task: str = "",
        **attrs: Any,
    ) -> Generator[_RunContext, None, None]:
        run_id = new_run_id()
        start_attrs: dict[str, Any] = {}
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
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                )
            )
