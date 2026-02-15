"""
Microsoft Semantic Kernel adapter: hooks into SK's filter/plugin system.

Targets: semantic-kernel >= 1.0

Semantic Kernel uses a filter pipeline for function invocations and
prompt rendering. The key extension points are:
  - FunctionInvocationFilter: before/after any function (tool) call
  - PromptRenderFilter: before/after prompt rendering
  - AutoFunctionInvocationFilter: before/after auto function calling

Mapping:
  kernel.invoke()             -> agent.lifecycle.start / end
  function invocation         -> agent.tool.call.start / end
  prompt render               -> agent.llm.call.start / end
  auto function calling loop  -> agent.step.start / end

Usage:
    from semantic_kernel import Kernel
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.semantic_kernel import (
        SKFunctionFilter,
        SKPromptFilter,
        SKAdapter,
    )

    init_telemetry()
    observer = AgentObserver()
    adapter = SKAdapter(observer)

    kernel = Kernel()
    kernel.add_filter("function_invocation", adapter.function_filter)
    kernel.add_filter("prompt_rendering", adapter.prompt_filter)

    # Or use the context-manager API:
    with adapter.run(task="Plan a trip") as run:
        result = await kernel.invoke(planner, input="Plan a trip to Paris")
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


class SKFunctionFilter:
    """
    Semantic Kernel function invocation filter.

    Wraps SK's FunctionInvocationFilter protocol. Each function call
    (plugin function, tool, etc.) is recorded as a tool call span.
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "semantic-kernel",
        run_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id_provider = run_id_provider or (lambda: new_run_id())
        self._active_calls: dict[str, str] = {}  # invocation_id -> tool_call_id

    async def on_function_invocation(
        self, context: Any, next_filter: Any
    ) -> None:
        """
        SK filter protocol: called around each function invocation.
        `context` is a FunctionInvocationContext.
        """
        func = getattr(context, "function", None)
        func_name = getattr(func, "name", "unknown") if func else "unknown"
        plugin_name = getattr(func, "plugin_name", "") if func else ""
        full_name = f"{plugin_name}.{func_name}" if plugin_name else func_name

        tc_id = new_tool_call_id()
        invocation_id = str(id(context))
        self._active_calls[invocation_id] = tc_id

        run_id = self._run_id_provider()
        arguments = getattr(context, "arguments", {})

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=run_id,
                tool_call_id=tc_id,
                tool_name=full_name,
                attributes={
                    "framework": "semantic-kernel",
                    "plugin": plugin_name,
                    "function": func_name,
                    "input": str(arguments)[:2000],
                },
            )
        )

        try:
            await next_filter(context)
            ok = True
            error = None
        except BaseException as e:
            ok = False
            error = e
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    tool_call_id=tc_id,
                    tool_name=full_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            self._active_calls.pop(invocation_id, None)
            result = getattr(context, "result", None)
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    tool_call_id=tc_id,
                    tool_name=full_name,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={"output": str(result)[:2000] if result else ""},
                )
            )


class SKPromptFilter:
    """
    Semantic Kernel prompt render filter.

    Wraps SK's PromptRenderFilter protocol. Each prompt render
    (which precedes an LLM call) is recorded as an LLM call span.
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "semantic-kernel",
        run_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id_provider = run_id_provider or (lambda: new_run_id())

    async def on_prompt_render(self, context: Any, next_filter: Any) -> None:
        """
        SK filter protocol: called around prompt rendering.
        `context` is a PromptRenderContext.
        """
        llm_id = new_llm_call_id()
        run_id = self._run_id_provider()

        func = getattr(context, "function", None)
        func_name = getattr(func, "name", "unknown") if func else "unknown"

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
                run_id=run_id,
                llm_call_id=llm_id,
                model_name=func_name,
                attributes={
                    "framework": "semantic-kernel",
                    "function": func_name,
                },
            )
        )

        try:
            await next_filter(context)
            ok = True
            error = None
        except BaseException as e:
            ok = False
            error = e
            raise
        finally:
            rendered = getattr(context, "rendered_prompt", "")
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    llm_call_id=llm_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={
                        "prompt_length": len(str(rendered)),
                    },
                )
            )


class SKAutoFunctionFilter:
    """
    Semantic Kernel auto function calling filter.

    Wraps the auto function invocation loop where SK automatically
    calls functions based on the LLM's tool_calls output.
    Each auto-invocation cycle is recorded as a step.
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "semantic-kernel",
        run_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id_provider = run_id_provider or (lambda: new_run_id())

    async def on_auto_function_invocation(
        self, context: Any, next_filter: Any
    ) -> None:
        step_id = new_step_id()
        run_id = self._run_id_provider()

        request_sequence = getattr(context, "request_sequence_index", 0)
        function_count = getattr(context, "function_count", 0)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=run_id,
                step_id=step_id,
                attributes={
                    "framework": "semantic-kernel",
                    "auto_invoke.sequence": request_sequence,
                    "auto_invoke.function_count": function_count,
                },
            )
        )

        try:
            await next_filter(context)
            ok = True
            error = None
        except BaseException as e:
            ok = False
            error = e
            raise
        finally:
            terminate = getattr(context, "terminate", False)
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    step_id=step_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={"terminate": terminate},
                )
            )


class _SKRunContext:
    """Internal run context with shared run_id for all filters."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class SKAdapter:
    """
    High-level Semantic Kernel adapter.

    Provides pre-configured filters that share a run_id, plus a
    context manager for explicit run boundaries.
    """

    def __init__(
        self, observer: AgentObserver, agent_id: str = "semantic-kernel"
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._current_run_id: Optional[str] = None

        def _get_run_id() -> str:
            return self._current_run_id or new_run_id()

        self.function_filter = SKFunctionFilter(
            observer, agent_id, run_id_provider=_get_run_id
        )
        self.prompt_filter = SKPromptFilter(
            observer, agent_id, run_id_provider=_get_run_id
        )
        self.auto_function_filter = SKAutoFunctionFilter(
            observer, agent_id, run_id_provider=_get_run_id
        )

    @contextmanager
    def run(
        self, task: str = "", **attrs: Any
    ) -> Generator[_SKRunContext, None, None]:
        """Wrap a kernel invocation with explicit run boundaries."""
        run_id = new_run_id()
        self._current_run_id = run_id

        start_attrs: dict[str, Any] = {"framework": "semantic-kernel"}
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

        ctx = _SKRunContext(run_id)
        ok = True
        error: Optional[BaseException] = None
        try:
            yield ctx
        except BaseException as e:
            ok = False
            error = e
            raise
        finally:
            self._current_run_id = None
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
