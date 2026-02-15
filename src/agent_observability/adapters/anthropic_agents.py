"""
Anthropic Claude Agent SDK adapter: hooks into the Claude agent lifecycle.

Targets: anthropic >= 0.49 (Anthropic SDK with agent/tool_use support)

The Anthropic SDK uses an agentic loop pattern where the client repeatedly
calls messages.create() with tool results until the model stops issuing
tool_use blocks. This adapter wraps that loop.

Two integration modes:

1. **Manual wrapper** (AgenticLoopAdapter): Wrap your own agentic loop.
   You call on_turn_start/on_turn_end/on_tool_call/on_tool_result explicitly.

2. **Message hooks** (AnthropicMessageHooksAdapter): Drop-in hooks for
   message-level events (on_message_start, on_message_end, on_tool_use).

Mapping:
  agentic loop start        -> agent.lifecycle.start
  agentic loop end          -> agent.lifecycle.end
  each LLM turn             -> agent.step.start / agent.step.end
  messages.create() call    -> agent.llm.call.start / agent.llm.call.end
  tool_use block            -> agent.tool.call.start
  tool_result               -> agent.tool.call.end

Usage (manual wrapper):
    from anthropic import Anthropic
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.anthropic_agents import AgenticLoopAdapter

    init_telemetry()
    client = Anthropic()
    observer = AgentObserver()
    adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")

    with adapter.run(task="Research AI safety") as run:
        messages = [{"role": "user", "content": "Research AI safety"}]
        while True:
            with run.turn() as turn:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    messages=messages,
                    tools=tools,
                )
                turn.record_llm_response(response)

                if response.stop_reason == "end_turn":
                    break

                for block in response.content:
                    if block.type == "tool_use":
                        with turn.tool_call(block.name, block.input, block.id) as tc:
                            result = execute_tool(block.name, block.input)
                            tc.set_result(result)
                        messages.append(...)
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
    """Context for an active tool call within a turn."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
        step_id: str,
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._step_id = step_id
        self._tool_call_id = tool_call_id
        self._tool_name = tool_name
        self._result: Optional[str] = None
        self._ok: bool = True

    def set_result(self, result: Any, ok: bool = True) -> None:
        self._result = str(result)[:4096]
        self._ok = ok


class _TurnContext:
    """Context for a single LLM turn (one messages.create() call)."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
        step_id: str,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._step_id = step_id
        self._llm_call_id: Optional[str] = None

    def record_llm_response(
        self,
        response: Any,
        model: Optional[str] = None,
    ) -> None:
        """Record the LLM response from messages.create()."""
        llm_id = new_llm_call_id()
        self._llm_call_id = llm_id

        # Extract metadata from Anthropic response object
        resp_model = model or getattr(response, "model", "unknown")
        usage = getattr(response, "usage", None)
        attrs: dict[str, Any] = {"model": resp_model}

        if usage:
            attrs["input_tokens"] = getattr(usage, "input_tokens", 0)
            attrs["output_tokens"] = getattr(usage, "output_tokens", 0)
            # Cache tokens if present
            cache_read = getattr(usage, "cache_read_input_tokens", None)
            cache_create = getattr(usage, "cache_creation_input_tokens", None)
            if cache_read is not None:
                attrs["cache_read_input_tokens"] = cache_read
            if cache_create is not None:
                attrs["cache_creation_input_tokens"] = cache_create

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason:
            attrs["stop_reason"] = stop_reason

        # Emit as a pair: LLM start + end (since we get the response synchronously)
        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                llm_call_id=llm_id,
                model_name=resp_model,
                attributes={"framework": "anthropic"},
            )
        )
        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                llm_call_id=llm_id,
                model_name=resp_model,
                ok=True,
                attributes=attrs,
            )
        )

    @contextmanager
    def tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_use_id: Optional[str] = None,
    ) -> Generator[_ToolCallContext, None, None]:
        """Context manager for a tool call within this turn."""
        tc_id = new_tool_call_id()

        attrs: dict[str, Any] = {"framework": "anthropic"}
        if tool_input is not None:
            attrs["input"] = str(tool_input)[:4096]
        if tool_use_id:
            attrs["anthropic.tool_use_id"] = tool_use_id

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                attributes=attrs,
            )
        )

        ctx = _ToolCallContext(
            self._observer,
            self._agent_id,
            self._run_id,
            self._step_id,
            tc_id,
            tool_name,
        )
        error: Optional[BaseException] = None
        try:
            yield ctx
        except BaseException as e:
            error = e
            ctx._ok = False
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            end_attrs: dict[str, Any] = {}
            if ctx._result is not None:
                end_attrs["output"] = ctx._result

            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    ok=ctx._ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes=end_attrs,
                )
            )


class _RunContext:
    """Context for a full agentic loop run."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._turn_count: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    @contextmanager
    def turn(self, **attrs: Any) -> Generator[_TurnContext, None, None]:
        """Context manager for a single LLM turn."""
        self._turn_count += 1
        step_id = new_step_id()

        step_attrs: dict[str, Any] = {
            "framework": "anthropic",
            "turn_number": self._turn_count,
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

        ctx = _TurnContext(self._observer, self._agent_id, self._run_id, step_id)
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


class AgenticLoopAdapter:
    """
    Manual wrapper for Anthropic's agentic loop pattern.

    Provides nested context managers: run() > turn() > tool_call().
    """

    def __init__(self, observer: AgentObserver, agent_id: str = "claude-agent") -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def run(
        self,
        task: str = "",
        model: Optional[str] = None,
        **attrs: Any,
    ) -> Generator[_RunContext, None, None]:
        run_id = new_run_id()

        start_attrs: dict[str, Any] = {"framework": "anthropic"}
        if task:
            start_attrs["task"] = task
        if model:
            start_attrs["model"] = model
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
                    attributes={"turn_count": ctx._turn_count},
                )
            )


class AnthropicMessageHooksAdapter:
    """
    Event-driven adapter for message-level hooks.

    Use when you want to instrument existing code without context managers.
    Call these methods from your agentic loop at the appropriate points.
    """

    def __init__(self, observer: AgentObserver, agent_id: str = "claude-agent") -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id: Optional[str] = None
        self._step_id: Optional[str] = None
        self._tool_calls: dict[str, str] = {}  # tool_use_id -> tool_call_id

    def on_run_start(self, task: str = "", **attrs: Any) -> str:
        """Call at the start of the agentic loop. Returns run_id."""
        self._run_id = new_run_id()
        start_attrs: dict[str, Any] = {"framework": "anthropic"}
        if task:
            start_attrs["task"] = task
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                attributes=start_attrs,
            )
        )
        return self._run_id

    def on_run_end(self, ok: bool = True, error: Optional[str] = None) -> None:
        """Call when the agentic loop completes."""
        if not self._run_id:
            return
        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                ok=ok,
                error_message=error,
            )
        )

    def on_message_start(self, **attrs: Any) -> str:
        """Call before each messages.create(). Returns step_id."""
        if not self._run_id:
            self._run_id = new_run_id()
        self._step_id = new_step_id()
        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                attributes=attrs,
            )
        )
        return self._step_id

    def on_message_end(
        self,
        response: Any = None,
        model: Optional[str] = None,
        ok: bool = True,
    ) -> None:
        """Call after messages.create() returns."""
        if not self._run_id or not self._step_id:
            return

        # Record LLM call
        if response is not None:
            llm_id = new_llm_call_id()
            resp_model = model or getattr(response, "model", "unknown")
            usage = getattr(response, "usage", None)
            attrs: dict[str, Any] = {}
            if usage:
                attrs["input_tokens"] = getattr(usage, "input_tokens", 0)
                attrs["output_tokens"] = getattr(usage, "output_tokens", 0)

            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    llm_call_id=llm_id,
                    model_name=resp_model,
                )
            )
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._step_id,
                    llm_call_id=llm_id,
                    model_name=resp_model,
                    ok=ok,
                    attributes=attrs,
                )
            )

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                ok=ok,
            )
        )

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_use_id: Optional[str] = None,
    ) -> str:
        """Call when a tool_use block is encountered. Returns tool_call_id."""
        if not self._run_id:
            self._run_id = new_run_id()
        tc_id = new_tool_call_id()
        if tool_use_id:
            self._tool_calls[tool_use_id] = tc_id

        attrs: dict[str, Any] = {}
        if tool_input is not None:
            attrs["input"] = str(tool_input)[:4096]
        if tool_use_id:
            attrs["anthropic.tool_use_id"] = tool_use_id

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                attributes=attrs,
            )
        )
        return tc_id

    def on_tool_result(
        self,
        tool_name: str,
        result: Any = None,
        tool_use_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        ok: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Call when a tool execution completes."""
        if not self._run_id:
            return

        tc_id = tool_call_id
        if not tc_id and tool_use_id:
            tc_id = self._tool_calls.pop(tool_use_id, None)
        if not tc_id:
            tc_id = new_tool_call_id()

        attrs: dict[str, Any] = {}
        if result is not None:
            attrs["output"] = str(result)[:4096]

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                ok=ok,
                error_message=error,
                attributes=attrs,
            )
        )
