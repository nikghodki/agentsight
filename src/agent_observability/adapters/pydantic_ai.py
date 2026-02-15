"""
PydanticAI adapter: hooks into PydanticAI's agent execution lifecycle.

Targets: pydantic-ai >= 0.1

PydanticAI is a type-safe agent framework using Pydantic for:
  - Structured tool definitions (via function signatures)
  - Result validation (typed responses)
  - Dependency injection
  - Multi-model support (OpenAI, Anthropic, Gemini, Groq, etc.)

Agent execution flow:
  1. agent.run() / agent.run_sync() starts
  2. Model generates response (may include tool calls)
  3. Tools are executed, results fed back
  4. Repeat until final result (validated by Pydantic model)

Mapping:
  agent.run()        -> agent.lifecycle.start / end
  model request      -> agent.llm.call.start / end
  tool execution     -> agent.tool.call.start / end
  retry on validation -> agent.step.start / end
  result validation  -> span event

Usage:
    from pydantic_ai import Agent
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.pydantic_ai import PydanticAIAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = PydanticAIAdapter(observer, agent_id="my-agent")

    agent = Agent("openai:gpt-4", result_type=MyResult)

    # Instrument via context manager:
    with adapter.run(task="Analyze data") as run:
        result = await agent.run("Analyze this data")
        run.record_result(result)

    # Or use the message history for post-hoc instrumentation:
    result = await agent.run("Do something")
    adapter.instrument_from_messages(result.all_messages())
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


class _RunContext:
    """Context for an active PydanticAI agent run."""

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
        self._current_step_id: Optional[str] = None
        self._result: Optional[Any] = None

    @property
    def run_id(self) -> str:
        return self._run_id

    def record_result(self, result: Any) -> None:
        """Record the agent result for end-span attributes."""
        self._result = result

    def on_model_request(
        self,
        model: str = "unknown",
        messages: Optional[list[Any]] = None,
        **attrs: Any,
    ) -> str:
        """Record a model request. Returns llm_call_id."""
        self._step_count += 1
        step_id = new_step_id()
        self._current_step_id = step_id

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                attributes={
                    "framework": "pydantic-ai",
                    "step_number": self._step_count,
                },
            )
        )

        llm_id = new_llm_call_id()
        call_attrs: dict[str, Any] = {"framework": "pydantic-ai"}
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
        tool_calls: Optional[list[str]] = None,
        **attrs: Any,
    ) -> None:
        """Record a model response."""
        resp_attrs: dict[str, Any] = {}
        if tokens:
            resp_attrs.update(tokens)
        if tool_calls:
            resp_attrs["tool_call_count"] = len(tool_calls)
            resp_attrs["tool_names"] = ", ".join(tool_calls)
        resp_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
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

        call_attrs: dict[str, Any] = {"framework": "pydantic-ai"}
        if tool_input is not None:
            call_attrs["input"] = str(tool_input)[:4096]
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
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
        retry_message: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        """Record a tool result."""
        res_attrs: dict[str, Any] = {}
        if result is not None:
            res_attrs["output"] = str(result)[:4096]
        if retry_message:
            res_attrs["retry_message"] = retry_message
        res_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                ok=ok,
                attributes=res_attrs,
            )
        )

    def on_step_end(self, ok: bool = True, **attrs: Any) -> None:
        """End the current model interaction step."""
        if self._current_step_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._current_step_id,
                    ok=ok,
                    attributes=attrs,
                )
            )
            self._current_step_id = None

    def on_validation_error(self, error: str, retry: bool = True) -> None:
        """Record a Pydantic validation error on the result."""
        self._observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                error_type="ValidationError",
                error_message=error,
                attributes={"will_retry": retry},
            )
        )


class PydanticAIAdapter:
    """PydanticAI agent observability adapter."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "pydantic-ai-agent",
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def run(
        self,
        task: str = "",
        model: Optional[str] = None,
        **attrs: Any,
    ) -> Generator[_RunContext, None, None]:
        """Wrap a full agent.run() execution."""
        run_id = new_run_id()

        start_attrs: dict[str, Any] = {"framework": "pydantic-ai"}
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
            # Close open step
            ctx.on_step_end(ok=ok)

            end_attrs: dict[str, Any] = {"step_count": ctx._step_count}
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

    def instrument_from_messages(
        self,
        messages: list[Any],
        model: str = "unknown",
    ) -> None:
        """
        Post-hoc instrumentation from PydanticAI's message history.

        PydanticAI's RunResult.all_messages() returns the full conversation.
        This method replays those messages as observability events.

        Supported message types:
          - ModelRequest (with parts: SystemPromptPart, UserPromptPart, ToolReturnPart, RetryPromptPart)
          - ModelResponse (with parts: TextPart, ToolCallPart)
        """
        run_id = new_run_id()

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=self._agent_id,
                run_id=run_id,
                attributes={
                    "framework": "pydantic-ai",
                    "mode": "post-hoc",
                    "message_count": len(messages),
                },
            )
        )

        step_count = 0
        for msg in messages:
            msg_kind = getattr(msg, "kind", type(msg).__name__)

            if msg_kind == "request" or "Request" in str(type(msg).__name__):
                step_count += 1
                step_id = new_step_id()

                # Check for tool returns in the request
                parts = getattr(msg, "parts", [])
                for part in parts:
                    part_kind = getattr(part, "part_kind", type(part).__name__)
                    if "tool_return" in str(part_kind).lower() or "ToolReturn" in str(type(part).__name__):
                        tool_name = getattr(part, "tool_name", "unknown")
                        content = getattr(part, "content", "")
                        tc_id = new_tool_call_id()
                        self._observer.emit(
                            AgentEvent(
                                name=EventName.TOOL_CALL_END,
                                agent_id=self._agent_id,
                                run_id=run_id,
                                step_id=step_id,
                                tool_call_id=tc_id,
                                tool_name=str(tool_name),
                                ok=True,
                                attributes={"output": str(content)[:2000]},
                            )
                        )

            elif msg_kind == "response" or "Response" in str(type(msg).__name__):
                llm_id = new_llm_call_id()
                self._observer.emit(
                    AgentEvent(
                        name=EventName.LLM_CALL_START,
                        agent_id=self._agent_id,
                        run_id=run_id,
                        llm_call_id=llm_id,
                        model_name=model,
                    )
                )

                # Check for tool calls in the response
                parts = getattr(msg, "parts", [])
                tool_names = []
                for part in parts:
                    part_kind = getattr(part, "part_kind", type(part).__name__)
                    if "tool_call" in str(part_kind).lower() or "ToolCall" in str(type(part).__name__):
                        tool_name = getattr(part, "tool_name", "unknown")
                        tool_names.append(str(tool_name))
                        args = getattr(part, "args", getattr(part, "args_as_dict", lambda: {})())
                        tc_id = new_tool_call_id()
                        self._observer.emit(
                            AgentEvent(
                                name=EventName.TOOL_CALL_START,
                                agent_id=self._agent_id,
                                run_id=run_id,
                                tool_call_id=tc_id,
                                tool_name=str(tool_name),
                                attributes={"input": str(args)[:2000]},
                            )
                        )

                # Token usage
                usage = getattr(msg, "usage", None) or getattr(msg, "token_usage", None)
                attrs: dict[str, Any] = {}
                if usage:
                    if hasattr(usage, "request_tokens"):
                        attrs["prompt_tokens"] = usage.request_tokens
                        attrs["completion_tokens"] = usage.response_tokens
                    elif isinstance(usage, dict):
                        attrs.update(usage)
                if tool_names:
                    attrs["tool_calls"] = ", ".join(tool_names)

                self._observer.emit(
                    AgentEvent(
                        name=EventName.LLM_CALL_END,
                        agent_id=self._agent_id,
                        run_id=run_id,
                        llm_call_id=llm_id,
                        model_name=model,
                        ok=True,
                        attributes=attrs,
                    )
                )

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id=self._agent_id,
                run_id=run_id,
                ok=True,
                attributes={"step_count": step_count},
            )
        )
