"""
Microsoft AutoGen adapter: hooks into AutoGen's multi-agent conversation system.

Targets: autogen-agentchat >= 0.4 (the new event-driven architecture)
Also supports: pyautogen >= 0.2 (legacy API via register_reply hooks)

AutoGen's architecture:
  - ConversableAgent: base agent with send/receive message hooks
  - GroupChat: multi-agent orchestration
  - In v0.4+: event-driven with AgentEvent, ChatCompletionEvent, ToolCallEvent

Mapping:
  group_chat.run()           -> agent.lifecycle.start / end
  agent message exchange     -> agent.step.start / end
  tool execution             -> agent.tool.call.start / end
  LLM call                   -> agent.llm.call.start / end
  agent handoff / selection  -> step attributes

Usage (v0.4+ event hooks):
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.autogen import AutoGenAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = AutoGenAdapter(observer)

    # Wrap a group chat
    with adapter.group_chat("research-team") as chat:
        chat.on_agent_message("researcher", "user", "Find papers on AI safety")
        chat.on_llm_call("researcher", model="gpt-4")
        chat.on_llm_response("researcher", response, tokens={...})
        chat.on_tool_call("researcher", "arxiv_search", {"query": "AI safety"})
        chat.on_tool_result("researcher", "arxiv_search", results, ok=True)
        chat.on_agent_message("researcher", "critic", "Here are the papers...")
        chat.on_agent_message("critic", "researcher", "Good analysis.")
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


class _GroupChatContext:
    """Tracks a multi-agent conversation."""

    def __init__(
        self,
        observer: AgentObserver,
        chat_name: str,
        run_id: str,
    ) -> None:
        self._observer = observer
        self._chat_name = chat_name
        self._run_id = run_id
        self._active_steps: dict[str, str] = {}      # agent_name -> step_id
        self._active_tools: dict[str, str] = {}       # key -> tool_call_id
        self._active_llms: dict[str, str] = {}        # agent_name -> llm_call_id
        self._message_count: int = 0

    def on_agent_message(
        self,
        sender: str,
        recipient: str,
        content: str = "",
        **attrs: Any,
    ) -> str:
        """
        Record a message exchange between agents.
        Each message from a new sender starts a new step.
        Returns step_id.
        """
        self._message_count += 1

        # End previous step for this sender if open
        old_step = self._active_steps.pop(sender, None)
        if old_step:
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=sender,
                    run_id=self._run_id,
                    step_id=old_step,
                    ok=True,
                )
            )

        # Start new step
        step_id = new_step_id()
        self._active_steps[sender] = step_id

        msg_attrs: dict[str, Any] = {
            "framework": "autogen",
            "sender": sender,
            "recipient": recipient,
            "message_number": self._message_count,
        }
        if content:
            msg_attrs["content_preview"] = content[:500]
        msg_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=sender,
                run_id=self._run_id,
                step_id=step_id,
                attributes=msg_attrs,
            )
        )
        return step_id

    def on_llm_call(
        self,
        agent_name: str,
        model: str = "unknown",
        **attrs: Any,
    ) -> str:
        """Record an LLM call by an agent. Returns llm_call_id."""
        llm_id = new_llm_call_id()
        self._active_llms[agent_name] = llm_id
        step_id = self._active_steps.get(agent_name)

        call_attrs: dict[str, Any] = {"framework": "autogen"}
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=agent_name,
                run_id=self._run_id,
                step_id=step_id,
                llm_call_id=llm_id,
                model_name=model,
                attributes=call_attrs,
            )
        )
        return llm_id

    def on_llm_response(
        self,
        agent_name: str,
        response: Any = None,
        tokens: Optional[dict[str, int]] = None,
        ok: bool = True,
        **attrs: Any,
    ) -> None:
        """Record an LLM response."""
        llm_id = self._active_llms.pop(agent_name, None)
        if not llm_id:
            return

        step_id = self._active_steps.get(agent_name)
        resp_attrs: dict[str, Any] = {}
        if tokens:
            resp_attrs.update(tokens)
        resp_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id=agent_name,
                run_id=self._run_id,
                step_id=step_id,
                llm_call_id=llm_id,
                ok=ok,
                attributes=resp_attrs,
            )
        )

    def on_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        tool_input: Any = None,
        **attrs: Any,
    ) -> str:
        """Record a tool call. Returns tool_call_id."""
        tc_id = new_tool_call_id()
        key = f"{agent_name}:{tool_name}:{tc_id}"
        self._active_tools[key] = tc_id
        step_id = self._active_steps.get(agent_name)

        call_attrs: dict[str, Any] = {"framework": "autogen"}
        if tool_input is not None:
            call_attrs["input"] = str(tool_input)[:4096]
        call_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=agent_name,
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
        agent_name: str,
        tool_name: str,
        result: Any = None,
        tool_call_id: Optional[str] = None,
        ok: bool = True,
        error: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        """Record a tool result."""
        # Find tool_call_id
        tc_id = tool_call_id
        if not tc_id:
            for key, tid in reversed(list(self._active_tools.items())):
                if key.startswith(f"{agent_name}:{tool_name}:"):
                    tc_id = tid
                    self._active_tools.pop(key)
                    break
        if not tc_id:
            tc_id = new_tool_call_id()

        step_id = self._active_steps.get(agent_name)
        res_attrs: dict[str, Any] = {}
        if result is not None:
            res_attrs["output"] = str(result)[:4096]
        res_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=agent_name,
                run_id=self._run_id,
                step_id=step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                ok=ok,
                error_message=error,
                attributes=res_attrs,
            )
        )

    def on_agent_selected(
        self, agent_name: str, selector: str = "group_chat_manager"
    ) -> None:
        """Record that an agent was selected to speak next."""
        step_id = self._active_steps.get(selector) or self._active_steps.get(agent_name)
        self._observer.emit(
            AgentEvent(
                name=EventName.MEMORY_WRITE,
                agent_id=selector,
                run_id=self._run_id,
                step_id=step_id,
                attributes={
                    "event": "agent_selected",
                    "selected_agent": agent_name,
                },
            )
        )

    def close_all_steps(self) -> None:
        """End all open steps. Called automatically at run end."""
        for agent_name, step_id in list(self._active_steps.items()):
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=agent_name,
                    run_id=self._run_id,
                    step_id=step_id,
                    ok=True,
                )
            )
        self._active_steps.clear()


class AutoGenAdapter:
    """
    Adapter for Microsoft AutoGen multi-agent conversations.

    Provides a context manager for group chats and explicit methods
    for all agent events.
    """

    def __init__(self, observer: AgentObserver) -> None:
        self._observer = observer

    @contextmanager
    def group_chat(
        self,
        chat_name: str = "autogen-group-chat",
        task: str = "",
        **attrs: Any,
    ) -> Generator[_GroupChatContext, None, None]:
        """
        Context manager wrapping a full group chat execution.
        """
        run_id = new_run_id()
        start_attrs: dict[str, Any] = {"framework": "autogen"}
        if task:
            start_attrs["task"] = task
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=chat_name,
                run_id=run_id,
                attributes=start_attrs,
            )
        )

        ctx = _GroupChatContext(self._observer, chat_name, run_id)
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
                    agent_id=chat_name,
                    run_id=run_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            ctx.close_all_steps()
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=chat_name,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={"message_count": ctx._message_count},
                )
            )

    @contextmanager
    def two_agent_chat(
        self,
        initiator: str,
        responder: str,
        task: str = "",
        **attrs: Any,
    ) -> Generator[_GroupChatContext, None, None]:
        """Convenience wrapper for two-agent conversations."""
        chat_name = f"{initiator}-{responder}"
        with self.group_chat(chat_name=chat_name, task=task, **attrs) as ctx:
            yield ctx
