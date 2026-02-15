"""
LangChain adapter: implements LangChain's BaseCallbackHandler to emit AgentEvents.

Usage:
    from agentsight import AgentObserver, init_telemetry
    from agentsight.adapters.langchain import LangChainAdapter

    init_telemetry()
    observer = AgentObserver()
    handler = LangChainAdapter(observer=observer, agent_id="my-langchain-agent")

    # Use as a LangChain callback handler
    chain.invoke(input, config={"callbacks": [handler]})
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

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

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    # Provide a stub so the module can be imported without langchain installed.
    # The adapter will fail at instantiation time with a clear error.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    _LANGCHAIN_AVAILABLE = False
else:
    _LANGCHAIN_AVAILABLE = True


class LangChainAdapter(BaseCallbackHandler):  # type: ignore[misc]
    """
    Translates LangChain callback hooks into AgentEvent emissions.

    Mapping:
      on_chain_start  -> agent.lifecycle.start (top-level) or agent.step.start (nested)
      on_chain_end    -> agent.lifecycle.end or agent.step.end
      on_tool_start   -> agent.tool.call.start
      on_tool_end     -> agent.tool.call.end
      on_llm_start    -> agent.llm.call.start
      on_llm_end      -> agent.llm.call.end
      on_chain_error  -> agent.error
      on_tool_error   -> agent.error
      on_llm_error    -> agent.error
    """

    def __init__(self, observer: AgentObserver, agent_id: str) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain adapter requires 'langchain-core'. "
                "Install with: pip install agentsight[langchain]"
            )
        super().__init__()
        self._observer = observer
        self._agent_id = agent_id

        # Track correlation IDs by LangChain's run UUIDs
        self._run_id: str = new_run_id()
        self._chain_depth: int = 0
        self._lc_run_to_step: dict[str, str] = {}
        self._lc_run_to_tool: dict[str, str] = {}
        self._lc_run_to_llm: dict[str, str] = {}

    @property
    def run_id(self) -> str:
        """The current observability run ID."""
        return self._run_id

    # --- Chain callbacks ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        if self._chain_depth == 0:
            # Top-level chain = agent lifecycle
            self._run_id = new_run_id()
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    attributes={
                        "chain.name": serialized.get("name", ""),
                        "chain.type": serialized.get("id", [""])[-1] if serialized.get("id") else "",
                    },
                )
            )
        else:
            # Nested chain = reasoning step
            step_id = new_step_id()
            self._lc_run_to_step[lc_id] = step_id
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=step_id,
                    attributes={
                        "chain.name": serialized.get("name", ""),
                    },
                )
            )
        self._chain_depth += 1

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._chain_depth = max(0, self._chain_depth - 1)
        lc_id = str(run_id)

        if self._chain_depth == 0:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    ok=True,
                )
            )
        else:
            step_id = self._lc_run_to_step.pop(lc_id, None)
            if step_id:
                self._observer.emit(
                    AgentEvent(
                        name=EventName.STEP_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        step_id=step_id,
                        ok=True,
                    )
                )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._chain_depth = max(0, self._chain_depth - 1)
        lc_id = str(run_id)

        self._observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._lc_run_to_step.get(lc_id),
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )

        if self._chain_depth == 0:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    ok=False,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

    # --- Tool callbacks ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        tool_id = new_tool_call_id()
        self._lc_run_to_tool[lc_id] = tool_id

        # Find parent step
        parent_step = self._lc_run_to_step.get(str(parent_run_id)) if parent_run_id else None

        tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=parent_step,
                tool_call_id=tool_id,
                tool_name=tool_name,
                attributes={"input": input_str},
            )
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        tool_id = self._lc_run_to_tool.pop(lc_id, None)
        parent_step = self._lc_run_to_step.get(str(parent_run_id)) if parent_run_id else None

        if tool_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=parent_step,
                    tool_call_id=tool_id,
                    ok=True,
                    attributes={"output": str(output)},
                )
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        tool_id = self._lc_run_to_tool.pop(lc_id, None)

        self._observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id=self._agent_id,
                run_id=self._run_id,
                tool_call_id=tool_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )
        if tool_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tool_id,
                    ok=False,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

    # --- LLM callbacks ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        llm_id = new_llm_call_id()
        self._lc_run_to_llm[lc_id] = llm_id

        parent_step = self._lc_run_to_step.get(str(parent_run_id)) if parent_run_id else None

        invocation_params = kwargs.get("invocation_params", {})
        model_name = invocation_params.get("model_name", serialized.get("name", "unknown"))

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=parent_step,
                llm_call_id=llm_id,
                model_name=model_name,
                attributes={
                    "prompt_count": len(prompts),
                },
            )
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        llm_id = self._lc_run_to_llm.pop(lc_id, None)
        parent_step = self._lc_run_to_step.get(str(parent_run_id)) if parent_run_id else None

        attrs: dict[str, Any] = {}
        # Extract token usage if available
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                attrs["prompt_tokens"] = usage.get("prompt_tokens", 0)
                attrs["completion_tokens"] = usage.get("completion_tokens", 0)
                attrs["total_tokens"] = usage.get("total_tokens", 0)

        if llm_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=parent_step,
                    llm_call_id=llm_id,
                    ok=True,
                    attributes=attrs,
                )
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        llm_id = self._lc_run_to_llm.pop(lc_id, None)

        self._observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id=self._agent_id,
                run_id=self._run_id,
                llm_call_id=llm_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )
        if llm_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    llm_call_id=llm_id,
                    ok=False,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
