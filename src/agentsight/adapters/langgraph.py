"""
LangGraph adapter: hooks into LangGraph's node/edge execution lifecycle.

Targets: langgraph >= 0.2

LangGraph builds stateful agent graphs where nodes are functions/runnables
and edges define transitions. This adapter maps graph execution to spans:

Mapping:
  graph.invoke() / graph.stream()  -> agent.lifecycle.start / end
  node execution                   -> agent.step.start / end
  tool node                        -> agent.tool.call.start / end
  LLM node (ChatModel call)       -> agent.llm.call.start / end
  conditional edge evaluation      -> span event on the step

Two integration modes:

1. **LangGraphCallbackAdapter**: Extends LangChain's BaseCallbackHandler
   since LangGraph inherits LangChain's callback system. Adds graph-aware
   context (node names, edge transitions).

2. **LangGraphEventAdapter**: Manual hooks for graph events. Use when you
   control the graph execution loop.

Usage (callback mode — recommended):
    from langgraph.graph import StateGraph
    from agentsight import AgentObserver, init_telemetry
    from agentsight.adapters.langgraph import LangGraphCallbackAdapter

    init_telemetry()
    observer = AgentObserver()
    handler = LangGraphCallbackAdapter(observer, agent_id="my-graph-agent")

    graph = StateGraph(...)
    app = graph.compile()
    result = app.invoke(input, config={"callbacks": [handler]})

Usage (manual mode):
    adapter = LangGraphEventAdapter(observer, agent_id="my-graph")
    with adapter.run(task="process request") as run:
        run.on_node_start("agent", state)
        run.on_node_end("agent", state)
        run.on_node_start("tools", state)
        run.on_node_end("tools", state)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional
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

    _LANGCHAIN_AVAILABLE = True
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    _LANGCHAIN_AVAILABLE = False


class LangGraphCallbackAdapter(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangGraph-aware callback handler.

    Extends LangChain's callback system with graph-specific tracking:
    - Tracks node names as step labels
    - Records edge transitions
    - Identifies tool nodes vs. LLM nodes
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "langgraph-agent",
        tool_node_names: Optional[set[str]] = None,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangGraph adapter requires 'langchain-core'. "
                "Install with: pip install agentsight[langgraph]"
            )
        super().__init__()
        self._observer = observer
        self._agent_id = agent_id
        # Names of nodes that represent tool execution (default: {"tools", "tool_node"})
        self._tool_node_names = tool_node_names or {"tools", "tool_node", "action"}
        self._run_id: str = new_run_id()
        self._chain_depth: int = 0

        # Track by LangChain run UUIDs
        self._lc_to_step: dict[str, str] = {}
        self._lc_to_tool: dict[str, str] = {}
        self._lc_to_llm: dict[str, str] = {}
        self._node_names: dict[str, str] = {}  # lc_run_id -> node_name

    @property
    def run_id(self) -> str:
        return self._run_id

    # --- Chain callbacks (graph + node level) ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        lc_id = str(run_id)
        name = serialized.get("name", "")

        # Detect node name from tags or serialized data
        node_name = None
        if tags:
            # LangGraph adds "graph:step:<N>" and node name tags
            for tag in tags:
                if tag.startswith("graph:step:"):
                    continue
                if tag not in ("seq:step", "langsmith:hidden"):
                    node_name = tag
                    break

        if not node_name:
            node_name = name or ""

        if node_name:
            self._node_names[lc_id] = node_name

        if self._chain_depth == 0:
            # Top-level = graph invocation
            self._run_id = new_run_id()
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    attributes={
                        "framework": "langgraph",
                        "graph.name": name,
                    },
                )
            )
        else:
            # Nested chain = node execution = step
            step_id = new_step_id()
            self._lc_to_step[lc_id] = step_id
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=step_id,
                    attributes={
                        "framework": "langgraph",
                        "node.name": node_name or "unknown",
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
            step_id = self._lc_to_step.pop(lc_id, None)
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

        self._node_names.pop(lc_id, None)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._chain_depth = max(0, self._chain_depth - 1)
        lc_id = str(run_id)

        self._observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._lc_to_step.get(lc_id),
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
        else:
            step_id = self._lc_to_step.pop(lc_id, None)
            if step_id:
                self._observer.emit(
                    AgentEvent(
                        name=EventName.STEP_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        step_id=step_id,
                        ok=False,
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
        tc_id = new_tool_call_id()
        self._lc_to_tool[lc_id] = tc_id

        tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        parent_step = self._lc_to_step.get(str(parent_run_id)) if parent_run_id else None

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=parent_step,
                tool_call_id=tc_id,
                tool_name=tool_name,
                attributes={
                    "framework": "langgraph",
                    "input": input_str[:2000],
                },
            )
        )

    def on_tool_end(
        self, output: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        lc_id = str(run_id)
        tc_id = self._lc_to_tool.pop(lc_id, None)
        if tc_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tc_id,
                    ok=True,
                    attributes={"output": str(output)[:2000]},
                )
            )

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        lc_id = str(run_id)
        tc_id = self._lc_to_tool.pop(lc_id, None)
        if tc_id:
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tc_id,
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
        self._lc_to_llm[lc_id] = llm_id

        parent_step = self._lc_to_step.get(str(parent_run_id)) if parent_run_id else None
        invocation_params = kwargs.get("invocation_params", {})
        model = invocation_params.get("model_name", serialized.get("name", "unknown"))

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=parent_step,
                llm_call_id=llm_id,
                model_name=model,
                attributes={
                    "framework": "langgraph",
                    "prompt_count": len(prompts),
                },
            )
        )

    def on_llm_end(
        self, response: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        lc_id = str(run_id)
        llm_id = self._lc_to_llm.pop(lc_id, None)
        if not llm_id:
            return

        attrs: dict[str, Any] = {}
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                attrs["prompt_tokens"] = usage.get("prompt_tokens", 0)
                attrs["completion_tokens"] = usage.get("completion_tokens", 0)
                attrs["total_tokens"] = usage.get("total_tokens", 0)

        self._observer.emit(
            AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                llm_call_id=llm_id,
                ok=True,
                attributes=attrs,
            )
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        lc_id = str(run_id)
        llm_id = self._lc_to_llm.pop(lc_id, None)
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


# --- Manual event adapter ---


class _GraphRunContext:
    """Context for a manual graph execution run."""

    def __init__(
        self, observer: AgentObserver, agent_id: str, run_id: str
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._open_steps: dict[str, str] = {}  # node_name -> step_id

    def on_node_start(
        self, node_name: str, state: Any = None, **attrs: Any
    ) -> str:
        """Record a node beginning execution. Returns step_id."""
        step_id = new_step_id()
        self._open_steps[node_name] = step_id
        node_attrs: dict[str, Any] = {
            "framework": "langgraph",
            "node.name": node_name,
        }
        node_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                attributes=node_attrs,
            )
        )
        return step_id

    def on_node_end(
        self,
        node_name: str,
        state: Any = None,
        ok: bool = True,
        **attrs: Any,
    ) -> None:
        """Record a node completing execution."""
        step_id = self._open_steps.pop(node_name, None)
        if not step_id:
            return
        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=step_id,
                ok=ok,
                attributes=attrs,
            )
        )

    def on_edge(
        self, from_node: str, to_node: str, condition: Optional[str] = None
    ) -> None:
        """Record an edge transition (informational span event)."""
        self._observer.emit(
            AgentEvent(
                name=EventName.MEMORY_WRITE,
                agent_id=self._agent_id,
                run_id=self._run_id,
                attributes={
                    "edge.from": from_node,
                    "edge.to": to_node,
                    "edge.condition": condition or "",
                },
            )
        )


class LangGraphEventAdapter:
    """Manual hooks for LangGraph graph execution."""

    def __init__(self, observer: AgentObserver, agent_id: str = "langgraph-agent") -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def run(
        self, task: str = "", **attrs: Any
    ) -> Generator[_GraphRunContext, None, None]:
        run_id = new_run_id()
        start_attrs: dict[str, Any] = {"framework": "langgraph"}
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

        ctx = _GraphRunContext(self._observer, self._agent_id, run_id)
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
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                )
            )
