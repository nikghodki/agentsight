"""
LlamaIndex adapter: implements LlamaIndex's callback handler interface.

Targets: llama-index-core >= 0.10

LlamaIndex uses a CallbackManager with event-type-based dispatching.
Each event has a start/end pair identified by a shared event_id.

Mapping:
  CBEventType.AGENT_STEP   -> agent.step.start / end
  CBEventType.LLM          -> agent.llm.call.start / end
  CBEventType.FUNCTION_CALL / TOOL -> agent.tool.call.start / end
  CBEventType.RETRIEVE     -> agent.tool.call.start / end (retrieval as tool)
  CBEventType.QUERY        -> agent.lifecycle.start / end
  CBEventType.EMBEDDING    -> span event on active step

Usage:
    from llama_index.core.callbacks import CallbackManager
    from agent_observability import AgentObserver, init_telemetry
    from agent_observability.adapters.llamaindex import LlamaIndexAdapter

    init_telemetry()
    observer = AgentObserver()
    handler = LlamaIndexAdapter(observer, agent_id="rag-agent")

    callback_manager = CallbackManager([handler])
    index = VectorStoreIndex.from_documents(docs, callback_manager=callback_manager)
    query_engine = index.as_query_engine(callback_manager=callback_manager)
    response = query_engine.query("What is AI safety?")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class CBEventType:  # type: ignore[no-redef]
        QUERY = "query"
        LLM = "llm"
        RETRIEVE = "retrieve"
        AGENT_STEP = "agent_step"
        FUNCTION_CALL = "function_call"
        EMBEDDING = "embedding"

    class EventPayload:  # type: ignore[no-redef]
        pass


# Event types we want to trace (others are logged as span events)
_TRACED_EVENT_TYPES = {
    "query",
    "llm",
    "retrieve",
    "agent_step",
    "function_call",
    "tool",
}


class LlamaIndexAdapter(BaseCallbackHandler):  # type: ignore[misc]
    """
    LlamaIndex callback handler that emits AgentEvents.

    Implements on_event_start() and on_event_end() which LlamaIndex's
    CallbackManager calls for every traced operation.
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "llamaindex-agent",
        event_starts_to_ignore: Optional[list[str]] = None,
        event_ends_to_ignore: Optional[list[str]] = None,
    ) -> None:
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex adapter requires 'llama-index-core'. "
                "Install with: pip install agent-observability[llamaindex]"
            )
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )
        self._observer = observer
        self._agent_id = agent_id
        self._run_id: str = new_run_id()

        # Track by LlamaIndex event_id
        self._event_to_step: dict[str, str] = {}       # event_id -> step_id
        self._event_to_tool: dict[str, str] = {}       # event_id -> tool_call_id
        self._event_to_llm: dict[str, str] = {}        # event_id -> llm_call_id
        self._event_to_run: dict[str, str] = {}        # event_id -> run_id
        self._query_depth: int = 0

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Called when a new trace starts (e.g., query begins)."""
        self._run_id = new_run_id()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Called when a trace ends."""
        pass

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called at the start of each traced event."""
        etype = str(event_type.value) if hasattr(event_type, "value") else str(event_type)
        safe_payload = payload or {}

        if etype == "query":
            self._query_depth += 1
            if self._query_depth == 1:
                self._run_id = new_run_id()
                self._event_to_run[event_id] = self._run_id
                query_str = safe_payload.get("query_str", safe_payload.get("query_bundle", ""))
                self._observer.emit(
                    AgentEvent(
                        name=EventName.LIFECYCLE_START,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        attributes={
                            "framework": "llamaindex",
                            "query": str(query_str)[:2000],
                        },
                    )
                )
            else:
                # Sub-query = step
                step_id = new_step_id()
                self._event_to_step[event_id] = step_id
                self._observer.emit(
                    AgentEvent(
                        name=EventName.STEP_START,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        step_id=step_id,
                        attributes={
                            "framework": "llamaindex",
                            "event_type": "sub_query",
                        },
                    )
                )

        elif etype == "agent_step":
            step_id = new_step_id()
            self._event_to_step[event_id] = step_id
            task_str = ""
            if "task" in safe_payload:
                task_obj = safe_payload["task"]
                task_str = str(getattr(task_obj, "input", task_obj))[:1000]

            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=step_id,
                    attributes={
                        "framework": "llamaindex",
                        "event_type": etype,
                        "task": task_str,
                    },
                )
            )

        elif etype == "llm":
            llm_id = new_llm_call_id()
            self._event_to_llm[event_id] = llm_id

            model = safe_payload.get("model_name", "unknown")
            messages = safe_payload.get("messages", [])
            serialized = safe_payload.get("serialized", {})
            if not model or model == "unknown":
                model = serialized.get("model", "unknown")

            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    llm_call_id=llm_id,
                    model_name=str(model),
                    attributes={
                        "framework": "llamaindex",
                        "message_count": len(messages),
                    },
                )
            )

        elif etype in ("function_call", "tool"):
            tc_id = new_tool_call_id()
            self._event_to_tool[event_id] = tc_id

            tool_name = safe_payload.get("tool", safe_payload.get("function_call", "unknown"))
            if hasattr(tool_name, "name"):
                tool_name = tool_name.name
            tool_input = safe_payload.get("function_call_args", safe_payload.get("tool_args", ""))

            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tc_id,
                    tool_name=str(tool_name),
                    attributes={
                        "framework": "llamaindex",
                        "input": str(tool_input)[:4096],
                    },
                )
            )

        elif etype == "retrieve":
            tc_id = new_tool_call_id()
            self._event_to_tool[event_id] = tc_id
            query_str = safe_payload.get("query_str", "")

            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tc_id,
                    tool_name="retriever",
                    attributes={
                        "framework": "llamaindex",
                        "query": str(query_str)[:2000],
                    },
                )
            )

        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called at the end of each traced event."""
        etype = str(event_type.value) if hasattr(event_type, "value") else str(event_type)
        safe_payload = payload or {}

        if etype == "query":
            self._query_depth = max(0, self._query_depth - 1)

            if event_id in self._event_to_run:
                self._event_to_run.pop(event_id)
                response = safe_payload.get("response", "")
                self._observer.emit(
                    AgentEvent(
                        name=EventName.LIFECYCLE_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        ok=True,
                        attributes={"response_preview": str(response)[:500]},
                    )
                )
            else:
                step_id = self._event_to_step.pop(event_id, None)
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

        elif etype == "agent_step":
            step_id = self._event_to_step.pop(event_id, None)
            if step_id:
                response = safe_payload.get("response", "")
                self._observer.emit(
                    AgentEvent(
                        name=EventName.STEP_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        step_id=step_id,
                        ok=True,
                        attributes={"response_preview": str(response)[:500]},
                    )
                )

        elif etype == "llm":
            llm_id = self._event_to_llm.pop(event_id, None)
            if llm_id:
                attrs: dict[str, Any] = {}
                completion = safe_payload.get("completion", safe_payload.get("response", ""))
                if completion:
                    attrs["response_preview"] = str(completion)[:500]
                # Token counts
                token_usage = safe_payload.get("token_usage", {})
                if isinstance(token_usage, dict):
                    attrs.update(token_usage)

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

        elif etype in ("function_call", "tool"):
            tc_id = self._event_to_tool.pop(event_id, None)
            if tc_id:
                output = safe_payload.get("function_call_response", safe_payload.get("tool_output", ""))
                self._observer.emit(
                    AgentEvent(
                        name=EventName.TOOL_CALL_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        tool_call_id=tc_id,
                        ok=True,
                        attributes={"output": str(output)[:4096]},
                    )
                )

        elif etype == "retrieve":
            tc_id = self._event_to_tool.pop(event_id, None)
            if tc_id:
                nodes = safe_payload.get("nodes", [])
                self._observer.emit(
                    AgentEvent(
                        name=EventName.TOOL_CALL_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        tool_call_id=tc_id,
                        tool_name="retriever",
                        ok=True,
                        attributes={"node_count": len(nodes)},
                    )
                )
