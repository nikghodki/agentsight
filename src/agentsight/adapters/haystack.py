"""
deepset Haystack adapter: hooks into Haystack's pipeline tracing system.

Targets: haystack-ai >= 2.0

Haystack 2.x uses a pipeline architecture where Components are connected
via an execution graph. The tracing integration point is the
`tracing.Tracer` interface with `trace()` context manager.

Mapping:
  pipeline.run()      -> agent.lifecycle.start / end
  component.run()     -> agent.step.start / end (or tool/llm based on type)
  generator component -> agent.llm.call.start / end
  retriever component -> agent.tool.call.start / end
  tool component      -> agent.tool.call.start / end

Usage:
    from haystack import Pipeline
    from agentsight import AgentObserver, init_telemetry
    from agentsight.adapters.haystack import HaystackAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = HaystackAdapter(observer, agent_id="rag-pipeline")

    with adapter.pipeline_run(pipeline_name="qa-pipeline") as run:
        result = pipeline.run({"query": "What is AI?"})
        run.record_result(result)

    # Or use component-level hooks:
    adapter.on_component_start("retriever", "BM25Retriever", {"query": "AI"})
    adapter.on_component_end("retriever", {"documents": [...]})
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

# Component types that map to specific span kinds
_LLM_COMPONENT_TYPES = {
    "generator",
    "openai",
    "anthropic",
    "huggingface",
    "azure",
    "ollama",
    "chatgenerator",
    "openai_generator",
    "chat_generator",
    "huggingface_local_generator",
    "huggingface_api_generator",
}

_RETRIEVER_COMPONENT_TYPES = {
    "retriever",
    "bm25retriever",
    "embedding_retriever",
    "in_memory_retriever",
    "elasticsearch_retriever",
    "pinecone_retriever",
    "weaviate_retriever",
    "qdrant_retriever",
}


def _classify_component(
    component_name: str, component_type: str
) -> str:
    """Classify a component as 'llm', 'retriever', or 'step'."""
    name_lower = component_name.lower()
    type_lower = component_type.lower()

    for pattern in _LLM_COMPONENT_TYPES:
        if pattern in name_lower or pattern in type_lower:
            return "llm"
    for pattern in _RETRIEVER_COMPONENT_TYPES:
        if pattern in name_lower or pattern in type_lower:
            return "retriever"
    return "step"


class _PipelineRunContext:
    """Context for a running pipeline."""

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

    @property
    def run_id(self) -> str:
        return self._run_id

    def record_result(self, result: Any) -> None:
        """Record the pipeline result for inclusion in the end span."""
        self._result = result


class HaystackAdapter:
    """
    Haystack 2.x pipeline observability adapter.

    Provides both a context-manager API for pipeline runs and
    explicit methods for component-level hooks.
    """

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "haystack-pipeline",
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id: Optional[str] = None
        self._active_steps: dict[str, str] = {}   # component_name -> step_id
        self._active_tools: dict[str, str] = {}   # component_name -> tool_call_id
        self._active_llms: dict[str, str] = {}    # component_name -> llm_call_id

    @contextmanager
    def pipeline_run(
        self,
        pipeline_name: str = "haystack-pipeline",
        task: str = "",
        **attrs: Any,
    ) -> Generator[_PipelineRunContext, None, None]:
        """Wrap a full pipeline.run() execution."""
        run_id = new_run_id()
        self._run_id = run_id

        start_attrs: dict[str, Any] = {
            "framework": "haystack",
            "pipeline.name": pipeline_name,
        }
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

        ctx = _PipelineRunContext(self._observer, self._agent_id, run_id)
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
            self._run_id = None
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

    def on_component_start(
        self,
        component_name: str,
        component_type: str = "",
        inputs: Optional[dict[str, Any]] = None,
        **attrs: Any,
    ) -> None:
        """Record a component starting execution."""
        if not self._run_id:
            self._run_id = new_run_id()

        kind = _classify_component(component_name, component_type)
        safe_inputs = inputs or {}

        if kind == "llm":
            llm_id = new_llm_call_id()
            self._active_llms[component_name] = llm_id
            call_attrs: dict[str, Any] = {
                "framework": "haystack",
                "component.name": component_name,
                "component.type": component_type,
            }
            call_attrs.update(attrs)
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    llm_call_id=llm_id,
                    model_name=component_type or component_name,
                    attributes=call_attrs,
                )
            )

        elif kind == "retriever":
            tc_id = new_tool_call_id()
            self._active_tools[component_name] = tc_id
            call_attrs = {
                "framework": "haystack",
                "component.name": component_name,
                "component.type": component_type,
                "input": str(safe_inputs)[:2000],
            }
            call_attrs.update(attrs)
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tc_id,
                    tool_name=component_name,
                    attributes=call_attrs,
                )
            )

        else:
            step_id = new_step_id()
            self._active_steps[component_name] = step_id
            step_attrs: dict[str, Any] = {
                "framework": "haystack",
                "component.name": component_name,
                "component.type": component_type,
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

    def on_component_end(
        self,
        component_name: str,
        outputs: Optional[dict[str, Any]] = None,
        ok: bool = True,
        error: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        """Record a component completing execution."""
        if not self._run_id:
            return

        # Check which category this component is in
        if component_name in self._active_llms:
            llm_id = self._active_llms.pop(component_name)
            end_attrs: dict[str, Any] = {}
            if outputs:
                # Extract token usage if available
                for key in ("meta", "metadata"):
                    if key in outputs:
                        meta = outputs[key]
                        if isinstance(meta, dict):
                            usage = meta.get("usage", {})
                            if usage:
                                end_attrs.update(usage)
            end_attrs.update(attrs)
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    llm_call_id=llm_id,
                    ok=ok,
                    error_message=error,
                    attributes=end_attrs,
                )
            )

        elif component_name in self._active_tools:
            tc_id = self._active_tools.pop(component_name)
            end_attrs = {}
            if outputs:
                doc_count = len(outputs.get("documents", []))
                end_attrs["document_count"] = doc_count
            end_attrs.update(attrs)
            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    tool_call_id=tc_id,
                    tool_name=component_name,
                    ok=ok,
                    error_message=error,
                    attributes=end_attrs,
                )
            )

        elif component_name in self._active_steps:
            step_id = self._active_steps.pop(component_name)
            self._observer.emit(
                AgentEvent(
                    name=EventName.STEP_END,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=step_id,
                    ok=ok,
                    error_message=error,
                    attributes=attrs,
                )
            )
