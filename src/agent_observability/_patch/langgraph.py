"""Auto-instrumentation patch for LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_invoke: Optional[Any] = None
_original_ainvoke: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_invoke, _original_ainvoke, _installed
    if _installed:
        return

    from langgraph.graph.graph import CompiledGraph

    from agent_observability.adapters.langgraph import LangGraphCallbackAdapter

    _original_invoke = CompiledGraph.invoke
    _original_ainvoke = CompiledGraph.ainvoke

    def _patched_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _ensure_callback(config, observer, LangGraphCallbackAdapter)
        return _original_invoke(self, input, config=config, **kwargs)

    async def _patched_ainvoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _ensure_callback(config, observer, LangGraphCallbackAdapter)
        return await _original_ainvoke(self, input, config=config, **kwargs)

    CompiledGraph.invoke = _patched_invoke  # type: ignore[assignment]
    CompiledGraph.ainvoke = _patched_ainvoke  # type: ignore[assignment]
    _installed = True
    logger.debug("LangGraph auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from langgraph.graph.graph import CompiledGraph

    if _original_invoke is not None:
        CompiledGraph.invoke = _original_invoke  # type: ignore[assignment]
    if _original_ainvoke is not None:
        CompiledGraph.ainvoke = _original_ainvoke  # type: ignore[assignment]
    _installed = False


def _ensure_callback(config: Any, observer: AgentObserver, adapter_cls: type) -> dict:
    config = dict(config) if config else {}
    callbacks = list(config.get("callbacks") or [])
    if not any(isinstance(cb, adapter_cls) for cb in callbacks):
        callbacks.insert(0, adapter_cls(observer, agent_id="langgraph-auto"))
        config["callbacks"] = callbacks
    return config
