"""Auto-instrumentation patch for LangChain."""

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

    from langchain_core.runnables import Runnable

    from agent_observability.adapters.langchain import LangChainAdapter

    _original_invoke = Runnable.invoke
    _original_ainvoke = Runnable.ainvoke

    def _patched_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _ensure_callback(config, observer, self, LangChainAdapter)
        return _original_invoke(self, input, config=config, **kwargs)

    async def _patched_ainvoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        config = _ensure_callback(config, observer, self, LangChainAdapter)
        return await _original_ainvoke(self, input, config=config, **kwargs)

    Runnable.invoke = _patched_invoke  # type: ignore[assignment]
    Runnable.ainvoke = _patched_ainvoke  # type: ignore[assignment]
    _installed = True
    logger.debug("LangChain auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from langchain_core.runnables import Runnable

    if _original_invoke is not None:
        Runnable.invoke = _original_invoke  # type: ignore[assignment]
    if _original_ainvoke is not None:
        Runnable.ainvoke = _original_ainvoke  # type: ignore[assignment]
    _installed = False


def _ensure_callback(config: Any, observer: AgentObserver, runnable: Any, adapter_cls: type) -> dict:
    config = dict(config) if config else {}
    callbacks = list(config.get("callbacks") or [])
    if not any(isinstance(cb, adapter_cls) for cb in callbacks):
        agent_id = getattr(runnable, "name", None) or type(runnable).__name__
        callbacks.insert(0, adapter_cls(observer, agent_id=str(agent_id)))
        config["callbacks"] = callbacks
    return config
