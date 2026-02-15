"""Auto-instrumentation patch for Microsoft Semantic Kernel."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_invoke: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_invoke, _installed
    if _installed:
        return

    from semantic_kernel import Kernel

    from agentsight.adapters.semantic_kernel import SKAdapter

    adapter = SKAdapter(observer, agent_id="semantic-kernel-auto")

    _original_invoke = Kernel.invoke

    async def _patched_invoke(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Register filters if not already present
        _register_filters_once(self, adapter)
        return await _original_invoke(self, *args, **kwargs)

    Kernel.invoke = _patched_invoke  # type: ignore[assignment]
    _installed = True
    logger.debug("Semantic Kernel auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from semantic_kernel import Kernel

    if _original_invoke is not None:
        Kernel.invoke = _original_invoke  # type: ignore[assignment]
    _installed = False


_FILTER_TAG = "_agent_obs_filters_registered"


def _register_filters_once(kernel: Any, adapter: Any) -> None:
    if getattr(kernel, _FILTER_TAG, False):
        return
    try:
        kernel.add_filter("function_invocation", adapter.function_filter)
        kernel.add_filter("prompt_rendering", adapter.prompt_filter)
        kernel.add_filter("auto_function_invocation", adapter.auto_function_filter)
    except Exception:
        pass
    setattr(kernel, _FILTER_TAG, True)
