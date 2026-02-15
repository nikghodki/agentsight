"""Auto-instrumentation patch for HuggingFace smolagents."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_run: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_run, _installed
    if _installed:
        return

    from smolagents import CodeAgent

    from agent_observability.adapters.smolagents import SmolagentsAdapter

    _original_run = CodeAgent.run

    def _patched_run(self: Any, task: str, *args: Any, **kwargs: Any) -> Any:
        adapter = SmolagentsAdapter(observer, agent_id="smolagents-auto")
        monitor = adapter.create_monitor()
        existing_callbacks = list(getattr(self, "step_callbacks", None) or [])
        if not any(hasattr(cb, "_agent_obs_marker") for cb in existing_callbacks):
            monitor._agent_obs_marker = True  # type: ignore[attr-defined]
            self.step_callbacks = existing_callbacks + [monitor]
        return _original_run(self, task, *args, **kwargs)

    CodeAgent.run = _patched_run  # type: ignore[assignment]
    _installed = True
    logger.debug("smolagents auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from smolagents import CodeAgent

    if _original_run is not None:
        CodeAgent.run = _original_run  # type: ignore[assignment]
    _installed = False
