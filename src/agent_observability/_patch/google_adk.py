"""Auto-instrumentation patch for Google Agent Development Kit."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_run_async: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_run_async, _installed
    if _installed:
        return

    from google.adk import Runner

    from agent_observability.adapters.google_adk import GoogleADKAdapter

    _original_run_async = Runner.run_async

    async def _patched_run_async(self: Any, *args: Any, **kwargs: Any) -> Any:
        adapter = GoogleADKAdapter(observer)
        agent_name = getattr(getattr(self, "agent", None), "name", "google-adk-auto")

        with adapter.run(agent_name=str(agent_name)) as run:
            async for event in _original_run_async(self, *args, **kwargs):
                run.on_event(event)
                yield event

    Runner.run_async = _patched_run_async  # type: ignore[assignment]
    _installed = True
    logger.debug("Google ADK auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from google.adk import Runner

    if _original_run_async is not None:
        Runner.run_async = _original_run_async  # type: ignore[assignment]
    _installed = False
