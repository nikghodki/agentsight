"""Auto-instrumentation patch for Phidata."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_run: Optional[Any] = None
_original_print_response: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_run, _original_print_response, _installed
    if _installed:
        return

    from phi.agent import Agent

    from agentsight.adapters.phidata import PhidataAdapter

    _original_run = Agent.run

    def _patched_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        adapter = PhidataAdapter(observer, agent_id=getattr(self, "name", "phidata-auto"))
        task = str(args[0])[:500] if args else ""
        with adapter.run(task=task):
            return _original_run(self, *args, **kwargs)

    Agent.run = _patched_run  # type: ignore[assignment]

    if hasattr(Agent, "print_response"):
        _original_print_response = Agent.print_response

        def _patched_print_response(self: Any, *args: Any, **kwargs: Any) -> Any:
            adapter = PhidataAdapter(observer, agent_id=getattr(self, "name", "phidata-auto"))
            task = str(args[0])[:500] if args else ""
            with adapter.run(task=task):
                return _original_print_response(self, *args, **kwargs)

        Agent.print_response = _patched_print_response  # type: ignore[assignment]

    _installed = True
    logger.debug("Phidata auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from phi.agent import Agent

    if _original_run is not None:
        Agent.run = _original_run  # type: ignore[assignment]
    if _original_print_response is not None:
        Agent.print_response = _original_print_response  # type: ignore[assignment]
    _installed = False
