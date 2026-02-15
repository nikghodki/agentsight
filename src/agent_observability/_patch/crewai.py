"""Auto-instrumentation patch for CrewAI."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_kickoff: Optional[Any] = None
_original_kickoff_async: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_kickoff, _original_kickoff_async, _installed
    if _installed:
        return

    from crewai import Crew

    from agent_observability.adapters.crewai import CrewAIAdapter

    _original_kickoff = Crew.kickoff

    def _patched_kickoff(self: Any, *args: Any, **kwargs: Any) -> Any:
        adapter = CrewAIAdapter(observer)
        crew_name = getattr(self, "name", None) or "crewai-auto"
        with adapter.observe_crew(crew_name=str(crew_name)):
            return _original_kickoff(self, *args, **kwargs)

    Crew.kickoff = _patched_kickoff  # type: ignore[assignment]

    if hasattr(Crew, "kickoff_async"):
        _original_kickoff_async = Crew.kickoff_async

        async def _patched_kickoff_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            adapter = CrewAIAdapter(observer)
            crew_name = getattr(self, "name", None) or "crewai-auto"
            with adapter.observe_crew(crew_name=str(crew_name)):
                return await _original_kickoff_async(self, *args, **kwargs)

        Crew.kickoff_async = _patched_kickoff_async  # type: ignore[assignment]

    _installed = True
    logger.debug("CrewAI auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from crewai import Crew

    if _original_kickoff is not None:
        Crew.kickoff = _original_kickoff  # type: ignore[assignment]
    if _original_kickoff_async is not None:
        Crew.kickoff_async = _original_kickoff_async  # type: ignore[assignment]
    _installed = False
