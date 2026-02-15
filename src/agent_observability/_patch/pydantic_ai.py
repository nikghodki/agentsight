"""Auto-instrumentation patch for PydanticAI."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_run_sync: Optional[Any] = None
_original_run: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_run_sync, _original_run, _installed
    if _installed:
        return

    from pydantic_ai import Agent

    from agent_observability.adapters.pydantic_ai import PydanticAIAdapter

    _original_run_sync = Agent.run_sync

    def _patched_run_sync(self: Any, prompt: str, *args: Any, **kwargs: Any) -> Any:
        adapter = PydanticAIAdapter(observer, agent_id=getattr(self, "name", "pydantic-ai-auto"))
        model = str(getattr(self, "model", "unknown"))
        with adapter.run(task=prompt[:500], model=model):
            return _original_run_sync(self, prompt, *args, **kwargs)

    Agent.run_sync = _patched_run_sync  # type: ignore[assignment]

    if hasattr(Agent, "run"):
        _original_run = Agent.run

        async def _patched_run(self: Any, prompt: str, *args: Any, **kwargs: Any) -> Any:
            adapter = PydanticAIAdapter(observer, agent_id=getattr(self, "name", "pydantic-ai-auto"))
            model = str(getattr(self, "model", "unknown"))
            with adapter.run(task=prompt[:500], model=model):
                return await _original_run(self, prompt, *args, **kwargs)

        Agent.run = _patched_run  # type: ignore[assignment]

    _installed = True
    logger.debug("PydanticAI auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from pydantic_ai import Agent

    if _original_run_sync is not None:
        Agent.run_sync = _original_run_sync  # type: ignore[assignment]
    if _original_run is not None:
        Agent.run = _original_run  # type: ignore[assignment]
    _installed = False
