"""Auto-instrumentation patch for OpenAI Agents SDK."""

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

    from agents import Runner

    from agent_observability.adapters.openai_agents import OpenAIRunHooksAdapter

    _original_run = Runner.run

    @staticmethod  # type: ignore[misc]
    async def _patched_run(agent: Any, input: Any, **kwargs: Any) -> Any:
        if "run_hooks" not in kwargs or kwargs["run_hooks"] is None:
            hooks = OpenAIRunHooksAdapter(observer)
            hooks.start_run()
            kwargs["run_hooks"] = hooks
            try:
                result = await _original_run(agent, input, **kwargs)
                hooks.end_run(ok=True)
                return result
            except Exception as e:
                hooks.end_run(ok=False, error=str(e))
                raise
        return await _original_run(agent, input, **kwargs)

    Runner.run = _patched_run  # type: ignore[assignment]
    _installed = True
    logger.debug("OpenAI Agents SDK auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from agents import Runner

    if _original_run is not None:
        Runner.run = _original_run  # type: ignore[assignment]
    _installed = False
