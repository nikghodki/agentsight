"""Auto-instrumentation patch for Haystack."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_run: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_run, _installed
    if _installed:
        return

    from haystack import Pipeline

    from agentsight.adapters.haystack import HaystackAdapter

    _original_run = Pipeline.run

    def _patched_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        adapter = HaystackAdapter(observer, agent_id="haystack-auto")
        pipeline_name = getattr(self, "name", None) or "haystack-pipeline"
        with adapter.pipeline_run(pipeline_name=str(pipeline_name)):
            return _original_run(self, *args, **kwargs)

    Pipeline.run = _patched_run  # type: ignore[assignment]
    _installed = True
    logger.debug("Haystack auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from haystack import Pipeline

    if _original_run is not None:
        Pipeline.run = _original_run  # type: ignore[assignment]
    _installed = False
