"""Auto-instrumentation patch for LlamaIndex."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)

_installed = False
_handler: Optional[Any] = None


def install(observer: AgentObserver) -> None:
    global _installed, _handler
    if _installed:
        return

    from llama_index.core import Settings

    from agentsight.adapters.llamaindex import LlamaIndexAdapter

    _handler = LlamaIndexAdapter(observer, agent_id="llamaindex-auto")
    Settings.callback_manager.add_handler(_handler)

    _installed = True
    logger.debug("LlamaIndex auto-instrumentation installed")


def uninstall() -> None:
    global _installed, _handler
    if not _installed:
        return
    try:
        from llama_index.core import Settings

        if _handler and _handler in Settings.callback_manager.handlers:
            Settings.callback_manager.handlers.remove(_handler)
    except Exception:
        pass
    _handler = None
    _installed = False
