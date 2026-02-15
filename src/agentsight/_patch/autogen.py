"""Auto-instrumentation patch for Microsoft AutoGen."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_initiate_chat: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_initiate_chat, _installed
    if _installed:
        return

    from autogen import ConversableAgent

    from agentsight.adapters.autogen import AutoGenAdapter

    _original_initiate_chat = ConversableAgent.initiate_chat

    def _patched_initiate_chat(self: Any, recipient: Any, *args: Any, **kwargs: Any) -> Any:
        adapter = AutoGenAdapter(observer)
        sender_name = getattr(self, "name", "unknown")
        recipient_name = getattr(recipient, "name", "unknown")
        message = kwargs.get("message", str(args[0]) if args else "")

        with adapter.two_agent_chat(sender_name, recipient_name, task=str(message)[:500]) as chat:
            return _original_initiate_chat(self, recipient, *args, **kwargs)

    ConversableAgent.initiate_chat = _patched_initiate_chat  # type: ignore[assignment]
    _installed = True
    logger.debug("AutoGen auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from autogen import ConversableAgent

    if _original_initiate_chat is not None:
        ConversableAgent.initiate_chat = _original_initiate_chat  # type: ignore[assignment]
    _installed = False
