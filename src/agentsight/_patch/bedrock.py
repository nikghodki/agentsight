"""Auto-instrumentation patch for Amazon Bedrock Agents."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_invoke_agent: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_invoke_agent, _installed
    if _installed:
        return

    import botocore.client

    from agentsight.adapters.bedrock_agents import BedrockAgentsAdapter

    _original_api_call = botocore.client.ClientCreator._create_api_method

    def _patched_create_api_method(self: Any, operation_name: str, *args: Any, **kwargs: Any) -> Any:
        method = _original_api_call(self, operation_name, *args, **kwargs)
        if operation_name != "InvokeAgent":
            return method

        def _wrapped(self_client: Any, **call_kwargs: Any) -> Any:
            adapter = BedrockAgentsAdapter(
                observer,
                agent_id=call_kwargs.get("agentId", "bedrock-auto"),
            )
            task = call_kwargs.get("inputText", "")
            response = method(self_client, **call_kwargs)

            # Wrap the completion stream to process trace events
            original_completion = response.get("completion", [])
            adapter_ctx = adapter.invocation(task=task)
            inv = adapter_ctx.__enter__()

            class _TracingStream:
                def __init__(self, stream: Any) -> None:
                    self._stream = iter(stream)

                def __iter__(self) -> "_TracingStream":
                    return self

                def __next__(self) -> Any:
                    try:
                        event = next(self._stream)
                        inv.process_event(event)
                        return event
                    except StopIteration:
                        adapter_ctx.__exit__(None, None, None)
                        raise

            response["completion"] = _TracingStream(original_completion)
            return response

        return _wrapped

    # Only patch if not already done
    if not getattr(botocore.client.ClientCreator, "_agent_obs_patched", False):
        botocore.client.ClientCreator._create_api_method = _patched_create_api_method  # type: ignore[attr-defined]
        botocore.client.ClientCreator._agent_obs_patched = True  # type: ignore[attr-defined]

    _installed = True
    logger.debug("Bedrock Agents auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    _installed = False
