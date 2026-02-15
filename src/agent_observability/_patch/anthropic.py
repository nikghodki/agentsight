"""Auto-instrumentation patch for Anthropic SDK."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_observability.events import AgentEvent, EventName, new_llm_call_id, new_run_id
from agent_observability.observer import AgentObserver

logger = logging.getLogger(__name__)

_original_create: Optional[Any] = None
_original_acreate: Optional[Any] = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_create, _original_acreate, _installed
    if _installed:
        return

    from anthropic.resources.messages import AsyncMessages, Messages

    _original_create = Messages.create

    def _patched_create(self: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        run_id = new_run_id()
        llm_id = new_llm_call_id()

        observer.emit(AgentEvent(
            name=EventName.LLM_CALL_START,
            agent_id="anthropic-auto",
            run_id=run_id,
            llm_call_id=llm_id,
            model_name=str(model),
            attributes={"framework": "anthropic", "auto_instrumented": True},
        ))

        ok = True
        error: Optional[Exception] = None
        try:
            result = _original_create(self, **kwargs)
        except Exception as e:
            ok = False
            error = e
            raise
        finally:
            attrs: dict[str, Any] = {}
            if ok and hasattr(result, "usage"):  # type: ignore[possibly-undefined]
                usage = result.usage  # type: ignore[possibly-undefined]
                attrs["input_tokens"] = getattr(usage, "input_tokens", 0)
                attrs["output_tokens"] = getattr(usage, "output_tokens", 0)
            observer.emit(AgentEvent(
                name=EventName.LLM_CALL_END,
                agent_id="anthropic-auto",
                run_id=run_id,
                llm_call_id=llm_id,
                model_name=str(model),
                ok=ok,
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
                attributes=attrs,
            ))

        return result

    Messages.create = _patched_create  # type: ignore[assignment]

    # Async variant
    if hasattr(AsyncMessages, "create"):
        _original_acreate = AsyncMessages.create

        async def _patched_acreate(self: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "unknown")
            run_id = new_run_id()
            llm_id = new_llm_call_id()

            observer.emit(AgentEvent(
                name=EventName.LLM_CALL_START,
                agent_id="anthropic-auto",
                run_id=run_id,
                llm_call_id=llm_id,
                model_name=str(model),
                attributes={"framework": "anthropic", "auto_instrumented": True},
            ))

            ok = True
            error: Optional[Exception] = None
            try:
                result = await _original_acreate(self, **kwargs)
            except Exception as e:
                ok = False
                error = e
                raise
            finally:
                attrs: dict[str, Any] = {}
                if ok and hasattr(result, "usage"):  # type: ignore[possibly-undefined]
                    usage = result.usage  # type: ignore[possibly-undefined]
                    attrs["input_tokens"] = getattr(usage, "input_tokens", 0)
                    attrs["output_tokens"] = getattr(usage, "output_tokens", 0)
                observer.emit(AgentEvent(
                    name=EventName.LLM_CALL_END,
                    agent_id="anthropic-auto",
                    run_id=run_id,
                    llm_call_id=llm_id,
                    model_name=str(model),
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes=attrs,
                ))

            return result

        AsyncMessages.create = _patched_acreate  # type: ignore[assignment]

    _installed = True
    logger.debug("Anthropic auto-instrumentation installed")


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from anthropic.resources.messages import AsyncMessages, Messages

    if _original_create is not None:
        Messages.create = _original_create  # type: ignore[assignment]
    if _original_acreate is not None:
        AsyncMessages.create = _original_acreate  # type: ignore[assignment]
    _installed = False
