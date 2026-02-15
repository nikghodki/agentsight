"""
Global singleton state for auto-instrumentation.

Manages a shared AgentObserver, OTel providers, and tracks which
frameworks have been instrumented.
"""

from __future__ import annotations

import atexit
import logging
from typing import Optional

from agentsight.observer import AgentObserver
from agentsight.otel_setup import ExporterType, init_telemetry, shutdown_telemetry
from agentsight.redaction import PayloadPolicy

logger = logging.getLogger(__name__)

_observer: Optional[AgentObserver] = None
_tracer_provider: Optional[object] = None
_meter_provider: Optional[object] = None
_initialized: bool = False
_instrumented: set[str] = set()
_atexit_registered: bool = False


def initialize(
    service_name: str = "agentsight",
    exporter: ExporterType = ExporterType.CONSOLE,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[dict[str, str]] = None,
    payload_policy: Optional[PayloadPolicy] = None,
) -> AgentObserver:
    """
    Initialize telemetry and create the global observer.

    Safe to call multiple times -- subsequent calls are no-ops
    and return the existing observer.
    """
    global _observer, _tracer_provider, _meter_provider, _initialized, _atexit_registered

    if _initialized and _observer is not None:
        return _observer

    tp, mp = init_telemetry(
        service_name=service_name,
        exporter=exporter,
        otlp_endpoint=otlp_endpoint,
        otlp_headers=otlp_headers,
    )
    _tracer_provider = tp
    _meter_provider = mp

    _observer = AgentObserver(payload_policy=payload_policy)
    _initialized = True

    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True

    return _observer


def get_observer() -> Optional[AgentObserver]:
    """Return the global observer, or None if not initialized."""
    return _observer


def mark_instrumented(framework: str) -> None:
    """Record that a framework has been instrumented."""
    _instrumented.add(framework)


def is_instrumented(framework: str) -> bool:
    """Check if a framework has already been instrumented."""
    return framework in _instrumented


def instrumented_frameworks() -> set[str]:
    """Return the set of instrumented framework names."""
    return set(_instrumented)


def shutdown() -> None:
    """Flush and shut down OTel providers. Called automatically at process exit."""
    global _initialized
    if not _initialized:
        return
    try:
        if _tracer_provider and _meter_provider:
            shutdown_telemetry(_tracer_provider, _meter_provider)  # type: ignore[arg-type]
    except Exception:
        logger.debug("Error during telemetry shutdown", exc_info=True)
    _initialized = False


def reset() -> None:
    """Reset all global state. For testing only."""
    global _observer, _tracer_provider, _meter_provider, _initialized, _instrumented
    _observer = None
    _tracer_provider = None
    _meter_provider = None
    _initialized = False
    _instrumented = set()
