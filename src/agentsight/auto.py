"""
One-line auto-instrumentation for agent frameworks.

Usage::

    from agentsight import auto_instrument
    auto_instrument()                          # instrument everything detected
    auto_instrument(frameworks=["langchain"])   # selective

After calling ``auto_instrument()``, all instrumented framework calls
emit OpenTelemetry spans automatically with zero changes to existing code.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional, Sequence

from agentsight._state import (
    get_observer,
    initialize,
    instrumented_frameworks,
    is_instrumented,
    mark_instrumented,
    reset as _reset_state,
    shutdown as _shutdown_state,
)
from agentsight.otel_setup import ExporterType
from agentsight.redaction import PayloadPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework registry
# ---------------------------------------------------------------------------

# Maps framework name -> (import_probe, patch_module)
#   import_probe : module path to try-import to detect availability
#   patch_module : dotted path under agentsight._patch

_REGISTRY: dict[str, tuple[str, str]] = {
    "langchain": ("langchain_core", "agentsight._patch.langchain"),
    "langgraph": ("langgraph", "agentsight._patch.langgraph"),
    "openai_agents": ("agents", "agentsight._patch.openai_agents"),
    "anthropic": ("anthropic", "agentsight._patch.anthropic"),
    "crewai": ("crewai", "agentsight._patch.crewai"),
    "autogen": ("autogen", "agentsight._patch.autogen"),
    "llamaindex": ("llama_index.core", "agentsight._patch.llamaindex"),
    "semantic_kernel": ("semantic_kernel", "agentsight._patch.semantic_kernel"),
    "google_adk": ("google.adk", "agentsight._patch.google_adk"),
    "bedrock": ("botocore", "agentsight._patch.bedrock"),
    "haystack": ("haystack", "agentsight._patch.haystack"),
    "smolagents": ("smolagents", "agentsight._patch.smolagents"),
    "pydantic_ai": ("pydantic_ai", "agentsight._patch.pydantic_ai"),
    "phidata": ("phi", "agentsight._patch.phidata"),
}


def _framework_available(import_probe: str) -> bool:
    """Return True if the framework's top-level package is importable."""
    try:
        importlib.import_module(import_probe)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _install_framework(name: str, patch_module_path: str) -> bool:
    """Import the patch module and call install(). Returns True on success."""
    observer = get_observer()
    if observer is None:
        logger.error("Cannot instrument %s: observer not initialized", name)
        return False

    try:
        mod = importlib.import_module(patch_module_path)
        mod.install(observer)
        mark_instrumented(name)
        logger.info("Instrumented: %s", name)
        return True
    except Exception:
        logger.warning("Failed to instrument %s", name, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_instrument(
    *,
    service_name: str = "agentsight",
    exporter: ExporterType = ExporterType.CONSOLE,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[dict[str, str]] = None,
    payload_policy: Optional[PayloadPolicy] = None,
    frameworks: Optional[Sequence[str]] = None,
) -> dict[str, bool]:
    """
    Detect installed agent frameworks and instrument them automatically.

    Parameters
    ----------
    service_name : str
        OpenTelemetry service name (default ``"agentsight"``).
    exporter : ExporterType
        Telemetry exporter: CONSOLE, OTLP_GRPC, or OTLP_HTTP.
    otlp_endpoint : str, optional
        OTLP collector endpoint.
    otlp_headers : dict, optional
        Extra headers for the OTLP exporter.
    payload_policy : PayloadPolicy, optional
        Redaction policy for payloads in spans.
    frameworks : list of str, optional
        Only instrument these frameworks. If ``None``, instrument all detected.

    Returns
    -------
    dict[str, bool]
        Maps framework name to whether it was successfully instrumented.

    Examples
    --------
    >>> from agentsight import auto_instrument
    >>> results = auto_instrument()
    >>> # All installed frameworks are now emitting spans.

    >>> results = auto_instrument(frameworks=["langchain", "anthropic"])
    >>> # Only LangChain and Anthropic are instrumented.

    >>> results = auto_instrument(
    ...     service_name="my-agent",
    ...     exporter=ExporterType.OTLP_GRPC,
    ...     otlp_endpoint="http://collector:4317",
    ... )
    """
    # Step 1: Initialize telemetry + observer
    initialize(
        service_name=service_name,
        exporter=exporter,
        otlp_endpoint=otlp_endpoint,
        otlp_headers=otlp_headers,
        payload_policy=payload_policy,
    )

    # Step 2: Determine which frameworks to try
    if frameworks is not None:
        targets = {name: _REGISTRY[name] for name in frameworks if name in _REGISTRY}
        unknown = set(frameworks) - set(_REGISTRY)
        if unknown:
            logger.warning("Unknown frameworks (ignored): %s", ", ".join(sorted(unknown)))
    else:
        targets = dict(_REGISTRY)

    # Step 3: Instrument each available framework
    results: dict[str, bool] = {}
    for name, (import_probe, patch_module) in targets.items():
        if is_instrumented(name):
            results[name] = True
            continue

        if not _framework_available(import_probe):
            logger.debug("Skipping %s (not installed)", name)
            results[name] = False
            continue

        results[name] = _install_framework(name, patch_module)

    instrumented = [k for k, v in results.items() if v]
    skipped = [k for k, v in results.items() if not v]

    if instrumented:
        logger.info(
            "Auto-instrumented %d framework(s): %s",
            len(instrumented),
            ", ".join(sorted(instrumented)),
        )
    if skipped:
        logger.debug("Not instrumented (not installed or failed): %s", ", ".join(sorted(skipped)))

    return results


def uninstrument(frameworks: Optional[Sequence[str]] = None) -> None:
    """
    Remove instrumentation patches.

    Parameters
    ----------
    frameworks : list of str, optional
        Only remove these. If ``None``, remove all.
    """
    targets = frameworks if frameworks else list(instrumented_frameworks())

    for name in targets:
        if not is_instrumented(name):
            continue

        entry = _REGISTRY.get(name)
        if not entry:
            continue

        _, patch_module_path = entry
        try:
            mod = importlib.import_module(patch_module_path)
            mod.uninstall()
            logger.info("Uninstalled: %s", name)
        except Exception:
            logger.warning("Failed to uninstall %s", name, exc_info=True)

    # Reset the state tracking
    _reset_state()
    _shutdown_state()


# ---------------------------------------------------------------------------
# Per-framework convenience functions
# ---------------------------------------------------------------------------


def _make_single_instrument(name: str) -> Callable[..., bool]:
    """Factory for per-framework instrument_*() functions."""

    def _instrument(
        *,
        service_name: str = "agentsight",
        exporter: ExporterType = ExporterType.CONSOLE,
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[dict[str, str]] = None,
        payload_policy: Optional[PayloadPolicy] = None,
    ) -> bool:
        initialize(
            service_name=service_name,
            exporter=exporter,
            otlp_endpoint=otlp_endpoint,
            otlp_headers=otlp_headers,
            payload_policy=payload_policy,
        )

        entry = _REGISTRY.get(name)
        if not entry:
            return False

        import_probe, patch_module = entry
        if is_instrumented(name):
            return True

        if not _framework_available(import_probe):
            logger.warning("Cannot instrument %s: package not installed", name)
            return False

        return _install_framework(name, patch_module)

    _instrument.__name__ = f"instrument_{name}"
    _instrument.__qualname__ = f"instrument_{name}"
    _instrument.__doc__ = f"""Instrument {name} with agentsight.

    Initializes telemetry (if needed) and patches {name} entry points
    to emit OpenTelemetry spans automatically.

    Returns True on success.
    """
    return _instrument


# Generate per-framework functions
instrument_langchain = _make_single_instrument("langchain")
instrument_langgraph = _make_single_instrument("langgraph")
instrument_openai_agents = _make_single_instrument("openai_agents")
instrument_anthropic = _make_single_instrument("anthropic")
instrument_crewai = _make_single_instrument("crewai")
instrument_autogen = _make_single_instrument("autogen")
instrument_llamaindex = _make_single_instrument("llamaindex")
instrument_semantic_kernel = _make_single_instrument("semantic_kernel")
instrument_google_adk = _make_single_instrument("google_adk")
instrument_bedrock = _make_single_instrument("bedrock")
instrument_haystack = _make_single_instrument("haystack")
instrument_smolagents = _make_single_instrument("smolagents")
instrument_pydantic_ai = _make_single_instrument("pydantic_ai")
instrument_phidata = _make_single_instrument("phidata")


def available_frameworks() -> list[str]:
    """Return names of frameworks that are importable in the current environment."""
    return [
        name
        for name, (import_probe, _) in _REGISTRY.items()
        if _framework_available(import_probe)
    ]
