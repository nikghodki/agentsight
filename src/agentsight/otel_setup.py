"""
OpenTelemetry configuration for agent observability.

Provides a single `init_telemetry()` entry point that configures
traces, metrics, and (optionally) logs exporters.

Supports:
  - Console exporters (development)
  - OTLP/gRPC exporters (production)
  - OTLP/HTTP exporters (when gRPC is not available)

Configuration is driven by arguments + environment variables:

  OTEL_EXPORTER_OTLP_ENDPOINT   - OTLP endpoint (e.g. http://localhost:4317)
  OTEL_EXPORTER_OTLP_HEADERS    - Extra headers (e.g. "Authorization=Bearer xxx")
  OTEL_SERVICE_NAME              - Overrides service_name argument
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


class ExporterType(str, Enum):
    CONSOLE = "console"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"


def init_telemetry(
    service_name: str = "agentsight",
    exporter: ExporterType = ExporterType.CONSOLE,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[dict[str, str]] = None,
    metric_export_interval_ms: int = 10_000,
) -> tuple[TracerProvider, MeterProvider]:
    """
    Initialize OpenTelemetry providers for traces and metrics.

    Args:
        service_name: Service name for the OTel resource.
        exporter: Which exporter backend to use.
        otlp_endpoint: OTLP endpoint. Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var.
        otlp_headers: OTLP headers. Falls back to OTEL_EXPORTER_OTLP_HEADERS env var.
        metric_export_interval_ms: How often to flush metrics.

    Returns:
        Tuple of (TracerProvider, MeterProvider) for testing/shutdown access.
    """
    effective_service = os.environ.get("OTEL_SERVICE_NAME", service_name)

    resource = Resource.create(
        {
            "service.name": effective_service,
            "service.version": "0.1.0",
            "telemetry.sdk.language": "python",
        }
    )

    # --- Traces ---
    tracer_provider = TracerProvider(resource=resource)

    if exporter == ExporterType.CONSOLE:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == ExporterType.OTLP_GRPC:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(_create_otlp_grpc_span_exporter(otlp_endpoint, otlp_headers))
        )
    elif exporter == ExporterType.OTLP_HTTP:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(_create_otlp_http_span_exporter(otlp_endpoint, otlp_headers))
        )

    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ---
    if exporter == ExporterType.CONSOLE:
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=metric_export_interval_ms,
        )
    elif exporter == ExporterType.OTLP_GRPC:
        metric_reader = PeriodicExportingMetricReader(
            _create_otlp_grpc_metric_exporter(otlp_endpoint, otlp_headers),
            export_interval_millis=metric_export_interval_ms,
        )
    elif exporter == ExporterType.OTLP_HTTP:
        metric_reader = PeriodicExportingMetricReader(
            _create_otlp_http_metric_exporter(otlp_endpoint, otlp_headers),
            export_interval_millis=metric_export_interval_ms,
        )

    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    return tracer_provider, meter_provider


def shutdown_telemetry(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    timeout_ms: int = 5_000,
) -> None:
    """Flush and shut down providers. Call on process exit."""
    tracer_provider.force_flush(timeout_millis=timeout_ms)
    tracer_provider.shutdown()
    meter_provider.shutdown()


# --- OTLP exporter factories ---


def _resolve_endpoint(endpoint: Optional[str]) -> str:
    resolved = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    return resolved


def _resolve_headers(headers: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
    if headers:
        return headers
    raw = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    if raw:
        parsed: dict[str, str] = {}
        for pair in raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                parsed[k.strip()] = v.strip()
        return parsed
    return None


def _create_otlp_grpc_span_exporter(
    endpoint: Optional[str], headers: Optional[dict[str, str]]
) -> "BatchSpanProcessor":
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as e:
        raise ImportError(
            "OTLP gRPC exporter requires 'opentelemetry-exporter-otlp-proto-grpc'. "
            "Install with: pip install agentsight[otlp]"
        ) from e

    kwargs: dict = {"endpoint": _resolve_endpoint(endpoint)}
    resolved_headers = _resolve_headers(headers)
    if resolved_headers:
        kwargs["headers"] = list(resolved_headers.items())
    return OTLPSpanExporter(**kwargs)  # type: ignore[return-value]


def _create_otlp_http_span_exporter(
    endpoint: Optional[str], headers: Optional[dict[str, str]]
) -> "BatchSpanProcessor":
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as e:
        raise ImportError(
            "OTLP HTTP exporter requires 'opentelemetry-exporter-otlp-proto-http'. "
            "Install with: pip install agentsight[otlp]"
        ) from e

    ep = _resolve_endpoint(endpoint)
    if not ep.endswith("/v1/traces"):
        ep = ep.rstrip("/") + "/v1/traces"

    kwargs: dict = {"endpoint": ep}
    resolved_headers = _resolve_headers(headers)
    if resolved_headers:
        kwargs["headers"] = resolved_headers
    return OTLPSpanExporter(**kwargs)  # type: ignore[return-value]


def _create_otlp_grpc_metric_exporter(
    endpoint: Optional[str], headers: Optional[dict[str, str]]
) -> "PeriodicExportingMetricReader":
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    except ImportError as e:
        raise ImportError(
            "OTLP gRPC exporter requires 'opentelemetry-exporter-otlp-proto-grpc'. "
            "Install with: pip install agentsight[otlp]"
        ) from e

    kwargs: dict = {"endpoint": _resolve_endpoint(endpoint)}
    resolved_headers = _resolve_headers(headers)
    if resolved_headers:
        kwargs["headers"] = list(resolved_headers.items())
    return OTLPMetricExporter(**kwargs)  # type: ignore[return-value]


def _create_otlp_http_metric_exporter(
    endpoint: Optional[str], headers: Optional[dict[str, str]]
) -> "PeriodicExportingMetricReader":
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    except ImportError as e:
        raise ImportError(
            "OTLP HTTP exporter requires 'opentelemetry-exporter-otlp-proto-http'. "
            "Install with: pip install agentsight[otlp]"
        ) from e

    ep = _resolve_endpoint(endpoint)
    if not ep.endswith("/v1/metrics"):
        ep = ep.rstrip("/") + "/v1/metrics"

    kwargs: dict = {"endpoint": ep}
    resolved_headers = _resolve_headers(headers)
    if resolved_headers:
        kwargs["headers"] = resolved_headers
    return OTLPMetricExporter(**kwargs)  # type: ignore[return-value]
