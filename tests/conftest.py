"""
Shared test fixtures for agentsight tests.

OTel global providers can only be set once per process. We use session-scoped
setup for the providers and clear the in-memory exporter before each test.
"""

from __future__ import annotations

import threading
from typing import Sequence

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from agentsight.observer import AgentObserver
from agentsight.redaction import PayloadPolicy


class InMemorySpanExporter(SpanExporter):
    """Minimal in-memory exporter for test assertions."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True


# Module-level singletons — set once, reused across all tests
_span_exporter = InMemorySpanExporter()
_metric_reader = InMemoryMetricReader()
_otel_initialized = False


def _ensure_otel() -> None:
    global _otel_initialized
    if _otel_initialized:
        return
    resource = Resource.create({"service.name": "test"})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(_span_exporter))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(resource=resource, metric_readers=[_metric_reader])
    metrics.set_meter_provider(meter_provider)

    _otel_initialized = True


@pytest.fixture(autouse=True)
def _reset_exporter():
    """Clear collected spans before each test so tests are isolated."""
    _ensure_otel()
    _span_exporter.clear()
    yield


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Access the shared in-memory span exporter."""
    return _span_exporter


@pytest.fixture
def metric_reader() -> InMemoryMetricReader:
    """Access the shared in-memory metric reader."""
    return _metric_reader


@pytest.fixture
def observer() -> AgentObserver:
    """Pre-configured observer for tests."""
    return AgentObserver(
        tracer_name="test-agent",
        meter_name="test-agent",
        payload_policy=PayloadPolicy(),
    )
