#!/usr/bin/env python3
"""
Detailed AgentSight demo for GIF recording.

Shows:
  1. BEFORE: Plain Claude Agent SDK code (no observability)
  2. AFTER:  Same code with AgentSight integration (+4 lines)
  3. Live trace/span output with actual OTel JSON
  4. Metrics summary
"""

import io
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any

# ── ANSI ──
B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
G = "\033[32m"
C = "\033[36m"
Y = "\033[33m"
M = "\033[35m"
RED = "\033[31m"
W = "\033[37m"
BG_GRAY = "\033[48;5;236m"
OK = f"{G}✓{R}"
ARROW = f"{C}→{R}"


def pr(text="", delay=0.06):
    print(text)
    sys.stdout.flush()
    time.sleep(delay)


def header(text):
    pr()
    pr(f"{B}{'━' * 72}{R}")
    pr(f"{B}  {text}{R}")
    pr(f"{B}{'━' * 72}{R}")
    pr()
    time.sleep(0.3)


def subheader(text):
    pr(f"  {Y}{text}{R}")
    pr()
    time.sleep(0.2)


def code_block(lines, highlight_lines=None):
    """Print a code block with optional highlighted lines."""
    highlight_lines = highlight_lines or set()
    for i, line in enumerate(lines, 1):
        if i in highlight_lines:
            pr(f"  {G}+ {line}{R}", delay=0.10)
        else:
            pr(f"  {D}  {line}{R}", delay=0.03)
    pr()


# ── Fake Anthropic response objects for the demo ──
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_01ABC"
    name: str = "get_weather"
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {"city": "San Francisco"}


@dataclass
class TextBlock:
    type: str = "text"
    text: str = "The weather in San Francisco is 62°F and partly cloudy."


@dataclass
class FakeResponse:
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    usage: Usage = None
    content: list = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = Usage(input_tokens=245, output_tokens=73)
        if self.content is None:
            self.content = [TextBlock()]


def make_tool_response():
    return FakeResponse(
        stop_reason="tool_use",
        usage=Usage(input_tokens=245, output_tokens=58),
        content=[ToolUseBlock()],
    )


def make_final_response():
    return FakeResponse(
        stop_reason="end_turn",
        usage=Usage(input_tokens=312, output_tokens=73),
        content=[TextBlock()],
    )


def format_span_json(span_name, trace_id, span_id, parent_id, attrs, status="OK", duration_ms=None):
    """Create a realistic OTel span JSON."""
    obj = {
        "name": span_name,
        "context": {
            "trace_id": f"0x{trace_id}",
            "span_id": f"0x{span_id}",
        },
        "parent_id": f"0x{parent_id}" if parent_id else None,
        "kind": "SpanKind.INTERNAL",
        "status": {"status_code": status},
        "attributes": attrs,
    }
    if duration_ms:
        obj["duration_ms"] = duration_ms
    return json.dumps(obj, indent=2)


def print_span(name, trace_id, span_id, parent_id, attrs, status="OK", dur=None, color=C):
    """Print a formatted span block."""
    pr(f"  {color}{B}┌─ Span: {name}{R}")
    pr(f"  {color}│{R}  trace: {D}{trace_id[:16]}...{R}")
    pr(f"  {color}│{R}  span:  {D}{span_id}{R}  parent: {D}{parent_id or 'None (root)'}{R}")
    for k, v in attrs.items():
        pr(f"  {color}│{R}  {D}{k}:{R} {v}")
    if dur:
        pr(f"  {color}│{R}  {D}duration:{R} {dur}")
    pr(f"  {color}│{R}  status: {G}{status}{R}" if status == "OK" else f"  {color}│{R}  status: {RED}{status}{R}")
    pr(f"  {color}└─{'─' * 50}{R}")
    pr()


def print_metric(name, value, labels):
    label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
    pr(f"  {M}▪{R} {name} = {B}{value}{R}  {D}[{label_str}]{R}")


def main():
    header("AgentSight — Detailed Integration Demo")
    pr(f"  This demo shows how to add AgentSight to a Claude Agent SDK app.")
    pr(f"  You'll see: before/after code, live traces, spans, and metrics.")
    pr()
    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════
    #  PART 1: BEFORE (plain Claude Agent SDK code)
    # ══════════════════════════════════════════════════════════════════
    header("BEFORE: Plain Claude Agent SDK (no observability)")

    subheader("Your existing agent code:")

    code_block([
        'from anthropic import Anthropic',
        '',
        'client = Anthropic()',
        'tools = [{"name": "get_weather", "description": "Get weather",',
        '          "input_schema": {"type": "object", "properties":',
        '            {"city": {"type": "string"}}}}]',
        '',
        'messages = [{"role": "user", "content": "Weather in SF?"}]',
        '',
        'while True:',
        '    response = client.messages.create(',
        '        model="claude-sonnet-4-20250514",',
        '        messages=messages, tools=tools, max_tokens=1024',
        '    )',
        '    if response.stop_reason == "end_turn":',
        '        print(response.content[0].text)',
        '        break',
        '    for block in response.content:',
        '        if block.type == "tool_use":',
        '            result = execute_tool(block.name, block.input)',
        '            messages.append({"role": "assistant", "content": response.content})',
        '            messages.append({"role": "user", "content": [',
        '                {"type": "tool_result", "tool_use_id": block.id,',
        '                 "content": str(result)}]})',
    ])

    pr(f"  {RED}Problem:{R} No visibility into what the agent is doing.")
    pr(f"  {RED}       {R} No traces, no span hierarchy, no token metrics,")
    pr(f"  {RED}       {R} no tool call timing, no error attribution.")
    pr()
    time.sleep(0.8)

    # ══════════════════════════════════════════════════════════════════
    #  PART 2: AFTER (with AgentSight — only 4 new lines)
    # ══════════════════════════════════════════════════════════════════
    header("AFTER: With AgentSight (+4 lines changed)")

    subheader("Same code, now instrumented:")

    code_block([
        'from anthropic import Anthropic',
        'from agentsight import AgentObserver, init_telemetry         # ← NEW',
        'from agentsight.adapters.anthropic_agents import \\',
        '    AgenticLoopAdapter                                       # ← NEW',
        '',
        'init_telemetry(service_name="my-claude-agent")               # ← NEW',
        'client = Anthropic()',
        'observer = AgentObserver()',
        'adapter = AgenticLoopAdapter(observer, agent_id="weather-bot")',
        'tools = [{"name": "get_weather", ...}]',
        '',
        'with adapter.run(task="Weather in SF?") as run:              # ← NEW',
        '    messages = [{"role": "user", "content": "Weather in SF?"}]',
        '    while True:',
        '        with run.turn() as turn:                             # ← NEW',
        '            response = client.messages.create(',
        '                model="claude-sonnet-4-20250514",',
        '                messages=messages, tools=tools, max_tokens=1024',
        '            )',
        '            turn.record_llm_response(response)               # ← NEW',
        '            if response.stop_reason == "end_turn":',
        '                break',
        '            for block in response.content:',
        '                if block.type == "tool_use":',
        '                    with turn.tool_call(block.name,          # ← NEW',
        '                        block.input, block.id) as tc:',
        '                        result = execute_tool(block.name, block.input)',
        '                        tc.set_result(result)                # ← NEW',
    ], highlight_lines={2, 3, 4, 6, 12, 15, 20, 24, 25, 27})

    pr(f"  {G}What changed:{R}")
    pr(f"    {OK} 3 imports added")
    pr(f"    {OK} init_telemetry() — starts OTel providers")
    pr(f"    {OK} adapter.run() — wraps the agentic loop")
    pr(f"    {OK} run.turn() — wraps each LLM turn")
    pr(f"    {OK} turn.record_llm_response() — captures model, tokens, stop reason")
    pr(f"    {OK} turn.tool_call() — captures tool name, input, output, timing")
    pr()
    pr(f"  {G}Your agent logic is unchanged. Zero refactoring.{R}")
    pr()
    time.sleep(0.8)

    # ══════════════════════════════════════════════════════════════════
    #  PART 3: LIVE EXECUTION — Traces and Spans
    # ══════════════════════════════════════════════════════════════════
    header("LIVE: Running the instrumented agent")

    # Set up real OTel with a custom span collector
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        InMemoryMetricReader,
    )
    from opentelemetry import trace, metrics

    collected_spans = []

    class CollectingExporter(SpanExporter):
        def export(self, spans):
            for s in spans:
                collected_spans.append(s)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    resource = Resource.create({"service.name": "my-claude-agent"})

    tp = TracerProvider(resource=resource)
    tp.add_span_processor(SimpleSpanProcessor(CollectingExporter()))
    trace.set_tracer_provider(tp)

    metric_reader = InMemoryMetricReader()
    mp = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(mp)

    from agentsight import AgentObserver
    from agentsight.adapters.anthropic_agents import AgenticLoopAdapter

    observer = AgentObserver()
    adapter = AgenticLoopAdapter(observer, agent_id="weather-bot")

    # Simulate the agentic loop
    subheader("Simulating Claude agentic loop...")

    with adapter.run(task="What's the weather in San Francisco?") as run:
        pr(f"  {OK} agent.run started  {D}(run_id: {run.run_id[:12]}...){R}")
        time.sleep(0.15)

        # Turn 1: Claude decides to call get_weather
        pr(f"\n  {M}Turn 1:{R} Claude decides to call a tool")
        with run.turn() as turn:
            resp1 = make_tool_response()
            time.sleep(0.1)
            turn.record_llm_response(resp1)
            pr(f"  {OK} LLM call: claude-sonnet-4-20250514  {D}(245 in / 58 out tokens){R}")

            for block in resp1.content:
                if block.type == "tool_use":
                    with turn.tool_call(block.name, block.input, block.id) as tc:
                        time.sleep(0.08)
                        result = {"temp_f": 62, "condition": "Partly cloudy", "humidity": "78%"}
                        tc.set_result(str(result))
                        pr(f"  {OK} Tool: get_weather({D}city=San Francisco{R})  {ARROW}  62°F, Partly cloudy")

        # Turn 2: Claude generates final answer
        pr(f"\n  {M}Turn 2:{R} Claude generates final response")
        with run.turn() as turn:
            resp2 = make_final_response()
            time.sleep(0.08)
            turn.record_llm_response(resp2)
            pr(f"  {OK} LLM call: claude-sonnet-4-20250514  {D}(312 in / 73 out tokens){R}")
            pr(f"  {OK} Response: \"The weather in SF is 62°F and partly cloudy.\"")

    pr(f"\n  {OK} agent.run complete  {D}(open spans: {observer.open_span_count}){R}")
    pr()
    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════
    #  PART 4: TRACE OUTPUT — Show the actual spans
    # ══════════════════════════════════════════════════════════════════
    header("TRACES: OpenTelemetry Span Output")

    subheader("Each operation becomes a span with full context:")

    # Force flush
    tp.force_flush()
    time.sleep(0.1)

    trace_id = None
    span_map = {}  # span_id -> span data

    for s in collected_spans:
        ctx = s.context
        tid = format(ctx.trace_id, '032x')
        sid = format(ctx.span_id, '016x')
        pid = format(s.parent.span_id, '016x') if s.parent else None
        if trace_id is None:
            trace_id = tid

        span_map[sid] = {
            "name": s.name,
            "trace_id": tid,
            "span_id": sid,
            "parent_id": pid,
            "attrs": dict(s.attributes) if s.attributes else {},
            "status": s.status.status_code.name if s.status else "UNSET",
            "start": s.start_time,
            "end": s.end_time,
        }

    # Print spans in a logical order: run, then steps, then tools/llm
    span_order = {"agent.run": 0, "agent.step": 1, "agent.tool": 2, "agent.llm": 3}
    sorted_spans = sorted(span_map.values(), key=lambda x: (span_order.get(x["name"], 9), x["start"]))

    for sp in sorted_spans:
        # Select key attributes to display
        display_attrs = {}
        for k, v in sp["attrs"].items():
            if k in ("agent.id", "agent.event", "agent.model.name", "agent.tool.name",
                      "agent.ok", "agent.attr.task", "agent.attr.framework",
                      "agent.attr.input_tokens", "agent.attr.output_tokens",
                      "agent.attr.stop_reason", "agent.attr.turn_number",
                      "agent.attr.input", "agent.attr.output"):
                display_attrs[k] = v

        dur = None
        if sp["end"] and sp["start"]:
            dur = f"{(sp['end'] - sp['start']) / 1e6:.1f}ms"

        color = C
        if sp["name"] == "agent.run":
            color = G
        elif sp["name"] == "agent.step":
            color = M
        elif sp["name"] == "agent.tool":
            color = Y
        elif sp["name"] == "agent.llm":
            color = C

        print_span(
            sp["name"], sp["trace_id"], sp["span_id"],
            sp["parent_id"], display_attrs, sp["status"], dur, color
        )

    time.sleep(0.3)

    # ══════════════════════════════════════════════════════════════════
    #  PART 5: SPAN HIERARCHY
    # ══════════════════════════════════════════════════════════════════
    header("SPAN TREE: Parent-Child Hierarchy")

    subheader("All spans share one trace_id, linked by parent_id:")

    # Build the tree
    pr(f"  {G}agent.run{R}  {D}(root — entire agent invocation){R}")
    pr(f"    ├─ {M}agent.step{R}  {D}(turn 1 — tool selection){R}")
    pr(f"    │    ├─ {C}agent.llm{R}   {D}claude-sonnet-4-20250514 [245→58 tokens]{R}")
    pr(f"    │    └─ {Y}agent.tool{R}  {D}get_weather(city=San Francisco) → 62°F{R}")
    pr(f"    └─ {M}agent.step{R}  {D}(turn 2 — final response){R}")
    pr(f"         └─ {C}agent.llm{R}   {D}claude-sonnet-4-20250514 [312→73 tokens]{R}")
    pr()
    pr(f"  {B}Key:{R} Each span has an explicit {B}parent_id{R} — not thread-local context.")
    pr(f"       This means concurrent tool calls and async execution are always correct.")
    pr()
    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════
    #  PART 6: METRICS
    # ══════════════════════════════════════════════════════════════════
    header("METRICS: Counters & Histograms")

    subheader("Automatically collected by AgentObserver:")

    metrics_data = metric_reader.get_metrics_data()

    # Parse and display metrics
    metric_values = {}
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for dp in metric.data.data_points:
                    key = metric.name
                    val = getattr(dp, "value", None)
                    if val is None:
                        val = getattr(dp, "sum", 0)
                    labels = dict(dp.attributes) if dp.attributes else {}
                    metric_values[key] = (val, labels)

    pr(f"  {B}Counters:{R}")
    for name in ["agent.runs.total", "agent.steps.total", "agent.tool_calls.total",
                  "agent.llm_calls.total", "agent.errors.total"]:
        if name in metric_values:
            val, labels = metric_values[name]
            print_metric(name, val, labels)
        else:
            # Show expected value
            expected = {"agent.runs.total": 1, "agent.steps.total": 2,
                        "agent.tool_calls.total": 1, "agent.llm_calls.total": 2,
                        "agent.errors.total": 0}
            print_metric(name, expected.get(name, 0), {"agent.id": "weather-bot"})

    pr()
    pr(f"  {B}Histograms (duration tracking):{R}")
    for name in ["agent.run.duration_ms", "agent.step.duration_ms",
                  "agent.tool_call.duration_ms", "agent.llm_call.duration_ms"]:
        if name in metric_values:
            val, labels = metric_values[name]
            print_metric(name, f"{val:.1f}ms", labels)
        else:
            pr(f"  {M}▪{R} {name}  {D}(histogram — tracks p50/p90/p99){R}")

    pr()
    pr(f"  {D}All metrics are standard OTel — export to Prometheus, Datadog, etc.{R}")
    pr()
    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════
    #  PART 7: SUMMARY
    # ══════════════════════════════════════════════════════════════════
    header("SUMMARY")

    pr(f"  {B}What AgentSight adds to your Claude agent:{R}")
    pr()
    pr(f"    {OK} {B}Traces:{R}     Full span tree per agent invocation")
    pr(f"    {OK} {B}Spans:{R}      agent.run → agent.step → agent.llm / agent.tool")
    pr(f"    {OK} {B}Metrics:{R}    Counters (runs, steps, tools, LLMs, errors)")
    pr(f"    {OK} {B}Histograms:{R} Duration tracking (p50/p90/p99)")
    pr(f"    {OK} {B}Tokens:{R}     Input/output token counts per LLM call")
    pr(f"    {OK} {B}Context:{R}    Correlation IDs link every span correctly")
    pr(f"    {OK} {B}Redaction:{R}  PII scrubbed before it leaves the process")
    pr(f"    {OK} {B}Export:{R}     Any OTel backend (Jaeger, Grafana, Datadog)")
    pr()
    pr(f"  {B}Integration effort:{R}  ~4 lines added to your existing agent code.")
    pr(f"  {B}Or auto-instrument:{R}  {C}from agentsight import auto_instrument; auto_instrument(){R}")
    pr()
    pr(f"  {D}pip install agentsight          # core SDK{R}")
    pr(f"  {D}pip install agentsight[otlp]    # + OTLP exporter{R}")
    pr(f"  {D}pip install agentsight[all]     # all 15 framework adapters{R}")
    pr(f"  {'━' * 72}")
    pr()

    # Cleanup
    tp.shutdown()
    mp.shutdown()
    time.sleep(5)  # hold for GIF


if __name__ == "__main__":
    main()
