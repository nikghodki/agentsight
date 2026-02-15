#!/usr/bin/env python3
"""Streamlined demo for GIF recording — shows AgentSight in action."""

import io
import os
import time
import sys

from agentsight import (
    AgentObserver,
    ExporterType,
    PayloadPolicy,
    init_telemetry,
    shutdown_telemetry,
)
from agentsight.adapters.generic import GenericAgentAdapter


def _init_quiet_telemetry(service_name: str):
    """Initialize OTel with exporters that write to /dev/null (no JSON noise)."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry import trace as trace_api, metrics as metrics_api

    devnull = open(os.devnull, "w")
    resource = Resource.create({"service.name": service_name})

    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter(out=devnull)))
    trace_api.set_tracer_provider(tp)

    reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(out=devnull), export_interval_millis=60_000
    )
    mp = MeterProvider(resource=resource, metric_readers=[reader])
    metrics_api.set_meter_provider(mp)

    return tp, mp


# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"


def typed_print(text, delay=0.08):
    """Print with a slight delay for the GIF effect."""
    print(text)
    sys.stdout.flush()
    time.sleep(delay)


def simulate_search(query: str) -> dict:
    time.sleep(0.05)
    return {"results": 10, "top_result": "Flight AA123 — $349"}


def simulate_llm(prompt: str) -> str:
    time.sleep(0.08)
    return "Based on search results, I recommend Flight AA123 departing at 2:30 PM."


def main():
    typed_print(f"\n{BOLD}{'=' * 62}{RESET}")
    typed_print(f"{BOLD}  AgentSight — Framework-agnostic Agent Observability SDK{RESET}")
    typed_print(f"{BOLD}{'=' * 62}{RESET}\n")

    # ── 1. Initialize ──
    typed_print(f"{DIM}# pip install agentsight{RESET}")
    typed_print(f"{CYAN}from{RESET} agentsight {CYAN}import{RESET} AgentObserver, init_telemetry, PayloadPolicy")
    typed_print(f"{CYAN}from{RESET} agentsight.adapters.generic {CYAN}import{RESET} GenericAgentAdapter\n")
    time.sleep(0.3)

    typed_print(f"{YELLOW}▸ Initializing OpenTelemetry...{RESET}")
    tracer_provider, meter_provider = _init_quiet_telemetry("flight-booking-agent")
    typed_print(f"  {CHECK} TracerProvider ready")
    typed_print(f"  {CHECK} MeterProvider ready\n")
    time.sleep(0.2)

    observer = AgentObserver(
        payload_policy=PayloadPolicy(
            max_str_len=512,
            redact_keys={"password", "api_key", "credit_card"},
        )
    )
    typed_print(f"{YELLOW}▸ Observer created with payload redaction{RESET}")
    typed_print(f"  {CHECK} Redacting: password, api_key, credit_card\n")
    time.sleep(0.2)

    agent = GenericAgentAdapter(observer, agent_id="flight-planner")

    # ── 2. Agent Run ──
    typed_print(f"{BOLD}{GREEN}▶ Starting agent run: \"Book SFO → JFK\"{RESET}\n")
    time.sleep(0.3)

    with agent.run(task="Book a flight from SFO to JFK") as run:
        typed_print(f"  {DIM}run_id: {run.run_id[:8]}...{RESET}\n")

        # Step 1
        typed_print(f"  {MAGENTA}┌─ Step 1: Search for flights{RESET}")
        with run.step(reason="Find available flights") as step:

            typed_print(f"  {MAGENTA}│{RESET}  {CYAN}🔧 tool:{RESET} flight_search(from=SFO, to=JFK)")
            with step.tool_call("flight_search", input={"from": "SFO", "to": "JFK", "date": "2025-03-15"}) as tc:
                results = simulate_search("SFO to JFK")
                tc.set_output(results, result_count=results["results"])
            typed_print(f"  {MAGENTA}│{RESET}     {CHECK} 10 results found")

            typed_print(f"  {MAGENTA}│{RESET}  {CYAN}🤖 llm:{RESET}  claude-3-opus")
            with step.llm_call(model="claude-3-opus") as llm:
                response = simulate_llm("SFO to JFK flights")
                llm.set_output(response, tokens={"prompt": 150, "completion": 45})
            typed_print(f"  {MAGENTA}│{RESET}     {CHECK} \"Recommend Flight AA123, 2:30 PM\"")
        typed_print(f"  {MAGENTA}└─ {CHECK} Step complete{RESET}\n")
        time.sleep(0.1)

        # Step 2
        typed_print(f"  {MAGENTA}┌─ Step 2: Book selected flight{RESET}")
        with run.step(reason="Book the recommended flight") as step:

            typed_print(f"  {MAGENTA}│{RESET}  {CYAN}🔧 tool:{RESET} booking_api(flight=AA123)")
            with step.tool_call("booking_api", input={"flight": "AA123", "passengers": 1}) as tc:
                time.sleep(0.05)
                tc.set_output({"confirmation": "BK-98765", "status": "confirmed"})
            typed_print(f"  {MAGENTA}│{RESET}     {CHECK} Confirmation: BK-98765")
        typed_print(f"  {MAGENTA}└─ {CHECK} Step complete{RESET}\n")

    typed_print(f"{BOLD}{GREEN}✅ Agent run complete{RESET}")
    typed_print(f"   Open spans: {observer.open_span_count} {DIM}(all closed correctly){RESET}\n")
    time.sleep(0.3)

    # ── 3. Error handling demo ──
    typed_print(f"{BOLD}{RED}▶ Error handling demo: impossible route{RESET}\n")
    time.sleep(0.2)

    try:
        with agent.run(task="Book MARS → MOON") as run:
            typed_print(f"  {MAGENTA}┌─ Step 1: Search{RESET}")
            with run.step(reason="Search for route") as step:
                typed_print(f"  {MAGENTA}│{RESET}  {CYAN}🔧 tool:{RESET} flight_search(from=MARS, to=MOON)")
                with step.tool_call("flight_search", input={"from": "MARS", "to": "MOON"}) as tc:
                    raise ValueError("No flights between MARS and MOON")
    except ValueError:
        typed_print(f"  {MAGENTA}│{RESET}     {CROSS} ValueError caught — span marked as error")
        typed_print(f"  {MAGENTA}└─ {CROSS} Run ended with error status{RESET}\n")

    typed_print(f"   Open spans: {observer.open_span_count} {DIM}(still zero — errors handled cleanly){RESET}\n")
    time.sleep(0.3)

    # ── 4. Trace summary ──
    typed_print(f"{BOLD}{'─' * 62}{RESET}")
    typed_print(f"{BOLD}  Captured Trace Hierarchy:{RESET}")
    typed_print(f"{BOLD}{'─' * 62}{RESET}")
    typed_print(f"  agent.run {DIM}(root span){RESET}")
    typed_print(f"    ├─ agent.step {DIM}(search){RESET}")
    typed_print(f"    │    ├─ agent.tool {DIM}(flight_search){RESET}")
    typed_print(f"    │    └─ agent.llm  {DIM}(claude-3-opus){RESET}")
    typed_print(f"    └─ agent.step {DIM}(booking){RESET}")
    typed_print(f"         └─ agent.tool {DIM}(booking_api){RESET}")
    typed_print(f"")
    typed_print(f"  {BOLD}Metrics:{RESET}  runs=2  steps=3  tools=3  llm_calls=1")
    typed_print(f"  {BOLD}Redacted:{RESET} password, api_key, credit_card fields auto-scrubbed")
    typed_print(f"  {BOLD}Export:{RESET}   OTel spans → Jaeger / Grafana / Datadog / any OTLP backend")
    typed_print(f"")
    typed_print(f"{BOLD}{'─' * 62}{RESET}")
    typed_print(f"  {DIM}pip install agentsight        # core SDK{RESET}")
    typed_print(f"  {DIM}pip install agentsight[otlp]  # + OTLP exporter{RESET}")
    typed_print(f"  {DIM}pip install agentsight[all]   # all framework adapters{RESET}")
    typed_print(f"{BOLD}{'─' * 62}{RESET}\n")

    shutdown_telemetry(tracer_provider, meter_provider)
    time.sleep(3)  # hold final frame for GIF


if __name__ == "__main__":
    main()
