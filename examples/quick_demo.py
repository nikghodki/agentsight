#!/usr/bin/env python3
"""
Quick Demo — agentsight SDK
=====================================
Simulates a realistic AI agent workflow and prints OpenTelemetry
traces + metrics to the console so you can see exactly what the
SDK captures.

Run:
    pip install -e .
    python examples/quick_demo.py

What you'll see:
    1. Trace spans printed as JSON (agent.run → agent.step → agent.tool / agent.llm)
    2. Metrics summary (counters + histograms)
"""

import time
from agentsight import (
    AgentEvent,
    AgentObserver,
    EventName,
    ExporterType,
    init_telemetry,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
    shutdown_telemetry,
)


def simulate_agent():
    """Simulates: User asks → LLM reasons → Tool called → LLM summarizes → Done."""

    # ── 1. Initialize OpenTelemetry with console exporter ──
    print("=" * 72)
    print("  agentsight SDK — Quick Demo")
    print("=" * 72)
    print()
    print("Initializing OpenTelemetry with CONSOLE exporter...")
    print("Spans and metrics will be printed below as JSON.\n")

    tracer_provider, meter_provider = init_telemetry(
        service_name="demo-agent-service",
        exporter=ExporterType.CONSOLE,
        metric_export_interval_ms=1_000,  # flush metrics quickly for demo
    )

    # ── 2. Create the observer ──
    observer = AgentObserver()

    # ── 3. Correlation IDs ──
    agent_id = "weather-assistant"
    run_id = new_run_id()

    # ══════════════════════════════════════════════════════════════════
    #  Agent Run Start
    # ══════════════════════════════════════════════════════════════════
    print("-" * 72)
    print("▶ Agent Run Started")
    print("-" * 72)

    observer.emit(AgentEvent(
        name=EventName.LIFECYCLE_START,
        agent_id=agent_id,
        run_id=run_id,
        attributes={"user.query": "What's the weather in San Francisco?"},
    ))

    time.sleep(0.05)  # simulate latency

    # ── Step 1: LLM decides to call a tool ──
    step1_id = new_step_id()
    llm1_id = new_llm_call_id()

    print("\n  📝 Step 1: LLM planning (deciding which tool to use)")

    observer.emit(AgentEvent(
        name=EventName.STEP_START,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step1_id,
        attributes={"step.description": "Planning — tool selection"},
    ))

    # LLM call: model decides to use weather tool
    observer.emit(AgentEvent(
        name=EventName.LLM_CALL_START,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step1_id,
        llm_call_id=llm1_id,
        model_name="gpt-4o-mini",
        attributes={
            "prompt.tokens": 128,
            "prompt.preview": "User wants weather in San Francisco...",
        },
    ))

    time.sleep(0.1)  # simulate LLM latency

    observer.emit(AgentEvent(
        name=EventName.LLM_CALL_END,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step1_id,
        llm_call_id=llm1_id,
        model_name="gpt-4o-mini",
        ok=True,
        attributes={
            "completion.tokens": 42,
            "total.tokens": 170,
            "decision": "call get_weather tool",
        },
    ))

    observer.emit(AgentEvent(
        name=EventName.STEP_END,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step1_id,
        ok=True,
    ))

    # ── Step 2: Tool execution ──
    step2_id = new_step_id()
    tool_id = new_tool_call_id()

    print("  🔧 Step 2: Calling get_weather tool")

    observer.emit(AgentEvent(
        name=EventName.STEP_START,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step2_id,
        attributes={"step.description": "Tool execution — get_weather"},
    ))

    observer.emit(AgentEvent(
        name=EventName.TOOL_CALL_START,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step2_id,
        tool_call_id=tool_id,
        tool_name="get_weather",
        attributes={"tool.input.city": "San Francisco"},
    ))

    time.sleep(0.15)  # simulate API call

    observer.emit(AgentEvent(
        name=EventName.TOOL_CALL_END,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step2_id,
        tool_call_id=tool_id,
        tool_name="get_weather",
        ok=True,
        attributes={
            "tool.output.temp_f": 62,
            "tool.output.condition": "Partly cloudy",
            "tool.output.humidity": "78%",
        },
    ))

    observer.emit(AgentEvent(
        name=EventName.STEP_END,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step2_id,
        ok=True,
    ))

    # ── Step 3: LLM generates final response ──
    step3_id = new_step_id()
    llm2_id = new_llm_call_id()

    print("  🤖 Step 3: LLM generating final response")

    observer.emit(AgentEvent(
        name=EventName.STEP_START,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step3_id,
        attributes={"step.description": "Response generation"},
    ))

    observer.emit(AgentEvent(
        name=EventName.LLM_CALL_START,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step3_id,
        llm_call_id=llm2_id,
        model_name="gpt-4o-mini",
        attributes={"prompt.tokens": 256},
    ))

    time.sleep(0.08)  # simulate LLM latency

    observer.emit(AgentEvent(
        name=EventName.LLM_CALL_END,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step3_id,
        llm_call_id=llm2_id,
        model_name="gpt-4o-mini",
        ok=True,
        attributes={
            "completion.tokens": 85,
            "total.tokens": 341,
            "response.preview": "The weather in San Francisco is 62°F and partly cloudy.",
        },
    ))

    observer.emit(AgentEvent(
        name=EventName.STEP_END,
        agent_id=agent_id,
        run_id=run_id,
        step_id=step3_id,
        ok=True,
    ))

    # ══════════════════════════════════════════════════════════════════
    #  Agent Run End
    # ══════════════════════════════════════════════════════════════════
    observer.emit(AgentEvent(
        name=EventName.LIFECYCLE_END,
        agent_id=agent_id,
        run_id=run_id,
        ok=True,
        attributes={"final.answer": "The weather in San Francisco is 62°F and partly cloudy."},
    ))

    print("\n" + "-" * 72)
    print("✅ Agent Run Complete")
    print("-" * 72)

    # ── 4. Flush and shutdown ──
    print("\nFlushing telemetry (spans + metrics will appear below)...\n")
    print("=" * 72)
    print("  OPENTELEMETRY OUTPUT")
    print("=" * 72)

    shutdown_telemetry(tracer_provider, meter_provider)

    print("\n" + "=" * 72)
    print("  DEMO COMPLETE")
    print("=" * 72)
    print()
    print("What was captured:")
    print("  Traces:")
    print("    • agent.run  (root span — entire agent invocation)")
    print("    •   └─ agent.step ×3  (planning, tool exec, response gen)")
    print("    •       ├─ agent.llm ×2  (GPT-4o-mini calls)")
    print("    •       └─ agent.tool ×1  (get_weather)")
    print()
    print("  Metrics:")
    print("    • agent.runs.total          = 1")
    print("    • agent.steps.total         = 3")
    print("    • agent.tool_calls.total    = 1")
    print("    • agent.llm_calls.total     = 2")
    print("    • agent.run.duration_ms     (histogram)")
    print("    • agent.step.duration_ms    (histogram)")
    print("    • agent.tool_call.duration_ms (histogram)")
    print("    • agent.llm_call.duration_ms  (histogram)")
    print()
    print("To send to Jaeger/Grafana/Datadog instead, change ExporterType.CONSOLE")
    print("to ExporterType.OTLP_GRPC and set OTEL_EXPORTER_OTLP_ENDPOINT.")


if __name__ == "__main__":
    simulate_agent()
