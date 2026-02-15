#!/usr/bin/env python3
"""
Demo: agentsight SDK in action.

Run with:
    python -m examples.demo

Or from the project root:
    python examples/demo.py

This emits spans and metrics to the console. To send to an OTLP
collector instead, change ExporterType.CONSOLE to ExporterType.OTLP_GRPC
and set OTEL_EXPORTER_OTLP_ENDPOINT.
"""

import time

from agentsight import (
    AgentObserver,
    ExporterType,
    PayloadPolicy,
    init_telemetry,
    shutdown_telemetry,
)
from agentsight.adapters.generic import GenericAgentAdapter


def simulate_search(query: str) -> dict:
    """Simulated tool: returns fake search results."""
    time.sleep(0.05)
    return {"results": 10, "query": query}


def simulate_llm_call(prompt: str) -> str:
    """Simulated LLM: returns a fake response."""
    time.sleep(0.1)
    return f"Based on the search results for '{prompt}', I recommend Flight AA123."


def main() -> None:
    # 1. Initialize OpenTelemetry (console exporter for demo)
    tracer_provider, meter_provider = init_telemetry(
        service_name="flight-booking-agent",
        exporter=ExporterType.CONSOLE,
    )

    # 2. Create observer with payload policy
    observer = AgentObserver(
        payload_policy=PayloadPolicy(
            max_str_len=1024,
            redact_keys={"password", "api_key", "credit_card"},
        )
    )

    # 3. Create adapter for our custom agent
    agent = GenericAgentAdapter(observer, agent_id="flight-planner")

    # 4. Run the agent
    print("--- Starting agent run ---\n")

    with agent.run(task="Book a flight from SFO to JFK") as run:
        print(f"Run ID: {run.run_id}\n")

        # Step 1: Search for flights
        with run.step(reason="Need to find available flights") as step:
            with step.tool_call("flight_search", input={"from": "SFO", "to": "JFK", "date": "2025-03-15"}) as tc:
                results = simulate_search("SFO to JFK")
                tc.set_output(results, result_count=results["results"])
                print(f"  Tool 'flight_search' returned {results['results']} results")

            with step.llm_call(model="claude-3-opus") as llm:
                response = simulate_llm_call("SFO to JFK flights")
                llm.set_output(response, tokens={"prompt": 150, "completion": 45})
                print(f"  LLM response: {response}")

        # Step 2: Book the selected flight
        with run.step(reason="Book the recommended flight") as step:
            with step.tool_call("booking_api", input={"flight": "AA123", "passengers": 1}) as tc:
                time.sleep(0.05)
                tc.set_output({"confirmation": "BK-98765", "status": "confirmed"})
                print("  Booking confirmed: BK-98765")

    print("\n--- Agent run complete ---")
    print(f"Open spans remaining: {observer.open_span_count}")

    # 5. Example with error handling
    print("\n--- Starting agent run with error ---\n")

    try:
        with agent.run(task="Book impossible route") as run:
            with run.step(reason="Search for route") as step:
                with step.tool_call("flight_search", input={"from": "MARS", "to": "MOON"}) as tc:
                    raise ValueError("No flights found between MARS and MOON")
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    print(f"Open spans remaining: {observer.open_span_count}")

    # 6. Flush and shutdown
    print("\n--- Flushing telemetry ---\n")
    shutdown_telemetry(tracer_provider, meter_provider)
    print("Done.")


if __name__ == "__main__":
    main()
