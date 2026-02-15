# agent-observability

Framework-agnostic observability SDK for AI agents, built on OpenTelemetry.

Instrument any agent framework with structured traces, metrics, and payload safety -- using a single canonical event protocol that works across 15+ frameworks.

## Why

AI agent frameworks each have their own callback systems, hook patterns, and logging approaches. Comparing traces across LangChain, OpenAI Agents, Anthropic Claude, CrewAI, and others requires a different integration for each.

`agent-observability` provides:

- **One event protocol** that all frameworks map to
- **Automatic span trees** with correct parent-child relationships (even for concurrent tool calls and async execution)
- **Built-in payload safety** with PII redaction, credential scrubbing, and truncation
- **Standard OTel export** to Jaeger, Grafana Tempo, Datadog, New Relic, or any OTLP-compatible backend

## Span Hierarchy

Every agent invocation produces:

```
agent.run                          # Full agent invocation
  +-- agent.step                   # Reasoning iteration / turn
  |     +-- agent.tool             # Tool call (search, API, code exec)
  |     +-- agent.tool             # Concurrent tool call
  |     +-- agent.llm              # LLM call (prompt -> completion)
  +-- agent.step                   # Second iteration
        +-- agent.llm
```

## Installation

```bash
# Core SDK only
pip install agent-observability

# With a framework adapter
pip install agent-observability[langchain]
pip install agent-observability[openai-agents]
pip install agent-observability[crewai]

# With OTLP export for production
pip install agent-observability[otlp]

# Everything
pip install agent-observability[all]
```

## Quick Example

```python
from agent_observability import AgentObserver, init_telemetry, shutdown_telemetry
from agent_observability.adapters.generic import GenericAgentAdapter

# Initialize OTel (console exporter for development)
tp, mp = init_telemetry(service_name="my-agent")
observer = AgentObserver()
agent = GenericAgentAdapter(observer, agent_id="my-agent")

with agent.run(task="Book a flight from SFO to JFK") as run:
    with run.step(reason="Search for flights") as step:
        with step.tool_call("flight_search", input={"from": "SFO", "to": "JFK"}) as tc:
            results = search_flights("SFO", "JFK")
            tc.set_output(results)
        with step.llm_call(model="gpt-4") as llm:
            response = call_llm("Pick the best flight...")
            llm.set_output(response, tokens={"prompt": 150, "completion": 45})

shutdown_telemetry(tp, mp)
```

## Supported Frameworks

| Framework | Adapter | Install Extra | Integration Mode |
|---|---|---|---|
| Custom / Generic | `GenericAgentAdapter` | (none) | Context managers |
| LangChain | `LangChainAdapter` | `langchain` | Callback handler |
| LangGraph | `LangGraphCallbackAdapter`, `LangGraphEventAdapter` | `langgraph` | Callback / Manual |
| OpenAI Agents SDK | `OpenAIRunHooksAdapter`, `OpenAIAgentHooksAdapter` | `openai-agents` | RunHooks / AgentHooks |
| Anthropic Claude | `AgenticLoopAdapter`, `AnthropicMessageHooksAdapter` | (none) | Context managers / Events |
| CrewAI | `CrewAIAdapter` | `crewai` | Context managers |
| AutoGen | `AutoGenAdapter` | (none) | Context managers |
| Google ADK | `GoogleADKAdapter` | `google-adk` | Context managers |
| Amazon Bedrock Agents | `BedrockAgentsAdapter` | (none) | Trace event processing |
| LlamaIndex | `LlamaIndexAdapter` | `llamaindex` | Callback handler |
| Semantic Kernel | `SKAdapter` | `semantic-kernel` | Filter protocol |
| Haystack | `HaystackAdapter` | `haystack` | Context managers |
| smolagents | `SmolagentsAdapter` | `smolagents` | Context managers / Monitor |
| PydanticAI | `PydanticAIAdapter` | `pydantic-ai` | Context managers |
| Phidata / Agno | `PhidataAdapter` | `phidata` | Context managers |

## Production Setup

Send traces and metrics to an OTLP collector:

```python
from agent_observability import ExporterType, init_telemetry

tp, mp = init_telemetry(
    service_name="production-agent",
    exporter=ExporterType.OTLP_GRPC,
    otlp_endpoint="http://otel-collector:4317",
)
```

Or configure via environment variables:

```bash
export OTEL_SERVICE_NAME=production-agent
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

## Payload Safety

Sensitive data is automatically redacted before reaching any exporter:

```python
from agent_observability import AgentObserver, PayloadPolicy

observer = AgentObserver(
    payload_policy=PayloadPolicy(
        max_str_len=1024,                             # Truncate long strings
        redact_keys={"password", "api_key", "ssn"},    # Redact by key name
        drop_keys={"internal_debug"},                  # Silently drop
    )
)
```

Default redaction catches passwords, API keys, tokens, AWS credentials, credit card numbers, and SSNs.

## Metrics

The SDK automatically records:

| Metric | Type | Description |
|---|---|---|
| `agent.runs.total` | Counter | Total agent invocations |
| `agent.steps.total` | Counter | Total reasoning steps |
| `agent.tool_calls.total` | Counter | Total tool calls |
| `agent.llm_calls.total` | Counter | Total LLM calls |
| `agent.errors.total` | Counter | Total errors |
| `agent.run.duration_ms` | Histogram | Run duration |
| `agent.step.duration_ms` | Histogram | Step duration |
| `agent.tool_call.duration_ms` | Histogram | Tool call duration |
| `agent.llm_call.duration_ms` | Histogram | LLM call duration |

## Project Structure

```
src/agent_observability/
  __init__.py          # Public API
  events.py            # AgentEvent protocol (frozen dataclass)
  observer.py          # OTel bridge (spans + metrics)
  redaction.py         # Payload sanitization
  otel_setup.py        # Exporter configuration
  adapters/            # 15 framework adapters
examples/
  demo.py              # Working end-to-end example
tests/                 # 58 tests
docs/
  ARCHITECTURE.md      # Design and internals
  QUICKSTART.md        # Installation and usage guide
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Documentation

- **[Integration Guide](docs/INTEGRATION_GUIDE.md)** - Before/after examples for all 15 frameworks (start here)
- **[Quick Start](docs/QUICKSTART.md)** - Installation, basic usage, and framework-specific examples
- **[Architecture](docs/ARCHITECTURE.md)** - Three-layer design, span correlation, payload hygiene, data flow

## Requirements

- Python 3.10+
- `opentelemetry-api >= 1.20.0`
- `opentelemetry-sdk >= 1.20.0`

## License

MIT
