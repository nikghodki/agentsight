# Quick Start

Get `agentsight` running with a single line of code.

## Installation

**Core SDK** (no framework dependencies):

```bash
pip install agentsight
```

**With a specific framework adapter:**

```bash
pip install agentsight[langchain]
pip install agentsight[openai-agents]
pip install agentsight[crewai]
# ... see full list below
```

**With OTLP exporters for production:**

```bash
pip install agentsight[otlp]
```

**Everything:**

```bash
pip install agentsight[all]
```

### Available Extras

| Extra | Frameworks |
|---|---|
| `langchain` | LangChain |
| `langgraph` | LangGraph |
| `crewai` | CrewAI |
| `openai-agents` | OpenAI Agents SDK |
| `llamaindex` | LlamaIndex |
| `semantic-kernel` | Microsoft Semantic Kernel |
| `haystack` | deepset Haystack |
| `smolagents` | HuggingFace smolagents |
| `google-adk` | Google Agent Development Kit |
| `pydantic-ai` | PydanticAI |
| `phidata` | Phidata / Agno |
| `otlp` | OTLP gRPC + HTTP exporters |
| `dev` | pytest, ruff, mypy |
| `all` | Everything above |

---

## One-Line Setup (Recommended)

```python
from agentsight import auto_instrument

auto_instrument()
```

That's it. All installed agent frameworks are now emitting OpenTelemetry spans automatically. Your existing code works unchanged:

```python
# LangChain -- spans emitted automatically
result = chain.invoke({"input": "Hello"})

# OpenAI Agents SDK -- spans emitted automatically
result = await Runner.run(agent, "Hello")

# Anthropic -- spans emitted automatically
response = client.messages.create(model="claude-sonnet-4-20250514", messages=messages)

# CrewAI -- spans emitted automatically
result = crew.kickoff()
```

### Selective Instrumentation

Only instrument specific frameworks:

```python
from agentsight import auto_instrument

auto_instrument(frameworks=["langchain", "anthropic"])
```

Or use per-framework functions:

```python
from agentsight import instrument_langchain, instrument_anthropic

instrument_langchain()
instrument_anthropic()
```

### See What's Available

```python
from agentsight import available_frameworks

print(available_frameworks())
# ['langchain', 'anthropic', 'bedrock']  -- only installed frameworks
```

### Production Setup

Send telemetry to an OTLP-compatible backend (Jaeger, Grafana Tempo, Datadog, etc.):

```python
from agentsight import auto_instrument, ExporterType, PayloadPolicy

auto_instrument(
    service_name="production-agent",
    exporter=ExporterType.OTLP_GRPC,
    otlp_endpoint="http://otel-collector:4317",
    payload_policy=PayloadPolicy(
        max_str_len=1024,
        redact_keys={"password", "api_key", "ssn"},
        drop_keys={"internal_debug"},
    ),
)
```

Or configure the endpoint via environment variables:

```bash
export OTEL_SERVICE_NAME=production-agent
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer token123"
```

```python
from agentsight import auto_instrument, ExporterType

auto_instrument(exporter=ExporterType.OTLP_GRPC)
```

### Cleanup

```python
from agentsight import uninstrument

uninstrument()  # Remove all patches and flush telemetry
```

---

## What You Get

Every instrumented call produces this span tree:

```
agent.run
  +-- agent.step
  |     +-- agent.tool (web_search)
  |     +-- agent.llm  (gpt-4)
  +-- agent.step
        +-- agent.llm
```

Plus counters (`agent.runs.total`, `agent.tool_calls.total`, etc.) and duration histograms -- all automatically exported to your OTel backend.

---

## Manual Integration (Advanced)

For fine-grained control over what gets traced, use adapters directly instead of `auto_instrument()`.

### Generic Adapter (Custom Agents)

```python
from agentsight import AgentObserver, init_telemetry, shutdown_telemetry
from agentsight.adapters.generic import GenericAgentAdapter

# 1. Initialize OpenTelemetry
tp, mp = init_telemetry(service_name="my-agent-service")

# 2. Create observer
observer = AgentObserver()

# 3. Create adapter
agent = GenericAgentAdapter(observer, agent_id="my-agent")

# 4. Instrument your agent
with agent.run(task="Answer user question") as run:
    with run.step(reason="Search for information") as step:
        with step.tool_call("web_search", input={"query": "Python OTel"}) as tc:
            result = web_search("Python OTel")
            tc.set_output(result)

        with step.llm_call(model="gpt-4") as llm:
            response = call_llm(prompt="Summarize: ...")
            llm.set_output(response, tokens={"prompt": 150, "completion": 50})

# 5. Shutdown cleanly
shutdown_telemetry(tp, mp)
```

### Framework-Specific Adapters

Each framework has a dedicated adapter. Here are the most common patterns:

**Callback handlers** (LangChain, LangGraph, LlamaIndex):

```python
from agentsight.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter(observer, agent_id="langchain-agent")
result = chain.invoke({"input": "What is AI?"}, config={"callbacks": [adapter]})
```

**Context managers** (Anthropic, CrewAI, Haystack, PydanticAI, Phidata):

```python
from agentsight.adapters.anthropic_agents import AgenticLoopAdapter

adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")
with adapter.run(task="Research AI") as run:
    with run.turn() as turn:
        response = client.messages.create(model="claude-sonnet-4-20250514", ...)
        turn.record_llm_response(response)
```

**Async hooks** (OpenAI Agents SDK):

```python
from agentsight.adapters.openai_agents import OpenAIRunHooksAdapter

hooks = OpenAIRunHooksAdapter(observer)
hooks.start_run()
result = await Runner.run(agent, "Hello", run_hooks=hooks)
hooks.end_run(ok=True)
```

**Filter registration** (Semantic Kernel):

```python
from agentsight.adapters.semantic_kernel import SKAdapter

adapter = SKAdapter(observer, agent_id="sk-agent")
kernel.add_filter("function_invocation", adapter.function_filter)
kernel.add_filter("prompt_rendering", adapter.prompt_filter)
```

See the [Integration Guide](INTEGRATION_GUIDE.md) for complete before/after examples for all 15 frameworks.

---

## Payload Redaction

Sensitive data is automatically redacted before reaching any exporter:

```python
from agentsight import AgentObserver, PayloadPolicy

observer = AgentObserver(
    payload_policy=PayloadPolicy(
        max_str_len=1024,                          # Truncate long strings
        redact_keys={"password", "api_key", "ssn"}, # Redact by key name
        drop_keys={"internal_debug"},               # Silently drop
    )
)
```

Default redaction catches: passwords, API keys, tokens, AWS credentials, credit card numbers, and SSNs.

With `auto_instrument()`, pass the policy directly:

```python
auto_instrument(
    payload_policy=PayloadPolicy(redact_keys={"password", "ssn"}),
)
```

---

## Verifying It Works

After instrumenting your agent, check the console output (default exporter) for spans:

```
{
    "name": "agent.run",
    "context": {"trace_id": "0x...", "span_id": "0x..."},
    "attributes": {"agent.id": "my-agent", "agent.ok": true}
}
```

You should see spans with names: `agent.run`, `agent.step`, `agent.tool`, `agent.llm` and proper parent-child relationships via trace and span IDs.

---

## Next Steps

- [Integration Guide](INTEGRATION_GUIDE.md) for before/after examples for all 15 frameworks
- [Architecture](ARCHITECTURE.md) for design details and data flow
- [Landscape Analysis](LANDSCAPE.md) for how this compares to other tools
- `examples/demo.py` for a complete working example
- Run tests: `pytest tests/ -v`
