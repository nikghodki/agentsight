# Quick Start

Get `agent-observability` running in under 5 lines of code. This guide covers installation, basic usage, and framework-specific integration examples.

## Installation

**Core SDK** (no framework dependencies):

```bash
pip install agent-observability
```

**With a specific framework adapter:**

```bash
pip install agent-observability[langchain]
pip install agent-observability[openai-agents]
pip install agent-observability[crewai]
# ... see full list below
```

**With OTLP exporters for production:**

```bash
pip install agent-observability[otlp]
```

**Everything:**

```bash
pip install agent-observability[all]
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

## Basic Usage (Generic Adapter)

For custom agents or any framework without a dedicated adapter:

```python
from agent_observability import AgentObserver, init_telemetry, shutdown_telemetry
from agent_observability.adapters.generic import GenericAgentAdapter

# 1. Initialize OpenTelemetry
tracer_provider, meter_provider = init_telemetry(
    service_name="my-agent-service",
)

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
shutdown_telemetry(tracer_provider, meter_provider)
```

This produces the span tree:

```
agent.run
  └── agent.step
        ├── agent.tool (web_search)
        └── agent.llm  (gpt-4)
```

## Production Setup

Send telemetry to an OTLP-compatible backend (Jaeger, Grafana Tempo, Datadog, etc.):

```python
from agent_observability import ExporterType, init_telemetry

tracer_provider, meter_provider = init_telemetry(
    service_name="production-agent",
    exporter=ExporterType.OTLP_GRPC,
    otlp_endpoint="http://otel-collector:4317",
)
```

Or via environment variables:

```bash
export OTEL_SERVICE_NAME=production-agent
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer token123"
```

```python
tracer_provider, meter_provider = init_telemetry(
    exporter=ExporterType.OTLP_GRPC,
)
```

## Payload Redaction

Configure what gets recorded in span attributes:

```python
from agent_observability import AgentObserver, PayloadPolicy

observer = AgentObserver(
    payload_policy=PayloadPolicy(
        max_str_len=1024,                          # Truncate long strings
        redact_keys={"password", "api_key", "ssn"}, # Redact by key name
        drop_keys={"internal_debug"},               # Silently drop
    )
)
```

## Framework-Specific Integration

### LangChain

```python
from langchain_core.language_models import BaseChatModel
from agent_observability import AgentObserver, init_telemetry
from agent_observability.adapters.langchain import LangChainAdapter

init_telemetry(service_name="langchain-agent")
observer = AgentObserver()
adapter = LangChainAdapter(observer, agent_id="langchain-agent")

# Use as a LangChain callback handler
result = chain.invoke(
    {"input": "What is AI?"},
    config={"callbacks": [adapter]},
)
```

### LangGraph

**Option A: Callback adapter** (automatic instrumentation):

```python
from agent_observability.adapters.langgraph import LangGraphCallbackAdapter

adapter = LangGraphCallbackAdapter(observer, agent_id="my-graph")

result = graph.invoke(
    {"messages": [("user", "Hello")]},
    config={"callbacks": [adapter]},
)
```

**Option B: Event adapter** (manual, finer control):

```python
from agent_observability.adapters.langgraph import LangGraphEventAdapter

adapter = LangGraphEventAdapter(observer, agent_id="my-graph")

with adapter.graph_run(task="Process query") as run:
    adapter.on_node_start("retriever", {"query": "AI"})
    adapter.on_node_end("retriever", {"documents": docs})
    adapter.on_node_start("generator", {"docs": docs})
    adapter.on_node_end("generator", {"response": "AI is..."})
```

### OpenAI Agents SDK

**Run-level hooks** (recommended):

```python
from agents import Agent, Runner
from agent_observability.adapters.openai_agents import OpenAIRunHooksAdapter

hooks = OpenAIRunHooksAdapter(observer)
hooks.start_run()

result = await Runner.run(
    agent,
    "Book me a flight",
    run_hooks=hooks,
)

hooks.end_run(ok=True)
```

**Per-agent hooks:**

```python
from agent_observability.adapters.openai_agents import OpenAIAgentHooksAdapter

hooks = OpenAIAgentHooksAdapter(observer, agent_id="travel-agent")
agent = Agent(name="travel-agent", hooks=hooks)
```

### Anthropic Claude

**Context-manager approach** (for your own agentic loop):

```python
from anthropic import Anthropic
from agent_observability.adapters.anthropic_agents import AgenticLoopAdapter

client = Anthropic()
adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")

with adapter.run(task="Research AI safety") as run:
    messages = [{"role": "user", "content": "Research AI safety"}]

    while True:
        with run.turn() as turn:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                messages=messages,
                tools=tools,
            )
            turn.record_llm_response(response)

            if response.stop_reason == "end_turn":
                break

            for block in response.content:
                if block.type == "tool_use":
                    with turn.tool_call(block.name, block.input, block.id) as tc:
                        result = execute_tool(block.name, block.input)
                        tc.set_result(result)
```

**Event-driven approach:**

```python
from agent_observability.adapters.anthropic_agents import AnthropicMessageHooksAdapter

adapter = AnthropicMessageHooksAdapter(observer, agent_id="claude")

run_id = adapter.on_run_start(task="Research AI")
adapter.on_message_start()

# After tool use
tc_id = adapter.on_tool_use("search", {"q": "AI"}, tool_use_id="tu_1")
adapter.on_tool_result("search", "results", tool_use_id="tu_1")

# After LLM response
adapter.on_message_end(response=response, model="claude-sonnet-4-20250514")
adapter.on_run_end(ok=True)
```

### CrewAI

```python
from agent_observability.adapters.crewai import CrewAIAdapter

adapter = CrewAIAdapter(observer, agent_id="research-crew")

with adapter.observe_crew(crew_name="research", task="Analyze market") as ctx:
    ctx.on_task_start("market-analysis", agent_role="researcher")
    ctx.on_tool_use("web_search", {"query": "market trends"}, "Results...")
    ctx.on_task_end("market-analysis", output="Analysis complete")
```

### AutoGen

```python
from agent_observability.adapters.autogen import AutoGenAdapter

adapter = AutoGenAdapter(observer)

with adapter.group_chat("research-team", task="Find papers") as chat:
    chat.on_agent_message("researcher", "user", "Finding AI papers...")
    tc_id = chat.on_tool_call("researcher", "arxiv_search", {"query": "AI"})
    chat.on_tool_result("researcher", "arxiv_search", "5 papers found", tool_call_id=tc_id)
    chat.on_agent_message("critic", "researcher", "Good work")
```

### Google ADK

```python
from agent_observability.adapters.google_adk import GoogleADKAdapter

adapter = GoogleADKAdapter(observer)

with adapter.run(agent_name="travel-agent", task="Book flight") as run:
    run.on_agent_turn_start()
    run.on_model_call(model="gemini-2.0-flash")
    run.on_model_response(tokens={"prompt_tokens": 100, "completion_tokens": 50})
    tc_id = run.on_tool_call(tool_name="search_flights", tool_input={"from": "SFO"})
    run.on_tool_result(tool_name="search_flights", result="Found 5 flights", tool_call_id=tc_id)
    run.on_agent_turn_end()
```

### Amazon Bedrock Agents

```python
from agent_observability.adapters.bedrock_agents import BedrockAgentsAdapter

adapter = BedrockAgentsAdapter(observer, agent_id="bedrock-agent")

with adapter.invocation(task="What is AI?") as inv:
    # Process streaming trace events from Bedrock
    for event in bedrock_response["completion"]:
        inv.process_event(event)
```

### LlamaIndex

```python
from agent_observability.adapters.llamaindex import LlamaIndexAdapter

adapter = LlamaIndexAdapter(observer, agent_id="rag-pipeline")

# Use as a LlamaIndex callback handler
from llama_index.core import Settings
Settings.callback_manager.add_handler(adapter)

# Queries are now automatically instrumented
response = index.as_query_engine().query("What is AI?")
```

### Microsoft Semantic Kernel

```python
from agent_observability.adapters.semantic_kernel import SKAdapter

adapter = SKAdapter(observer, agent_id="sk-agent")

# Get filter instances
function_filter = adapter.create_function_filter()
prompt_filter = adapter.create_prompt_filter()
auto_filter = adapter.create_auto_function_filter()

# Register with Semantic Kernel
kernel.add_filter("function_invocation", function_filter)
kernel.add_filter("prompt_rendering", prompt_filter)
kernel.add_filter("auto_function_invocation", auto_filter)
```

### Haystack

```python
from agent_observability.adapters.haystack import HaystackAdapter

adapter = HaystackAdapter(observer, agent_id="rag-pipeline")

with adapter.pipeline_run(pipeline_name="qa") as run:
    adapter.on_component_start("retriever", "BM25Retriever", {"query": "AI"})
    adapter.on_component_end("retriever", {"documents": docs})
    adapter.on_component_start("generator", "OpenAIGenerator")
    adapter.on_component_end("generator", {"meta": {"usage": {"prompt_tokens": 100}}})
```

### HuggingFace smolagents

**Context-manager approach:**

```python
from agent_observability.adapters.smolagents import SmolagentsAdapter

adapter = SmolagentsAdapter(observer, agent_id="code-agent")

with adapter.run(task="Analyze data") as run:
    step_id = run.on_step_start(step_number=1)
    llm_id = run.on_llm_call(model_name="qwen-2.5-72b")
    run.on_llm_response(llm_id, tokens={"prompt": 200, "completion": 100})
    tc_id = run.on_tool_call("python_executor", "print('hello')", step_id=step_id)
    run.on_tool_result(tc_id, "python_executor", "hello", step_id=step_id)
    run.on_step_end(step_id)
```

**Monitor callback:**

```python
adapter = SmolagentsAdapter(observer, agent_id="code-agent")
monitor = adapter.create_monitor()

# Pass as step_callbacks to smolagents
agent = CodeAgent(tools=[...], step_callbacks=[monitor])
```

### PydanticAI

```python
from agent_observability.adapters.pydantic_ai import PydanticAIAdapter

adapter = PydanticAIAdapter(observer, agent_id="pydantic-agent")

with adapter.run(task="Analyze data", model="openai:gpt-4") as run:
    llm_id = run.on_model_request(model="gpt-4", messages=["user msg"])
    run.on_model_response(llm_id, tool_calls=["search", "calculate"])
    tc1 = run.on_tool_call("search", {"query": "data"})
    run.on_tool_result(tc1, "search", "results")
```

### Phidata / Agno

```python
from agent_observability.adapters.phidata import PhidataAdapter

adapter = PhidataAdapter(observer, agent_id="research-agent")

with adapter.run(task="Research trends") as run:
    llm_id = run.on_model_call(model="gpt-4", messages=["msg1"])
    run.on_model_response(llm_id, tokens={"prompt_tokens": 100})
    tc_id = run.on_tool_call("web_search", {"query": "AI trends 2025"})
    run.on_tool_result(tc_id, "web_search", "Results...")
    run.on_knowledge_search("AI trends", results=[1, 2, 3])
    run.on_memory_write("chat_history", message_count=5)
```

## Verifying It Works

After instrumenting your agent, check the console output (default exporter) for spans:

```python
# Verify no spans were leaked
assert observer.open_span_count == 0
```

You should see spans with names: `agent.run`, `agent.step`, `agent.tool`, `agent.llm` and proper parent-child relationships via trace and span IDs.

## Next Steps

- [Architecture](ARCHITECTURE.md) for design details and data flow
- `examples/demo.py` for a complete working example
- Run tests: `pytest tests/ -v`
