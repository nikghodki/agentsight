# Auto-Instrumentation: How It Works

This document explains the internals of `auto_instrument()` — the one-line function that adds OpenTelemetry observability to your agent code with zero changes to your existing application logic.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Step-by-Step Execution Flow](#step-by-step-execution-flow)
4. [Framework Detection](#framework-detection)
5. [Monkey-Patching: How Frameworks Get Instrumented](#monkey-patching-how-frameworks-get-instrumented)
6. [Patching Strategies by Framework](#patching-strategies-by-framework)
7. [The Event Pipeline](#the-event-pipeline)
8. [Span Hierarchy and Correlation](#span-hierarchy-and-correlation)
9. [Global State Management](#global-state-management)
10. [Uninstrumentation and Cleanup](#uninstrumentation-and-cleanup)
11. [Adding a New Framework](#adding-a-new-framework)

---

## Overview

When you call:

```python
from agentsight import auto_instrument

auto_instrument()
```

The SDK performs three things behind the scenes:

1. **Initializes OpenTelemetry** — sets up `TracerProvider`, `MeterProvider`, and exporters (console, OTLP/gRPC, or OTLP/HTTP).
2. **Detects installed frameworks** — scans `sys.modules` / attempts imports for 14 supported agent frameworks.
3. **Monkey-patches entry points** — replaces key methods on each detected framework with instrumented wrappers that emit `AgentEvent` objects, which the `AgentObserver` converts into OTel spans and metrics.

Your application code remains completely unchanged. The patching is transparent.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        auto_instrument()                             │
│                                                                      │
│  1. Initialize          2. Detect              3. Patch              │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────┐  │
│  │ init_telemetry│    │ For each entry   │    │ import _patch.<fw> │  │
│  │ → TracerProv  │    │ in _REGISTRY:    │    │ call install()     │  │
│  │ → MeterProv   │    │   try import     │    │ → monkey-patch     │  │
│  │ → Exporter    │    │   framework pkg  │    │   framework methods│  │
│  │              │    │   → available?    │    │                    │  │
│  │ AgentObserver │    │                  │    │ Wrapper emits      │  │
│  │ → Span maps  │    │   yes → patch    │    │ AgentEvent objects │  │
│  │ → Metrics    │    │   no  → skip     │    │                    │  │
│  └──────────────┘    └─────────────────┘    └────────────────────┘  │
│                                                                      │
│  4. At runtime: patched methods emit events → AgentObserver          │
│                                                                      │
│  ┌─────────────┐     ┌──────────────┐     ┌───────────────────────┐ │
│  │ Your Code   │ ──→ │ Patched      │ ──→ │ AgentObserver.emit()  │ │
│  │ chain.invoke│     │ Wrapper      │     │  → OTel Span          │ │
│  │ crew.kickoff│     │ emits events │     │  → OTel Metrics       │ │
│  │ Runner.run  │     │              │     │  → Console / OTLP     │ │
│  └─────────────┘     └──────────────┘     └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Execution Flow

Here is exactly what happens when `auto_instrument()` is called:

### Step 1: Initialize Telemetry (`_state.initialize()`)

```python
# _state.py
def initialize(service_name, exporter, ...):
    # Called only once (idempotent via _initialized flag)

    # 1. Create OTel Resource with service metadata
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "telemetry.sdk.language": "python",
    })

    # 2. Create TracerProvider with exporter
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())  # or OTLP exporter
    )
    trace.set_tracer_provider(tracer_provider)

    # 3. Create MeterProvider with periodic exporter
    meter_provider = MeterProvider(resource=resource, metric_readers=[...])
    metrics.set_meter_provider(meter_provider)

    # 4. Create the global AgentObserver singleton
    _observer = AgentObserver(payload_policy=payload_policy)

    # 5. Register atexit handler for clean shutdown
    atexit.register(shutdown)
```

The `AgentObserver` creates:
- A **tracer** (for creating spans)
- A **meter** (for recording metrics)
- 4 span tracking dictionaries: `_run_spans`, `_step_spans`, `_tool_spans`, `_llm_spans`
- 8 metric instruments: 4 counters + 4 histograms

### Step 2: Detect Installed Frameworks

The SDK maintains a **framework registry** — a dictionary mapping framework names to detection probes:

```python
# auto.py
_REGISTRY = {
    "langchain":    ("langchain_core",   "agentsight._patch.langchain"),
    "langgraph":    ("langgraph",        "agentsight._patch.langgraph"),
    "openai_agents":("agents",           "agentsight._patch.openai_agents"),
    "anthropic":    ("anthropic",        "agentsight._patch.anthropic"),
    "crewai":       ("crewai",           "agentsight._patch.crewai"),
    "autogen":      ("autogen",          "agentsight._patch.autogen"),
    "llamaindex":   ("llama_index.core", "agentsight._patch.llamaindex"),
    "semantic_kernel":("semantic_kernel","agentsight._patch.semantic_kernel"),
    "google_adk":   ("google.adk",       "agentsight._patch.google_adk"),
    "bedrock":      ("botocore",         "agentsight._patch.bedrock"),
    "haystack":     ("haystack",         "agentsight._patch.haystack"),
    "smolagents":   ("smolagents",       "agentsight._patch.smolagents"),
    "pydantic_ai":  ("pydantic_ai",      "agentsight._patch.pydantic_ai"),
    "phidata":      ("phi",              "agentsight._patch.phidata"),
}
#                      ^                   ^
#                      │                   └── patch module to load on success
#                      └── package to try-import for detection
```

For each entry, the SDK calls:

```python
def _framework_available(import_probe: str) -> bool:
    try:
        importlib.import_module(import_probe)  # e.g., "langchain_core"
        return True
    except (ImportError, ModuleNotFoundError):
        return False
```

Only frameworks whose probe package is importable get patched. The rest are silently skipped.

### Step 3: Install Patches

For each detected framework:

```python
def _install_framework(name, patch_module_path):
    observer = get_observer()                        # global singleton
    mod = importlib.import_module(patch_module_path)  # e.g., _patch.langchain
    mod.install(observer)                             # monkey-patch the framework
    mark_instrumented(name)                           # track in _instrumented set
```

Each patch module follows the same contract:
- `install(observer: AgentObserver)` — patches framework methods
- `uninstall()` — restores original methods

---

## Framework Detection

The detection is **lazy and non-intrusive**:

- It uses `importlib.import_module()` to check if the framework's top-level package exists.
- If the package is not installed, the SDK skips it with zero overhead — no error, no warning at INFO level.
- Detection does NOT import your application code or trigger any side effects in the framework.

You can check what's available at any time:

```python
from agentsight import available_frameworks

print(available_frameworks())
# ['langchain', 'anthropic', 'crewai']  # only installed ones
```

---

## Monkey-Patching: How Frameworks Get Instrumented

Each patch module in `_patch/` follows the same pattern:

```python
# _patch/<framework>.py

_original_method = None   # stores the real method
_installed = False        # idempotent guard

def install(observer: AgentObserver):
    global _original_method, _installed
    if _installed:
        return

    from <framework> import TargetClass

    # 1. Save the original method
    _original_method = TargetClass.entry_method

    # 2. Define a wrapper that emits events
    def _patched_method(self, *args, **kwargs):
        # Emit START event
        observer.emit(AgentEvent(name=EventName.LIFECYCLE_START, ...))

        try:
            result = _original_method(self, *args, **kwargs)  # call original
            # Emit END event (success)
            observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, ok=True, ...))
            return result
        except Exception as e:
            # Emit END event (failure)
            observer.emit(AgentEvent(name=EventName.LIFECYCLE_END, ok=False, ...))
            raise

    # 3. Replace the method on the class
    TargetClass.entry_method = _patched_method
    _installed = True

def uninstall():
    global _installed
    if _original_method is not None:
        TargetClass.entry_method = _original_method
    _installed = False
```

Key properties:
- **The original method is always saved** — uninstall restores it exactly.
- **The wrapper calls the original** — your code runs identically; the wrapper only adds event emission before/after.
- **Idempotent** — calling install twice is a no-op.
- **Both sync and async** — most patches cover both `.invoke()` and `.ainvoke()` (or equivalent).

---

## Patching Strategies by Framework

Different frameworks require different patching approaches. The SDK uses three strategies:

### Strategy 1: Callback Injection (LangChain, LangGraph)

LangChain/LangGraph use a callback system. The patch injects an adapter as a callback:

```
Runnable.invoke(input, config)
    ↓ patched to:
Runnable.invoke(input, config={callbacks: [LangChainAdapter(observer)]})
```

- Patches: `Runnable.invoke`, `Runnable.ainvoke`
- The adapter receives lifecycle events from LangChain's built-in callback system.
- No wrapping of the return value — the adapter is passive.

### Strategy 2: Context Manager Wrapping (CrewAI)

CrewAI doesn't have a callback system. The patch wraps the entry point with a context manager:

```
Crew.kickoff(*args)
    ↓ patched to:
with adapter.observe_crew(crew_name=...):
    original_kickoff(self, *args)
```

- Patches: `Crew.kickoff`, `Crew.kickoff_async`
- The context manager emits `LIFECYCLE_START` on enter, `LIFECYCLE_END` on exit.

### Strategy 3: Direct Event Emission (Anthropic, Bedrock)

For SDK clients without callbacks or middleware, the patch emits events directly:

```
Messages.create(**kwargs)
    ↓ patched to:
emit(LLM_CALL_START)
try:
    result = original_create(self, **kwargs)
    emit(LLM_CALL_END, ok=True, tokens=result.usage)
except:
    emit(LLM_CALL_END, ok=False)
    raise
```

- Patches: `Messages.create`, `AsyncMessages.create`
- Events carry extracted metadata (model name, token counts) from the response.

### Strategy 4: Hook Registration (OpenAI Agents SDK)

The OpenAI Agents SDK supports run hooks. The patch registers an adapter as a hook:

```
Runner.run(agent, input, **kwargs)
    ↓ patched to:
Runner.run(agent, input, run_hooks=OpenAIRunHooksAdapter(observer), **kwargs)
```

- Patches: `Runner.run`
- Only adds hooks if none were provided (doesn't override user hooks).

### Summary Table

| Framework | Patched Method(s) | Strategy |
|---|---|---|
| LangChain | `Runnable.invoke`, `ainvoke` | Callback injection |
| LangGraph | `Runnable.invoke`, `ainvoke` | Callback injection |
| OpenAI Agents | `Runner.run` | Hook registration |
| Anthropic | `Messages.create`, `AsyncMessages.create` | Direct event emission |
| CrewAI | `Crew.kickoff`, `kickoff_async` | Context manager |
| AutoGen | `GroupChat.run`, `initiate_chat` | Direct event emission |
| LlamaIndex | `QueryEngine.query`, `aquery` | Direct event emission |
| Semantic Kernel | `Kernel.invoke`, `invoke_prompt` | Direct event emission |
| Google ADK | `Agent.run`, `arun` | Direct event emission |
| Bedrock | `bedrock-runtime.invoke_model` | Direct event emission |
| Haystack | `Pipeline.run` | Direct event emission |
| Smolagents | `Agent.run` | Direct event emission |
| Pydantic AI | `Agent.run`, `run_sync` | Direct event emission |
| Phidata | `Assistant.run` | Direct event emission |

---

## The Event Pipeline

Every patched method emits `AgentEvent` objects. Here's the full pipeline from framework call to exported span:

```
Your Code                  Patched Wrapper              AgentObserver              OTel SDK
────────                   ───────────────              ─────────────              ────────
chain.invoke("Hi")
        │
        ▼
  _patched_invoke()
        │
        ├── AgentEvent(LIFECYCLE_START) ──→ emit() ──→ tracer.start_span("agent.run")
        │                                                    │
        │                                              _run_spans[run_id] = span_handle
        │                                              _run_counter.add(1)
        │
        ├── AgentEvent(LLM_CALL_START) ──→ emit() ──→ tracer.start_span("agent.llm")
        │                                                    │ parent = run span
        │                                              _llm_spans[llm_id] = span_handle
        │                                              _llm_counter.add(1)
        │
        ├── AgentEvent(LLM_CALL_END) ────→ emit() ──→ span.set_status(OK)
        │                                              span.end()
        │                                              _llm_duration.record(elapsed_ms)
        │
        ├── AgentEvent(LIFECYCLE_END) ───→ emit() ──→ span.set_status(OK)
        │                                              span.end()
        │                                              _run_duration.record(elapsed_ms)
        │
        ▼                                                              │
  return result                                                        ▼
                                                              BatchSpanProcessor
                                                                       │
                                                              ConsoleSpanExporter
                                                              (or OTLP exporter)
                                                                       │
                                                                       ▼
                                                              JSON output / Collector
```

### Event Types

| Event | When Emitted | Span Created |
|---|---|---|
| `agent.lifecycle.start` | Agent invocation begins | `agent.run` (root) |
| `agent.lifecycle.end` | Agent invocation completes | Ends `agent.run` |
| `agent.step.start` | Reasoning step begins | `agent.step` (child of run) |
| `agent.step.end` | Reasoning step completes | Ends `agent.step` |
| `agent.tool.call.start` | Tool invocation begins | `agent.tool` (child of step) |
| `agent.tool.call.end` | Tool invocation completes | Ends `agent.tool` |
| `agent.llm.call.start` | LLM API call begins | `agent.llm` (child of step) |
| `agent.llm.call.end` | LLM API call completes | Ends `agent.llm` |
| `agent.memory.read` | Memory/context retrieval | Event on parent span |
| `agent.memory.write` | Memory/context storage | Event on parent span |
| `agent.error` | Error occurred | Sets ERROR status on active span |

---

## Span Hierarchy and Correlation

Spans are nested using explicit correlation IDs (NOT OTel's implicit current-span context):

```
agent.run (run_id="abc123")                           ← root span
├── agent.step (step_id="step-1", parent=run_id)      ← child of run
│   ├── agent.llm (llm_call_id="llm-1", parent=step_id)  ← child of step
│   └── agent.tool (tool_call_id="tool-1", parent=step_id)
├── agent.step (step_id="step-2", parent=run_id)
│   └── agent.llm (llm_call_id="llm-2", parent=step_id)
```

This design is intentional:
- **Thread-safe**: Multiple tools can execute concurrently within a step
- **Async-safe**: Works with `asyncio`, no context propagation issues
- **DAG-safe**: Handles out-of-order events from graph-based execution

The `AgentObserver` tracks open spans in dictionaries keyed by correlation ID:

```python
_run_spans:  {run_id    → SpanHandle}
_step_spans: {step_id   → SpanHandle}
_tool_spans: {tool_call_id → SpanHandle}
_llm_spans:  {llm_call_id  → SpanHandle}
```

When a child span starts, it looks up its parent's OTel context by correlation ID:

```python
parent_ctx = self._get_parent_context(self._run_spans, event.run_id)
span = self._tracer.start_span("agent.step", context=parent_ctx)
```

---

## Global State Management

The `_state.py` module manages a singleton:

```
_observer          : AgentObserver  (singleton, created once)
_tracer_provider   : TracerProvider (OTel trace provider)
_meter_provider    : MeterProvider  (OTel metrics provider)
_initialized       : bool          (idempotent guard)
_instrumented      : set[str]      (which frameworks are patched)
```

Key behaviors:
- **Idempotent**: Calling `auto_instrument()` multiple times is safe — it returns the existing observer after the first call.
- **Process exit cleanup**: An `atexit` handler flushes pending spans/metrics before the process terminates.
- **Thread-safe**: The observer uses a lock for span tracking operations.

---

## Uninstrumentation and Cleanup

To remove all patches and restore original behavior:

```python
from agentsight import uninstrument

uninstrument()  # restores all original methods
```

Selectively:

```python
uninstrument(frameworks=["langchain"])  # only restore LangChain
```

What `uninstrument()` does:
1. For each instrumented framework, imports its `_patch` module and calls `uninstall()`.
2. `uninstall()` restores the saved original method on the framework class.
3. Resets global state (`_initialized`, `_instrumented`, `_observer`).
4. Calls `shutdown_telemetry()` which flushes and shuts down OTel providers.

After `uninstrument()`, framework methods behave exactly as they did before instrumentation.

---

## Adding a New Framework

To add auto-instrumentation for a new framework:

### 1. Add to the registry in `auto.py`:

```python
_REGISTRY["my_framework"] = ("my_framework_pkg", "agentsight._patch.my_framework")
#                              ^                     ^
#                              package to detect      patch module path
```

### 2. Create `_patch/my_framework.py`:

```python
"""Auto-instrumentation patch for MyFramework."""

from agentsight.events import AgentEvent, EventName, new_run_id
from agentsight.observer import AgentObserver

_original_run = None
_installed = False


def install(observer: AgentObserver) -> None:
    global _original_run, _installed
    if _installed:
        return

    from my_framework_pkg import Agent

    _original_run = Agent.run

    def _patched_run(self, *args, **kwargs):
        run_id = new_run_id()
        agent_id = getattr(self, "name", "my-framework-auto")

        observer.emit(AgentEvent(
            name=EventName.LIFECYCLE_START,
            agent_id=agent_id,
            run_id=run_id,
        ))

        try:
            result = _original_run(self, *args, **kwargs)
            observer.emit(AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id=agent_id,
                run_id=run_id,
                ok=True,
            ))
            return result
        except Exception as e:
            observer.emit(AgentEvent(
                name=EventName.LIFECYCLE_END,
                agent_id=agent_id,
                run_id=run_id,
                ok=False,
                error_type=type(e).__name__,
                error_message=str(e),
            ))
            raise

    Agent.run = _patched_run
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from my_framework_pkg import Agent
    if _original_run is not None:
        Agent.run = _original_run
    _installed = False
```

### 3. Add convenience function in `auto.py`:

```python
instrument_my_framework = _make_single_instrument("my_framework")
```

### 4. Export from `__init__.py`:

```python
from agentsight.auto import instrument_my_framework
```

That's it. The framework will be auto-detected and patched whenever `auto_instrument()` is called.
