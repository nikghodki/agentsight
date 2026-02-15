# Agent Observability Landscape

A comprehensive survey of every library, framework, and platform that provides observability for AI agents and LLM applications -- and where `agent-observability` fits.

Last updated: February 2026

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Landscape Map](#landscape-map)
- [Detailed Analysis](#detailed-analysis)
  - [1. Traceloop OpenLLMetry](#1-traceloop-openllmetry)
  - [2. Arize OpenInference / Phoenix](#2-arize-openinference--phoenix)
  - [3. AgentOps](#3-agentops)
  - [4. Langfuse](#4-langfuse)
  - [5. Opik by Comet](#5-opik-by-comet)
  - [6. MLflow Tracing](#6-mlflow-tracing)
  - [7. Pydantic Logfire](#7-pydantic-logfire)
  - [8. OpenLIT](#8-openlit)
  - [9. LangSmith](#9-langsmith)
  - [10. W&B Weave](#10-wb-weave)
  - [11. DeepEval / Confident AI](#11-deepeval--confident-ai)
  - [12. Laminar](#12-laminar)
  - [13. OpenTelemetry GenAI Semantic Conventions](#13-opentelemetry-genai-semantic-conventions)
  - [14. MCP Observability](#14-mcp-observability)
  - [15. Enterprise APM Platforms](#15-enterprise-apm-platforms)
  - [16. Cloud Provider Native](#16-cloud-provider-native)
  - [17. Other Notable Tools](#17-other-notable-tools)
- [Comparison Matrix](#comparison-matrix)
- [Span Hierarchy Comparison](#span-hierarchy-comparison)
- [Instrumentation Approach Comparison](#instrumentation-approach-comparison)
- [Where agent-observability Fits](#where-agent-observability-fits)
- [What Exists vs. What We Do Differently](#what-exists-vs-what-we-do-differently)
- [Recommendations](#recommendations)

---

## Executive Summary

The AI agent observability space has grown significantly. As of February 2026, there are **12+ open-source tools**, **4 enterprise APM platforms**, **3 cloud-native offerings**, and one emerging standard:

| Category | Tools | Status |
|---|---|---|
| **OTel-native instrumentation SDKs** | OpenLLMetry (6.8k stars), OpenInference (855 stars), OpenLIT (2.2k stars) | Mature, widely adopted |
| **Agent-specific monitoring** | AgentOps (5.3k stars), Laminar (2.6k stars) | Growing, agent-centric design |
| **Full observability platforms** | Langfuse (21.9k stars), Opik (17.7k stars), MLflow (24.1k stars), Logfire (4k stars) | Mature, broad feature sets |
| **LLM evaluation + tracing** | DeepEval (13.7k stars), LangSmith, W&B Weave (1.1k stars) | Evaluation-first with tracing |
| **Enterprise APM** | Datadog, New Relic, Dynatrace | Production-grade, proprietary |
| **Cloud-native** | Azure AI Foundry, AWS CloudWatch, Google Vertex AI | OTel-aligned, platform-specific |
| **Emerging standard** | OTel GenAI Semantic Conventions v1.39.0 | Development (not Stable) |

**No existing tool** combines all three of:
1. A unified event protocol that all framework adapters map to
2. Correlation-ID-based span parenting (instead of OTel's thread-local context)
3. Built-in payload redaction at the SDK level

The closest tool is **OpenLLMetry** (6.8k GitHub stars), which is OTel-native and supports ~25+ frameworks via auto-instrumentation, but uses independent per-framework instrumentors without a shared event contract.

### Key Trends (February 2026)

1. **OTel GenAI Semantic Conventions are becoming the standard.** OpenLLMetry's conventions were upstreamed to OTel. Logfire adopted `gen_ai.*` attributes. Microsoft/Cisco are collaborating on multi-agent conventions within OTel.
2. **Multi-agent observability is the emerging frontier.** New OTel conventions define `agent_to_agent_interaction`, `agent_planning`, `agent_orchestration`, and `agent.state.management` spans.
3. **MCP semantic conventions are now defined** in OTel v1.39.0, but no standalone instrumentation library exists yet.
4. **Content redaction is still immature.** Only Datadog, Langfuse (server-side), and `agent-observability` offer meaningful redaction. Most tools rely on a binary content-capture toggle.
5. **Auto-instrumentation is table stakes.** Every major tool now offers one-line setup.

---

## Landscape Map

```
                        +--------------------------------------+
                        |    OTel GenAI Semantic Conventions     |
                        |    v1.39.0 (Development)              |
                        |    gen_ai.* + MCP conventions          |
                        |    7 official instrumentor packages    |
                        +------------------+-------------------+
                                           |
            +------------------------------+------------------------------+
            |                              |                              |
   +--------v--------+       +------------v-----------+       +---------v---------+
   |  OTel-Native     |       |  Observability          |       |  Agent-Specific    |
   |  SDKs            |       |  Platforms               |       |  Monitoring        |
   |                  |       |                          |       |                    |
   |  OpenLLMetry     |       |  Langfuse (21.9k stars)  |       |  AgentOps (5.3k)   |
   |  (6.8k stars)    |       |  Opik (17.7k stars)      |       |  Laminar (2.6k)    |
   |                  |       |  MLflow (24.1k stars)    |       |                    |
   |  OpenInference   |       |  Logfire (4k stars)      |       |                    |
   |  (855 stars)     |       |                          |       |                    |
   |                  |       |  LangSmith (commercial)  |       |                    |
   |  OpenLIT         |       |  Weave (1.1k stars)      |       |                    |
   |  (2.2k stars)    |       |  DeepEval (13.7k stars)  |       |                    |
   +------------------+       +--------------------------+       +--------------------+
            |                              |                              |
   +--------v------------------------------v------------------------------v--------+
   |                                                                               |
   |                     agent-observability (this project)                         |
   |                                                                               |
   |   auto_instrument() + 15 adapters + Unified event protocol +                  |
   |   Correlation-ID spans + Payload redaction                                    |
   |                                                                               |
   +-------------------------------------------------------------------------------+
            |                              |                              |
   +--------v--------+       +------------v-----------+       +---------v---------+
   |  Enterprise APM  |       |  Cloud-Native           |       |  OTel GenAI SIG    |
   |                  |       |                          |       |                    |
   |  Datadog         |       |  Azure AI Foundry        |       |  Multi-agent       |
   |  New Relic       |       |  AWS CloudWatch          |       |  conventions       |
   |  Dynatrace       |       |  Google Vertex AI        |       |  (Microsoft/Cisco) |
   +------------------+       +--------------------------+       +--------------------+
```

---

## Detailed Analysis

### 1. Traceloop OpenLLMetry

**The most established OTel-native LLM instrumentation library.**

| | |
|---|---|
| **GitHub** | [traceloop/openllmetry](https://github.com/traceloop/openllmetry) |
| **Stars** | 6.8k |
| **License** | Apache-2.0 |
| **Contributors** | 102 |
| **Latest** | v0.52.3 (Feb 2026) |
| **Language** | Python (separate JS repo: openllmetry-js) |

**What it does:** A set of OpenTelemetry extensions that auto-instruments LLM applications. Call `Traceloop.init()` and it monkey-patches detected libraries to emit OTel spans automatically. All output is standard OTel data consumable by any backend.

**Framework coverage:**

| Category | Supported |
|---|---|
| Agent frameworks | Agno, AWS Strands, CrewAI, Haystack, LangChain, Langflow, LangGraph, LiteLLM, LlamaIndex, OpenAI Agents |
| LLM providers | Anthropic, AWS Bedrock, Cohere, Google Gemini, Groq, HuggingFace, IBM Watsonx, Mistral, Ollama, OpenAI/Azure, Replicate, SageMaker, Together AI, Vertex AI, Voyage AI, WRITER |
| Vector databases | Chroma, LanceDB, Marqo, Milvus, Pinecone, Qdrant, Weaviate |

**OTel GenAI conventions:** **Adopted and contributed.** OpenLLMetry's README states: *"Our semantic conventions are now part of OpenTelemetry!"* Their conventions were upstreamed into the official OTel GenAI semantic conventions, making Traceloop a foundational contributor to the standard.

**Recent additions (mid-2025 -- Feb 2026):**
- v0.52.3 -- OpenAI Agents: dual instrumentation mode flag
- v0.52.0 -- Voyage AI instrumentation; evals output schema alignment
- v0.51.0 -- Google GenerativeAI metrics; CSV/JSON experiment format
- v0.50.0 -- Guardrail decorator
- v0.49.8 -- OpenAI WebSocket / Realtime API support

**Span hierarchy:**

```
WORKFLOW (orchestration container)
  +-- TASK (discrete operation)
       +-- AGENT (autonomous decision-maker)
            +-- TOOL (function invocation)
```

Set via `TraceloopSpanKindValues` attribute. Manual annotation through decorators: `@workflow(name="...")`, `@task(name="...")`, `@agent(name="...")`, `@tool(name="...")`.

**Payload handling:** Limited. Provides `enable_content_tracing` flag (when disabled, message content is not recorded), `span_postprocess_callback` for custom filtering, and custom exporter support. No built-in fine-grained field-level redaction (regex-based PII masking, key-based scrubbing, etc.).

**Strengths:**
- Most mature OTel-native option
- Auto-instrumentation (zero code changes)
- 7 vector database instrumentors (unique)
- Conventions upstreamed to OTel standard
- Large contributor base

**Limitations:**
- Each framework instrumentor is independent -- no shared event protocol
- No built-in payload redaction (binary on/off for content capture)
- Relies on OTel's thread-local context for span parenting
- Auto-instrumentation via monkey-patching can break across version upgrades

---

### 2. Arize OpenInference / Phoenix

**The richest span kind taxonomy with 10 span types.**

| | |
|---|---|
| **GitHub** | [Arize-AI/openinference](https://github.com/Arize-AI/openinference) |
| **Stars** | 855 |
| **License** | Apache-2.0 |
| **Language** | Python, JavaScript, Java |

**What it does:** OTel-compatible instrumentation plugins plus the OpenInference semantic conventions. Designed as the instrumentation layer for Arize Phoenix, but outputs standard OTel data.

**Framework coverage (30+ in Python, growing JS/TS):**

Python: OpenAI, Anthropic, Bedrock, Mistral, Groq, Google GenAI, Vertex AI, LangChain, LlamaIndex, DSPy, Haystack, CrewAI, AutoGen, OpenAI Agents, Agno, BeeAI, PydanticAI, MCP, Smolagents, Pipecat, Portkey, Guardrails, LiteLLM

JavaScript/TypeScript: OpenAI, Anthropic, LangChain.js, AWS Bedrock, BeeAI, MCP, Vercel AI SDK

**Recent additions:**
- Pipecat instrumentation (voice AI framework)
- Agno 2.5 support
- PydanticAI improvements
- OpenLLMetry/OpenLIT span processors (normalize traces from other libraries into OpenInference format)

**Span kinds (10 types -- most granular in the space):**

| Span Kind | Description |
|---|---|
| `LLM` | Language model calls |
| `EMBEDDING` | Embedding generation |
| `CHAIN` | Links between application steps |
| `RETRIEVER` | Data retrieval (RAG) |
| `RERANKER` | Document relevance reranking |
| `TOOL` | External tool/function invocations |
| `AGENT` | Reasoning blocks coordinating LLMs and tools |
| `GUARDRAIL` | Content safety checks |
| `EVALUATOR` | Output quality assessment |
| `PROMPT` | Template rendering |

**Strengths:**
- Richest span taxonomy (GUARDRAIL, EVALUATOR, PROMPT are unique)
- Multi-language (Python, JS, Java)
- 30+ framework instrumentors
- MCP instrumentation support

**Limitations:**
- Tightly associated with Arize Phoenix platform
- Each framework instrumentor is independent
- No shared event protocol
- No built-in payload redaction

---

### 3. AgentOps

**The most agent-centric monitoring tool with session replays.**

| | |
|---|---|
| **GitHub** | [AgentOps-AI/agentops](https://github.com/AgentOps-AI/agentops) |
| **Stars** | 5.3k |
| **License** | MIT |
| **Language** | Python, TypeScript |

**What it does:** Agent monitoring platform with session replays, LLM cost tracking, benchmarking, and compliance monitoring.

**Framework coverage:** OpenAI Agents SDK (Python & TypeScript), CrewAI, AG2 (AutoGen), LangChain, LangGraph, Agno, Smolagents, Mem0, Camel AI, LlamaIndex, Llama Stack, Haystack 2.x, DSPy, Cohere, Anthropic, Mistral, LiteLLM

**Recent additions (mid-2025 -- Feb 2026):**
- OpenAI Agents SDK support (Python & TypeScript)
- LangGraph integration
- Haystack 2.x auto-instrumentation with Azure support
- AG2 migration (from PyAutoGen)
- DSPy callback handler
- Input/output guardrail decorator
- `@track_endpoint` decorator for HTTP tracing

**Span hierarchy (6 levels -- most agent-centric):**

```
SESSION (root container for an entire agent session)
  +-- AGENT (autonomous entity)
       +-- WORKFLOW (logical grouping)
            +-- OPERATION / TASK (specific function)
                 +-- LLM (language model interaction)
                 +-- TOOL (tool/API usage)
```

**Strengths:**
- Session-level tracking with replays (unique feature)
- 6-level hierarchy designed for multi-agent systems
- Cost tracking per session
- Growing framework support (17+ frameworks)

**Limitations:**
- SaaS backend required for full features
- Not OTel-native (own SDK)
- No built-in payload redaction
- No unified event protocol

---

### 4. Langfuse

**The most popular open-source LLM observability platform.**

| | |
|---|---|
| **GitHub** | [langfuse/langfuse](https://github.com/langfuse/langfuse) |
| **Stars** | 21.9k |
| **License** | MIT (except `ee` folders) |
| **Latest** | v3.153.0 (Feb 2026) |
| **Language** | Python, JavaScript/TypeScript |

**What it does:** Full LLM engineering platform: tracing, prompt management, evaluation, and production monitoring. Acts as an OTel backend -- consumes traces via an OTLP endpoint.

**Framework coverage:** 50+ integrations via native SDKs and OTel endpoint.

**Recent additions (mid-2025 -- Feb 2026):**
- **Server-side ingestion masking for OTel traces** (v3.152.0) -- data redaction at the server level
- LLM-as-a-judge evaluations running on observations
- Dataset versioning across APIs and experiment runs
- Events table v4 with event-based views
- Corrections feature with JSON validation and diff viewer
- Trace comments for inline annotation
- Custom `trace_id` mapping for LiteLLM (v3.147.0)

**Trace model:**

```
Session (groups multi-turn conversations)
  +-- Trace (single request)
       +-- Span (generic operation)
       +-- Generation (LLM call: model, tokens, cost)
       +-- Event (point-in-time marker)
```

**Strengths:**
- Highest GitHub stars (21.9k) -- largest community
- Full platform: tracing + prompt management + evaluation + monitoring
- OTel-native backend (accepts any OTel traces)
- Self-hostable
- Server-side OTel trace masking (new)

**Limitations:**
- It's a platform/backend, not an instrumentation SDK
- Does not produce OTel traces -- it consumes them
- Requires running the Langfuse server
- `Generation` type is Langfuse-specific, not portable

---

### 5. Opik by Comet

**Broadest framework integration count with the strongest evaluation story.**

| | |
|---|---|
| **GitHub** | [comet-ml/opik](https://github.com/comet-ml/opik) |
| **Stars** | 17.7k |
| **License** | Apache-2.0 |
| **Latest** | v1.10.13 (Feb 2026) |
| **Language** | Python, TypeScript |

**What it does:** End-to-end LLM evaluation and observability platform. Covers development (tracing, debugging), testing (automated evaluation, CI/CD integration), and production monitoring (online evaluation rules).

**Framework coverage (70+ integrations):**

| Category | Frameworks |
|---|---|
| LLM Providers | OpenAI, Anthropic, Google Gemini, Cohere, Mistral, Groq, Together AI, Bedrock, xAI Grok |
| Agent Frameworks | LangChain, LlamaIndex, CrewAI, AutoGen, PydanticAI, LangGraph, DSPy, Haystack, Google ADK, Flowise, Langflow, n8n, Dify |

**Recent additions:**
- OpikAssist backend integration for AI capabilities
- G-Eval metric in TypeScript SDK
- JSON path explorer for trace inspection
- Annotation queue management
- Service profiles (v1.7.0) for flexible dev setups
- Merged traces, threads, and spans into unified "Logs" tab
- Google ADK, AutoGen, and Flowise AI integrations

**Strengths:**
- 70+ framework integrations (broadest count)
- LLM-as-a-judge evaluation metrics
- CI/CD integration via PyTest
- Self-hostable (Docker/Kubernetes)

**Limitations:**
- Platform-centric (requires Opik server)
- `@opik.track` decorator is Opik-specific, not portable
- No shared event protocol across frameworks
- No built-in payload redaction at the SDK level

---

### 6. MLflow Tracing

**Broadest auto-instrumentation coverage in the ML lifecycle space.**

| | |
|---|---|
| **GitHub** | [mlflow/mlflow](https://github.com/mlflow/mlflow) |
| **Stars** | 24.1k |
| **License** | Apache-2.0 |
| **Latest** | v3.10.0rc0 (Feb 2026) |
| **Language** | Python, TypeScript, Java, R |

**What it does:** GenAI tracing module integrated into the broader MLflow ecosystem (experiment tracking, model registry, deployment).

**Framework coverage (50+ total):**

| Category | Count | Examples |
|---|---|---|
| Python agent frameworks | 24+ | LangChain, LangGraph, OpenAI Agents, DSPy, PydanticAI, Google ADK, CrewAI, LlamaIndex, AutoGen, Smolagents, Semantic Kernel, Haystack |
| TypeScript frameworks | 5 | LangChain, LangGraph, Vercel AI SDK, Mastra, VoltAgent |
| Model providers | 18+ | OpenAI, Anthropic, Gemini, Bedrock, LiteLLM, Mistral, Ollama, Groq, DeepSeek |
| Gateways/Tools | 12+ | LiteLLM Proxy, Vercel AI Gateway, OpenRouter, Portkey, Helicone, Kong, Claude Code |

**Major additions (mid-2025 -- Feb 2026):**
- **Trace Cost Tracking** (v3.10.0rc0) -- automatic model extraction from LLM spans with cost calculation
- **Gateway Usage Tracking** -- AI Gateway endpoint monitoring
- **Multi-turn Conversation Simulation** -- custom scenario creation
- **MLflow Assistant** (v3.9.0) -- in-product AI chatbot for debugging
- **Online Monitoring with LLM Judges** -- automatic evaluation on traces
- **Distributed Tracing** -- cross-service context propagation
- **Trace Overview Dashboard** -- pre-built statistics

**Strengths:**
- Broadest framework coverage (50+)
- Deep MLflow ecosystem integration
- OTel-compatible output
- Distributed tracing for cross-service observability

**Limitations:**
- Tied to MLflow ecosystem
- No agent-specific span hierarchy
- No shared event protocol
- No built-in payload redaction

---

### 7. Pydantic Logfire

**The cleanest OTel wrapper for Python/Pydantic stacks.**

| | |
|---|---|
| **GitHub** | [pydantic/logfire](https://github.com/pydantic/logfire) |
| **Stars** | 4k |
| **License** | MIT (SDKs open-source, backend proprietary) |
| **Latest** | v4.24.0 (Feb 2026) |
| **Language** | Python, TypeScript, Rust |

**What it does:** Observability platform built by the Pydantic team. Opinionated OTel wrapper with deep Python integration and special support for LLM/AI monitoring.

**LLM framework support:** OpenAI, Anthropic, Google GenAI, PydanticAI, LangChain, LiteLLM, DSPy, LlamaIndex, Mirascope, Magentic, Claude Agent SDK, OpenAI Agents, LangSmith

**OTel GenAI conventions:** **Adopted.** v4.21.0 added OTel GenAI semantic convention scalar attributes to LLM instrumentations, and v4.23.0 added semantic convention message attributes.

**Recent additions (mid-2025 -- Feb 2026):**
- v4.24.0 -- Latest release
- v4.23.0 -- OpenAI Agents and LangSmith integrations
- v4.22.0 -- Google GenAI integration
- v4.21.0 -- LangChain reasoning summaries, OTel GenAI semconv attributes
- v4.20.0 -- Anthropic tool call ID tracking, Pytest integration
- v4.18.0 -- Claude SDK instrumentation, aiohttp request body capture

**Strengths:**
- Cleanest Python developer experience
- Standard OTel output (not locked to Logfire backend)
- Multi-language SDKs (Python, TypeScript, Rust)
- OTel GenAI semconv compliance

**Limitations:**
- Backend is proprietary (SDKs are open-source)
- Narrower framework coverage (13 vs. 50+ for MLflow)
- No agent-specific span hierarchy
- No built-in payload redaction

---

### 8. OpenLIT

**OTel-native SDK + self-hosted platform with GPU monitoring.**

| | |
|---|---|
| **GitHub** | [openlit/openlit](https://github.com/openlit/openlit) |
| **Stars** | 2.2k |
| **License** | Apache-2.0 |
| **Contributors** | 57 |
| **Latest** | v1.36.8 (Feb 2026) |
| **Language** | Python, TypeScript |

**What it does:** Open-source platform for AI engineering providing OTel-native LLM observability, GPU monitoring (NVIDIA, AMD), guardrails, evaluations, prompt management (Prompt Hub), API key management (Vault), LLM testing/comparison (OpenGround), and collector management (Fleet Hub via OpAMP).

**Key differentiators from OpenLLMetry:** Both are OTel-native, but OpenLLMetry is purely an instrumentation library (no UI). OpenLIT bundles an instrumentation SDK **plus** a self-hosted observability platform with UI, prompt management, guardrails, and playground. OpenLLMetry has broader community adoption (6.8k vs 2.2k stars).

**Framework coverage:** 50+ integrations including LLM providers, vector databases, agent frameworks, and GPUs.

**Recent additions:**
- PromptFlow OTel tracing instrumentation
- psycopg3 database driver support
- Agno framework integration
- Qdrant client 1.16.0+ compatibility
- LangGraph fixes
- OpenGround V2 UI overhaul

**How it works:**

```python
import openlit

openlit.init()  # One line -- OTel traces emitted automatically
```

**Strengths:**
- Fully OTel-native from the ground up
- Self-hosted platform with UI included
- GPU monitoring (unique in the space)
- 50+ integrations
- Prompt Hub, Vault, and OpenGround features

**Limitations:**
- Smaller community than OpenLLMetry (2.2k vs 6.8k stars)
- No built-in fine-grained payload redaction
- No unified event protocol across frameworks

---

### 9. LangSmith

**Commercial platform with OTel interoperability and LangChain-native tracing.**

| | |
|---|---|
| **GitHub** | [langchain-ai/langsmith-sdk](https://github.com/langchain-ai/langsmith-sdk) |
| **Stars** | 778 (SDK only; platform is commercial) |
| **License** | MIT (SDK), proprietary (platform) |
| **Latest** | v0.7.3 (Feb 2026) |
| **Language** | Python, TypeScript |

**What it does:** Framework-agnostic platform for developing, debugging, and deploying AI agents and LLM applications. Tight integration with LangChain/LangGraph ecosystems.

**OTel interoperability:** Not natively OTel-based, but has **full OTel interoperability**. LangSmith accepts traces via an OTLP endpoint (`https://api.smith.langchain.com/otel`). For LangChain/LangGraph apps, set `LANGSMITH_OTEL_ENABLED=true` for automatic OTel instrumentation. Supports hybrid fan-out to multiple backends.

**Recent additions:**
- Agent observability for OpenAI and Claude Agent SDKs
- Improved span tracking and timing for OpenAI Agents
- Global singleton for prompt caching (enabled by default)
- JWT authentication support for replicas
- Annotation queue rubric support

**Payload handling:** Yes. LangSmith provides input/output hiding at the project or per-run level.

**Strengths:**
- Deep LangChain/LangGraph integration
- OTel compatibility (accepts OTLP traces)
- Full platform features (evaluation, prompt management, annotation)
- HIPAA, SOC 2 Type 2, GDPR compliance

**Limitations:**
- Commercial platform (SDK is open source, platform is proprietary)
- `@traceable` decorator is LangSmith-specific
- Star count reflects SDK only, not platform user base
- Primarily LangChain-ecosystem focused

---

### 10. W&B Weave

**Experiment tracking meets LLM tracing.**

| | |
|---|---|
| **GitHub** | [wandb/weave](https://github.com/wandb/weave) |
| **Stars** | 1.1k |
| **License** | Apache-2.0 |
| **Latest** | v0.52.28 (Feb 2026) |
| **Language** | Python |

**What it does:** Toolkit for developing Generative AI applications, providing tracing, evaluation, and experiment tracking. Part of the Weights & Biases ecosystem.

**Framework coverage:** OpenAI, Anthropic, Google AI Studio, Hugging Face, LangChain, LlamaIndex, CrewAI, DSPy, Bedrock Agents

**Recent additions:**
- OpenAI Realtime API support (GA)
- Bedrock Agents support
- Google GenAI with system instructions and thinking tokens
- ClickHouse-based call stats for performance
- Op kinds and colors for visual differentiation
- Migration from `requests` to `httpx`

**Strengths:**
- Deep W&B ecosystem integration
- Evaluation-first design
- Active development cadence

**Limitations:**
- Not OTel-native (own tracing protocol)
- Decorator-based (`@weave.op()`) -- not zero-code
- Tied to W&B platform
- No built-in payload redaction

---

### 11. DeepEval / Confident AI

**The most popular open-source LLM evaluation framework.**

| | |
|---|---|
| **GitHub** | [confident-ai/deepeval](https://github.com/confident-ai/deepeval) |
| **Stars** | 13.7k |
| **License** | Apache-2.0 |
| **Language** | Python |

**What it does:** LLM evaluation framework ("Pytest for LLMs"). Provides 14+ evaluation metrics, including hallucination detection, faithfulness, answer relevancy, contextual recall, and bias. Confident AI is the commercial cloud platform for dashboards and hosted evaluation runs.

**Tracing:** Uses `@observe` decorator for tracing LLM calls, retrievers, tool calls, and agents. Not OTel-native.

**Framework coverage:** Framework-agnostic evaluation (input/output pairs from any source). Specific integrations for LlamaIndex and Hugging Face.

**Strengths:**
- Most popular evaluation-first tool (13.7k stars)
- 14+ evaluation metrics out of the box
- Pytest integration for CI/CD
- Framework-agnostic evaluation

**Limitations:**
- Evaluation-first, not observability-first
- Not OTel-native
- `@observe` decorator requires code modification
- No built-in payload redaction

---

### 12. Laminar

**Rust-based, agent-first observability platform.**

| | |
|---|---|
| **GitHub** | [lmnr-ai/lmnr](https://github.com/lmnr-ai/lmnr) |
| **Stars** | 2.6k |
| **License** | Apache-2.0 |
| **YC Batch** | S24 |
| **Language** | Python, TypeScript, Rust (backend) |

**What it does:** Open-source observability platform purpose-built for AI agents. Differentiates with Rust-based performance backend, SQL access to traces, and agent-first design.

**Framework coverage:** OpenAI, Anthropic, Gemini, LangChain (growing list).

**Strengths:**
- Rust-based backend for performance
- Agent-first design
- SQL access to traces, metrics, events
- Real-time trace viewing with full-text search
- Custom event tracking

**Limitations:**
- Smaller community (2.6k stars)
- Narrower framework coverage
- Not fully OTel-native (partial)
- Newer entrant (launched 2024)

---

### 13. OpenTelemetry GenAI Semantic Conventions

**The emerging standard the industry is converging toward.**

| | |
|---|---|
| **Source** | [opentelemetry.io/docs/specs/semconv/gen-ai](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
| **Version** | v1.39.0 (January 12, 2025) |
| **Status** | Development (not Stable) |
| **Scope** | Specification + 7 official instrumentor packages |

**What it standardizes:** The official `gen_ai.*` attribute namespace for AI/LLM observability across all OTel implementations.

#### Span Types (5+)

| Span Type | SpanKind | Operation Name | Description |
|---|---|---|---|
| Inference | CLIENT | `chat`, `text_completion`, `generate_content` | Client calls to GenAI models |
| Embeddings | CLIENT | `embeddings` | Embedding generation |
| Execute Tool | INTERNAL | `execute_tool` | Tool execution within agent flows |
| Create Agent | CLIENT | `create_agent` | Agent instantiation |
| Invoke Agent | CLIENT or INTERNAL | `invoke_agent` | Agent execution |
| Retrieval | CLIENT | -- | Vector store / knowledge base access |

#### Agent Attributes (4, all Development)

| Attribute | Type | Requirement | Description |
|---|---|---|---|
| `gen_ai.agent.id` | string | Conditionally Required | Unique agent identifier |
| `gen_ai.agent.name` | string | Conditionally Required | Human-readable agent name |
| `gen_ai.agent.description` | string | Conditionally Required | Free-form agent description |
| `gen_ai.agent.version` | string | Recommended | Agent version string |

#### Official Agent Span Hierarchy

```
Invoke Agent
  +-- Inference (LLM calls)
  +-- Execute Tool (tool calls)
  +-- Invoke Agent (sub-agent calls, nested)
```

#### Key Standardized Attributes

| Attribute | Description |
|---|---|
| `gen_ai.operation.name` | Operation type (required) |
| `gen_ai.provider.name` | Provider identifier (required) |
| `gen_ai.request.model` | Model requested |
| `gen_ai.response.model` | Model actually used |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.tool.call.id` | Tool call identifier |
| `gen_ai.tool.name` | Tool name |
| `gen_ai.conversation.id` | Session/thread identifier |
| `gen_ai.request.temperature` | Temperature parameter |
| `gen_ai.request.max_tokens` | Max tokens parameter |

#### Content Recording Strategy (Three Tiers)

1. **Default:** Don't capture content (PII-aware by default)
2. **In-process recording:** Record as span attributes after completion, opt-in via `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
3. **External storage:** Upload content separately, record references on spans

Opt-in attributes: `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`

#### Metrics (5 Histograms, all Development)

| Metric | Unit | Requirement |
|---|---|---|
| `gen_ai.client.operation.duration` | seconds | Required |
| `gen_ai.client.token.usage` | tokens | Recommended |
| `gen_ai.server.request.duration` | seconds | Recommended |
| `gen_ai.server.time_per_output_token` | seconds | Recommended |
| `gen_ai.server.time_to_first_token` | seconds | Recommended |

#### Official OTel Instrumentor Packages (7)

All in `opentelemetry-python-contrib/instrumentation-genai/`, all Development status:

| Package | Library | Min Version | Metrics |
|---|---|---|---|
| `opentelemetry-instrumentation-openai-v2` | openai | >= 1.26.0 | Yes |
| `opentelemetry-instrumentation-anthropic` | anthropic | >= 0.16.0 | No |
| `opentelemetry-instrumentation-google-genai` | google-genai | >= 1.0.0 | No |
| `opentelemetry-instrumentation-langchain` | langchain | >= 0.3.21 | No |
| `opentelemetry-instrumentation-openai-agents-v2` | openai-agents | >= 0.3.3 | No |
| `opentelemetry-instrumentation-vertexai` | google-cloud-aiplatform | >= 1.64 | No |
| `opentelemetry-instrumentation-weaviate` | weaviate-client | >= 3.0.0 | No |

The **OpenAI v2** instrumentor is the most mature (Beta, v2.3b0, Dec 2025). The **OpenAI Agents v2** instrumentor generates spans for agents, tools, generations, guardrails, and handoffs with configurable content capture modes (`span_only`, `event_only`, `span_and_event`, `no_content`).

#### Multi-Agent Conventions (Emerging)

Microsoft and Cisco (Outshift) are collaborating on **multi-agent observability semantic conventions** within OTel, introducing spans for:
- `execute_task` -- Task execution within agent orchestration
- `agent_to_agent_interaction` -- Inter-agent communication
- `agent_planning` -- Agent planning/reasoning steps
- `agent_orchestration` -- Multi-agent coordination
- `agent.state.management` -- Agent state tracking

These are being integrated into Microsoft Foundry, Semantic Kernel, and Azure AI packages.

**Provider-specific conventions exist for:** OpenAI, Anthropic, AWS Bedrock, Azure AI Inference, MCP.

---

### 14. MCP Observability

**Model Context Protocol semantic conventions are defined but no standalone instrumentation library exists yet.**

#### MCP Protocol Built-in Observability

The MCP specification (version 2025-03-26) does **not** define tracing or telemetry. It includes:
- Logging as a utility (server-to-client log notifications)
- Progress tracking for long-running operations
- Error reporting via JSON-RPC error codes

The MCP Python SDK has **no built-in OpenTelemetry integration**.

#### OTel MCP Semantic Conventions (v1.39.0)

As of v1.39.0 (January 12, 2025), OTel defines MCP semantic conventions at `docs/gen-ai/mcp.md`:

**Span Types:**
- **MCP Client Spans** -- Track requests from the client perspective. Name: `{mcp.method.name} {target}`
- **MCP Server Spans** -- Track request processing from the server perspective

**Key Attributes (~20):**
- `mcp.method.name` (Required) -- e.g., `tools/call`, `initialize`, `resources/read`
- `mcp.session.id` -- Session identifier
- `mcp.protocol.version` -- MCP protocol version
- `mcp.resource.uri` -- Resource URI
- `gen_ai.tool.name` -- Bridges GenAI and MCP conventions
- `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` -- Tool I/O
- `jsonrpc.request.id` -- JSON-RPC request ID
- Network attributes: `network.transport`, `network.protocol.name`, `server.address`, `server.port`

**Context Propagation:**
W3C Trace Context (`traceparent`, `tracestate`) and optionally W3C Baggage injected into MCP messages via `params._meta`. MCP server instrumentation extracts this context as parent for server spans.

**Metrics (4 Histograms):**

| Metric | Description |
|---|---|
| `mcp.client.operation.duration` | Client-side request duration |
| `mcp.server.operation.duration` | Server-side processing duration |
| `mcp.client.session.duration` | Client session lifetime |
| `mcp.server.session.duration` | Server session lifetime |

#### Current State

No widely adopted standalone MCP instrumentation library has emerged. The conventions are new (January 2025), and the MCP Python SDK does not include native OTel integration. However:

- OpenInference has MCP instrumentation in its Python and JS/TS packages
- The conventions bridge MCP and GenAI telemetry through shared `gen_ai.tool.*` attributes
- `agent-observability` instruments MCP tool calls through its framework adapters when agents use MCP tools

---

### 15. Enterprise APM Platforms

#### Datadog LLM Observability

| | |
|---|---|
| **Type** | Proprietary platform |
| **OTel** | Partial (via `ddtrace` + OTel ingestion) |
| **Auto-instrumentation** | Yes (`DD_LLMOBS_ENABLED=1`) |
| **Frameworks** | OpenAI, Anthropic, Bedrock, Azure OpenAI, LangChain, Vertex AI |
| **Payload redaction** | Yes (`DD_LLMOBS_PROMPT_REDACTION_ENABLED`) |

Extends Datadog's APM with LLM-specific tracing, token usage, cost estimation, and quality metrics. The `ddtrace` library (BSD-3-Clause, 3k+ stars) auto-patches supported libraries.

#### New Relic AI Monitoring

| | |
|---|---|
| **Type** | Proprietary platform |
| **OTel** | Partial (agents + OTel ingestion) |
| **Auto-instrumentation** | Yes (agent configuration toggle) |
| **Frameworks** | OpenAI, Bedrock, LangChain |
| **Payload redaction** | Yes (`ai_monitoring.record_content.enabled: false`) |

Extends New Relic's observability with AI model performance, token usage, cost tracking, and response quality. Python and Node.js agents supported.

#### Dynatrace AI Observability

| | |
|---|---|
| **Type** | Proprietary platform |
| **OTel** | Yes (major OTel contributor, supports GenAI semconv) |
| **Auto-instrumentation** | Yes (OneAgent auto-discovery) |
| **Frameworks** | OpenAI, Azure OpenAI, Bedrock, Vertex AI; Java, .NET, Node.js, Go, Python |
| **Payload redaction** | Yes (data masking and sensitive data protection rules) |

Most OTel-aligned of the enterprise APMs. Dynatrace is a major contributor to the OTel project and supports the emerging GenAI semantic conventions. OneAgent auto-detects AI service calls without code changes.

---

### 16. Cloud Provider Native

#### Microsoft Azure AI Foundry

**Most advanced cloud-native agent observability offering.**

- OTel-native tracing using GenAI semantic conventions
- Native integrations: Microsoft Agent Framework, Semantic Kernel, LangChain, LangGraph, OpenAI Agents SDK
- Content recording control: `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED`
- Export to Azure Monitor Application Insights
- Multi-agent observability with new semantic conventions
- Thread-level visibility in Agents Playground
- AI Toolkit for VS Code integration for local OTLP tracing
- User feedback attachment to traces via OTel conventions

Azure has the most advanced cloud-native offering, with Microsoft actively contributing multi-agent semantic conventions to OTel.

#### AWS CloudWatch Application Signals

- Built on OTel via AWS Distro for OpenTelemetry (ADOT)
- Auto-instrumentation for AWS Bedrock (InvokeModel, Converse)
- Prompt/completion content not captured by default
- Limited to AWS ecosystem (Bedrock, SageMaker)

#### Google Cloud Vertex AI

- OTel-native via Google Cloud Trace
- Uses OTel GenAI semantic conventions
- Support for Vertex AI SDK, Gemini models, Reasoning Engine agents, LangChain-on-Vertex
- DLP API integration for data protection
- Primarily Google-ecosystem focused

---

### 17. Other Notable Tools

| Tool | Type | Stars | Notes |
|---|---|---|---|
| **Helicone** | Proxy-based platform | -- | Sits as proxy between app and LLM APIs. No code changes. Not OTel-based. |
| **Braintrust** | Evaluation platform | -- | Evaluation and dataset management. Own tracing format. |
| **LiteLLM** | LLM proxy + observability | -- | Unified API for 100+ LLMs. Built-in spend tracking and callbacks. |
| **Patronus AI** | Evaluation SDK | ~7 | LLM evaluation company. `@traced()` decorator. Not OTel-based. |
| **HoneyHive** | Platform + SDK | -- | AI observability + evaluation. Partial OTel support. Configurable data redaction. |
| **Galileo** | Platform + SDK | -- | LLM evaluation + observability. PII detection and redaction. Not OTel-native. |

---

## Comparison Matrix

| Feature | OpenLLMetry | OpenInference | AgentOps | Langfuse | Opik | MLflow | Logfire | OpenLIT | **agent-observability** |
|---|---|---|---|---|---|---|---|---|---|
| **GitHub Stars** | 6.8k | 855 | 5.3k | 21.9k | 17.7k | 24.1k | 4k | 2.2k | -- |
| **Type** | SDK | SDK | Platform+SDK | Platform | Platform | Platform+SDK | Platform+SDK | Platform+SDK | **SDK** |
| **OTel Native** | Yes | Yes | No | Yes (v3) | Accepts | Compatible | Yes | Yes | **Yes** |
| **GenAI SemConv** | Yes (contrib.) | Own spec | No | Partial | No | No | Yes | Community | **`agent.*` namespace** |
| **Unified Event Protocol** | No | No | No | No | No | No | No | No | **Yes** |
| **Correlation-ID Spans** | No | No | No | No | No | No | No | No | **Yes** |
| **Built-in Redaction** | Binary flag | No | No | Server-side | No | No | No | No | **Yes (fine-grained)** |
| **Auto-Instrumentation** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |
| **Agent Frameworks** | 10 | 20+ | 17+ | 50+ | 70+ | 50+ | 13+ | 50+ | **15** |
| **LLM Providers** | 16+ | 7+ | 9+ | 50+ | 9+ | 18+ | 3 | 50+ | -- |
| **Vector DB Support** | 7 | No | No | No | No | No | No | Yes | No |
| **Evaluation Tools** | No | Yes (Phoenix) | No | Yes | Yes | Yes | No | Yes | No |
| **MCP Support** | No | Yes | No | No | No | No | No | No | Via adapters |
| **Self-Hosted** | N/A (SDK) | Yes (Phoenix) | No | Yes | Yes | Yes | No | Yes | **N/A (SDK)** |
| **Open Source** | Yes | Yes | Yes | Partial | Yes | Yes | SDKs only | Yes | **Yes** |
| **License** | Apache-2.0 | Apache-2.0 | MIT | MIT* | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 | **MIT** |

### Enterprise / Cloud Matrix

| Feature | Datadog | New Relic | Dynatrace | Azure AI | AWS CloudWatch | Google Vertex |
|---|---|---|---|---|---|---|
| **OTel Native** | Partial | Partial | Yes | Yes | Yes (ADOT) | Yes |
| **GenAI SemConv** | No | No | Yes | Yes | No | Yes |
| **Auto-Instrumentation** | Yes | Yes | Yes | Yes | Yes (Bedrock) | Partial |
| **Payload Redaction** | Yes | Yes | Yes | Toggle | Default off | Via DLP |
| **Multi-Agent** | No | No | No | Yes (active) | No | No |
| **Self-Hosted** | No | No | No | No | N/A | N/A |

---

## Span Hierarchy Comparison

Every tool defines its own span hierarchy. Here they are side-by-side:

### OpenLLMetry
```
WORKFLOW
  +-- TASK
       +-- AGENT
            +-- TOOL
```

### OpenInference (10 span kinds)
```
AGENT
  +-- CHAIN
  |    +-- LLM
  |    +-- RETRIEVER
  |    +-- RERANKER
  |    +-- TOOL
  +-- GUARDRAIL
  +-- EVALUATOR
  +-- PROMPT --> EMBEDDING
```

### AgentOps (6 levels)
```
SESSION
  +-- AGENT
       +-- WORKFLOW
            +-- OPERATION / TASK
                 +-- LLM
                 +-- TOOL
```

### OTel GenAI Semantic Conventions
```
Invoke Agent
  +-- Inference (LLM)
  +-- Execute Tool
  +-- Invoke Agent (sub-agent, nested)
```

### Langfuse
```
Session
  +-- Trace
       +-- Span (generic)
       +-- Generation (LLM-specific)
       +-- Event (point-in-time)
```

### agent-observability (this project)
```
agent.run (keyed by run_id)
  +-- agent.step (keyed by step_id)
       +-- agent.tool (keyed by tool_call_id)
       +-- agent.tool (concurrent, same parent)
       +-- agent.llm (keyed by llm_call_id)
```

**Key difference:** Our hierarchy uses explicit correlation IDs for span parenting rather than OTel's implicit thread-local context. This means concurrent tool calls within a step, out-of-order events, and async frameworks produce correct parent-child trees without workarounds.

---

## Instrumentation Approach Comparison

### Auto-Instrumentation (monkey-patching)

Used by: OpenLLMetry, OpenInference, AgentOps, MLflow, Logfire, OpenLIT, **agent-observability**

```python
# One line instruments everything
Traceloop.init()                  # OpenLLMetry
openlit.init()                    # OpenLIT
mlflow.langchain.autolog()        # MLflow
logfire.instrument_openai()       # Logfire
agentops.init()                   # AgentOps
OpenAIInstrumentor().instrument() # OpenInference
auto_instrument()                 # agent-observability
```

**Pros:** Zero code changes to existing agent logic. Instant observability.

**Cons:** Monkey-patching is fragile across version upgrades. Opaque -- you can't see what gets captured. Can conflict with other instrumentors.

### Decorator-Based

Used by: OpenLLMetry, AgentOps, Opik, MLflow, Langfuse, DeepEval, W&B Weave

```python
@workflow(name="research")     # OpenLLMetry
@agentops.agent                # AgentOps
@opik.track                    # Opik
@mlflow.trace                  # MLflow
@observe()                     # Langfuse / DeepEval
@weave.op()                    # W&B Weave
def my_function():
    ...
```

**Pros:** Explicit control over what gets traced. Works with custom code.

**Cons:** Requires modifying function signatures. All-or-nothing per function. Limited control over child spans.

### Callback/Hook Implementation

Used by: agent-observability (LangChain, LangGraph, LlamaIndex adapters)

```python
handler = LangChainAdapter(observer, agent_id="my-agent")
result = chain.invoke(input, config={"callbacks": [handler]})
```

**Pros:** Uses the framework's native extension mechanism. No monkey-patching. No function modification.

**Cons:** Requires the framework to have a callback system.

### Context Managers (Explicit Wrapping)

Used by: agent-observability (Generic, Anthropic, CrewAI, AutoGen, Google ADK, Bedrock, Haystack, smolagents, PydanticAI, Phidata adapters)

```python
with adapter.run(task="Book flight") as run:
    with run.turn() as turn:
        response = client.messages.create(...)
        turn.record_llm_response(response)
        with turn.tool_call(block.name, block.input, block.id) as tc:
            result = execute_tool(...)
            tc.set_result(result)
```

**Pros:** Fully explicit -- every span is visible and debuggable. Works with any framework. No monkey-patching. Correct parent-child relationships via correlation IDs.

**Cons:** More lines of code than auto-instrumentation (average 4.6 lines added per integration).

---

## Where agent-observability Fits

`agent-observability` bridges convenience and control:

```
+----------------------------------------------------------------+
|                                                                |
|   Convenience <------------------------------------> Control   |
|                                                                |
|   Traceloop    MLflow    AgentOps     agent-obs                |
|   (auto only)  (auto)   (decorators) (auto + manual)          |
|                                                                |
|   "Just works" <-------------------------------> "I see all"  |
|                       ^                                        |
|               agent-obs sits here:                             |
|               auto_instrument() for quick start                |
|               manual adapters for full control                 |
|                                                                |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
|                                                                |
|   SDK only <------------------------------------> Full platform|
|                                                                |
|   OpenLLMetry   agent-obs   OpenInference   Langfuse   Opik   |
|   (pure SDK)    (pure SDK)  (SDK+Phoenix)   (platform) (plat) |
|                                                                |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
|                                                                |
|   No redaction <-------------------------------> Full redaction|
|                                                                |
|   OpenLLMetry   Langfuse    Datadog         agent-obs          |
|   (flag only)   (server)    (enterprise)    (SDK-level)        |
|                                                                |
+----------------------------------------------------------------+
```

Our position: **Pure SDK with both auto and manual instrumentation, built-in payload safety, zero platform dependency.**

Unlike other tools that force a choice between convenience (auto only) and control (manual only), `agent-observability` provides both:

```python
# Quick start: one line, zero code changes
auto_instrument()

# Production: selective frameworks, custom config
auto_instrument(
    frameworks=["langchain", "anthropic"],
    payload_policy=PayloadPolicy(redact_keys={"password", "ssn"}),
)

# Full control: manual adapters with explicit span boundaries
with adapter.run(task="...") as run:
    with run.step() as step:
        with step.tool_call("search", input={...}) as tc:
            ...
```

---

## What Exists vs. What We Do Differently

### Three things no existing tool does:

#### 1. Unified Event Protocol

Every existing tool has independent per-framework instrumentors. OpenLLMetry's LangChain instrumentor knows nothing about its CrewAI instrumentor. They share OTel as an output format, but there is no intermediate representation.

`agent-observability` has a single `AgentEvent` frozen dataclass that all 15 adapters map to:

```
LangChain callback  --+
OpenAI RunHooks     --+
Anthropic loop      --+--> AgentEvent --> AgentObserver --> OTel spans
CrewAI hooks        --+
AutoGen messages    --+
Google ADK events   --+
```

**Why this matters:** You can swap frameworks without changing your observability pipeline. You can write custom processing logic against `AgentEvent` that works for any framework. You can test adapter correctness by inspecting events before they become spans.

#### 2. Correlation-ID-Based Span Parenting

Every existing tool relies on OTel's implicit `Context.attach()` for span parenting. This is thread-local, which breaks in:

| Scenario | Thread-local context | Correlation IDs |
|---|---|---|
| Concurrent tool calls in one step | Wrong parent (last attached wins) | Correct (each tool has explicit `step_id` parent) |
| Async frameworks (`await` switches coroutines) | Context lost on switch | Correct (IDs are on the event, not thread-local) |
| Out-of-order events (LangGraph DAG) | Incorrect nesting | Correct (explicit lookup by ID) |
| Multi-agent handoffs | Ambiguous parentage | Correct (`run_id` stays constant) |

Our observer stores spans in four dictionaries keyed by correlation ID:

```python
_run_spans:  dict[str, _SpanHandle]   # run_id --> span
_step_spans: dict[str, _SpanHandle]   # step_id --> span
_tool_spans: dict[str, _SpanHandle]   # tool_call_id --> span
_llm_spans:  dict[str, _SpanHandle]   # llm_call_id --> span
```

Parent lookup is explicit: `_get_parent_context(self._step_spans, event.step_id)`.

#### 3. Built-in Payload Redaction

No existing instrumentation SDK has fine-grained payload redaction built in. The landscape:

| Tool | Redaction Approach |
|---|---|
| OpenLLMetry | Binary flag (`enable_content_tracing`) + `span_postprocess_callback` |
| Langfuse | Server-side masking (v3.152.0) -- data still leaves the process |
| Datadog | Enterprise config (`DD_LLMOBS_PROMPT_REDACTION_ENABLED`) |
| New Relic | Agent config toggle (`ai_monitoring.record_content.enabled: false`) |
| Azure AI | Environment variable toggle |
| OTel GenAI spec | Three-tier strategy but no implementation |
| **agent-observability** | **Fine-grained SDK-level redaction before data leaves the process** |

`agent-observability` has `PayloadPolicy` at the SDK level:

```python
observer = AgentObserver(
    payload_policy=PayloadPolicy(
        max_str_len=1024,
        redact_keys={"password", "api_key", "ssn"},
        redact_patterns=[re.compile(r"\b\d{3}-\d{2}-\d{4}\b")],  # SSN
        drop_keys={"internal_debug"},
    )
)
```

**Why this matters:** Sensitive data never leaves the process. It doesn't travel to the OTel Collector, doesn't get written to disk, and doesn't appear in any exporter. This is a stronger security posture than server-side or backend-side redaction.

### What existing tools do better:

| Capability | Best tool | Our status |
|---|---|---|
| Framework coverage | Opik (70+), MLflow (50+), OpenLIT (50+) | 15 frameworks (14 auto-instrumented) |
| Evaluation/testing | Opik, DeepEval, Langfuse, Phoenix | Pure instrumentation (by design) |
| Vector DB tracing | OpenLLMetry (7 DBs) | Not supported |
| Dashboards/visualization | Langfuse, Opik, Phoenix, OpenLIT | We rely on OTel backends |
| Cost tracking | Langfuse, AgentOps, MLflow (v3.10) | Not built-in |
| Session replays | AgentOps | Not supported |
| GenAI semantic conventions | OpenLLMetry (contributed upstream) | We use `agent.*` namespace |
| GPU monitoring | OpenLIT | Not supported |
| Community size | MLflow (24.1k), Langfuse (21.9k), Opik (17.7k) | New project |
| Enterprise APM integration | Datadog, New Relic, Dynatrace | Pure SDK, works with any OTel backend |

---

## Recommendations

### When to use agent-observability

- You need **both auto-instrumentation and manual control** over what gets instrumented
- You run agents with **concurrent tool calls** or **async execution** where thread-local context breaks
- You have **compliance requirements** that mandate payload redaction at the source (HIPAA, GDPR, SOC 2)
- You want a **pure SDK** with zero platform dependencies
- You want to **swap agent frameworks** without changing your observability setup
- You want to **test and debug** your instrumentation by inspecting the event protocol
- You need **fine-grained redaction** (key-based, pattern-based, truncation) not just content on/off

### When to use something else

| If you need... | Use... |
|---|---|
| The broadest framework coverage | MLflow, Opik, or OpenLIT |
| A full observability platform with dashboards | Langfuse, Opik, or OpenLIT |
| Agent session replays | AgentOps |
| LLM evaluation and testing | DeepEval, Opik, or Langfuse |
| PydanticAI-native tracing | Logfire |
| Vector database tracing | OpenLLMetry |
| GPU monitoring | OpenLIT |
| Enterprise APM integration | Datadog, New Relic, or Dynatrace |
| Cloud-native agent tracing | Azure AI Foundry, Vertex AI, or CloudWatch |
| MCP-specific instrumentation | OpenInference (has MCP instrumentor) |

### Potential convergence

The OTel GenAI Semantic Conventions (`gen_ai.*` namespace) are still in Development status but represent the standard the industry will converge on. A future version of `agent-observability` could:

1. **Adopt `gen_ai.*` attributes** alongside or instead of `agent.*` -- map `AgentEvent` fields to `gen_ai.agent.id`, `gen_ai.usage.input_tokens`, `gen_ai.tool.name`, etc.
2. **Adopt MCP conventions** -- emit `mcp.client.*` and `mcp.server.*` spans for MCP tool calls with `params._meta` context propagation
3. **Adopt multi-agent conventions** -- emit `agent_to_agent_interaction`, `agent_planning`, `agent_orchestration` spans as the Microsoft/Cisco proposals stabilize
4. **Keep correlation-ID-based span parenting** -- complementary to semantic conventions, not conflicting
5. **Keep `PayloadPolicy`** -- the OTel spec's three-tier content recording strategy has a similar model but no implementation. Our SDK-level redaction remains differentiated.

This would give `agent-observability` the unique combination of **standard semantic conventions** + **correlation-ID spans** + **built-in redaction** -- a position no existing tool occupies.
