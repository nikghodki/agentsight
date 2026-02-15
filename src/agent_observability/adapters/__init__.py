"""
Framework adapters: translate framework-specific hooks into AgentEvents.

Each adapter converts a framework's callback/hook system into the
canonical AgentEvent protocol, then forwards to an AgentObserver.

Available adapters (install optional deps for framework-specific ones):

  Generic (no extra deps):
    - GenericAgentAdapter       Context-manager API for custom frameworks

  LangChain ecosystem:
    - LangChainAdapter          langchain-core BaseCallbackHandler
    - LangGraphCallbackAdapter  LangGraph-aware callback handler
    - LangGraphEventAdapter     Manual hooks for LangGraph graphs

  OpenAI:
    - OpenAIRunHooksAdapter     OpenAI Agents SDK RunHooks
    - OpenAIAgentHooksAdapter   OpenAI Agents SDK per-Agent hooks

  Anthropic:
    - AgenticLoopAdapter        Context-manager for Claude agentic loops
    - AnthropicMessageHooksAdapter  Event-driven message hooks

  Microsoft:
    - AutoGenAdapter            AutoGen multi-agent conversations
    - SKAdapter                 Semantic Kernel filters (function, prompt, auto)

  Google:
    - GoogleADKAdapter          Google Agent Development Kit

  Amazon:
    - BedrockAgentsAdapter      Amazon Bedrock Agents runtime

  HuggingFace:
    - SmolagentsAdapter         smolagents (CodeAgent, ToolCallingAgent)

  LlamaIndex:
    - LlamaIndexAdapter         LlamaIndex CallbackManager handler

  PydanticAI:
    - PydanticAIAdapter         PydanticAI agent runs + message replay

  Phidata / Agno:
    - PhidataAdapter            Phidata agent monitoring

  CrewAI:
    - CrewAIAdapter             CrewAI crew/task/tool hooks

  deepset:
    - HaystackAdapter           Haystack 2.x pipeline components
"""
