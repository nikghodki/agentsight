# Integration Guide

Adding `agent-observability` to existing agent code is a minimal change. Your core agent logic stays untouched -- you wrap it with a few context managers or pass an adapter as a callback.

This guide shows **before** and **after** for every supported framework. Lines marked `# <-- NEW` are the only additions.

## How Invasive Is It?

| Integration Mode | Lines Added | Logic Changes | Framework Examples |
|---|---|---|---|
| **Callback handler** | 3-5 | Zero. Pass adapter as a callback parameter. | LangChain, LangGraph, LlamaIndex |
| **Context managers** | 5-10 | Zero. `with` blocks wrap existing code. | Generic, Anthropic, CrewAI, AutoGen, Google ADK, Bedrock, Haystack, smolagents, PydanticAI, Phidata |
| **Async hooks** | 3-5 | Zero. Assign adapter to hooks parameter. | OpenAI Agents SDK |
| **Filter registration** | 4-6 | Zero. Register filters with kernel. | Semantic Kernel |

In all cases: **zero changes to your agent's core logic**. The observability layer is purely additive.

---

## Common Setup (All Frameworks)

Every integration starts with the same 3 lines:

```python
from agent_observability import AgentObserver, init_telemetry, shutdown_telemetry

tp, mp = init_telemetry(service_name="my-agent")
observer = AgentObserver()
```

And ends with:

```python
shutdown_telemetry(tp, mp)
```

The examples below omit these for brevity, but they are always required.

---

## 1. Anthropic Claude (Agentic Loop)

### Before

```python
from anthropic import Anthropic

client = Anthropic()
tools = [{"name": "web_search", "description": "Search the web", "input_schema": {...}}]

def execute_tool(name, input):
    if name == "web_search":
        return f"Results for: {input['query']}"
    return "Unknown tool"

def run_agent(task):
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})
```

### After

```python
from anthropic import Anthropic
from agent_observability.adapters.anthropic_agents import AgenticLoopAdapter     # <-- NEW

client = Anthropic()
adapter = AgenticLoopAdapter(observer, agent_id="claude-agent")                  # <-- NEW
tools = [{"name": "web_search", "description": "Search the web", "input_schema": {...}}]

def execute_tool(name, input):
    if name == "web_search":
        return f"Results for: {input['query']}"
    return "Unknown tool"

def run_agent(task):
    messages = [{"role": "user", "content": task}]

    with adapter.run(task=task) as run:                                          # <-- NEW
        while True:
            with run.turn() as turn:                                             # <-- NEW
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    tools=tools,
                    messages=messages,
                )
                turn.record_llm_response(response)                              # <-- NEW
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    return "".join(b.text for b in response.content if b.type == "text")

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        with turn.tool_call(block.name, block.input, block.id) as tc:  # <-- NEW
                            result = execute_tool(block.name, block.input)
                            tc.set_result(result)                               # <-- NEW
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
```

**Lines added: 7** | **Lines of logic changed: 0**

---

## 2. OpenAI Agents SDK

### Before

```python
from agents import Agent, Runner

search_agent = Agent(
    name="search-assistant",
    instructions="You help users search for information.",
    tools=[web_search_tool],
)

async def main():
    result = await Runner.run(search_agent, "Find recent AI papers")
    print(result.final_output)
```

### After

```python
from agents import Agent, Runner
from agent_observability.adapters.openai_agents import OpenAIRunHooksAdapter     # <-- NEW

hooks = OpenAIRunHooksAdapter(observer)                                          # <-- NEW

search_agent = Agent(
    name="search-assistant",
    instructions="You help users search for information.",
    tools=[web_search_tool],
)

async def main():
    hooks.start_run()                                                            # <-- NEW
    try:                                                                         # <-- NEW
        result = await Runner.run(search_agent, "Find recent AI papers", run_hooks=hooks)  # <-- CHANGED: added run_hooks
        print(result.final_output)
        hooks.end_run(ok=True)                                                   # <-- NEW
    except Exception as e:                                                       # <-- NEW
        hooks.end_run(ok=False, error=str(e))                                    # <-- NEW
        raise                                                                    # <-- NEW
```

**Lines added: 7** | **Lines of logic changed: 1** (added `run_hooks=hooks` parameter)

---

## 3. LangChain

### Before

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([("system", "You are helpful."), ("human", "{input}")])
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "What is the weather in SF?"})
```

### After

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from agent_observability.adapters.langchain import LangChainAdapter              # <-- NEW

handler = LangChainAdapter(observer, agent_id="langchain-agent")                 # <-- NEW

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([("system", "You are helpful."), ("human", "{input}")])
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "What is the weather in SF?"}, config={"callbacks": [handler]})  # <-- CHANGED: added config
```

**Lines added: 2** | **Lines of logic changed: 1** (added `config={"callbacks": [handler]}`)

---

## 4. LangGraph

### Before

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def agent_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = StateGraph(dict)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", should_continue)
graph.set_entry_point("agent")
app = graph.compile()

result = app.invoke({"messages": [("user", "What is AI?")]})
```

### After

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from agent_observability.adapters.langgraph import LangGraphCallbackAdapter      # <-- NEW

handler = LangGraphCallbackAdapter(observer, agent_id="my-graph")                # <-- NEW

llm = ChatOpenAI(model="gpt-4")

def agent_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = StateGraph(dict)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", should_continue)
graph.set_entry_point("agent")
app = graph.compile()

result = app.invoke({"messages": [("user", "What is AI?")]}, config={"callbacks": [handler]})  # <-- CHANGED: added config
```

**Lines added: 2** | **Lines of logic changed: 1** (added `config={"callbacks": [handler]}`)

---

## 5. CrewAI

### Before

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find papers", backstory="Expert researcher")
writer = Agent(role="Writer", goal="Summarize findings", backstory="Technical writer")

research_task = Task(description="Find AI safety papers", agent=researcher)
write_task = Task(description="Write summary", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

### After

```python
from crewai import Agent, Task, Crew
from agent_observability.adapters.crewai import CrewAIAdapter                    # <-- NEW

adapter = CrewAIAdapter(observer)                                                # <-- NEW

researcher = Agent(role="Researcher", goal="Find papers", backstory="Expert researcher")
writer = Agent(role="Writer", goal="Summarize findings", backstory="Technical writer")

research_task = Task(description="Find AI safety papers", agent=researcher)
write_task = Task(description="Write summary", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])

with adapter.observe_crew(crew_name="research-crew"):                            # <-- NEW
    result = crew.kickoff()
```

**Lines added: 3** | **Lines of logic changed: 0** (kickoff moved inside `with` block)

---

## 6. AutoGen

### Before

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

researcher = ConversableAgent("researcher", llm_config=llm_config)
critic = ConversableAgent("critic", llm_config=llm_config)

group_chat = GroupChat(agents=[researcher, critic], messages=[], max_round=5)
manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

researcher.initiate_chat(manager, message="Find AI safety papers")
```

### After

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager
from agent_observability.adapters.autogen import AutoGenAdapter                  # <-- NEW

adapter = AutoGenAdapter(observer)                                               # <-- NEW

researcher = ConversableAgent("researcher", llm_config=llm_config)
critic = ConversableAgent("critic", llm_config=llm_config)

group_chat = GroupChat(agents=[researcher, critic], messages=[], max_round=5)
manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

with adapter.group_chat("research-team", task="Find AI safety papers") as chat:  # <-- NEW
    researcher.initiate_chat(manager, message="Find AI safety papers")
    # Record messages as they happen:
    for msg in group_chat.messages:                                              # <-- NEW
        chat.on_agent_message(msg["name"], "group", msg["content"])              # <-- NEW
```

**Lines added: 5** | **Lines of logic changed: 0**

Note: For deeper integration, AutoGen v0.4+ supports event hooks where `on_agent_message`, `on_tool_call`, and `on_llm_call` can be called from `register_reply` hooks.

---

## 7. Google ADK

### Before

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

agent = Agent(name="travel-agent", model="gemini-2.0-flash", tools=[search_flights])
runner = Runner(agent=agent, app_name="travel", session_service=InMemorySessionService())

session = await runner.session_service.create_session(app_name="travel", user_id="u1")

async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=message):
    if event.is_final_response:
        print(event.content.parts[0].text)
```

### After

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from agent_observability.adapters.google_adk import GoogleADKAdapter             # <-- NEW

adapter = GoogleADKAdapter(observer)                                             # <-- NEW

agent = Agent(name="travel-agent", model="gemini-2.0-flash", tools=[search_flights])
runner = Runner(agent=agent, app_name="travel", session_service=InMemorySessionService())

session = await runner.session_service.create_session(app_name="travel", user_id="u1")

with adapter.run(agent_name="travel-agent", task="Book flight") as run:          # <-- NEW
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=message):
        run.on_event(event)                                                      # <-- NEW
        if event.is_final_response:
            print(event.content.parts[0].text)
```

**Lines added: 4** | **Lines of logic changed: 0**

---

## 8. Amazon Bedrock Agents

### Before

```python
import boto3

client = boto3.client("bedrock-agent-runtime")

response = client.invoke_agent(
    agentId="AGENT_ID",
    agentAliasId="ALIAS_ID",
    sessionId="session-123",
    inputText="What is AI safety?",
    enableTrace=True,
)

for event in response["completion"]:
    if "chunk" in event:
        print(event["chunk"]["bytes"].decode(), end="")
```

### After

```python
import boto3
from agent_observability.adapters.bedrock_agents import BedrockAgentsAdapter     # <-- NEW

adapter = BedrockAgentsAdapter(observer, agent_id="my-bedrock-agent")            # <-- NEW

client = boto3.client("bedrock-agent-runtime")

response = client.invoke_agent(
    agentId="AGENT_ID",
    agentAliasId="ALIAS_ID",
    sessionId="session-123",
    inputText="What is AI safety?",
    enableTrace=True,
)

with adapter.invocation(task="What is AI safety?") as inv:                       # <-- NEW
    for event in response["completion"]:
        inv.process_event(event)                                                 # <-- NEW
        if "chunk" in event:
            print(event["chunk"]["bytes"].decode(), end="")
```

**Lines added: 4** | **Lines of logic changed: 0**

---

## 9. LlamaIndex

### Before

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is AI safety?")
print(response)
```

### After

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.callbacks import CallbackManager                           # <-- NEW
from agent_observability.adapters.llamaindex import LlamaIndexAdapter            # <-- NEW

handler = LlamaIndexAdapter(observer, agent_id="rag-agent")                      # <-- NEW
callback_manager = CallbackManager([handler])                                    # <-- NEW

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, callback_manager=callback_manager)      # <-- CHANGED: added callback_manager
query_engine = index.as_query_engine(callback_manager=callback_manager)          # <-- CHANGED: added callback_manager

response = query_engine.query("What is AI safety?")
print(response)
```

**Lines added: 4** | **Lines of logic changed: 2** (added `callback_manager` parameter)

---

## 10. Microsoft Semantic Kernel

### Before

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4"))
kernel.add_plugin(MathPlugin(), "math")

result = await kernel.invoke(
    kernel.get_function("math", "Add"),
    a=5, b=3,
)
print(result)
```

### After

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from agent_observability.adapters.semantic_kernel import SKAdapter               # <-- NEW

adapter = SKAdapter(observer, agent_id="sk-agent")                               # <-- NEW

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4"))
kernel.add_plugin(MathPlugin(), "math")

kernel.add_filter("function_invocation", adapter.function_filter)                # <-- NEW
kernel.add_filter("prompt_rendering", adapter.prompt_filter)                     # <-- NEW
kernel.add_filter("auto_function_invocation", adapter.auto_function_filter)      # <-- NEW

with adapter.run(task="Calculate sum") as run:                                   # <-- NEW
    result = await kernel.invoke(
        kernel.get_function("math", "Add"),
        a=5, b=3,
    )
print(result)
```

**Lines added: 6** | **Lines of logic changed: 0**

---

## 11. Haystack

### Before

```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipeline.add_component("generator", OpenAIGenerator(model="gpt-4"))
pipeline.connect("retriever.documents", "generator.documents")

result = pipeline.run({"retriever": {"query": "What is AI?"}})
print(result["generator"]["replies"][0])
```

### After

```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from agent_observability.adapters.haystack import HaystackAdapter                # <-- NEW

adapter = HaystackAdapter(observer, agent_id="rag-pipeline")                     # <-- NEW

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipeline.add_component("generator", OpenAIGenerator(model="gpt-4"))
pipeline.connect("retriever.documents", "generator.documents")

with adapter.pipeline_run(pipeline_name="qa") as run:                            # <-- NEW
    result = pipeline.run({"retriever": {"query": "What is AI?"}})
print(result["generator"]["replies"][0])
```

**Lines added: 3** | **Lines of logic changed: 0**

For deeper component-level tracing, add explicit hooks:

```python
with adapter.pipeline_run(pipeline_name="qa") as run:
    adapter.on_component_start("retriever", "BM25Retriever", {"query": "What is AI?"})
    result = pipeline.run({"retriever": {"query": "What is AI?"}})
    adapter.on_component_end("retriever", {"documents": result["retriever"]["documents"]})
    adapter.on_component_end("generator", result["generator"])
```

---

## 12. HuggingFace smolagents

### Before

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

model = HfApiModel("Qwen/Qwen2.5-72B-Instruct")
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

result = agent.run("What is the latest news about AI?")
print(result)
```

### After (monitor callback)

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from agent_observability.adapters.smolagents import SmolagentsAdapter            # <-- NEW

adapter = SmolagentsAdapter(observer, agent_id="code-agent")                     # <-- NEW
monitor = adapter.create_monitor()                                               # <-- NEW

model = HfApiModel("Qwen/Qwen2.5-72B-Instruct")
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, step_callbacks=[monitor])  # <-- CHANGED: added step_callbacks

result = agent.run("What is the latest news about AI?")
print(result)
```

**Lines added: 3** | **Lines of logic changed: 1** (added `step_callbacks=[monitor]`)

### After (context manager, for more control)

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from agent_observability.adapters.smolagents import SmolagentsAdapter            # <-- NEW

adapter = SmolagentsAdapter(observer, agent_id="code-agent")                     # <-- NEW

model = HfApiModel("Qwen/Qwen2.5-72B-Instruct")
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

with adapter.run(task="Latest AI news") as run:                                  # <-- NEW
    result = agent.run("What is the latest news about AI?")
print(result)
```

---

## 13. PydanticAI

### Before

```python
from pydantic_ai import Agent

agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant.")

result = agent.run_sync("What is AI safety?")
print(result.data)
```

### After

```python
from pydantic_ai import Agent
from agent_observability.adapters.pydantic_ai import PydanticAIAdapter           # <-- NEW

adapter = PydanticAIAdapter(observer, agent_id="pydantic-agent")                 # <-- NEW

agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant.")

with adapter.run(task="What is AI safety?", model="openai:gpt-4") as run:       # <-- NEW
    result = agent.run_sync("What is AI safety?")
print(result.data)
```

**Lines added: 3** | **Lines of logic changed: 0**

For deeper tracing with tool calls and retries, use the explicit methods:

```python
with adapter.run(task="Analyze data", model="openai:gpt-4") as run:
    # Before each LLM call
    llm_id = run.on_model_request(model="gpt-4", messages=[...])
    result = agent.run_sync("Analyze data")
    run.on_model_response(llm_id, tool_calls=["search"])

    # For each tool call
    tc_id = run.on_tool_call("search", {"query": "data"})
    run.on_tool_result(tc_id, "search", "results")
```

---

## 14. Phidata / Agno

### Before

```python
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
)

agent.print_response("What are the latest AI trends?")
```

### After

```python
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from agent_observability.adapters.phidata import PhidataAdapter                  # <-- NEW

adapter = PhidataAdapter(observer, agent_id="research-agent")                    # <-- NEW

agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
)

with adapter.run(task="What are the latest AI trends?") as run:                  # <-- NEW
    agent.print_response("What are the latest AI trends?")
```

**Lines added: 3** | **Lines of logic changed: 0**

For richer telemetry, add explicit hooks in your agent's callback:

```python
with adapter.run(task="Research AI trends") as run:
    llm_id = run.on_model_call(model="gpt-4", messages=["..."])
    response = agent.run("Research AI trends")
    run.on_model_response(llm_id, tokens={"prompt_tokens": 100})

    tc_id = run.on_tool_call("duckduckgo", {"query": "AI trends"})
    run.on_tool_result(tc_id, "duckduckgo", "Results...")

    run.on_knowledge_search("AI trends", results=[...])
    run.on_memory_write("chat_history", message_count=5)
```

---

## 15. Generic / Custom Agent

### Before

```python
def run_my_agent(task):
    # Step 1: Search
    results = web_search(task)

    # Step 2: Analyze with LLM
    response = call_llm(model="gpt-4", prompt=f"Analyze: {results}")

    # Step 3: Take action
    action_result = execute_action(response.action)

    return response.text
```

### After

```python
from agent_observability.adapters.generic import GenericAgentAdapter              # <-- NEW

agent = GenericAgentAdapter(observer, agent_id="my-agent")                        # <-- NEW

def run_my_agent(task):
    with agent.run(task=task) as run:                                             # <-- NEW
        # Step 1: Search
        with run.step(reason="Search for information") as step:                   # <-- NEW
            with step.tool_call("web_search", input={"query": task}) as tc:       # <-- NEW
                results = web_search(task)
                tc.set_output(results)                                            # <-- NEW

        # Step 2: Analyze with LLM
        with run.step(reason="Analyze results") as step:                          # <-- NEW
            with step.llm_call(model="gpt-4") as llm:                            # <-- NEW
                response = call_llm(model="gpt-4", prompt=f"Analyze: {results}")
                llm.set_output(response.text, tokens={"prompt": 150})            # <-- NEW

        # Step 3: Take action
        with run.step(reason="Execute action") as step:                           # <-- NEW
            with step.tool_call("execute_action", input={"action": response.action}) as tc:  # <-- NEW
                action_result = execute_action(response.action)
                tc.set_output(action_result)                                      # <-- NEW

        return response.text
```

**Lines added: 13** | **Lines of logic changed: 0**

The generic adapter requires the most lines because it instruments every operation explicitly. Framework-specific adapters do most of this automatically.

---

## Summary

| # | Framework | Lines Added | Logic Changed | Integration Method |
|---|---|---|---|---|
| 1 | **Anthropic Claude** | 7 | 0 | `with adapter.run()` > `with run.turn()` > `with turn.tool_call()` |
| 2 | **OpenAI Agents SDK** | 7 | 1 | `run_hooks=hooks` parameter |
| 3 | **LangChain** | 2 | 1 | `config={"callbacks": [handler]}` |
| 4 | **LangGraph** | 2 | 1 | `config={"callbacks": [handler]}` |
| 5 | **CrewAI** | 3 | 0 | `with adapter.observe_crew()` |
| 6 | **AutoGen** | 5 | 0 | `with adapter.group_chat()` |
| 7 | **Google ADK** | 4 | 0 | `with adapter.run()` + `run.on_event(event)` |
| 8 | **Bedrock Agents** | 4 | 0 | `with adapter.invocation()` + `inv.process_event(event)` |
| 9 | **LlamaIndex** | 4 | 2 | `CallbackManager([handler])` |
| 10 | **Semantic Kernel** | 6 | 0 | `kernel.add_filter(...)` |
| 11 | **Haystack** | 3 | 0 | `with adapter.pipeline_run()` |
| 12 | **smolagents** | 3 | 1 | `step_callbacks=[monitor]` |
| 13 | **PydanticAI** | 3 | 0 | `with adapter.run()` |
| 14 | **Phidata** | 3 | 0 | `with adapter.run()` |
| 15 | **Generic** | 13 | 0 | `with agent.run()` > `with run.step()` > `with step.tool_call()` |

**Average: 4.6 lines added, 0.4 lines of existing logic changed.**

## What All Integrations Produce

Regardless of framework, every integration produces the same span tree:

```
agent.run
  +-- agent.step
  |     +-- agent.tool
  |     +-- agent.llm
  +-- agent.step
        +-- agent.llm
```

Plus counters (`agent.runs.total`, `agent.tool_calls.total`, etc.) and duration histograms -- all automatically exported to your OTel backend.
