"""
Amazon Bedrock Agents adapter: hooks into Bedrock's agent runtime events.

Targets: boto3 (bedrock-agent-runtime) and the Bedrock Agents response stream.

Bedrock Agents use an invoke_agent API that returns a streaming response
with trace events. The trace contains:
  - PreProcessingTrace: input parsing
  - OrchestrationTrace: reasoning + action selection
  - PostProcessingTrace: output formatting
  - FailureTrace: errors

Inside OrchestrationTrace:
  - ModelInvocationInput/Output: LLM calls
  - InvocationInput (ActionGroupInvocation, KnowledgeBaseLookup): tool calls
  - Observation: tool results
  - Rationale: agent reasoning

Mapping:
  invoke_agent()                  -> agent.lifecycle.start / end
  OrchestrationTrace iteration    -> agent.step.start / end
  ModelInvocationInput/Output     -> agent.llm.call.start / end
  ActionGroupInvocation           -> agent.tool.call.start / end
  KnowledgeBaseLookup             -> agent.tool.call.start / end (retrieval)
  Rationale                       -> span event

Usage:
    import boto3
    from agentsight import AgentObserver, init_telemetry
    from agentsight.adapters.bedrock_agents import BedrockAgentsAdapter

    init_telemetry()
    observer = AgentObserver()
    adapter = BedrockAgentsAdapter(observer, agent_id="bedrock-agent")

    client = boto3.client("bedrock-agent-runtime")
    response = client.invoke_agent(
        agentId="AGENT_ID",
        agentAliasId="ALIAS_ID",
        sessionId="SESSION_ID",
        inputText="What is AI safety?",
    )

    with adapter.invocation(task="What is AI safety?") as inv:
        for event in response["completion"]:
            inv.process_event(event)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from agentsight.events import (
    AgentEvent,
    EventName,
    new_llm_call_id,
    new_run_id,
    new_step_id,
    new_tool_call_id,
)
from agentsight.observer import AgentObserver

logger = logging.getLogger(__name__)


class _InvocationContext:
    """Context for a single Bedrock agent invocation."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str,
        run_id: str,
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id
        self._run_id = run_id
        self._current_step_id: Optional[str] = None
        self._current_llm_id: Optional[str] = None
        self._current_tool_id: Optional[str] = None
        self._step_count: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def process_event(self, event: dict[str, Any]) -> None:
        """
        Process a single event from the Bedrock agent response stream.

        Each event dict may contain one of:
          - "trace": trace information (orchestration, preprocessing, etc.)
          - "chunk": response text chunk
        """
        trace = event.get("trace", {}).get("trace", {})

        if "orchestrationTrace" in trace:
            self._process_orchestration(trace["orchestrationTrace"])
        elif "preProcessingTrace" in trace:
            self._process_preprocessing(trace["preProcessingTrace"])
        elif "postProcessingTrace" in trace:
            self._process_postprocessing(trace["postProcessingTrace"])
        elif "failureTrace" in trace:
            self._process_failure(trace["failureTrace"])

    def _process_orchestration(self, orch: dict[str, Any]) -> None:
        """Process an orchestration trace event."""
        # Model invocation input -> LLM call start
        if "modelInvocationInput" in orch:
            self._start_orchestration_step()
            mii = orch["modelInvocationInput"]
            llm_id = new_llm_call_id()
            self._current_llm_id = llm_id

            model_type = mii.get("type", "ORCHESTRATION")
            self._observer.emit(
                AgentEvent(
                    name=EventName.LLM_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._current_step_id,
                    llm_call_id=llm_id,
                    model_name=f"bedrock-{model_type.lower()}",
                    attributes={
                        "framework": "bedrock-agents",
                        "invocation_type": model_type,
                    },
                )
            )

        # Model invocation output -> LLM call end
        if "modelInvocationOutput" in orch:
            mio = orch["modelInvocationOutput"]
            if self._current_llm_id:
                metadata = mio.get("metadata", {}).get("usage", {})
                attrs: dict[str, Any] = {}
                if metadata:
                    attrs["input_tokens"] = metadata.get("inputTokens", 0)
                    attrs["output_tokens"] = metadata.get("outputTokens", 0)

                self._observer.emit(
                    AgentEvent(
                        name=EventName.LLM_CALL_END,
                        agent_id=self._agent_id,
                        run_id=self._run_id,
                        step_id=self._current_step_id,
                        llm_call_id=self._current_llm_id,
                        ok=True,
                        attributes=attrs,
                    )
                )
                self._current_llm_id = None

        # Rationale -> span event
        if "rationale" in orch:
            rationale = orch["rationale"]
            text = rationale.get("text", "")
            self._observer.emit(
                AgentEvent(
                    name=EventName.MEMORY_WRITE,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._current_step_id,
                    attributes={"rationale": text[:2000]},
                )
            )

        # Invocation input -> tool call start
        if "invocationInput" in orch:
            inv = orch["invocationInput"]
            self._process_invocation_input(inv)

        # Observation -> tool call end
        if "observation" in orch:
            obs = orch["observation"]
            self._process_observation(obs)
            # End step after observation
            self._end_current_step()

    def _process_invocation_input(self, inv: dict[str, Any]) -> None:
        """Process a tool invocation input."""
        tc_id = new_tool_call_id()
        self._current_tool_id = tc_id

        if "actionGroupInvocationInput" in inv:
            ag = inv["actionGroupInvocationInput"]
            tool_name = ag.get("function", ag.get("apiPath", "action_group"))
            group_name = ag.get("actionGroupName", "")
            params = ag.get("parameters", ag.get("requestBody", {}))

            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._current_step_id,
                    tool_call_id=tc_id,
                    tool_name=str(tool_name),
                    attributes={
                        "framework": "bedrock-agents",
                        "action_group": group_name,
                        "input": str(params)[:4096],
                    },
                )
            )

        elif "knowledgeBaseLookupInput" in inv:
            kb = inv["knowledgeBaseLookupInput"]
            query = kb.get("text", "")
            kb_id = kb.get("knowledgeBaseId", "")

            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._current_step_id,
                    tool_call_id=tc_id,
                    tool_name="knowledge_base_lookup",
                    attributes={
                        "framework": "bedrock-agents",
                        "knowledge_base_id": kb_id,
                        "query": query[:2000],
                    },
                )
            )

        elif "codeInterpreterInvocationInput" in inv:
            ci = inv["codeInterpreterInvocationInput"]
            code = ci.get("code", "")

            self._observer.emit(
                AgentEvent(
                    name=EventName.TOOL_CALL_START,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    step_id=self._current_step_id,
                    tool_call_id=tc_id,
                    tool_name="code_interpreter",
                    attributes={
                        "framework": "bedrock-agents",
                        "code": code[:4096],
                    },
                )
            )

    def _process_observation(self, obs: dict[str, Any]) -> None:
        """Process an observation (tool result)."""
        if not self._current_tool_id:
            return

        tc_id = self._current_tool_id
        self._current_tool_id = None
        tool_name = "unknown"
        output = ""

        if "actionGroupInvocationOutput" in obs:
            ag_out = obs["actionGroupInvocationOutput"]
            output = ag_out.get("text", str(ag_out))
            tool_name = "action_group"

        elif "knowledgeBaseLookupOutput" in obs:
            kb_out = obs["knowledgeBaseLookupOutput"]
            refs = kb_out.get("retrievedReferences", [])
            output = f"{len(refs)} references retrieved"
            tool_name = "knowledge_base_lookup"

        elif "codeInterpreterInvocationOutput" in obs:
            ci_out = obs["codeInterpreterInvocationOutput"]
            execution_output = ci_out.get("executionOutput", "")
            output = str(execution_output)[:2000]
            tool_name = "code_interpreter"

        elif "finalResponse" in obs:
            # Final response is not a tool result
            return

        self._observer.emit(
            AgentEvent(
                name=EventName.TOOL_CALL_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                tool_call_id=tc_id,
                tool_name=tool_name,
                ok=True,
                attributes={"output": output[:4096]},
            )
        )

    def _process_preprocessing(self, pre: dict[str, Any]) -> None:
        """Process preprocessing trace."""
        model_output = pre.get("modelInvocationOutput", {})
        parsed = model_output.get("parsedResponse", {})
        is_valid = parsed.get("isValid", True)

        if not is_valid:
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    error_type="PreprocessingValidationError",
                    error_message="Input validation failed",
                )
            )

    def _process_postprocessing(self, post: dict[str, Any]) -> None:
        """Process postprocessing trace."""
        # Post-processing is informational
        model_output = post.get("modelInvocationOutput", {})
        parsed = model_output.get("parsedResponse", {})
        if parsed:
            self._observer.emit(
                AgentEvent(
                    name=EventName.MEMORY_WRITE,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    attributes={"postprocessing": str(parsed)[:1000]},
                )
            )

    def _process_failure(self, failure: dict[str, Any]) -> None:
        """Process a failure trace."""
        reason = failure.get("failureReason", "Unknown failure")
        trace_id = failure.get("traceId", "")

        self._observer.emit(
            AgentEvent(
                name=EventName.ERROR,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                error_type="BedrockAgentFailure",
                error_message=reason,
                attributes={"bedrock.trace_id": trace_id},
            )
        )

    def _start_orchestration_step(self) -> None:
        """Start a new orchestration step if not already in one."""
        if self._current_step_id:
            return
        self._step_count += 1
        self._current_step_id = new_step_id()
        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_START,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                attributes={
                    "framework": "bedrock-agents",
                    "step_number": self._step_count,
                },
            )
        )

    def _end_current_step(self) -> None:
        """End the current orchestration step."""
        if not self._current_step_id:
            return
        self._observer.emit(
            AgentEvent(
                name=EventName.STEP_END,
                agent_id=self._agent_id,
                run_id=self._run_id,
                step_id=self._current_step_id,
                ok=True,
            )
        )
        self._current_step_id = None


class BedrockAgentsAdapter:
    """Amazon Bedrock Agents observability adapter."""

    def __init__(
        self,
        observer: AgentObserver,
        agent_id: str = "bedrock-agent",
    ) -> None:
        self._observer = observer
        self._agent_id = agent_id

    @contextmanager
    def invocation(
        self,
        task: str = "",
        agent_alias_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **attrs: Any,
    ) -> Generator[_InvocationContext, None, None]:
        """Wrap a single invoke_agent call."""
        run_id = new_run_id()

        start_attrs: dict[str, Any] = {"framework": "bedrock-agents"}
        if task:
            start_attrs["task"] = task
        if agent_alias_id:
            start_attrs["agent_alias_id"] = agent_alias_id
        if session_id:
            start_attrs["session_id"] = session_id
        start_attrs.update(attrs)

        self._observer.emit(
            AgentEvent(
                name=EventName.LIFECYCLE_START,
                agent_id=self._agent_id,
                run_id=run_id,
                attributes=start_attrs,
            )
        )

        ctx = _InvocationContext(self._observer, self._agent_id, run_id)
        ok = True
        error: Optional[BaseException] = None
        try:
            yield ctx
        except BaseException as e:
            ok = False
            error = e
            self._observer.emit(
                AgentEvent(
                    name=EventName.ERROR,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            raise
        finally:
            # Clean up open step
            if ctx._current_step_id:
                ctx._end_current_step()
            self._observer.emit(
                AgentEvent(
                    name=EventName.LIFECYCLE_END,
                    agent_id=self._agent_id,
                    run_id=run_id,
                    ok=ok,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error) if error else None,
                    attributes={"step_count": ctx._step_count},
                )
            )
