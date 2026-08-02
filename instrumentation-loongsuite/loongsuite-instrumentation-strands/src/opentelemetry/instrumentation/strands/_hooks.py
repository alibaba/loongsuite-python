# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

from opentelemetry import context as otel_context
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

_CURRENT_INVOCATION_KEY: ContextVar[int | None] = ContextVar(
    "loongsuite_strands_invocation_key", default=None
)


@dataclass
class _InvocationState:
    agent: InvokeAgentInvocation
    agent_error: BaseException | None = None
    cycle_id: Any = None
    round: int = 0
    step: ReactStepInvocation | None = None
    step_error: Exception | None = None
    llm: LLMInvocation | None = None
    tool_invocations: dict[str, ExecuteToolInvocation] = field(
        default_factory=dict
    )


class LoongsuiteHook:
    """Create LoongSuite GenAI spans from the public Strands hook API."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._states: dict[int, _InvocationState] = {}
        self._lock = threading.RLock()

    def register_hooks(self, registry: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before_invocation)
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)

    @hook_advice("strands", "before_invocation")
    def _before_invocation(self, event: BeforeInvocationEvent) -> None:
        agent = event.agent
        invocation = InvokeAgentInvocation(
            provider=_provider_name(agent),
            agent_name=getattr(agent, "name", None) or "Strands Agents",
            agent_id=getattr(agent, "agent_id", None),
            agent_description=getattr(agent, "description", None),
            request_model=_model_name(agent),
            input_messages=_convert_messages(event.messages or []),
            system_instruction=_system_instruction(agent),
            tool_definitions=_tool_definitions(agent),
            conversation_id=_conversation_id(event.invocation_state),
        )
        self._handler.start_invoke_agent(invocation)
        state_key = id(event.invocation_state)
        with self._lock:
            self._states[state_key] = _InvocationState(agent=invocation)
        _CURRENT_INVOCATION_KEY.set(state_key)

    @hook_advice("strands", "before_model")
    def _before_model(self, event: BeforeModelCallEvent) -> None:
        state = self._state(event.invocation_state)
        if state is None:
            return
        cycle_id = event.invocation_state.get("event_loop_cycle_id")
        if state.step is None or cycle_id != state.cycle_id:
            self._finish_step(state)
            state.round += 1
            state.cycle_id = cycle_id
            state.step = ReactStepInvocation(
                round=state.round,
            )
            state.step_error = None
            self._handler.start_react_step(
                state.step, context=set_span_in_context(state.agent.span)
            )

        if not _create_llm_span(event.agent):
            return
        invocation = LLMInvocation(
            request_model=_model_name(event.agent),
            provider=_provider_name(event.agent),
            operation_name="chat",
            input_messages=_convert_messages(
                event.invocation_state.get("messages", [])
            ),
            system_instruction=_system_instruction(event.agent),
            tool_definitions=_tool_definitions(event.agent),
            conversation_id=_conversation_id(event.invocation_state),
        )
        self._handler.start_llm(
            invocation, context=set_span_in_context(state.step.span)
        )
        state.llm = invocation

    @hook_advice("strands", "after_model")
    def _after_model(self, event: AfterModelCallEvent) -> None:
        state = self._state(event.invocation_state)
        if state is None or state.llm is None:
            return
        invocation, state.llm = state.llm, None
        response = event.stop_response
        if response is not None:
            reason = _finish_reason(response.stop_reason)
            invocation.finish_reasons = [reason]
            invocation.output_messages = _convert_output_message(
                response.message, reason
            )
            _apply_usage(invocation, response.message.get("metadata", {}))
            if state.step is not None:
                state.step.finish_reason = reason
        if event.exception is not None:
            state.agent_error = event.exception
            if state.step is not None:
                state.step.finish_reason = "error"
                state.step_error = event.exception
            self._finalize(
                self._handler.fail_llm,
                invocation,
                _error(event.exception),
            )
        else:
            self._finalize(self._handler.stop_llm, invocation)

    @hook_advice("strands", "before_tool")
    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        state = self._state(event.invocation_state)
        if state is None:
            return
        tool_use = event.tool_use
        tool_id = str(tool_use.get("toolUseId", ""))
        invocation = ExecuteToolInvocation(
            tool_name=tool_use.get("name", "unknown"),
            tool_call_id=tool_id or None,
            tool_call_arguments=tool_use.get("input"),
            tool_description=_tool_description(event.selected_tool),
            tool_type="function",
            provider="strands",
        )
        parent = (
            state.step.span if state.step is not None else state.agent.span
        )
        self._handler.start_execute_tool(
            invocation, context=set_span_in_context(parent)
        )
        with self._lock:
            state.tool_invocations[tool_id] = invocation

    @hook_advice("strands", "after_tool")
    def _after_tool(self, event: AfterToolCallEvent) -> None:
        state = self._state(event.invocation_state)
        if state is None:
            return
        tool_id = str(event.tool_use.get("toolUseId", ""))
        with self._lock:
            invocation = state.tool_invocations.pop(tool_id, None)
        if invocation is None:
            return
        invocation.tool_call_result = event.result
        if event.exception is not None:
            self._finalize(
                self._handler.fail_execute_tool,
                invocation,
                _error(event.exception),
            )
        else:
            self._finalize(self._handler.stop_execute_tool, invocation)

    @hook_advice("strands", "after_invocation")
    def _after_invocation(self, event: AfterInvocationEvent) -> None:
        state_key = id(event.invocation_state)
        with self._lock:
            state = self._states.pop(state_key, None)
        if _CURRENT_INVOCATION_KEY.get() == state_key:
            _CURRENT_INVOCATION_KEY.set(None)
        if state is None:
            return
        if state.llm is not None:
            unfinished_error = RuntimeError("model call did not finish")
            self._finalize(
                self._handler.fail_llm,
                state.llm,
                _error(unfinished_error),
            )
            state.llm = None
            state.step_error = unfinished_error
            state.agent_error = unfinished_error
        for invocation in list(state.tool_invocations.values()):
            self._finalize(
                self._handler.fail_execute_tool,
                invocation,
                _error(RuntimeError("tool call did not finish")),
            )
        state.tool_invocations.clear()
        self._finish_step(state)

        result = event.result
        if result is not None:
            reason = _finish_reason(result.stop_reason)
            state.agent.finish_reasons = [reason]
            state.agent.output_messages = _convert_output_message(
                result.message, reason
            )
            _apply_usage(
                state.agent, {"usage": result.metrics.accumulated_usage}
            )
            self._finalize(self._handler.stop_invoke_agent, state.agent)
        else:
            self._finalize(
                self._handler.fail_invoke_agent,
                state.agent,
                _error(
                    state.agent_error
                    or RuntimeError(
                        "Strands agent invocation did not produce a result"
                    )
                ),
            )

    def current_invocation_key(self) -> int | None:
        return _CURRENT_INVOCATION_KEY.get()

    @hook_advice("strands", "stream_context_attach")
    def attach_stream_contexts(self, state_key: int | None) -> None:
        state = self._state_by_key(state_key)
        if state is None:
            return
        invocations = [state.agent, state.step, state.llm]
        invocations.extend(state.tool_invocations.values())
        for invocation in invocations:
            if (
                invocation is not None
                and invocation.span is not None
                and invocation.context_token is None
            ):
                invocation.context_token = otel_context.attach(
                    set_span_in_context(invocation.span)
                )

    @hook_advice("strands", "stream_context_detach")
    def detach_stream_contexts(self, state_key: int | None) -> None:
        state = self._state_by_key(state_key)
        if state is None:
            return
        invocations = list(state.tool_invocations.values())
        invocations.extend([state.llm, state.step, state.agent])
        for invocation in invocations:
            if invocation is not None and invocation.context_token is not None:
                token, invocation.context_token = (
                    invocation.context_token,
                    None,
                )
                otel_context.detach(token)
        if _CURRENT_INVOCATION_KEY.get() == state_key:
            _CURRENT_INVOCATION_KEY.set(None)

    def abandon_stream_context_tokens(self, state_key: int | None) -> None:
        """Drop unusable tokens after a failed cross-context detach."""
        state = self._state_by_key(state_key)
        if state is None:
            return
        invocations = [state.agent, state.step, state.llm]
        invocations.extend(state.tool_invocations.values())
        for invocation in invocations:
            if invocation is not None:
                invocation.context_token = None
        if _CURRENT_INVOCATION_KEY.get() == state_key:
            _CURRENT_INVOCATION_KEY.set(None)

    @hook_advice("strands", "stream_close")
    def finish_closed_stream(self, state_key: int | None) -> None:
        if state_key is None:
            return
        with self._lock:
            state = self._states.pop(state_key, None)
        if _CURRENT_INVOCATION_KEY.get() == state_key:
            _CURRENT_INVOCATION_KEY.set(None)
        if state is None:
            return

        close_error = RuntimeError(
            "Strands stream closed before invocation completed"
        )
        if state.llm is not None:
            self._finalize(
                self._handler.fail_llm, state.llm, _error(close_error)
            )
            state.llm = None
        for invocation in list(state.tool_invocations.values()):
            self._finalize(
                self._handler.fail_execute_tool,
                invocation,
                _error(close_error),
            )
        state.tool_invocations.clear()
        state.step_error = state.step_error or close_error
        state.agent_error = state.agent_error or close_error
        self._finish_step(state)
        self._finalize(
            self._handler.fail_invoke_agent,
            state.agent,
            _error(state.agent_error),
        )

    def _state(
        self, invocation_state: dict[str, Any]
    ) -> _InvocationState | None:
        return self._state_by_key(id(invocation_state))

    def _state_by_key(self, state_key: int | None) -> _InvocationState | None:
        if state_key is None:
            return None
        with self._lock:
            return self._states.get(state_key)

    def _finish_step(self, state: _InvocationState) -> None:
        if state.step is not None:
            if state.step_error is not None:
                self._finalize(
                    self._handler.fail_react_step,
                    state.step,
                    _error(state.step_error),
                )
            else:
                self._finalize(self._handler.stop_react_step, state.step)
            state.step = None
            state.step_error = None

    @staticmethod
    def _finalize(callback: Any, invocation: Any, *args: Any) -> None:
        """End a span if a util finalizer fails before cleaning its context."""
        try:
            callback(invocation, *args)
        except Exception as exc:
            token, invocation.context_token = invocation.context_token, None
            if token is not None:
                otel_context.detach(token)
            span = invocation.span
            if span is not None and span.is_recording():
                span.set_attribute(
                    "gen_ai.span.kind", _fallback_span_kind(invocation)
                )
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.end()
            raise


def _error(exception: BaseException) -> Error:
    return Error(message=str(exception), type=type(exception))


def _fallback_span_kind(invocation: Any) -> str:
    if isinstance(invocation, InvokeAgentInvocation):
        return "AGENT"
    if isinstance(invocation, ReactStepInvocation):
        return "STEP"
    if isinstance(invocation, ExecuteToolInvocation):
        return "TOOL"
    return "LLM"


def _model_name(agent: Any) -> str | None:
    model = getattr(agent, "model", None)
    if model is None:
        return None
    config = model.get_config()
    if isinstance(config, dict):
        for name in ("model_id", "modelId", "model", "model_name"):
            value = config.get(name)
            if value:
                return str(value)
    for name in ("model_id", "model_name", "name"):
        value = getattr(model, name, None)
        if value:
            return str(value)
    return type(model).__name__


def _provider_name(agent: Any) -> str:
    model = getattr(agent, "model", None)
    if model is None:
        return "strands"
    client_args = getattr(model, "client_args", {})
    base_url_host = (
        (urlparse(str(client_args.get("base_url", ""))).hostname or "").lower()
        if isinstance(client_args, dict)
        else ""
    )
    model_name = (_model_name(agent) or "").lower()
    if (
        base_url_host
        in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}
        or "qwen" in model_name
    ):
        return "dashscope"
    module = type(model).__module__.lower()
    class_name = type(model).__name__.lower()
    for needle, provider in (
        ("bedrock", "aws.bedrock"),
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("gemini", "gcp.gemini"),
        ("ollama", "ollama"),
    ):
        if needle in module or needle in class_name:
            return provider
    return module.split(".")[0] or "strands"


def _create_llm_span(agent: Any) -> bool:
    mode = os.getenv(
        "OTEL_INSTRUMENTATION_STRANDS_LLM_SPAN_MODE", "auto"
    ).lower()
    if mode == "never":
        return False
    if mode == "always":
        return True
    model = getattr(agent, "model", None)
    return not bool(
        getattr(model, "_is_instrumented_by_opentelemetry", False)
        or getattr(type(model), "_is_instrumented_by_opentelemetry", False)
    )


def _conversation_id(invocation_state: dict[str, Any]) -> str | None:
    for key in ("conversation_id", "session_id", "thread_id"):
        value = invocation_state.get(key)
        if value:
            return str(value)
    return None


def _system_instruction(agent: Any) -> list[Any]:
    prompt = getattr(agent, "_system_prompt", None)
    return [Text(content=prompt)] if isinstance(prompt, str) and prompt else []


def _tool_definitions(agent: Any) -> list[FunctionToolDefinition]:
    registry = getattr(agent, "tool_registry", None)
    if registry is None:
        return []
    try:
        specs = registry.get_all_tool_specs()
    except (AttributeError, TypeError):
        return []
    return [
        FunctionToolDefinition(
            name=spec.get("name", "unknown"),
            description=spec.get("description"),
            parameters=spec.get("inputSchema", {}).get("json", {}),
        )
        for spec in specs
    ]


def _tool_description(selected_tool: Any) -> str | None:
    spec = getattr(selected_tool, "tool_spec", None)
    return spec.get("description") if isinstance(spec, dict) else None


def _convert_messages(messages: list[Any]) -> list[InputMessage]:
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        converted.append(
            InputMessage(
                role=str(message.get("role", "user")),
                parts=_convert_content(message.get("content", [])),
            )
        )
    return converted


def _convert_output_message(message: Any, reason: str) -> list[OutputMessage]:
    if not isinstance(message, dict):
        return []
    return [
        OutputMessage(
            role=str(message.get("role", "assistant")),
            parts=_convert_content(message.get("content", [])),
            finish_reason=reason,
        )
    ]


def _convert_content(blocks: Any) -> list[Any]:
    if isinstance(blocks, str):
        return [Text(content=blocks)]
    parts = []
    for block in blocks or []:
        if isinstance(block, str):
            parts.append(Text(content=block))
        elif "text" in block:
            parts.append(Text(content=block["text"]))
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            parts.append(
                ToolCall(
                    name=tool_use.get("name", ""),
                    arguments=tool_use.get("input"),
                    id=tool_use.get("toolUseId"),
                )
            )
        elif "toolResult" in block:
            tool_result = block["toolResult"]
            parts.append(
                ToolCallResponse(
                    response=tool_result.get("content"),
                    id=tool_result.get("toolUseId"),
                )
            )
        else:
            parts.append(block)
    return parts


def _finish_reason(reason: Any) -> str:
    value = str(reason or "stop")
    if value == "tool_use":
        return "tool_calls"
    if value in {"max_tokens", "limit_output_tokens", "limit_total_tokens"}:
        return "length"
    if value in {"content_filtered", "guardrail_intervened"}:
        return "content_filter"
    return "stop" if value in {"end_turn", "stop_sequence"} else value


def _apply_usage(invocation: Any, metadata: Any) -> None:
    if not isinstance(metadata, dict):
        return
    usage = metadata.get("usage", {})
    if isinstance(usage, dict):
        invocation.input_tokens = usage.get("inputTokens")
        invocation.output_tokens = usage.get("outputTokens")
        invocation.usage_cache_read_input_tokens = usage.get(
            "cacheReadInputTokens"
        )
        invocation.usage_cache_creation_input_tokens = usage.get(
            "cacheWriteInputTokens"
        )
    metrics = metadata.get("metrics", {})
    if isinstance(metrics, dict) and invocation.monotonic_start_s is not None:
        ttft_ms = metrics.get("timeToFirstByteMs")
        if ttft_ms is not None:
            invocation.monotonic_first_token_s = (
                invocation.monotonic_start_s + float(ttft_ms) / 1000
            )
