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

"""Shared helpers for the LoongSuite agentUniverse instrumentation.

Nothing in this module -- or anywhere else in this package -- imports or
delegates to agentUniverse's own OTel instrumentors. The wrapper extension
points in ``agentuniverse.base.annotation.trace`` are claimed by this package,
and the span, attribute, metric and token-usage work is done here from scratch.
Only the framework's non-telemetry business surfaces are reused: the
``_get_*_info`` argument readers, ``ConversationMemoryModule``,
``InvocationChainContext``/``Monitor`` and the ``AuTraceManager`` token API.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import json
import logging
import time
import traceback
from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, Optional, Type

from agentuniverse.base.tracing.au_trace_manager import (
    add_current_token_usage,
    add_current_token_usage_to_parent,
    get_current_token_usage,
    init_new_token_usage,
)
from agentuniverse.base.util.monitor.monitor import Monitor
from agentuniverse.llm.llm_output import TokenUsage

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
    is_experimental_mode,
)

logger = logging.getLogger(__name__)

INSTRUMENTOR_NAME = "loongsuite-instrumentation-agentuniverse"

#: Framework name reported on every ``gen_ai.*`` span this package creates.
FRAMEWORK_NAME = "agentuniverse"

#: ``gen_ai.framework`` is a LoongSuite extension attribute. The shared handler
#: does not set it, so each layer records it here.
GEN_AI_FRAMEWORK = "gen_ai.framework"

#: The wrapper extension points this package claims. These are the module-level
#: globals the ``@trace_agent``/``@trace_llm``/``@trace_tool`` decorators resolve
#: on every call, so replacing them intercepts every call.
LAYER_WRAPPER_GLOBALS: tuple[tuple[str, str], ...] = (
    ("_agent_wrapper_sync", "_agent_wrapper_async"),
    ("_llm_wrapper_sync", "_llm_wrapper_async"),
    ("_tool_wrapper_sync", "_tool_wrapper_async"),
)

WRAPPER_GLOBAL_NAMES: tuple[str, ...] = tuple(
    name for pair in LAYER_WRAPPER_GLOBALS for name in pair
)

#: Span status values used by the framework's ``au.*.status`` attribute.
SUCCESS_STATUS = "success"
ERROR_STATUS = "error"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _fallback(value: Any) -> Any:
    """JSON fallback for values the stdlib encoder rejects."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)


def safe_json_dumps(value: Any) -> str:
    """Serialize ``value`` for a span attribute, never raising."""
    try:
        return json.dumps(value, default=_fallback, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive
        return str(value)


def error_to_string(error: BaseException) -> str:
    """Render an exception for the ``au.*.error.message`` attribute.

    The framework's own instrumentation always writes the full traceback. A
    traceback quotes the values flowing through the failing call, and so does an
    exception's own message, so with content capture disabled this package
    reports the exception type alone -- see the privacy section of the README.
    """
    if not captures_span_content():
        return type(error).__name__
    try:
        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    except Exception:  # pragma: no cover - defensive
        return type(error).__name__


def error_status_text(error: BaseException) -> str:
    """The span status description for a failed call.

    Uses the same privacy rule as the ``au.*.error.message`` attribute, so a
    status description can never carry content the attributes may not.
    """
    return error_to_string(error)


def build_safely(
    factory: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Build one layer's call record, or ``None`` when that is not possible.

    Reading the framework's call information can fail -- an unusual call shape, a
    component that is only half configured. Telemetry must never turn that into a
    failing application call, so the caller runs the original function untouched
    when this returns ``None``.
    """
    try:
        return factory(*args, **kwargs)
    except Exception:
        logger.warning(
            "Could not prepare agentUniverse instrumentation for a call; "
            "running it without instrumentation",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Content capture / privacy
# ---------------------------------------------------------------------------


def content_capturing_mode() -> ContentCapturingMode:
    """The effective GenAI content capturing mode, fail-safe.

    The shared util raises when the GenAI stability opt-in is not experimental,
    which is exactly the "no content" case here.
    """
    try:
        if not is_experimental_mode():
            return ContentCapturingMode.NO_CONTENT
        return get_content_capturing_mode()
    except Exception:  # pragma: no cover - defensive
        return ContentCapturingMode.NO_CONTENT


def captures_span_content() -> bool:
    """Whether user content may be written to span attributes.

    Evaluated per call from the same shared configuration the GenAI handler
    reads, so the ``au.*`` compatibility carriers and the handler's
    ``gen_ai.*`` content attributes are always gated identically. ``EVENT_ONLY``
    keeps span content redacted.
    """
    return content_capturing_mode() in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


def maybe_content(value: Any) -> Optional[str]:
    """Serialize ``value`` for a span content carrier, or ``None``."""
    if not captures_span_content():
        return None
    return safe_json_dumps(value)


def maybe_content_text(value: Any) -> Optional[str]:
    """Render ``value`` as span content text, or ``None`` when redacted."""
    if not captures_span_content():
        return None
    if isinstance(value, str):
        return value
    return safe_json_dumps(value)


#: LLM request parameters that are safe to report with content capture disabled.
#: They are scalars by nature, so they cannot carry prompt or message content.
SAFE_LLM_PARAM_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_new_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "n",
        "seed",
        "stream",
        "logprobs",
        "top_logprobs",
        "response_format",
        "stop",
    }
)


def llm_params_attribute(params: Any) -> str:
    """The ``au.llm.llm_params`` payload for a call.

    With content capture enabled this is the whole parameter mapping, exactly
    what the framework's own instrumentation reports. With capture disabled only
    the scalar request parameters in :data:`SAFE_LLM_PARAM_KEYS` are kept, so a
    call that passes content through its parameters cannot leak it.
    """
    if captures_span_content():
        return safe_json_dumps(params)
    safe: Dict[str, Any] = {}
    if isinstance(params, dict):
        for key, value in params.items():
            if key not in SAFE_LLM_PARAM_KEYS:
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                safe[key] = value
    return safe_json_dumps(safe)


# ---------------------------------------------------------------------------
# Token usage (framework API; the usage bookkeeping belongs to our spans)
# ---------------------------------------------------------------------------


def reset_token_usage(span_id: Optional[int] = None) -> None:
    init_new_token_usage(span_id)


def token_usage_for(span_id: Optional[int] = None) -> TokenUsage:
    """Token usage accumulated for ``span_id``, defaulting to an empty one."""
    try:
        return get_current_token_usage(span_id)
    except Exception:  # pragma: no cover - defensive
        return TokenUsage()


def record_token_usage(
    usage: TokenUsage, span_id: Optional[int] = None
) -> None:
    """Accumulate ``usage`` onto ``span_id`` exactly once."""
    if usage is None:
        return
    try:
        add_current_token_usage(usage, span_id)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Failed to accumulate token usage", exc_info=True)


def promote_token_usage_to_parent(
    usage: Optional[TokenUsage] = None,
    parent_span_id: Optional[int] = None,
) -> None:
    """Move a finished layer span's usage up to its parent span."""
    try:
        add_current_token_usage_to_parent(usage, parent_span_id)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Failed to promote token usage", exc_info=True)


# ---------------------------------------------------------------------------
# Span lifecycle
# ---------------------------------------------------------------------------


class LayerSpan:
    """One LoongSuite-owned layer span plus its attached context token."""

    def __init__(
        self,
        tracer: trace.Tracer,
        span_name: str,
        span_kind: SpanKind = SpanKind.INTERNAL,
    ) -> None:
        self.name = span_name
        self.span: Span = tracer.start_span(
            name=span_name,
            kind=span_kind,
            context=otel_context.get_current(),
        )
        self.context_token = None
        self.start_time = time.time()
        self.monotonic_start_s = time.monotonic()
        self.first_token_time: Optional[float] = None
        self.first_token_monotonic: Optional[float] = None

    def attach(self) -> "LayerSpan":
        self.context_token = otel_context.attach(
            trace.set_span_in_context(self.span)
        )
        try:
            # The framework's token bookkeeping is keyed by the span that is
            # current when a layer starts, so seed it right after attaching.
            reset_token_usage(self.span.context.span_id)
            self.mark_gen_ai_identity()
        except BaseException:
            # A half-attached span is worse than no span: the context token
            # would stay attached and the span open, so every later span of
            # this trace would be filed under a span that never finished.
            # Undo both, then let the caller run the call uninstrumented.
            self.detach()
            self.end()
            raise
        return self

    def take_context_token(self) -> Any:
        """Hand the context token to the handler that will detach it."""
        token, self.context_token = self.context_token, None
        return token

    @property
    def span_id(self) -> Optional[int]:
        return self.span.context.span_id

    @property
    def parent_span_id(self) -> Optional[int]:
        parent = self.span.parent
        return parent.span_id if parent is not None else None

    def elapsed(self) -> float:
        """Seconds since the layer started, measured off the call clock."""
        return time.time() - self.start_time

    def record_first_token(self) -> float:
        """Record and return the elapsed seconds up to the first token."""
        self.first_token_monotonic = time.monotonic()
        self.first_token_time = time.time()
        return self.first_token_time - self.start_time

    def apply_timing(self, invocation: Any) -> None:
        """Hand the shared handler the clocks behind its timing attributes.

        ``gen_ai.response.time_to_first_token`` is derived by the shared handler
        from the monotonic start and first-token times, so both are recorded
        here. A call that never saw a streamed token treats its response as the
        first token, measured when the call finishes.
        """
        invocation.monotonic_start_s = self.monotonic_start_s
        if self.first_token_monotonic is None:
            self.first_token_monotonic = time.monotonic()
        invocation.monotonic_first_token_s = self.first_token_monotonic

    def mark_gen_ai_identity(self) -> None:
        self.span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
        session_id = agent_name_attributes().get(GEN_AI_SESSION_ID)
        if session_id:
            self.span.set_attribute(GEN_AI_SESSION_ID, session_id)

    def end(self) -> None:
        if self.span.is_recording():
            self.span.end()

    def detach(self) -> None:
        token, self.context_token = self.context_token, None
        detach_context_token(token)


def detach_context_token(token: Any) -> None:
    """Detach a context token once, never raising."""
    if token is None:
        return
    try:
        otel_context.detach(token)
    except Exception:  # pragma: no cover - cross-context teardown
        logger.debug("Failed to detach span context", exc_info=True)


def monotonic_now() -> float:
    return time.monotonic()


class NamePreservingSpan(Span):
    """Keeps the framework-native span name while the shared handler finalizes.

    The shared GenAI handler normalises a span name to its own ``gen_ai`` form
    (``invoke_agent X``/``chat ...``/``execute_tool X``) and ends the span
    itself, which would replace the ``au.agent.*``/``au.llm.*``/``au.tool.*``
    span name that agentUniverse's own instrumentation exports. Handing the
    handler this thin proxy instead -- every other call is forwarded to the real
    span, ``end()`` included -- keeps the native span-name contract while the
    handler still writes its ``gen_ai.*`` attributes, records its own metrics
    and closes the span.
    """

    def __init__(self, span: Span) -> None:
        self._span = span

    def update_name(self, name: str) -> None:
        """Drop the handler's name normalisation."""

    def get_span_context(self):
        return self._span.get_span_context()

    def set_attribute(self, key, value):
        return self._span.set_attribute(key, value)

    def set_attributes(self, attributes):
        return self._span.set_attributes(attributes)

    def add_event(self, *args, **kwargs):
        return self._span.add_event(*args, **kwargs)

    def set_status(self, *args, **kwargs):
        return self._span.set_status(*args, **kwargs)

    def record_exception(self, *args, **kwargs):
        return self._span.record_exception(*args, **kwargs)

    def is_recording(self):
        return self._span.is_recording()

    def end(self, *args, **kwargs):
        return self._span.end(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._span, item)


def set_error_status(span: Span, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))


# ---------------------------------------------------------------------------
# The agentUniverse ``au.*`` attribute contract, per layer
# ---------------------------------------------------------------------------


class AgentAttributes:
    SPAN_KIND = "au.span.kind"
    NAME = "au.agent.name"
    INPUT = "au.agent.input"
    OUTPUT = "au.agent.output"
    DURATION = "au.agent.duration"
    STATUS = "au.agent.status"
    PAIR_ID = "au.agent.pair_id"
    STREAMING = "au.agent.streaming"
    FIRST_TOKEN_DURATION = "au.agent.first_token.duration"
    ERROR_TYPE = "au.agent.error.type"
    ERROR_MESSAGE = "au.agent.error.message"
    CALLER_NAME = "au.trace.caller_name"
    CALLER_TYPE = "au.trace.caller_type"
    USAGE_TOTAL_TOKENS = "au.agent.usage.total_tokens"
    USAGE_PROMPT_TOKENS = "au.agent.usage.prompt_tokens"
    USAGE_COMPLETION_TOKENS = "au.agent.usage.completion_tokens"
    USAGE_DETAIL_TOKENS = "au.agent.usage.detail_tokens"


class LLMAttributes:
    SPAN_KIND = "au.span.kind"
    NAME = "au.llm.name"
    CHANNEL_NAME = "au.llm.channel_name"
    INPUT = "au.llm.input"
    OUTPUT = "au.llm.output"
    LLM_PARAMS = "au.llm.llm_params"
    STREAMING = "au.llm.streaming"
    DURATION = "au.llm.duration"
    STATUS = "au.llm.status"
    FIRST_TOKEN_DURATION = "au.llm.first_token.duration"
    ERROR_TYPE = "au.llm.error.type"
    ERROR_MESSAGE = "au.llm.error.message"
    CALLER_NAME = "au.trace.caller_name"
    CALLER_TYPE = "au.trace.caller_type"
    USAGE_TOTAL_TOKENS = "au.llm.usage.total_tokens"
    USAGE_PROMPT_TOKENS = "au.llm.usage.prompt_tokens"
    USAGE_COMPLETION_TOKENS = "au.llm.usage.completion_tokens"
    USAGE_DETAIL_TOKENS = "au.llm.usage.detail_tokens"


class ToolAttributes:
    SPAN_KIND = "au.span.kind"
    NAME = "au.tool.name"
    INPUT = "au.tool.input"
    OUTPUT = "au.tool.output"
    DURATION = "au.tool.duration"
    STATUS = "au.tool.status"
    PAIR_ID = "au.tool.pair_id"
    ERROR_TYPE = "au.tool.error.type"
    ERROR_MESSAGE = "au.tool.error.message"
    CALLER_NAME = "au.trace.caller_name"
    CALLER_TYPE = "au.trace.caller_type"
    USAGE_TOTAL_TOKENS = "au.tool.usage.total_tokens"
    USAGE_PROMPT_TOKENS = "au.tool.usage.prompt_tokens"
    USAGE_COMPLETION_TOKENS = "au.tool.usage.completion_tokens"
    USAGE_DETAIL_TOKENS = "au.tool.usage.detail_tokens"


class MetricLabels:
    """Metric label keys, shared by the three layers."""

    CALLER_NAME = "au_trace_caller_name"
    CALLER_TYPE = "au_trace_caller_type"
    AGENT_NAME = "au_agent_name"
    AGENT_STATUS = "au_agent_status"
    AGENT_STREAMING = "au_agent_streaming"
    LLM_NAME = "au_llm_name"
    LLM_CHANNEL_NAME = "au_llm_channel_name"
    LLM_STATUS = "au_llm_status"
    LLM_STREAMING = "au_llm_streaming"
    TOOL_NAME = "au_tool_name"
    TOOL_STATUS = "au_tool_status"


def caller_labels(caller_info: Dict[str, Any]) -> Dict[str, Any]:
    """The caller labels every layer puts on its metrics."""
    return {
        MetricLabels.CALLER_NAME: caller_info.get("source", "unknown"),
        MetricLabels.CALLER_TYPE: caller_info.get("type", "user"),
    }


def set_caller_attributes(span: Span, caller_info: Dict[str, Any]) -> None:
    span.set_attribute(
        AgentAttributes.CALLER_NAME, caller_info.get("source", "unknown")
    )
    span.set_attribute(
        AgentAttributes.CALLER_TYPE, caller_info.get("type", "user")
    )


def set_usage_attributes(
    span: Span, names: Type[Any], usage: TokenUsage
) -> None:
    """Record the ``au.*.usage.*`` attributes for a layer."""
    span.set_attribute(names.USAGE_TOTAL_TOKENS, usage.total_tokens)
    span.set_attribute(names.USAGE_PROMPT_TOKENS, usage.prompt_tokens)
    span.set_attribute(names.USAGE_COMPLETION_TOKENS, usage.completion_tokens)
    span.set_attribute(
        names.USAGE_DETAIL_TOKENS, safe_json_dumps(usage.to_dict())
    )


def set_layer_error_attributes(
    span: Span, names: Type[Any], error: BaseException, duration: float
) -> None:
    """Record the ``au.*`` error contract for a layer."""
    span.set_attribute(names.STATUS, ERROR_STATUS)
    span.set_attribute(names.ERROR_TYPE, type(error).__name__)
    span.set_attribute(names.ERROR_MESSAGE, error_to_string(error))
    span.set_attribute(names.DURATION, duration)


# ---------------------------------------------------------------------------
# GenAI helpers
# ---------------------------------------------------------------------------


def run_safely(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one telemetry step without ever breaking the application.

    Telemetry must not turn a working agent call into a failing one, so every
    tracing/metrics step the layers run goes through here. Anything that goes
    wrong is logged and dropped.
    """
    try:
        return action(*args, **kwargs)
    except Exception:  # pragma: no cover - telemetry must never raise
        logger.debug(
            "agentUniverse instrumentation step failed", exc_info=True
        )
        return None


def user_message(content: str) -> InputMessage:
    return InputMessage(role="user", parts=[Text(content=content)])


def assistant_message(
    content: str, finish_reason: str = "stop"
) -> OutputMessage:
    return OutputMessage(
        role="assistant",
        parts=[Text(content=content)],
        finish_reason=finish_reason,
    )


def input_messages_for(payload: Any) -> list:
    """Preserve public message roles, tool IDs and the actual user input."""
    if isinstance(payload, dict):
        nested = payload.get("kwargs")
        values = nested if isinstance(nested, dict) else payload
        for key in ("messages", "prompt", "input"):
            if key in values:
                return input_messages_for(values[key])
        positional = payload.get("args")
        if isinstance(positional, (list, tuple)) and positional:
            return input_messages_for(positional[0])
    if isinstance(payload, (list, tuple)):
        messages = []
        for message in payload:
            role = message_field(message, "role") or message_field(
                message, "type"
            )
            role = getattr(role, "value", role)
            if role not in (
                "user",
                "assistant",
                "system",
                "developer",
                "tool",
            ):
                continue
            content = message_field(message, "content")
            if role == "tool":
                parts = [
                    ToolCallResponse(
                        response=content,
                        id=message_field(message, "tool_call_id"),
                    )
                ]
            else:
                parts = content_parts(content)
                reasoning = message_field(message, "reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    parts.insert(0, Reasoning(content=reasoning))
                parts.extend(
                    tool_call_parts(message_field(message, "tool_calls"))
                )
            messages.append(InputMessage(role=role, parts=parts))
        return messages
    if isinstance(payload, str):
        text = payload
    else:
        text = safe_json_dumps(payload)
    return [user_message(text)] if text else []


def message_field(message: Any, name: str) -> Any:
    return (
        message.get(name)
        if isinstance(message, dict)
        else getattr(message, name, None)
    )


def content_parts(content: Any) -> list:
    if isinstance(content, str):
        return [Text(content=content)] if content else []
    if isinstance(content, list):
        return [
            Text(content=item["text"])
            for item in content
            if isinstance(item, dict)
            and item.get("type") in ("text", "input_text", "output_text")
            and isinstance(item.get("text"), str)
        ]
    return []


def parse_tool_arguments(value: Any) -> Any:
    """Parse the framework's JSON tool arguments for a GenAI tool call.

    The framework carries tool arguments as the provider's JSON string. The
    shared GenAI type holds them parsed, so a string that is valid JSON becomes
    the object it encodes and anything else is passed through unchanged.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        parsed = json.loads(text)
    except Exception:
        return value
    return parsed if isinstance(parsed, dict) else value


def tool_call_function(call: Any) -> Any:
    """The ``FunctionCall`` of a framework tool call, when it has one.

    The framework's own ``ToolCall`` keeps the name and the JSON arguments one
    level down, in ``function`` (``{"id": ..., "type": "function",
    "function": {"name": ..., "arguments": ...}}``), which is also the shape the
    provider sends.
    """
    if isinstance(call, dict):
        return call.get("function")
    return getattr(call, "function", None)


def tool_call_field(call: Any, name: str) -> Any:
    """One field of a tool call, wherever its shape keeps it."""
    function = tool_call_function(call)
    if function is not None:
        value = (
            function.get(name)
            if isinstance(function, dict)
            else getattr(function, name, None)
        )
        if value is not None:
            return value
    if isinstance(call, dict):
        return call.get(name)
    return getattr(call, name, None)


def tool_call_parts(payload: Any) -> list:
    """Map the framework's tool calls onto the shared GenAI tool-call parts."""
    if not payload:
        return []
    parts = []
    for call in payload:
        parts.append(
            ToolCall(
                arguments=parse_tool_arguments(
                    tool_call_field(call, "arguments")
                ),
                name=str(tool_call_field(call, "name") or ""),
                id=tool_call_field(call, "id"),
            )
        )
    return parts


def output_messages_for(
    payload: Any,
    finish_reason: str = "stop",
    tool_calls: Any = None,
    reasoning: Optional[str] = None,
) -> list:
    """The assistant message a layer reports for its own output.

    A round that answers with tool calls has no text: the calls themselves are
    the output, so ``tool_calls`` becomes a tool-call part of the same message.
    """
    parts: list = []
    if isinstance(reasoning, str) and reasoning:
        parts.append(Reasoning(content=reasoning))
    if payload is not None:
        text = (
            payload if isinstance(payload, str) else safe_json_dumps(payload)
        )
        if text:
            parts.append(Text(content=text))
    parts.extend(tool_call_parts(tool_calls))
    if not parts:
        return []
    return [
        OutputMessage(
            role="assistant", parts=parts, finish_reason=finish_reason
        )
    ]


def final_answer_of(result: Any) -> Any:
    """The agent's final answer, without the framework's own call envelope.

    ``AgentTemplate.customized_execute`` returns the caller's whole agent input
    with the answer under ``output``. Only that answer is the agent's output;
    echoing the input back would publish the prompt and the framework's
    bookkeeping as the agent's output.
    """
    if isinstance(result, dict):
        return result.get("output")
    getter = getattr(result, "get_data", None)
    if callable(getter):
        answer = run_safely(getter, "output")
        if answer is not None:
            return answer
    return getattr(result, "output", result)


_current_agent_name: ContextVar[tuple] = ContextVar(
    "agentuniverse_current_agent_name", default=(None, None)
)


def push_current_agent_name(
    name: Optional[str], session_id: Optional[str] = None
) -> Optional[Token]:
    """Track the agent a nested LLM/tool call belongs to."""
    if not name:
        return None
    try:
        if session_id is None:
            session_id = _current_agent_name.get()[1]
        return _current_agent_name.set((name, session_id))
    except Exception:  # pragma: no cover - defensive
        return None


def pop_current_agent_name(token: Optional[Token]) -> None:
    if token is None:
        return
    try:
        _current_agent_name.reset(token)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Failed to reset current agent name", exc_info=True)


def agent_name_attributes() -> Dict[str, Any]:
    """``gen_ai.agent.name`` for a nested layer span, when there is one."""
    name, session_id = _current_agent_name.get()
    attributes = {GenAI.GEN_AI_AGENT_NAME: name} if name else {}
    if session_id:
        attributes[GEN_AI_SESSION_ID] = session_id
    return attributes


def push_invocation_chain(source: str, node_type: str) -> bool:
    """Add this layer's node to the framework's invocation chain.

    The chain lives in the framework context under the current trace id, so the
    layer's span has to be the current span before its node is added: pushing it
    earlier would file the node under a different trace id than the one the
    instrumented call's own children read. ``True`` means this call added the
    node and is therefore the one that may remove it again.
    """
    try:
        Monitor.init_invocation_chain()
        Monitor.add_invocation_chain({"source": source, "type": node_type})
    except Exception:
        logger.debug(
            "Could not extend the agentUniverse invocation chain",
            exc_info=True,
        )
        return False
    return True


def pop_invocation_chain() -> None:
    """Remove this layer's node; on failure a parent's entry stays put."""
    try:
        Monitor.pop_invocation_chain()
    except Exception:
        logger.debug(
            "Could not leave the agentUniverse invocation chain",
            exc_info=True,
        )


def describe_wrapper_owner(wrapper: Any) -> Optional[str]:
    """Name the class a wrapper extension point is bound to, if any.

    The wrapper extension points are plain module globals, so whoever
    instrumented the framework before this package either installed a plain
    function or bound a method of its instrumentor class. This is used for
    logging only -- nothing is imported, constructed or called.
    """
    if wrapper is None:
        return None
    owner = getattr(wrapper, "__self__", None)
    if owner is None:
        return None
    owner_type = type(owner)
    return f"{owner_type.__module__}.{owner_type.__name__}"
