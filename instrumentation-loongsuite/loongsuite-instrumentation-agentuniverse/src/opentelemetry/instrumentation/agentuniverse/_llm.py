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

"""The LLM layer of the LoongSuite agentUniverse instrumentation.

This module claims the ``_llm_wrapper_sync``/``_llm_wrapper_async`` extension
points the ``@trace_llm`` decorator resolves and builds the ``au.llm.*`` span,
metrics and token bookkeeping itself. ``gen_ai.*`` attributes and metrics on the
same span come from the shared ``ExtendedTelemetryHandler``.

Streaming keeps one intentional difference from the framework's own
instrumentation: the token usage reported by the stream is aggregated onto the
parent span exactly once (see README).

No agentUniverse OTel class is imported or called here.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Generator, Optional

from agentuniverse.base.annotation import trace as trace_module
from agentuniverse.base.annotation.trace import _get_llm_info
from agentuniverse.base.util.monitor.monitor import Monitor
from agentuniverse.llm.llm_output import LLMOutput, TokenUsage

from opentelemetry.metrics import Meter
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    LLMInvocation,
)

from ._common import (
    SUCCESS_STATUS,
    LayerSpan,
    LLMAttributes,
    MetricLabels,
    NamePreservingSpan,
    agent_name_attributes,
    build_safely,
    caller_labels,
    error_status_text,
    input_messages_for,
    llm_params_attribute,
    maybe_content,
    maybe_content_text,
    monotonic_now,
    output_messages_for,
    pop_invocation_chain,
    promote_token_usage_to_parent,
    push_invocation_chain,
    record_token_usage,
    run_safely,
    set_caller_attributes,
    set_error_status,
    set_layer_error_attributes,
    set_usage_attributes,
    token_usage_for,
)
from ._usage import provider_usage, usage_call_context

logger = logging.getLogger(__name__)

SPAN_KIND_VALUE = "llm"


def usage_of(result: Any) -> Optional[TokenUsage]:
    """Prefer inclusive provider totals; never mutate SDK usage objects."""
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict):
        payload = raw.get("usage")
        provider = (
            "anthropic"
            if isinstance(payload, dict)
            and (
                "cache_read_input_tokens" in payload
                or "cache_creation_input_tokens" in payload
            )
            else "openai"
        )
        derived = provider_usage(payload, provider)
        if derived is not None:
            return derived
    return getattr(result, "usage", None)


def chunk_record(chunk: Any) -> tuple:
    """One streamed chunk as ``(text, usage)``, from either framework shape.

    A chunk of the stable ``LLMOutput`` shape carries its text in ``text``, and
    an empty string there is a real value: a role-only or usage-only chunk
    contributes nothing rather than its repr. From 0.1.0a2 on a streamed chunk --
    or the ``LLMStreamEvent`` wrapper around it -- carries the delta in
    ``text_delta`` instead, and ``LLMStreamEvent.text`` is a class method rather
    than the payload, so a reader that only knows ``text`` drops the whole
    stream's output. A chunk with neither field is not a framework payload and is
    rendered as it is.
    """
    text = ""
    for name in ("text", "text_delta"):
        value = getattr(chunk, name, None)
        if isinstance(value, str):
            text = value
            break
    else:
        if not hasattr(chunk, "text") and not hasattr(chunk, "text_delta"):
            text = str(chunk)
    return text, getattr(chunk, "usage", None)


def chunk_text(chunk: Any) -> str:
    """The text one streamed chunk contributes to the LLM span's output."""
    return chunk_record(chunk)[0]


def raw_choice(result: Any) -> dict:
    """The first public OpenAI choice retained by stable LLMOutput.raw."""
    raw = getattr(result, "raw", None)
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def raw_message(result: Any) -> dict:
    choice = raw_choice(result)
    message = choice.get("delta") or choice.get("message")
    return message if isinstance(message, dict) else {}


def tool_calls_of(result: Any) -> list:
    """The tool calls an LLM result asked for, from either framework shape.

    ``LLMOutput.get_tool_calls()`` is the 0.1.0a2 accessor; the stable shape
    keeps the provider message in ``LLMOutput.message``. A plain answer has
    neither and reports no calls.
    """
    getter = getattr(result, "get_tool_calls", None)
    if callable(getter):
        calls = run_safely(getter)
        if calls:
            return list(calls)
    calls = getattr(result, "tool_calls", None)
    if calls:
        return list(calls)
    message = getattr(result, "message", None)
    if isinstance(message, dict) and message.get("tool_calls"):
        return list(message["tool_calls"])
    return list(raw_message(result).get("tool_calls") or [])


def _streamed_tool_call_deltas(chunk: Any) -> list:
    """The tool-call deltas one streamed chunk carries, if any."""
    delta = getattr(chunk, "tool_call_delta", None)
    if delta is not None:
        return [delta]
    deltas = getattr(chunk, "tool_calls", None)
    if deltas:
        return list(deltas)
    message = getattr(chunk, "message", None)
    if isinstance(message, dict) and message.get("tool_calls"):
        return list(message["tool_calls"])
    return list(raw_message(chunk).get("tool_calls") or [])


class ToolCallAccumulator:
    """Assembles the tool calls a stream spreads across its chunks.

    A streamed tool call arrives in fragments: the first delta carries the id
    and the name, every later one carries another slice of the JSON arguments.
    The calls are kept in the order of their indices, which is the order the
    provider numbered them.
    """

    def __init__(self) -> None:
        self._calls: Dict[int, Dict[str, Any]] = {}

    def feed(self, chunk: Any) -> None:
        for position, delta in enumerate(_streamed_tool_call_deltas(chunk)):
            index = getattr(delta, "index", None)
            identifier = getattr(delta, "id", None)
            name = getattr(delta, "name", None)
            fragment = getattr(delta, "arguments_delta", None)
            if isinstance(delta, dict):
                index = delta.get("index", index)
                identifier = delta.get("id", identifier)
                name = delta.get("name", name)
                fragment = delta.get("arguments_delta", fragment)
                function = delta.get("function")
                if isinstance(function, dict):
                    name = function.get("name", name)
                    fragment = function.get("arguments", fragment)
            if not isinstance(index, int):
                index = len(self._calls) if index is None else position
            call = self._calls.setdefault(
                index, {"id": None, "name": "", "arguments": ""}
            )
            if identifier:
                call["id"] = identifier
            if name:
                call["name"] = name
            if isinstance(fragment, str):
                call["arguments"] += fragment

    def result(self) -> list:
        return [self._calls[index] for index in sorted(self._calls)]


class ClaimedSyncStream:
    """Finalize an explicitly closed stream even before its first iteration."""

    def __init__(self, call, stream, original):
        self._call = call
        self._stream = stream
        self._original = original
        self._started = False
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        self._started = True
        return next(self._stream)

    def send(self, value):
        self._started = True
        return self._stream.send(value)

    def throw(self, *args):
        self._started = True
        return self._stream.throw(*args)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._original, "close", None)
            if callable(close):
                close()
        finally:
            try:
                self._stream.close()
            finally:
                if not self._started:
                    run_safely(self._call.finish_stream_success, "", None)
                run_safely(self._call.finalize)


class ClaimedAsyncStream:
    """The async twin of ``process_sync_stream``: claimed, then finalized once.

    ``process_async_stream`` cannot itself be an async generator, because an
    async generator's body -- ``mark_streaming()`` included -- only runs on the
    caller's first ``__anext__``. A stream that is awaited and then closed or
    dropped without being read would therefore keep the caller's context token
    attached and its span recording. This iterator claims the stream when it is
    created, so the caller's context is released before it resumes, and it
    finalizes the span as soon as consumption stops, however it stops.
    """

    def __init__(
        self,
        call: "_LLMCall",
        stream: AsyncGenerator[Any, None],
        original: Any,
    ) -> None:
        self._call = call
        self._stream = stream
        self._original = original
        self._started = False
        self._closed = False

    def __aiter__(self) -> "ClaimedAsyncStream":
        return self

    def __anext__(self) -> Any:
        self._started = True
        return self._stream.__anext__()

    def asend(self, value: Any) -> Any:
        self._started = True
        return self._stream.asend(value)

    def athrow(self, *args: Any, **kwargs: Any) -> Any:
        self._started = True
        return self._stream.athrow(*args, **kwargs)

    async def aclose(self) -> None:
        """Stop the stream early; the span closes exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._original, "aclose", None)
            if callable(close):
                await close()
        finally:
            try:
                await self._stream.aclose()
            finally:
                if not self._started:
                    run_safely(self._call.finish_stream_success, "", None)
                run_safely(self._call.finalize)


def _apply_llm_plugins(func: Any) -> Any:
    """Apply the framework's LLM plugin hook, resolved per call.

    The framework looks its plugin hook up through the trace module's globals,
    so it is resolved here on every call too: an application that replaces it --
    or a test that spies on it -- sees the call. A missing or failing hook falls
    back to the plain function, so telemetry can never swallow the call.
    """
    try:
        plugins = getattr(trace_module, "_llm_plugins", None)
        return plugins(func) if plugins else func
    except Exception:
        logger.warning("Could not apply the LLM plugin hook", exc_info=True)
        return func


class LLMMetrics:
    """The agentUniverse LLM metric set, recorded by this package."""

    def __init__(self, meter: Meter) -> None:
        self._calls = meter.create_counter(
            name="llm_calls_total",
            description="Total number of LLM calls",
            unit="1",
        )
        self._errors = meter.create_counter(
            name="llm_errors_total",
            description="Total number of LLM errors",
            unit="1",
        )
        self._duration = meter.create_histogram(
            name="llm_call_duration",
            description="Duration of LLM calls in seconds",
            unit="s",
        )
        self._first_token = meter.create_histogram(
            name="llm_first_token_duration",
            description="Duration of LLM first token in seconds",
            unit="s",
        )
        self._total_tokens = meter.create_histogram(
            name="llm_total_tokens",
            description="Total token nums used in llm",
            unit="1",
        )
        self._prompt_tokens = meter.create_histogram(
            name="llm_prompt_tokens",
            description="Prompt token nums used in llm",
            unit="1",
        )
        self._completion_tokens = meter.create_histogram(
            name="llm_completion_tokens",
            description="Completion token nums used in llm",
            unit="1",
        )
        self._cached_tokens = meter.create_histogram(
            name="llm_cached_tokens",
            description="Cached token nums used in llm",
            unit="1",
        )
        self._reasoning_tokens = meter.create_histogram(
            name="llm_reasoning_tokens",
            description="Reasoning token nums used in llm",
            unit="1",
        )

    def record_call(self, labels: Dict[str, Any]) -> None:
        self._calls.add(1, labels)

    def record_duration(self, duration: float, labels: Dict[str, Any]) -> None:
        self._duration.record(duration, labels)

    def record_first_token(
        self, duration: float, labels: Dict[str, Any], streaming: bool
    ) -> None:
        self._first_token.record(
            duration, {**labels, MetricLabels.LLM_STREAMING: streaming}
        )

    def record_error(
        self, error: BaseException, duration: float, labels: Dict[str, Any]
    ) -> None:
        error_labels = {
            **labels,
            MetricLabels.LLM_STATUS: type(error).__name__,
        }
        self._errors.add(1, error_labels)
        self._duration.record(duration, error_labels)

    def record_tokens(self, usage: TokenUsage, labels: Dict[str, Any]) -> None:
        self._total_tokens.record(usage.total_tokens, labels)
        self._completion_tokens.record(usage.completion_tokens, labels)
        self._prompt_tokens.record(usage.prompt_tokens, labels)
        self._cached_tokens.record(usage.cached_tokens, labels)
        self._reasoning_tokens.record(usage.reasoning_tokens, labels)


class _LLMCall:
    """Everything one wrapped LLM call needs, sync, async or streaming."""

    def __init__(
        self,
        layer: "LLMLayer",
        func: Any,
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> None:
        self._layer = layer
        #: Kept because the per-call overrides are read from the call's own
        #: arguments: ``_get_llm_info`` only reports what the component is
        #: configured with.
        self.kwargs = dict(kwargs or {})
        (
            self.llm_instance,
            self.source,
            self.channel_name,
            self.llm_input,
            self.params,
            self.caller_info,
        ) = _get_llm_info(func, *args, **kwargs)
        self.span_name = f"au.llm.{self.source}"
        self.labels: Dict[str, Any] = {
            MetricLabels.LLM_NAME: self.source,
            **caller_labels(self.caller_info),
            MetricLabels.LLM_STATUS: SUCCESS_STATUS,
        }
        self.layer_span: Optional[LayerSpan] = None
        self.streaming = False
        self.provider_usage: Optional[TokenUsage] = None
        self.reasoning_chunks: list[str] = []
        self.stream_finish_reason: Optional[str] = None
        self.finished = False
        #: Only a node this call actually added may be popped again.
        self.invocation_chain_pushed = False
        self.invocation = LLMInvocation(
            request_model=self._request_model(),
            provider=self.channel_name or None,
            attributes=agent_name_attributes(),
            temperature=self._temperature(),
        )

    def _call_model_override(self) -> Optional[str]:
        """The model this one call asks for, when it names one itself.

        Every framework LLM implementation pops ``model`` out of the call's own
        arguments and falls back to its configured name
        (``model=kwargs.pop("model", self.model_name)``), so a call that names a
        model really does go to that model. Telemetry has to report the model the
        call went to, not the one the component defaults to.
        """
        for source in (
            self.kwargs,
            self.llm_input.get("kwargs"),
            self.llm_input,
        ):
            if not isinstance(source, dict):
                continue
            value = source.get("model")
            if isinstance(value, str) and value:
                return value
        return None

    def _request_model(self) -> Optional[str]:
        """The model the call asks for, for ``gen_ai.request.model``.

        The framework's ``LLM.model_name`` carries the provider model (``gpt-4``
        and friends); a component that does not configure one reports its own
        name, which is what the framework's instrumentation uses as well.
        ``au.llm.name`` keeps that component name in either case.
        """
        override = self._call_model_override()
        if override:
            return override
        channel = getattr(self.llm_instance, "_channel_instance", None)
        model_name = (
            getattr(channel, "channel_model_name", None)
            or getattr(self.llm_instance, "channel_model_name", None)
            or getattr(self.llm_instance, "model_name", None)
        )
        if isinstance(model_name, str) and model_name:
            return model_name
        return None

    def _temperature(self) -> Optional[float]:
        """The configured temperature, ignoring the framework's -1 sentinel."""
        channel = (
            getattr(self.llm_instance, "_channel_instance", None)
            or self.llm_instance
        )
        config = getattr(channel, "channel_model_config", None) or {}
        nested = self.llm_input.get("kwargs") or {}
        for values in (self.kwargs, nested, config):
            if "temperature" in values:
                temperature = values["temperature"]
                return (
                    temperature
                    if isinstance(temperature, (int, float))
                    and temperature != -1
                    else None
                )
        temperature = getattr(self.llm_instance, "temperature", None)
        return (
            temperature
            if isinstance(temperature, (int, float)) and temperature != -1
            else None
        )

    def _refresh_request(self) -> None:
        # A channel may be lazily initialized inside the first business call.
        channel = (
            getattr(self.llm_instance, "_channel_instance", None)
            or self.llm_instance
        )
        provider = getattr(channel, "channel_name", None)
        self.invocation.provider = (
            provider if provider and provider != "default_channel" else None
        )
        self.invocation.request_model = self._request_model()
        tools = (
            self.kwargs.get("tools")
            or (self.llm_input.get("kwargs") or {}).get("tools")
            or []
        )
        self.invocation.tool_definitions = [
            FunctionToolDefinition(
                name=tool["function"]["name"],
                description=tool["function"].get("description"),
                parameters=tool["function"].get("parameters"),
            )
            for tool in tools
            if isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        ]
        self.invocation.temperature = self._temperature()
        if self.invocation.temperature is not None:
            self.params["temperature"] = self.invocation.temperature
        if self.layer_span is not None:
            self.layer_span.span.set_attribute(
                LLMAttributes.LLM_PARAMS, llm_params_attribute(self.params)
            )
            if self.invocation.provider:
                self.layer_span.span.set_attribute(
                    LLMAttributes.CHANNEL_NAME, self.invocation.provider
                )

    def _collect_chunk_metadata(self, chunk: Any) -> None:
        raw = getattr(chunk, "raw", None)
        raw = raw if isinstance(raw, dict) else {}
        reasoning = getattr(chunk, "reasoning_delta", None) or raw_message(
            chunk
        ).get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self.reasoning_chunks.append(reasoning)
        finish = getattr(chunk, "finish_reason", None) or raw_choice(
            chunk
        ).get("finish_reason")
        if isinstance(finish, str) and finish:
            self.stream_finish_reason = finish
        identifier = getattr(chunk, "response_id", None) or raw.get("id")
        if isinstance(identifier, str) and identifier:
            self.invocation.response_id = identifier
        if isinstance(raw.get("model"), str):
            self.invocation.response_model_name = raw["model"]

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        run_safely(Monitor.trace_llm_input, self.source, self.llm_input)

        self.layer_span = self._layer.start_span(self.span_name)
        # The chain is keyed by the current trace id, so the span has to be
        # current before this layer's node is added; children read it there.
        if push_invocation_chain(self.source, "llm"):
            self.invocation_chain_pushed = True
        span = self.layer_span.span
        # The shared handler finalizes this span on behalf of the invocation, so
        # it has to know the span and the context token it may detach. The proxy
        # keeps the framework-native span name while it does so.
        self.invocation.span = NamePreservingSpan(span)
        span.set_attribute(LLMAttributes.SPAN_KIND, SPAN_KIND_VALUE)
        span.set_attribute(LLMAttributes.NAME, self.source)
        span.set_attribute(LLMAttributes.CHANNEL_NAME, self.channel_name)
        content = maybe_content(self.llm_input)
        if content is not None:
            span.set_attribute(LLMAttributes.INPUT, content)
        span.set_attribute(
            LLMAttributes.LLM_PARAMS, llm_params_attribute(self.params)
        )
        set_caller_attributes(span, self.caller_info)
        self.invocation.input_messages = self._input_messages()
        self._refresh_request()

        run_safely(self._layer.metrics.record_call, self.labels)

    def _input_messages(self) -> list:
        return input_messages_for(self.llm_input)

    def mark_streaming(self) -> None:
        """Mark the span streaming and release the context for the caller.

        The streaming flag only rides on the first-token metric, exactly like
        the framework's own instrumentation; the remaining metrics keep the
        labels the call started with.
        """
        self.streaming = True
        self.layer_span.span.set_attribute(LLMAttributes.STREAMING, True)
        self.invocation.context_token = self.layer_span.take_context_token()
        run_safely(self._layer.handler.detach_llm_context, self.invocation)

    def _finish_first_token(self) -> None:
        duration = self.layer_span.record_first_token()
        run_safely(
            self._layer.metrics.record_first_token,
            duration,
            self.labels,
            True,
        )
        self.layer_span.span.set_attribute(
            LLMAttributes.FIRST_TOKEN_DURATION, duration
        )

    def _record_usage(self, usage: TokenUsage) -> None:
        """Record usage on the metrics and accumulate it exactly once."""
        run_safely(self._layer.metrics.record_tokens, usage, self.labels)
        run_safely(record_token_usage, usage, self.layer_span.span_id)

    def finish_success(self, result: LLMOutput) -> None:
        if self.layer_span is None:
            return
        span = self.layer_span.span
        span.set_attribute(LLMAttributes.STREAMING, False)
        duration = self.layer_span.elapsed()
        usage = self.provider_usage or usage_of(result)
        raw = getattr(result, "raw", None)
        self.invocation.response_id = getattr(result, "response_id", None) or (
            raw.get("id") if isinstance(raw, dict) else None
        )
        if isinstance(raw, dict) and isinstance(raw.get("model"), str):
            self.invocation.response_model_name = raw["model"]

        run_safely(
            Monitor.trace_llm_invocation,
            self.source,
            self.llm_input,
            result.text,
            duration,
        )
        run_safely(
            Monitor().trace_llm_token_usage,
            self.llm_instance,
            self.llm_input,
            result,
        )

        run_safely(self._layer.metrics.record_duration, duration, self.labels)
        run_safely(
            self._layer.metrics.record_first_token,
            duration,
            self.labels,
            False,
        )
        if usage:
            self._record_usage(usage)

        span.set_attribute(LLMAttributes.FIRST_TOKEN_DURATION, duration)
        span.set_attribute(LLMAttributes.DURATION, duration)
        span.set_attribute(LLMAttributes.STATUS, SUCCESS_STATUS)
        content = maybe_content_text(result.text)
        if content is not None:
            span.set_attribute(LLMAttributes.OUTPUT, content)
        if usage:
            set_usage_attributes(span, LLMAttributes, usage)

        self._set_invocation_success(
            result.text,
            usage,
            getattr(result, "finish_reason", None)
            or raw_choice(result).get("finish_reason"),
            tool_calls_of(result),
            getattr(result, "reasoning_text", None)
            or raw_message(result).get("reasoning_content"),
        )

    def finish_stream_success(
        self,
        output: str,
        usage: Optional[TokenUsage],
        tool_calls: Optional[list] = None,
    ) -> None:
        """Finalize a consumed stream. Usage reaches the parent once.

        ``tool_calls`` carries the calls a streamed tool-planning round assembled:
        that round has no text at all, and its calls are its output.
        """
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()
        usage = self.provider_usage or usage

        run_safely(
            Monitor().trace_llm_invocation,
            source=self.source,
            llm_input=self.llm_input,
            llm_output=output,
            cost_time=duration,
        )
        pseudo_result = LLMOutput(text=output, usage=usage)
        run_safely(
            Monitor().trace_llm_token_usage,
            self.llm_instance,
            self.llm_input,
            pseudo_result,
        )

        run_safely(self._layer.metrics.record_duration, duration, self.labels)
        if usage:
            self._record_usage(usage)

        span.set_attribute(LLMAttributes.DURATION, duration)
        span.set_attribute(LLMAttributes.STATUS, SUCCESS_STATUS)
        content = maybe_content_text(output)
        if content is not None:
            span.set_attribute(LLMAttributes.OUTPUT, content)
        if usage:
            set_usage_attributes(span, LLMAttributes, usage)

        self._set_invocation_success(
            output,
            usage,
            self.stream_finish_reason,
            tool_calls,
            "".join(self.reasoning_chunks),
        )

    def _set_invocation_success(
        self,
        output: str,
        usage: Optional[TokenUsage],
        finish_reason: Optional[str] = None,
        tool_calls: Optional[list] = None,
        reasoning: Optional[str] = None,
    ) -> None:
        """Hand the shared handler this call's output, calls included.

        A round that answers with tool calls carries no text; without the calls
        its ``gen_ai.output.messages`` would be empty even though the round did
        produce the tool plan.
        """
        reason = finish_reason or ("tool_calls" if tool_calls else "stop")
        self.invocation.output_messages = output_messages_for(
            output, reason, tool_calls, reasoning
        )
        self.invocation.finish_reasons = [reason]
        if usage is not None:
            self.invocation.input_tokens = usage.prompt_tokens
            self.invocation.output_tokens = usage.completion_tokens
            if usage.cached_in:
                self.invocation.usage_cache_read_input_tokens = usage.cached_in
        self._refresh_request()
        self.layer_span.apply_timing(self.invocation)

    def finish_error(self, error: BaseException) -> None:
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()
        run_safely(
            self._layer.metrics.record_error, error, duration, self.labels
        )
        set_layer_error_attributes(span, LLMAttributes, error, duration)
        set_error_status(span, error_status_text(error))

    def finalize(self, error: Optional[BaseException] = None) -> None:
        if self.finished:
            return
        self.finished = True
        layer_span = self.layer_span
        if layer_span is None:
            return

        run_safely(
            promote_token_usage_to_parent,
            token_usage_for(layer_span.span_id),
            layer_span.parent_span_id,
        )

        if self.invocation.context_token is None:
            self.invocation.context_token = layer_span.take_context_token()
        self.invocation.monotonic_end_s = monotonic_now()
        if error is None:
            run_safely(self._layer.handler.stop_llm, self.invocation)
        else:
            run_safely(
                self._layer.handler.fail_llm,
                self.invocation,
                Error(message=error_status_text(error), type=type(error)),
            )
        # Everything left over is closed by the handler's own cleanup rather
        # than by hand: ``abandon_llm`` detaches whatever token is still pending
        # and ends the span if it is still recording, and it is a no-op on an
        # invocation ``stop_llm``/``fail_llm`` already finalized. The plugin
        # therefore hands the token over before finalizing and owns neither
        # step afterwards.
        run_safely(self._layer.handler.abandon_llm, self.invocation)
        if self.invocation_chain_pushed:
            self.invocation_chain_pushed = False
            pop_invocation_chain()

    # -- streaming ----------------------------------------------------

    def process_sync_stream(self, result: Any) -> ClaimedSyncStream:
        self.mark_streaming()
        return ClaimedSyncStream(self, self._sync_stream(result), result)

    def _sync_stream(self, result: Any) -> Generator[Any, None, None]:
        output: list[str] = []
        usage: Optional[TokenUsage] = None
        tool_calls = ToolCallAccumulator()
        error: Optional[BaseException] = None
        first = True
        try:
            iterator = iter(result)
            while True:
                try:
                    with usage_call_context(self):
                        chunk = next(iterator)
                except StopIteration:
                    break
                if first:
                    first = False
                    run_safely(self._finish_first_token)
                # The chunk is read here, before the caller receives it, so every
                # read is guarded: a collector that fails must not turn a working
                # stream into a failing one, and must not disturb the chunk the
                # caller is waiting for -- it is yielded unchanged and in order.
                text, chunk_usage = run_safely(chunk_record, chunk) or (
                    "",
                    None,
                )
                if text:
                    output.append(text)
                if chunk_usage:
                    usage = chunk_usage
                run_safely(tool_calls.feed, chunk)
                run_safely(self._collect_chunk_metadata, chunk)
                yield chunk
        except GeneratorExit:
            # The caller stopped consuming the stream early: the span still
            # closes exactly once, with whatever the stream produced so far.
            raise
        except BaseException as exc:
            error = exc
            run_safely(self.finish_error, exc)
            raise
        finally:
            if error is None:
                run_safely(
                    self.finish_stream_success,
                    "".join(output),
                    usage,
                    tool_calls.result(),
                )
            run_safely(self.finalize, error)

    def process_async_stream(self, result: Any) -> ClaimedAsyncStream:
        """Claim the stream before the caller gets it, like the sync twin.

        ``mark_streaming`` releases the caller's context token, so it has to run
        now rather than on the caller's first iteration; the returned iterator
        finalizes the span whenever consumption stops.
        """
        self.mark_streaming()
        return ClaimedAsyncStream(self, self._async_stream(result), result)

    async def _async_stream(self, result: Any) -> AsyncGenerator[Any, None]:
        output: list[str] = []
        usage: Optional[TokenUsage] = None
        tool_calls = ToolCallAccumulator()
        error: Optional[BaseException] = None
        first = True
        try:
            iterator = result.__aiter__()
            while True:
                try:
                    with usage_call_context(self):
                        chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                if first:
                    first = False
                    run_safely(self._finish_first_token)
                # Same rule as the sync twin: the collector is guarded, and the
                # chunk the caller awaited is delivered unchanged and in order.
                text, chunk_usage = run_safely(chunk_record, chunk) or (
                    "",
                    None,
                )
                if text:
                    output.append(text)
                if chunk_usage:
                    usage = chunk_usage
                run_safely(tool_calls.feed, chunk)
                run_safely(self._collect_chunk_metadata, chunk)
                yield chunk
        except GeneratorExit:
            # The caller stopped consuming the stream early: the span still
            # closes exactly once, with whatever the stream produced so far.
            raise
        except BaseException as exc:
            error = exc
            run_safely(self.finish_error, exc)
            raise
        finally:
            if error is None:
                run_safely(
                    self.finish_stream_success,
                    "".join(output),
                    usage,
                    tool_calls.result(),
                )
            run_safely(self.finalize, error)


class LLMLayer:
    """The LLM wrapper pair plus the spans and metrics it owns."""

    def __init__(
        self,
        tracer: Any,
        handler: ExtendedTelemetryHandler,
        meter: Meter,
    ) -> None:
        self._tracer = tracer
        self.handler = handler
        self.metrics = LLMMetrics(meter)

    def start_span(self, span_name: str) -> LayerSpan:
        return LayerSpan(self._tracer, span_name, SpanKind.INTERNAL).attach()

    def wrap_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_LLMCall, self, func, args, kwargs)
        if call is None:
            return _apply_llm_plugins(func)(*args, **kwargs)
        run_safely(call.start)
        try:
            with usage_call_context(call):
                result = _apply_llm_plugins(func)(*args, **kwargs)
        except BaseException as exc:
            run_safely(call.finish_error, exc)
            run_safely(call.finalize, exc)
            raise
        if isinstance(result, LLMOutput):
            run_safely(call.finish_success, result)
            run_safely(call.finalize)
            return result
        if call.layer_span is None:
            return result
        stream = run_safely(call.process_sync_stream, result)
        if stream is None:
            run_safely(call.finalize)
            return result
        return stream

    async def wrap_async(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_LLMCall, self, func, args, kwargs)
        if call is None:
            return await _apply_llm_plugins(func)(*args, **kwargs)
        run_safely(call.start)
        try:
            with usage_call_context(call):
                result = await _apply_llm_plugins(func)(*args, **kwargs)
        except BaseException as exc:
            run_safely(call.finish_error, exc)
            run_safely(call.finalize, exc)
            raise
        if isinstance(result, LLMOutput):
            run_safely(call.finish_success, result)
            run_safely(call.finalize)
            return result
        if call.layer_span is None:
            return result
        stream = run_safely(call.process_async_stream, result)
        if stream is None:
            run_safely(call.finalize)
            return result
        return stream
