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
from opentelemetry.util.genai.types import Error, LLMInvocation

from ._common import (
    SUCCESS_STATUS,
    LayerSpan,
    LLMAttributes,
    MetricLabels,
    NamePreservingSpan,
    agent_name_attributes,
    build_safely,
    caller_labels,
    detach_context_token,
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

logger = logging.getLogger(__name__)

SPAN_KIND_VALUE = "llm"


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
        self.finished = False
        #: Only a node this call actually added may be popped again.
        self.invocation_chain_pushed = False
        self.invocation = LLMInvocation(
            request_model=self.source,
            provider=self.channel_name or None,
            attributes=agent_name_attributes(),
            temperature=self._temperature(),
        )

    def _temperature(self) -> Optional[float]:
        """The configured temperature, ignoring the framework's -1 sentinel."""
        temperature = (self.params or {}).get("temperature")
        if temperature is None or temperature == -1:
            return None
        return temperature

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

        run_safely(self._layer.metrics.record_call, self.labels)

    def _input_messages(self) -> list:
        prompt = None
        if isinstance(self.llm_input, dict):
            prompt = self.llm_input.get("prompt")
        if isinstance(prompt, str) and prompt:
            return input_messages_for(prompt)
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
        usage = getattr(result, "usage", None)

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
            result.text, usage, getattr(result, "finish_reason", None)
        )

    def finish_stream_success(
        self, output: str, usage: Optional[TokenUsage]
    ) -> None:
        """Finalize a consumed stream. Usage reaches the parent once."""
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()

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

        self._set_invocation_success(output, usage)

    def _set_invocation_success(
        self,
        output: str,
        usage: Optional[TokenUsage],
        finish_reason: Optional[str] = None,
    ) -> None:
        reason = finish_reason or "stop"
        self.invocation.output_messages = output_messages_for(output, reason)
        self.invocation.finish_reasons = [reason]
        if usage:
            if usage.prompt_tokens:
                self.invocation.input_tokens = usage.prompt_tokens
            if usage.completion_tokens:
                self.invocation.output_tokens = usage.completion_tokens
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
        self._release_context_token(layer_span)
        layer_span.end()
        if self.invocation_chain_pushed:
            self.invocation_chain_pushed = False
            pop_invocation_chain()

    def _release_context_token(self, layer_span: LayerSpan) -> None:
        """Detach the span context unless the handler already did.

        The shared handler detaches the token and ends the span itself whenever
        it finalizes an invocation; detaching it a second time would raise. A
        streaming call has already handed its token over before this point.
        """
        if not layer_span.span.is_recording():
            self.invocation.context_token = None
        leftover = self.invocation.context_token
        if leftover is not None:
            self.invocation.context_token = None
            run_safely(detach_context_token, leftover)

    # -- streaming ----------------------------------------------------

    def process_sync_stream(self, result: Any) -> Generator[Any, None, None]:
        self.mark_streaming()
        return self._sync_stream(result)

    def _sync_stream(self, result: Any) -> Generator[Any, None, None]:
        output: list[str] = []
        usage: Optional[TokenUsage] = None
        error: Optional[BaseException] = None
        first = True
        try:
            for chunk in result:
                if first:
                    first = False
                    run_safely(self._finish_first_token)
                output.append(
                    chunk.text if hasattr(chunk, "text") else str(chunk)
                )
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = chunk_usage
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
                run_safely(self.finish_stream_success, "".join(output), usage)
            run_safely(self.finalize, error)

    async def process_async_stream(
        self, result: Any
    ) -> AsyncGenerator[Any, None]:
        self.mark_streaming()
        async for chunk in self._async_stream(result):
            yield chunk

    async def _async_stream(self, result: Any) -> AsyncGenerator[Any, None]:
        output: list[str] = []
        usage: Optional[TokenUsage] = None
        error: Optional[BaseException] = None
        first = True
        try:
            async for chunk in result:
                if first:
                    first = False
                    run_safely(self._finish_first_token)
                output.append(getattr(chunk, "text", None) or str(chunk))
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = chunk_usage
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
                run_safely(self.finish_stream_success, "".join(output), usage)
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
        return call.process_sync_stream(result)

    async def wrap_async(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_LLMCall, self, func, args, kwargs)
        if call is None:
            return await _apply_llm_plugins(func)(*args, **kwargs)
        run_safely(call.start)
        try:
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
        return call.process_async_stream(result)
