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

"""The tool layer of the LoongSuite agentUniverse instrumentation.

This module claims the ``_tool_wrapper_sync``/``_tool_wrapper_async`` extension
points the ``@trace_tool`` decorator resolves and builds the ``au.tool.*`` span,
metrics and token bookkeeping itself. ``gen_ai.*`` attributes and metrics on the
same span come from the shared ``ExtendedTelemetryHandler``.

No agentUniverse OTel class is imported or called here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agentuniverse.agent.memory.conversation_memory.conversation_memory_module import (  # noqa: E501
    ConversationMemoryModule,
)
from agentuniverse.base.annotation.trace import _get_tool_info
from agentuniverse.llm.llm_output import TokenUsage

from opentelemetry.metrics import Meter
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
from opentelemetry.util.genai.types import Error

from ._common import (
    SUCCESS_STATUS,
    LayerSpan,
    MetricLabels,
    NamePreservingSpan,
    ToolAttributes,
    agent_name_attributes,
    build_safely,
    caller_labels,
    detach_context_token,
    error_status_text,
    maybe_content,
    monotonic_now,
    pop_invocation_chain,
    promote_token_usage_to_parent,
    push_invocation_chain,
    run_safely,
    set_caller_attributes,
    set_error_status,
    set_layer_error_attributes,
    set_usage_attributes,
    token_usage_for,
)

logger = logging.getLogger(__name__)

SPAN_KIND_VALUE = "tool"

#: The framework exposes tools as plain callables to the model.
TOOL_TYPE = "function"


def _memory_module() -> Any:
    """The framework's conversation memory, or ``None`` when unavailable."""
    try:
        return ConversationMemoryModule()
    except Exception:  # pragma: no cover - application config not ready
        logger.debug("ConversationMemoryModule unavailable", exc_info=True)
        return None


class ToolMetrics:
    """The agentUniverse tool metric set, recorded by this package."""

    def __init__(self, meter: Meter) -> None:
        self._calls = meter.create_counter(
            name="tool_calls_total",
            description="Total number of Tool calls",
            unit="1",
        )
        self._errors = meter.create_counter(
            name="tool_errors_total",
            description="Total number of Tool errors",
            unit="1",
        )
        self._duration = meter.create_histogram(
            name="tool_call_duration",
            description="Duration of Tool calls in seconds",
            unit="s",
        )
        self._total_tokens = meter.create_histogram(
            name="tool_total_tokens",
            description="Total token nums used in tool",
            unit="1",
        )
        self._prompt_tokens = meter.create_histogram(
            name="tool_prompt_tokens",
            description="Prompt token nums used in tool",
            unit="1",
        )
        self._completion_tokens = meter.create_histogram(
            name="tool_completion_tokens",
            description="Completion token nums used in tool",
            unit="1",
        )
        self._cached_tokens = meter.create_histogram(
            name="tool_cached_tokens",
            description="Cached token nums used in tool",
            unit="1",
        )
        self._reasoning_tokens = meter.create_histogram(
            name="tool_reasoning_tokens",
            description="Reasoning token nums used in tool",
            unit="1",
        )

    def record_call(self, labels: Dict[str, Any]) -> None:
        self._calls.add(1, labels)

    def record_duration(self, duration: float, labels: Dict[str, Any]) -> None:
        self._duration.record(duration, labels)

    def record_error(
        self, error: BaseException, duration: float, labels: Dict[str, Any]
    ) -> None:
        error_labels = {
            **labels,
            MetricLabels.TOOL_STATUS: type(error).__name__,
        }
        self._errors.add(1, error_labels)
        self._duration.record(duration, error_labels)

    def record_tokens(self, usage: TokenUsage, labels: Dict[str, Any]) -> None:
        self._total_tokens.record(usage.total_tokens, labels)
        self._completion_tokens.record(usage.completion_tokens, labels)
        self._prompt_tokens.record(usage.prompt_tokens, labels)
        self._cached_tokens.record(usage.cached_tokens, labels)
        self._reasoning_tokens.record(usage.reasoning_tokens, labels)


class _ToolCall:
    """Everything one wrapped tool call needs, sync or async."""

    def __init__(
        self,
        layer: "ToolLayer",
        func: Any,
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> None:
        self._layer = layer
        (
            self.tool_instance,
            self.source,
            self.tool_input,
            self.caller_info,
            self.pair_id,
        ) = _get_tool_info(func, *args, **kwargs)
        self.span_name = f"au.tool.{self.source}"
        self.labels: Dict[str, Any] = {
            MetricLabels.TOOL_NAME: self.source,
            **caller_labels(self.caller_info),
            MetricLabels.TOOL_STATUS: SUCCESS_STATUS,
        }
        self.layer_span: Optional[LayerSpan] = None
        self.finished = False
        #: Only a node this call actually added may be popped again.
        self.invocation_chain_pushed = False
        self.invocation = ExecuteToolInvocation(
            tool_name=self.source,
            tool_type=TOOL_TYPE,
            tool_call_id=self.pair_id,
            attributes=agent_name_attributes(),
        )

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        self.layer_span = self._layer.start_span(self.span_name)
        # The chain is keyed by the current trace id, so the span has to be
        # current before this layer's node is added; children read it there.
        if push_invocation_chain(self.source, "tool"):
            self.invocation_chain_pushed = True
        span = self.layer_span.span
        # The shared handler finalizes this span on behalf of the invocation, so
        # it has to know the span and the context token it may detach. The proxy
        # keeps the framework-native span name while it does so.
        self.invocation.span = NamePreservingSpan(span)
        span.set_attribute(ToolAttributes.SPAN_KIND, SPAN_KIND_VALUE)
        span.set_attribute(ToolAttributes.NAME, self.source)
        content = maybe_content(self.tool_input)
        if content is not None:
            span.set_attribute(ToolAttributes.INPUT, content)
        set_caller_attributes(span, self.caller_info)
        span.set_attribute(ToolAttributes.PAIR_ID, self.pair_id)

        run_safely(self._layer.metrics.record_call, self.labels)
        memory = _memory_module()
        if memory is not None:
            run_safely(
                memory.add_tool_input_info,
                self.caller_info,
                self.source,
                self.tool_input,
                self.pair_id,
            )

    def finish_success(self, result: Any) -> None:
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()
        usage = token_usage_for(self.layer_span.span_id)

        memory = _memory_module()
        if memory is not None:
            run_safely(
                memory.add_tool_output_info,
                self.caller_info,
                self.source,
                params=result,
                pair_id=self.pair_id,
            )
        run_safely(self._layer.metrics.record_duration, duration, self.labels)
        run_safely(self._layer.metrics.record_tokens, usage, self.labels)

        span.set_attribute(ToolAttributes.DURATION, duration)
        span.set_attribute(ToolAttributes.STATUS, SUCCESS_STATUS)
        content = maybe_content(result)
        if content is not None:
            span.set_attribute(ToolAttributes.OUTPUT, content)
        set_usage_attributes(span, ToolAttributes, usage)

        self.invocation.tool_call_arguments = self.tool_input
        self.invocation.tool_call_result = result

    def finish_error(self, error: BaseException) -> None:
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()
        usage = token_usage_for(self.layer_span.span_id)
        run_safely(
            self._layer.metrics.record_error, error, duration, self.labels
        )
        run_safely(
            self._layer.metrics.record_tokens,
            usage,
            self.labels,
        )
        set_layer_error_attributes(span, ToolAttributes, error, duration)
        # The framework records the (zero) usage of a failed tool call too.
        set_usage_attributes(span, ToolAttributes, usage)
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
        if self.invocation_chain_pushed:
            self.invocation_chain_pushed = False
            pop_invocation_chain()
        self.invocation.context_token = layer_span.take_context_token()
        self.invocation.monotonic_end_s = monotonic_now()
        if error is None:
            run_safely(self._layer.handler.stop_execute_tool, self.invocation)
        else:
            run_safely(
                self._layer.handler.fail_execute_tool,
                self.invocation,
                Error(message=error_status_text(error), type=type(error)),
            )
        self._release_context_token(layer_span)
        layer_span.end()

    def _release_context_token(self, layer_span: LayerSpan) -> None:
        """Detach the span context unless the handler already did.

        The shared handler detaches the token and ends the span itself whenever
        it finalizes an invocation; detaching it a second time would raise. If
        the handler did not finish the span, the token is still ours to drop.
        """
        if not layer_span.span.is_recording():
            self.invocation.context_token = None
        leftover = self.invocation.context_token
        if leftover is not None:
            self.invocation.context_token = None
            run_safely(detach_context_token, leftover)


class ToolLayer:
    """The tool wrapper pair plus the spans and metrics it owns."""

    def __init__(
        self,
        tracer: Any,
        handler: ExtendedTelemetryHandler,
        meter: Meter,
    ) -> None:
        self._tracer = tracer
        self.handler = handler
        self.metrics = ToolMetrics(meter)

    def start_span(self, span_name: str) -> LayerSpan:
        return LayerSpan(self._tracer, span_name, SpanKind.INTERNAL).attach()

    def wrap_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_ToolCall, self, func, args, kwargs)
        if call is None:
            return func(*args, **kwargs)
        error: Optional[BaseException] = None
        try:
            run_safely(call.start)
            result = func(*args, **kwargs)
        except BaseException as exc:
            error = exc
            run_safely(call.finish_error, exc)
            raise
        else:
            run_safely(call.finish_success, result)
            return result
        finally:
            run_safely(call.finalize, error)

    async def wrap_async(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_ToolCall, self, func, args, kwargs)
        if call is None:
            return await func(*args, **kwargs)
        error: Optional[BaseException] = None
        try:
            run_safely(call.start)
            result = await func(*args, **kwargs)
        except BaseException as exc:
            error = exc
            run_safely(call.finish_error, exc)
            raise
        else:
            run_safely(call.finish_success, result)
            return result
        finally:
            run_safely(call.finalize, error)
