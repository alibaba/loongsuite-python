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

"""The agent layer of the LoongSuite agentUniverse instrumentation.

This module claims the ``_agent_wrapper_sync``/``_agent_wrapper_async``
extension points the ``@trace_agent`` decorator resolves, and builds the
``au.agent.*`` span, metrics and token bookkeeping itself. The ``gen_ai.*``
attributes on that same span come from the shared ``ExtendedTelemetryHandler``,
which finalizes the invocation this layer starts.

No agentUniverse OTel class is imported or called here.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Any, Dict, Optional

from agentuniverse.agent.memory.conversation_memory.conversation_memory_module import (  # noqa: E501
    ConversationMemoryModule,
)
from agentuniverse.base.annotation.trace import _get_agent_info
from agentuniverse.llm.llm_output import TokenUsage

from opentelemetry.metrics import Meter
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.types import Error

from ._common import (
    FRAMEWORK_NAME,
    SUCCESS_STATUS,
    AgentAttributes,
    LayerSpan,
    MetricLabels,
    NamePreservingSpan,
    build_safely,
    caller_labels,
    detach_context_token,
    error_status_text,
    input_messages_for,
    maybe_content,
    monotonic_now,
    output_messages_for,
    pop_current_agent_name,
    pop_invocation_chain,
    promote_token_usage_to_parent,
    push_current_agent_name,
    push_invocation_chain,
    run_safely,
    set_caller_attributes,
    set_error_status,
    set_layer_error_attributes,
    set_usage_attributes,
    token_usage_for,
)

logger = logging.getLogger(__name__)

SPAN_KIND_VALUE = "agent"


def _memory_module() -> Any:
    """The framework's conversation memory, or ``None`` when unavailable.

    The framework's own instrumentation builds this module unguarded; this
    package never lets a telemetry side effect break a working agent call, so a
    module that cannot be built (or a memory call that fails) is only logged.
    """
    try:
        return ConversationMemoryModule()
    except Exception:  # pragma: no cover - application config not ready
        logger.debug("ConversationMemoryModule unavailable", exc_info=True)
        return None


class AgentMetrics:
    """The agentUniverse agent metric set, recorded by this package."""

    def __init__(self, meter: Meter) -> None:
        self._calls = meter.create_counter(
            name="agent_calls_total",
            description="Total number of Agent calls",
            unit="1",
        )
        self._errors = meter.create_counter(
            name="agent_errors_total",
            description="Total number of Agent errors",
            unit="1",
        )
        self._duration = meter.create_histogram(
            name="agent_call_duration",
            description="Duration of Agent calls in seconds",
            unit="s",
        )
        self._first_token = meter.create_histogram(
            name="agent_first_token_duration",
            description="Duration of Agent first token in seconds",
            unit="s",
        )
        self._total_tokens = meter.create_histogram(
            name="agent_total_tokens",
            description="Total token nums used in agent",
            unit="1",
        )
        self._prompt_tokens = meter.create_histogram(
            name="agent_prompt_tokens",
            description="Prompt token nums used in agent",
            unit="1",
        )
        self._completion_tokens = meter.create_histogram(
            name="agent_completion_tokens",
            description="Completion token nums used in agent",
            unit="1",
        )
        self._cached_tokens = meter.create_histogram(
            name="agent_cached_tokens",
            description="Cached token nums used in agent",
            unit="1",
        )
        self._reasoning_tokens = meter.create_histogram(
            name="agent_reasoning_tokens",
            description="Reasoning token nums used in agent",
            unit="1",
        )

    def record_call(self, labels: Dict[str, Any]) -> None:
        self._calls.add(1, labels)

    def record_duration(self, duration: float, labels: Dict[str, Any]) -> None:
        self._duration.record(duration, labels)

    def record_first_token(
        self, duration: float, labels: Dict[str, Any]
    ) -> None:
        self._first_token.record(duration, labels)

    def record_error(
        self, error: BaseException, duration: float, labels: Dict[str, Any]
    ) -> None:
        error_labels = {
            **labels,
            MetricLabels.AGENT_STATUS: type(error).__name__,
        }
        self._errors.add(1, error_labels)
        self._duration.record(duration, error_labels)

    def record_tokens(self, usage: TokenUsage, labels: Dict[str, Any]) -> None:
        self._total_tokens.record(usage.total_tokens, labels)
        self._completion_tokens.record(usage.completion_tokens, labels)
        self._prompt_tokens.record(usage.prompt_tokens, labels)
        self._cached_tokens.record(usage.cached_tokens, labels)
        self._reasoning_tokens.record(usage.reasoning_tokens, labels)


class FirstTokenStream:
    """Proxy for the agent's ``output_stream`` queue.

    The agent writes its first token straight onto the caller's queue, so the
    proxy is what turns that write into the first-token timing. Everything else
    is delegated to the wrapped queue.
    """

    def __init__(self, wrapped: Any, on_first_item: Any) -> None:
        self._wrapped = wrapped
        self._on_first_item = on_first_item
        self._fired = False

    def _fire(self) -> None:
        if self._fired:
            return
        self._fired = True
        run_safely(self._on_first_item)

    def put(self, item: Any, block: bool = True, timeout: Any = None) -> Any:
        self._fire()
        return self._wrapped.put(item, block, timeout)

    def put_nowait(self, item: Any) -> Any:
        self._fire()
        return self._wrapped.put_nowait(item)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class AsyncFirstTokenStream(FirstTokenStream):
    """``FirstTokenStream`` for an ``asyncio.Queue``."""

    async def put(self, item: Any) -> Any:  # type: ignore[override]
        self._fire()
        return await self._wrapped.put(item)

    def put_nowait(self, item: Any) -> Any:  # type: ignore[override]
        self._fire()
        return self._wrapped.put_nowait(item)


def wrap_first_token_stream(output_stream: Any, on_first_item: Any) -> Any:
    """Wrap a recognised stream queue, leaving anything else untouched."""
    if isinstance(output_stream, asyncio.Queue):
        return AsyncFirstTokenStream(output_stream, on_first_item)
    if isinstance(output_stream, (queue.Queue, queue.SimpleQueue)):
        return FirstTokenStream(output_stream, on_first_item)
    return output_stream


class _AgentCall:
    """Everything one wrapped agent call needs, sync or async."""

    def __init__(
        self,
        layer: "AgentLayer",
        func: Any,
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> None:
        self._layer = layer
        (
            self.agent_instance,
            self.source,
            self.agent_input,
            self.caller_info,
            self.pair_id,
        ) = _get_agent_info(func, *args, **kwargs)
        self.span_name = f"au.agent.{self.source}"
        self.labels: Dict[str, Any] = {
            MetricLabels.AGENT_NAME: self.source,
            **caller_labels(self.caller_info),
            MetricLabels.AGENT_STATUS: SUCCESS_STATUS,
        }
        self.layer_span: Optional[LayerSpan] = None
        self.invocation = InvokeAgentInvocation(
            provider=FRAMEWORK_NAME, agent_name=self.source
        )
        self.streaming = False
        self.finished = False
        #: Only a node this call actually added may be popped again.
        self.invocation_chain_pushed = False

    # -- lifecycle ----------------------------------------------------

    def start(self, kwargs: Dict[str, Any]) -> None:
        self.layer_span = self._layer.start_span(self.span_name)
        # The chain is keyed by the current trace id, so the span has to be
        # current before this layer's node is added; children read it there.
        if push_invocation_chain(self.source, "agent"):
            self.invocation_chain_pushed = True
        span = self.layer_span.span
        # The shared handler finalizes this span on behalf of the invocation, so
        # it has to know the span and the context token it may detach. The proxy
        # keeps the framework-native span name while it does so.
        self.invocation.span = NamePreservingSpan(span)
        span.set_attribute(AgentAttributes.SPAN_KIND, SPAN_KIND_VALUE)
        span.set_attribute(AgentAttributes.NAME, self.source)
        content = maybe_content(self.agent_input)
        if content is not None:
            span.set_attribute(AgentAttributes.INPUT, content)
        set_caller_attributes(span, self.caller_info)
        span.set_attribute(AgentAttributes.PAIR_ID, self.pair_id)
        self.invocation.input_messages = input_messages_for(self.agent_input)

        run_safely(self._layer.metrics.record_call, self.labels)

        kwargs["memory_source_info"] = self.caller_info
        memory = _memory_module()
        if memory is not None:
            run_safely(
                memory.add_agent_input_info,
                self.caller_info,
                self.agent_instance,
                self.agent_input,
                self.pair_id,
            )

        output_stream = kwargs.get("output_stream")
        if output_stream is not None:
            self.streaming = True
            self.labels[MetricLabels.AGENT_STREAMING] = True
            span.set_attribute(AgentAttributes.STREAMING, True)
            kwargs["output_stream"] = wrap_first_token_stream(
                output_stream, self._on_first_token
            )
        else:
            self.labels[MetricLabels.AGENT_STREAMING] = False
            span.set_attribute(AgentAttributes.STREAMING, False)

    def _on_first_token(self) -> None:
        if self.layer_span is None:
            return
        duration = self.layer_span.record_first_token()
        run_safely(
            self._layer.metrics.record_first_token, duration, self.labels
        )
        self.layer_span.span.set_attribute(
            AgentAttributes.FIRST_TOKEN_DURATION, duration
        )

    def finish_success(self, result: Any) -> None:
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()
        usage = token_usage_for(self.layer_span.span_id)

        run_safely(self._layer.metrics.record_duration, duration, self.labels)
        run_safely(self._layer.metrics.record_tokens, usage, self.labels)
        if not self.streaming:
            run_safely(
                self._layer.metrics.record_first_token, duration, self.labels
            )
            span.set_attribute(AgentAttributes.FIRST_TOKEN_DURATION, duration)

        span.set_attribute(AgentAttributes.DURATION, duration)
        span.set_attribute(AgentAttributes.STATUS, SUCCESS_STATUS)
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        content = maybe_content(payload)
        if content is not None:
            span.set_attribute(AgentAttributes.OUTPUT, content)
        set_usage_attributes(span, AgentAttributes, usage)

        self.invocation.output_messages = output_messages_for(payload)
        if usage.prompt_tokens:
            self.invocation.input_tokens = usage.prompt_tokens
        if usage.completion_tokens:
            self.invocation.output_tokens = usage.completion_tokens
        self.layer_span.apply_timing(self.invocation)
        memory = _memory_module()
        if memory is not None:
            run_safely(
                memory.add_agent_result_info,
                self.agent_instance,
                result,
                self.caller_info,
                self.pair_id,
            )

    def finish_error(self, error: BaseException) -> None:
        if self.layer_span is None:
            return
        span = self.layer_span.span
        duration = self.layer_span.elapsed()
        run_safely(
            self._layer.metrics.record_error, error, duration, self.labels
        )
        run_safely(
            self._layer.metrics.record_tokens,
            token_usage_for(self.layer_span.span_id),
            self.labels,
        )
        set_layer_error_attributes(span, AgentAttributes, error, duration)
        set_usage_attributes(
            span, AgentAttributes, token_usage_for(self.layer_span.span_id)
        )
        set_error_status(span, error_status_text(error))

    def finalize(self, error: Optional[BaseException] = None) -> None:
        """Promote token usage, hand the span to the handler and end it."""
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
            run_safely(self._layer.handler.stop_invoke_agent, self.invocation)
        else:
            run_safely(
                self._layer.handler.fail_invoke_agent,
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


class AgentLayer:
    """The agent wrapper pair plus the spans and metrics it owns."""

    def __init__(
        self,
        tracer: Any,
        handler: ExtendedTelemetryHandler,
        meter: Meter,
    ) -> None:
        self._tracer = tracer
        self.handler = handler
        self.metrics = AgentMetrics(meter)

    def start_span(self, span_name: str) -> LayerSpan:
        return LayerSpan(self._tracer, span_name, SpanKind.INTERNAL).attach()

    def wrap_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_AgentCall, self, func, args, kwargs)
        if call is None:
            return func(*args, **kwargs)
        error: Optional[BaseException] = None
        agent_name_token = None
        try:
            run_safely(call.start, kwargs)
            agent_name_token = push_current_agent_name(call.source)
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
            pop_current_agent_name(agent_name_token)

    async def wrap_async(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        call = build_safely(_AgentCall, self, func, args, kwargs)
        if call is None:
            return await func(*args, **kwargs)
        error: Optional[BaseException] = None
        agent_name_token = None
        try:
            run_safely(call.start, kwargs)
            agent_name_token = push_current_agent_name(call.source)
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
            pop_current_agent_name(agent_name_token)
