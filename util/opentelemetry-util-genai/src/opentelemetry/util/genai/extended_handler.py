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

"""
Extended Telemetry Handler for GenAI invocations.

This module extends the base `TelemetryHandler` to support additional GenAI (Generative AI)
invocations such as embedding, execute_tool, invoke_agent, rerank, and memory operations,
which are not supported by the base handler.

This is an extension module that does not modify the original `handler.py`,
allowing for easy upstream synchronization without conflicts.

Classes:
    - ExtendedTelemetryHandler: Extended Telemetry Handler that supports additional GenAI invocations.

Functions:
    - get_extended_telemetry_handler: Returns a singleton `ExtendedTelemetryHandler` instance.

Usage:
    handler = get_extended_telemetry_handler()

    # Create an invocation object with your request data
    # The span and context_token attributes are set by the TelemetryHandler, and
    # managed by the TelemetryHandler during the lifecycle of the span.

    # Use the context manager to manage the lifecycle of an LLM invocation.
    with handler.llm(invocation) as invocation:
        # Populate outputs and any additional attributes
        invocation.output_messages = [...]
        invocation.attributes.update({"more": "attrs"})

    # Or, if you prefer to manage the lifecycle manually
    invocation = LLMInvocation(
        request_model="my-model",
        input_messages=[...],
        provider="my-provider",
        attributes={"custom": "attr"},
    )

    # Start the invocation (opens a span)
    handler.start_llm(invocation)

    # Populate outputs and any additional attributes, then stop (closes the span)
    invocation.output_messages = [...]
    invocation.attributes.update({"more": "attrs"})
    handler.stop_llm(invocation)

    # Or, in case of error
    handler.fail_llm(invocation, Error(type="...", message="..."))
"""

from __future__ import annotations

import timeit
from contextlib import contextmanager
from typing import Iterator, Optional, Union

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry._logs import LoggerProvider
from opentelemetry.context import Context
from opentelemetry.metrics import MeterProvider, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import (
    Span,
    SpanKind,
    TracerProvider,
    set_span_in_context,
)
from opentelemetry.util.genai._multimodal_processing import (
    MultimodalProcessingMixin,
)
from opentelemetry.util.genai.extended_memory import (
    MemoryInvocation,
    _apply_memory_finish_attributes,
    _maybe_emit_memory_event,
)
from opentelemetry.util.genai.extended_metrics import (
    ExtendedInvocationMetricsRecorder,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID as _GEN_AI_SESSION_ID,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_USER_ID as _GEN_AI_USER_ID,
)
from opentelemetry.util.genai.extended_span_utils import (
    _apply_create_agent_finish_attributes,
    _apply_embedding_finish_attributes,
    _apply_entry_finish_attributes,
    _apply_execute_tool_finish_attributes,
    _apply_invoke_agent_finish_attributes,
    _apply_react_step_finish_attributes,
    _apply_rerank_finish_attributes,
    _apply_retrieval_finish_attributes,
    _maybe_emit_invoke_agent_event,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
    RerankInvocation,
    RetrievalInvocation,
)
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    _current_context,
    _inject_agent_name_from_baggage,
    _safe_detach,
)
from opentelemetry.util.genai.span_utils import _apply_error_attributes
from opentelemetry.util.genai.types import Error, LLMInvocation

_ENTRY_ACTIVE_CONTEXT_KEY = "loongsuite.gen_ai.entry.active"


def is_entry_context_active(context: Context | None = None) -> bool:
    """Return whether the current context is inside a LoongSuite ENTRY."""

    return bool(
        otel_context.get_value(_ENTRY_ACTIVE_CONTEXT_KEY, context=context)
    )


class ExtendedTelemetryHandler(MultimodalProcessingMixin, TelemetryHandler):  # pylint: disable=too-many-public-methods
    """
    Extended Telemetry Handler that supports additional GenAI operations.

    This class extends the base TelemetryHandler to support:
    - Create agent operations
    - Embedding operations
    - Execute tool operations
    - Invoke agent operations
    - Retrieval operations
    - Rerank documents operations
    - Memory operations
    - Entry operations (AI application system entry point)
    - ReAct Step operations (Reasoning-Acting iteration)
    - All operations supported by the base TelemetryHandler (LLM/chat)
    - Async multimodal processing (via MultimodalProcessingMixin)
    """

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        logger_provider: LoggerProvider | None = None,
    ):
        """Initialize the extended telemetry handler with metrics support."""
        super().__init__(tracer_provider, meter_provider, logger_provider)

        # Replace the base metrics recorder with extended one
        meter = get_meter(__name__, meter_provider=meter_provider)
        self._metrics_recorder = ExtendedInvocationMetricsRecorder(meter)

        # Initialize multimodal processing (from Mixin)
        self._init_multimodal()
        self.__class__._ensure_multimodal_shutdown_atexit_registered()  # pylint: disable=no-member

    # ==================== Metrics Helper ====================

    def _record_extended_metrics(
        self,
        span: Span,
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            ExecuteToolInvocation,
            InvokeAgentInvocation,
            CreateAgentInvocation,
            RetrievalInvocation,
            RerankInvocation,
            MemoryInvocation,
            EntryInvocation,
            ReactStepInvocation,
        ],
        *,
        error_type: str | None = None,
    ) -> None:
        """Record extended metrics for any invocation type."""
        if self._metrics_recorder is not None and isinstance(
            self._metrics_recorder, ExtendedInvocationMetricsRecorder
        ):
            self._metrics_recorder.record_extended(
                span, invocation, error_type=error_type
            )

    # ==================== LLM Operations Override (Async Multimodal) ====================
    # Note: start_llm is inherited from TelemetryHandler.
    # We only override stop_llm/fail_llm for async multimodal processing.

    def stop_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        """Stop an LLM invocation, with async multimodal processing if needed.

        For invocations with multimodal data, this method:
        1. Records the actual end time
        2. Detaches context immediately (non-blocking)
        3. Queues the multimodal processing for async execution

        Principle: Never block the user application.
        """
        if invocation.context_token is None or invocation.span is None:
            return invocation

        # Record actual end time
        invocation.monotonic_end_s = timeit.default_timer()

        # Try async multimodal processing
        if self.process_multimodal_stop(invocation, method="stop_llm"):  # pylint: disable=unexpected-keyword-arg
            return invocation

        # No multimodal: use parent's sync path
        return super().stop_llm(invocation)

    def fail_llm(
        self, invocation: LLMInvocation, error: Error
    ) -> LLMInvocation:
        """Fail an LLM invocation, with async multimodal processing if needed.

        Similar to stop_llm but includes error handling.
        Principle: Never block the user application.
        """
        if invocation.context_token is None or invocation.span is None:
            return invocation

        invocation.monotonic_end_s = timeit.default_timer()

        # Try async multimodal processing
        if self.process_multimodal_fail(invocation, error, method="fail_llm"):  # pylint: disable=unexpected-keyword-arg
            return invocation

        # No multimodal: use parent's sync path
        return super().fail_llm(invocation, error)

    # ==================== Create Agent Operations ====================

    def start_create_agent(
        self,
        invocation: CreateAgentInvocation,
        context: Context | None = None,
    ) -> CreateAgentInvocation:
        """Start an agent creation invocation and create a pending span entry."""
        if invocation.agent_name:
            span_name = f"{GenAI.GenAiOperationNameValues.CREATE_AGENT.value} {invocation.agent_name}"
        else:
            span_name = GenAI.GenAiOperationNameValues.CREATE_AGENT.value

        # Create agent is typically a CLIENT operation to remote services
        span = self._tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_create_agent(  # pylint: disable=no-self-use
        self, invocation: CreateAgentInvocation
    ) -> CreateAgentInvocation:
        """Finalize an agent creation invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_create_agent_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_create_agent(  # pylint: disable=no-self-use
        self, invocation: CreateAgentInvocation, error: Error
    ) -> CreateAgentInvocation:
        """Fail an agent creation invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_create_agent_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def create_agent(
        self, invocation: CreateAgentInvocation | None = None
    ) -> Iterator[CreateAgentInvocation]:
        """Context manager for agent creation invocations."""
        if invocation is None:
            invocation = CreateAgentInvocation(provider="")
        self.start_create_agent(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_create_agent(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_create_agent(invocation)

    # ==================== Embedding Operations ====================

    def start_embedding(
        self,
        invocation: EmbeddingInvocation,
        context: Context | None = None,
    ) -> EmbeddingInvocation:
        """Start an embedding invocation and create a pending span entry."""
        span = self._tracer.start_span(
            name=f"{GenAI.GenAiOperationNameValues.EMBEDDINGS.value} {invocation.request_model}",
            kind=SpanKind.CLIENT,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_embedding(  # pylint: disable=no-self-use
        self, invocation: EmbeddingInvocation
    ) -> EmbeddingInvocation:
        """Finalize an embedding invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_embedding_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_embedding(  # pylint: disable=no-self-use
        self, invocation: EmbeddingInvocation, error: Error
    ) -> EmbeddingInvocation:
        """Fail an embedding invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_embedding_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def embedding(
        self, invocation: EmbeddingInvocation | None = None
    ) -> Iterator[EmbeddingInvocation]:
        """Context manager for embedding invocations."""
        if invocation is None:
            invocation = EmbeddingInvocation(request_model="")
        self.start_embedding(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_embedding(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_embedding(invocation)

    # ==================== Execute Tool Operations ====================

    def start_execute_tool(
        self,
        invocation: ExecuteToolInvocation,
        context: Context | None = None,
    ) -> ExecuteToolInvocation:
        """Start a tool execution invocation and create a pending span entry."""
        span = self._tracer.start_span(
            name=f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {invocation.tool_name}",
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_execute_tool(  # pylint: disable=no-self-use
        self, invocation: ExecuteToolInvocation
    ) -> ExecuteToolInvocation:
        """Finalize a tool execution invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_execute_tool_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_execute_tool(  # pylint: disable=no-self-use
        self, invocation: ExecuteToolInvocation, error: Error
    ) -> ExecuteToolInvocation:
        """Fail a tool execution invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_execute_tool_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def execute_tool(
        self, invocation: ExecuteToolInvocation | None = None
    ) -> Iterator[ExecuteToolInvocation]:
        """Context manager for tool execution invocations."""
        if invocation is None:
            invocation = ExecuteToolInvocation(tool_name="")
        self.start_execute_tool(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_execute_tool(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_execute_tool(invocation)

    # ==================== Invoke Agent Operations ====================

    def start_invoke_agent(
        self,
        invocation: InvokeAgentInvocation,
        context: Context | None = None,
    ) -> InvokeAgentInvocation:
        """Start an agent invocation and create a pending span entry."""
        if invocation.agent_name:
            span_name = f"{GenAI.GenAiOperationNameValues.INVOKE_AGENT.value} {invocation.agent_name}"
        else:
            span_name = GenAI.GenAiOperationNameValues.INVOKE_AGENT.value

        # Span kind should be INTERNAL for in-process agents, CLIENT for remote services
        # Default to INTERNAL as most frameworks run agents in-process
        span = self._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        ctx = set_span_in_context(span, current_context)
        if invocation.agent_name:
            ctx = baggage.set_baggage(
                GenAI.GEN_AI_AGENT_NAME,
                invocation.agent_name,
                ctx,
            )
        invocation.context_token = otel_context.attach(ctx)
        return invocation

    def stop_invoke_agent(
        self, invocation: InvokeAgentInvocation
    ) -> InvokeAgentInvocation:  # pylint: disable=no-self-use
        """Finalize an agent invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        # Record actual end time
        invocation.monotonic_end_s = timeit.default_timer()

        # Try async multimodal processing
        if self.process_multimodal_stop(invocation, method="stop_agent"):  # pylint: disable=unexpected-keyword-arg
            return invocation

        # No multimodal: sync path
        _apply_invoke_agent_finish_attributes(invocation.span, invocation)
        _maybe_emit_invoke_agent_event(
            self._logger, invocation.span, invocation
        )
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_invoke_agent(  # pylint: disable=no-self-use
        self, invocation: InvokeAgentInvocation, error: Error
    ) -> InvokeAgentInvocation:
        """Fail an agent invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        # Record actual end time
        invocation.monotonic_end_s = timeit.default_timer()

        # Try async multimodal processing
        if self.process_multimodal_fail(  # pylint: disable=unexpected-keyword-arg
            invocation, error, method="fail_agent"
        ):
            return invocation

        # No multimodal: sync path
        span = invocation.span
        _apply_invoke_agent_finish_attributes(span, invocation)
        _apply_error_attributes(span, error)
        _maybe_emit_invoke_agent_event(self._logger, span, invocation, error)  # pylint: disable=too-many-function-args
        self._record_extended_metrics(
            span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        span.end()
        return invocation

    @contextmanager
    def invoke_agent(
        self, invocation: InvokeAgentInvocation | None = None
    ) -> Iterator[InvokeAgentInvocation]:
        """Context manager for agent invocations."""
        if invocation is None:
            invocation = InvokeAgentInvocation(provider="")
        self.start_invoke_agent(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_invoke_agent(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_invoke_agent(invocation)

    # ==================== Retrieval Operations ====================

    def start_retrieval(
        self,
        invocation: RetrievalInvocation,
        context: Context | None = None,
    ) -> RetrievalInvocation:
        """Start a retrieval invocation and create a pending span entry."""
        span = self._tracer.start_span(
            name="retrieval",
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_retrieval(  # pylint: disable=no-self-use
        self, invocation: RetrievalInvocation
    ) -> RetrievalInvocation:
        """Finalize a retrieval invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_retrieval_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_retrieval(  # pylint: disable=no-self-use
        self, invocation: RetrievalInvocation, error: Error
    ) -> RetrievalInvocation:
        """Fail a retrieval invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_retrieval_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def retrieval(
        self, invocation: RetrievalInvocation | None = None
    ) -> Iterator[RetrievalInvocation]:
        """Context manager for retrieval invocations."""
        if invocation is None:
            invocation = RetrievalInvocation()
        self.start_retrieval(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_retrieval(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_retrieval(invocation)

    # ==================== Rerank Documents Operations ====================

    def start_rerank(
        self,
        invocation: RerankInvocation,
        context: Context | None = None,
    ) -> RerankInvocation:
        """Start a rerank documents invocation and create a pending span entry."""
        span = self._tracer.start_span(
            name="rerank_documents",
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_rerank(self, invocation: RerankInvocation) -> RerankInvocation:  # pylint: disable=no-self-use
        """Finalize a rerank documents invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_rerank_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_rerank(  # pylint: disable=no-self-use
        self, invocation: RerankInvocation, error: Error
    ) -> RerankInvocation:
        """Fail a rerank documents invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_rerank_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def rerank(
        self, invocation: RerankInvocation | None = None
    ) -> Iterator[RerankInvocation]:
        """Context manager for rerank documents invocations."""
        if invocation is None:
            invocation = RerankInvocation(provider="")
        self.start_rerank(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_rerank(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_rerank(invocation)

    # ==================== Memory Operations ====================

    def start_memory(
        self,
        invocation: MemoryInvocation,
        context: Context | None = None,
    ) -> MemoryInvocation:
        """Start a memory operation invocation and create a pending span entry."""
        span_name = f"memory_operation {invocation.operation}"

        # Memory operations are CLIENT operations to remote services
        span = self._tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            context=context,
        )
        # Record a monotonic start timestamp (seconds) for duration
        # calculation using timeit.default_timer.
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_memory(self, invocation: MemoryInvocation) -> MemoryInvocation:  # pylint: disable=no-self-use
        """Finalize a memory operation invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_memory_finish_attributes(invocation.span, invocation)
        _maybe_emit_memory_event(self._logger, invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_memory(  # pylint: disable=no-self-use
        self, invocation: MemoryInvocation, error: Error
    ) -> MemoryInvocation:
        """Fail a memory operation invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        span = invocation.span
        _apply_memory_finish_attributes(span, invocation)
        _apply_error_attributes(span, error)
        _maybe_emit_memory_event(self._logger, span, invocation, error)  # pylint: disable=too-many-function-args
        _safe_detach(invocation.context_token)
        span.end()
        return invocation

    @contextmanager
    def memory(
        self, invocation: MemoryInvocation | None = None
    ) -> Iterator[MemoryInvocation]:
        """Context manager for memory operation invocations."""
        if invocation is None:
            invocation = MemoryInvocation(operation="")
        self.start_memory(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_memory(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_memory(invocation)

    # ==================== Entry Operations ====================

    def start_entry(
        self,
        invocation: EntryInvocation,
        context: Context | None = None,
    ) -> EntryInvocation:
        """Start an entry invocation and create a pending span entry.

        Entry identifies the call entry point to an AI application system.
        Span name: enter_ai_application_system (per LoongSuite semantic conventions).

        If ``session_id`` or ``user_id`` are set on the invocation, they are
        also propagated into Baggage (keys ``gen_ai.session.id`` /
        ``gen_ai.user.id``).  Combined with a ``BaggageSpanProcessor``, this
        enables traffic coloring: every child span created inside the entry
        block will automatically inherit these values as span attributes.

        .. note::

           For baggage propagation to take effect on child spans,
           ``session_id`` / ``user_id`` must be set **before**
           ``start_entry`` is called (i.e. at construction time).
        """
        span = self._tracer.start_span(
            name="enter_ai_application_system",
            kind=SpanKind.INTERNAL,
            context=context,
        )
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span

        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        ctx = set_span_in_context(span, current_context)
        ctx = otel_context.set_value(_ENTRY_ACTIVE_CONTEXT_KEY, True, ctx)
        if invocation.session_id is not None:
            ctx = baggage.set_baggage(
                _GEN_AI_SESSION_ID, invocation.session_id, ctx
            )
        if invocation.user_id is not None:
            ctx = baggage.set_baggage(_GEN_AI_USER_ID, invocation.user_id, ctx)
        invocation.context_token = otel_context.attach(ctx)
        return invocation

    def stop_entry(self, invocation: EntryInvocation) -> EntryInvocation:  # pylint: disable=no-self-use
        """Finalize an entry invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_entry_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_entry(
        self, invocation: EntryInvocation, error: Error
    ) -> EntryInvocation:  # pylint: disable=no-self-use
        """Fail an entry invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_entry_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def entry(
        self, invocation: EntryInvocation | None = None
    ) -> Iterator[EntryInvocation]:
        """Context manager for entry invocations."""
        if invocation is None:
            invocation = EntryInvocation()
        self.start_entry(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_entry(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_entry(invocation)

    # ==================== ReAct Step Operations ====================

    def start_react_step(
        self,
        invocation: ReactStepInvocation,
        context: Context | None = None,
    ) -> ReactStepInvocation:
        """Start a ReAct step invocation and create a pending span entry.

        ReAct Step identifies one Reasoning-Acting iteration in an Agent.
        Span name: react step (per LoongSuite semantic conventions).
        """
        span = self._tracer.start_span(
            name="react step",
            kind=SpanKind.INTERNAL,
            context=context,
        )
        invocation.monotonic_start_s = timeit.default_timer()
        invocation.span = span
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)
        invocation.context_token = otel_context.attach(
            set_span_in_context(span, current_context)
        )
        return invocation

    def stop_react_step(
        self, invocation: ReactStepInvocation
    ) -> ReactStepInvocation:  # pylint: disable=no-self-use
        """Finalize a ReAct step invocation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_react_step_finish_attributes(invocation.span, invocation)
        self._record_extended_metrics(invocation.span, invocation)

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    def fail_react_step(
        self, invocation: ReactStepInvocation, error: Error
    ) -> ReactStepInvocation:  # pylint: disable=no-self-use
        """Fail a ReAct step invocation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        _apply_react_step_finish_attributes(invocation.span, invocation)
        _apply_error_attributes(invocation.span, error)
        self._record_extended_metrics(
            invocation.span, invocation, error_type=error.type.__qualname__
        )

        _safe_detach(invocation.context_token)
        invocation.span.end()
        return invocation

    @contextmanager
    def react_step(
        self, invocation: ReactStepInvocation | None = None
    ) -> Iterator[ReactStepInvocation]:
        """Context manager for ReAct step invocations."""
        if invocation is None:
            invocation = ReactStepInvocation()
        self.start_react_step(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_react_step(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise
        self.stop_react_step(invocation)


def get_extended_telemetry_handler(
    tracer_provider: TracerProvider | None = None,
    logger_provider: LoggerProvider | None = None,
) -> ExtendedTelemetryHandler:
    """
    Returns a singleton ExtendedTelemetryHandler instance.
    This handler supports all operations from the base TelemetryHandler
    plus additional operations like embedding and rerank.
    """
    handler: Optional[ExtendedTelemetryHandler] = getattr(
        get_extended_telemetry_handler, "_default_handler", None
    )
    if handler is None:
        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )
        setattr(get_extended_telemetry_handler, "_default_handler", handler)
    return handler
