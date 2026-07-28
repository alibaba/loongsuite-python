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
Telemetry handler for GenAI invocations.

This module exposes the `TelemetryHandler` class, which manages the lifecycle of
GenAI (Generative AI) invocations and emits telemetry data (spans and related attributes).
It supports starting, stopping, and failing LLM invocations.

Classes:
    - TelemetryHandler: Manages GenAI invocation lifecycles and emits telemetry.

Functions:
    - get_telemetry_handler: Returns a singleton `TelemetryHandler` instance.

Usage:
    handler = get_telemetry_handler()

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

# LoongSuite Extension
import logging
import timeit
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry._logs import (
    LoggerProvider,
    get_logger,
)
from opentelemetry.context import (  # LoongSuite Extension
    _RUNTIME_CONTEXT,
    Context,
)
from opentelemetry.metrics import MeterProvider, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import (
    Span,
    SpanKind,
    TracerProvider,
    get_tracer,
    set_span_in_context,
)
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.span_utils import (
    _apply_error_attributes,
    _apply_llm_finish_attributes,
    _maybe_emit_llm_event,
)
from opentelemetry.util.genai.types import (
    Error,
    LLMInvocation,
)
from opentelemetry.util.genai.version import __version__

# LoongSuite Extension
logger = logging.getLogger(__name__)

# LoongSuite Extension
_AGENT_NAME_BAGGAGE_KEY = GenAI.GEN_AI_AGENT_NAME


class _InvocationWithAttributes(Protocol):
    attributes: dict[str, Any]


# LoongSuite Extension
def _safe_detach(token: object) -> None:
    """Safely detach an OpenTelemetry context token.

    In async or cross-thread scenarios the token may already have been
    consumed or belong to a different context, causing ``detach`` to raise.
    This helper uses ``_RUNTIME_CONTEXT.detach`` directly to avoid the
    noisy ERROR log that ``otel_context.detach`` emits before re-raising.
    """
    if token is None:
        return
    try:
        _RUNTIME_CONTEXT.detach(token)  # type: ignore[arg-type]
    except (ValueError, RuntimeError) as exc:
        logger.debug(
            "Context detach failed (cross-thread/async scenario): %s", exc
        )


def _current_context(context: Context | None = None) -> Context:
    if context is not None:
        return context
    return otel_context.get_current()


def _inject_agent_name_from_baggage(
    invocation: _InvocationWithAttributes, context: Context
) -> None:
    if GenAI.GEN_AI_AGENT_NAME in invocation.attributes:
        return
    agent_name = baggage.get_baggage(_AGENT_NAME_BAGGAGE_KEY, context=context)
    if agent_name:
        invocation.attributes[GenAI.GEN_AI_AGENT_NAME] = agent_name


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events.
    """

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        logger_provider: LoggerProvider | None = None,
    ):
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_37_0.value,
        )
        self._metrics_recorder: InvocationMetricsRecorder | None = None
        meter = get_meter(__name__, meter_provider=meter_provider)
        self._metrics_recorder = InvocationMetricsRecorder(meter)
        self._logger = get_logger(
            __name__,
            __version__,
            logger_provider,
            schema_url=Schemas.V1_37_0.value,
        )

    def _record_llm_metrics(
        self,
        invocation: LLMInvocation,
        span: Span | None = None,
        *,
        error_type: str | None = None,
    ) -> None:
        if self._metrics_recorder is None or span is None:
            return
        self._metrics_recorder.record(
            span,
            invocation,
            error_type=error_type,
        )

    def start_llm(
        self,
        invocation: LLMInvocation,
        context: Context | None = None,  # LoongSuite Extension
    ) -> LLMInvocation:
        """Start an LLM invocation and create a pending span entry."""
        invocation._lifecycle_finalized = False
        current_context = _current_context(context)
        _inject_agent_name_from_baggage(invocation, current_context)

        # Create a span and attach it as current; keep the token to detach later
        span = self._tracer.start_span(
            name=f"{invocation.operation_name} {invocation.request_model}",
            kind=SpanKind.CLIENT,
            context=current_context,  # LoongSuite Extension
        )
        invocation.span = span
        try:
            # Record a monotonic start timestamp (seconds) for duration
            # calculation using timeit.default_timer.
            invocation.monotonic_start_s = timeit.default_timer()
            invocation.context_token = otel_context.attach(
                set_span_in_context(span, current_context)
            )
        except Exception:
            self.abandon_llm(invocation)
            raise
        return invocation

    # LoongSuite Extension
    def detach_llm_context(  # pylint: disable=no-self-use
        self, invocation: LLMInvocation
    ) -> LLMInvocation:
        """Detach an invocation from the context where it was started.

        Streaming instrumentations should call this before returning the stream
        to application code. The span remains recording and can be finalized
        later, even when iteration happens in another task or context.
        """
        token = invocation.context_token
        invocation.context_token = None
        _safe_detach(token)
        return invocation

    # LoongSuite Extension
    def abandon_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        """Best-effort cleanup for a partially processed LLM invocation."""
        invocation._lifecycle_finalized = True
        # LoongSuite Extension: cleanup must survive probe-owned failures.
        try:
            self.detach_llm_context(invocation)
        finally:
            span = invocation.span
            if span is not None and span.is_recording():
                span.end()
        return invocation

    def stop_llm(self, invocation: LLMInvocation) -> LLMInvocation:  # pylint: disable=no-self-use
        """Finalize an LLM invocation successfully and end its span."""
        if invocation.span is None or invocation._lifecycle_finalized:
            # TODO: Provide feedback that this invocation was not started
            return invocation

        invocation._lifecycle_finalized = True
        span = invocation.span
        # LoongSuite Extension: cleanup must survive probe-owned failures.
        try:
            _apply_llm_finish_attributes(span, invocation)
            self._record_llm_metrics(invocation, span)
            _maybe_emit_llm_event(self._logger, span, invocation)
        finally:
            try:
                self.detach_llm_context(invocation)
            finally:
                if span.is_recording():
                    span.end()
        return invocation

    def fail_llm(  # pylint: disable=no-self-use
        self, invocation: LLMInvocation, error: Error
    ) -> LLMInvocation:
        """Fail an LLM invocation and end its span with error status."""
        if invocation.span is None or invocation._lifecycle_finalized:
            # TODO: Provide feedback that this invocation was not started
            return invocation

        invocation._lifecycle_finalized = True
        span = invocation.span
        # LoongSuite Extension: cleanup must survive probe-owned failures.
        try:
            _apply_llm_finish_attributes(span, invocation)
            _apply_error_attributes(span, error)
            error_type = getattr(error.type, "__qualname__", None)
            self._record_llm_metrics(invocation, span, error_type=error_type)
            _maybe_emit_llm_event(self._logger, span, invocation, error)
        finally:
            try:
                self.detach_llm_context(invocation)
            finally:
                if span.is_recording():
                    span.end()
        return invocation

    @contextmanager
    def llm(
        self, invocation: LLMInvocation | None = None
    ) -> Iterator[LLMInvocation]:
        """Context manager for LLM invocations.

        Only set data attributes on the invocation object, do not modify the span or context.

        Starts the span on entry. On normal exit, finalizes the invocation and ends the span.
        If an exception occurs inside the context, marks the span as error, ends it, and
        re-raises the original exception.
        """
        if invocation is None:
            invocation = LLMInvocation(
                request_model="",
            )
        self.start_llm(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise
        self.stop_llm(invocation)


def get_telemetry_handler(
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    logger_provider: LoggerProvider | None = None,
) -> TelemetryHandler:
    """
    Returns a singleton TelemetryHandler instance.
    """
    handler: TelemetryHandler | None = getattr(
        get_telemetry_handler, "_default_handler", None
    )
    if handler is None:
        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        setattr(get_telemetry_handler, "_default_handler", handler)
    return handler
