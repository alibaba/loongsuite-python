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

from unittest.mock import patch

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, LLMInvocation


@pytest.fixture(name="handler_and_exporter")
def fixture_handler_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TelemetryHandler(tracer_provider=provider), exporter


@pytest.mark.parametrize("method", ["stop_llm", "fail_llm"])
def test_finalize_failure_still_detaches_and_ends_span(
    handler_and_exporter, method
):
    handler, exporter = handler_and_exporter
    before = trace.get_current_span()
    invocation = LLMInvocation(request_model="model")
    handler.start_llm(invocation)

    with patch.object(
        handler,
        "_record_llm_metrics",
        side_effect=RuntimeError("probe metrics boom"),
    ):
        with pytest.raises(RuntimeError, match="probe metrics boom"):
            if method == "stop_llm":
                handler.stop_llm(invocation)
            else:
                handler.fail_llm(
                    invocation,
                    Error(message="business boom", type=RuntimeError),
                )

    assert invocation.context_token is None
    assert trace.get_current_span() is before
    assert invocation.span is not None
    assert invocation.span.is_recording() is False
    assert len(exporter.get_finished_spans()) == 1


def test_start_attach_failure_ends_partial_span(handler_and_exporter):
    handler, exporter = handler_and_exporter
    invocation = LLMInvocation(request_model="model")

    with patch(
        "opentelemetry.util.genai.handler.otel_context.attach",
        side_effect=RuntimeError("probe attach boom"),
    ):
        with pytest.raises(RuntimeError, match="probe attach boom"):
            handler.start_llm(invocation)

    assert invocation.context_token is None
    assert invocation.span is not None
    assert invocation.span.is_recording() is False
    assert len(exporter.get_finished_spans()) == 1


def test_detached_stream_span_can_be_finalized_in_another_context(
    handler_and_exporter,
):
    handler, exporter = handler_and_exporter
    invocation = LLMInvocation(request_model="model")
    handler.start_llm(invocation)
    handler.detach_llm_context(invocation)

    handler.stop_llm(invocation)

    assert invocation.context_token is None
    assert invocation.span is not None
    assert invocation.span.is_recording() is False
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("method", ["stop_llm", "fail_llm"])
def test_llm_finalization_is_idempotent(handler_and_exporter, method):
    handler, exporter = handler_and_exporter
    invocation = LLMInvocation(request_model="model")
    handler.start_llm(invocation)

    with patch.object(
        handler, "_record_llm_metrics", wraps=handler._record_llm_metrics
    ) as record_metrics:
        if method == "stop_llm":
            handler.stop_llm(invocation)
            handler.stop_llm(invocation)
        else:
            error = Error(message="business boom", type=RuntimeError)
            handler.fail_llm(invocation, error)
            handler.fail_llm(invocation, error)

    assert record_metrics.call_count == 1
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("method", ["stop_llm", "fail_llm"])
def test_multimodal_dispatch_failure_allows_finalize_retry(
    handler_and_exporter, method
):
    _handler, exporter = handler_and_exporter
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = ExtendedTelemetryHandler(tracer_provider=provider)
    invocation = LLMInvocation(request_model="model")
    handler.start_llm(invocation)

    process_method = (
        "process_multimodal_stop"
        if method == "stop_llm"
        else "process_multimodal_fail"
    )
    error = Error(message="business boom", type=RuntimeError)
    with patch.object(
        handler,
        process_method,
        side_effect=[RuntimeError("probe dispatch boom"), False],
    ) as process:
        with pytest.raises(RuntimeError, match="probe dispatch boom"):
            if method == "stop_llm":
                handler.stop_llm(invocation)
            else:
                handler.fail_llm(invocation, error)

        if method == "stop_llm":
            handler.stop_llm(invocation)
        else:
            handler.fail_llm(invocation, error)

    assert process.call_count == 2
    assert invocation.span is not None
    assert invocation.span.is_recording() is False
    assert len(exporter.get_finished_spans()) == 1
