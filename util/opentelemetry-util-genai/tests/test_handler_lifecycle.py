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
from datetime import date
from unittest.mock import patch

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.util.genai._configuration import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    is_experimental_mode,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)
from opentelemetry.util.genai.extended_memory import MemoryInvocation
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
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    ToolCall,
)


@pytest.fixture(name="handler_and_exporter")
def fixture_handler_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TelemetryHandler(tracer_provider=provider), exporter


@pytest.fixture(name="extended_handler")
def fixture_extended_handler():
    """An extended handler, a tracer for the caller's spans, the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = ExtendedTelemetryHandler(tracer_provider=provider)
    return handler, provider.get_tracer(__name__), exporter


@pytest.fixture(name="content_capture")
def fixture_content_capture():
    with patch.dict(
        os.environ,
        {
            OTEL_SEMCONV_STABILITY_OPT_IN: "gen_ai_latest_experimental",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
        },
    ):
        is_experimental_mode.cache_clear()
        yield
    is_experimental_mode.cache_clear()


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
def test_detached_non_recording_span_is_finalized_once(method):
    handler = TelemetryHandler(
        tracer_provider=TracerProvider(sampler=ALWAYS_OFF)
    )
    invocation = LLMInvocation(request_model="model")
    handler.start_llm(invocation)
    handler.detach_llm_context(invocation)
    assert invocation.span is not None
    assert invocation.span.is_recording() is False

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


@pytest.mark.parametrize("method", ["stop_llm", "fail_llm"])
def test_detached_stream_span_can_be_finalized_by_extended_handler(method):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = ExtendedTelemetryHandler(tracer_provider=provider)
    invocation = LLMInvocation(request_model="model")
    handler.start_llm(invocation)
    handler.detach_llm_context(invocation)

    if method == "stop_llm":
        handler.stop_llm(invocation)
    else:
        handler.fail_llm(
            invocation,
            Error(message="business boom", type=RuntimeError),
        )

    assert invocation.context_token is None
    assert invocation.span is not None
    assert invocation.span.is_recording() is False
    assert len(exporter.get_finished_spans()) == 1


# Each extended invocation type, by the name its handler methods use.
_EXTENDED_INVOCATIONS = {
    "create_agent": lambda: CreateAgentInvocation(provider="provider"),
    "embedding": lambda: EmbeddingInvocation(request_model="model"),
    "execute_tool": lambda: ExecuteToolInvocation(tool_name="tool"),
    "invoke_agent": lambda: InvokeAgentInvocation(provider="provider"),
    "retrieval": RetrievalInvocation,
    "rerank": lambda: RerankInvocation(provider="provider"),
    "memory": lambda: MemoryInvocation(operation="search"),
    "entry": EntryInvocation,
    "react_step": ReactStepInvocation,
}

_DAY = date(2024, 1, 1)


def _finalize(handler, kind, finish, invocation):
    if finish == "stop":
        getattr(handler, f"stop_{kind}")(invocation)
    else:
        getattr(handler, f"fail_{kind}")(
            invocation, Error(message="business boom", type=RuntimeError)
        )


def _assert_ended_and_detached(exporter, invocation, caller, current, later):
    """The span ended and was exported, and the caller's context is back."""
    assert invocation.span is not None
    assert invocation.span.is_recording() is False
    finished = [span.context.span_id for span in exporter.get_finished_spans()]
    assert invocation.span.get_span_context().span_id in finished
    assert current is caller
    assert later.parent.span_id == caller.get_span_context().span_id
    assert invocation.context_token is None


@pytest.mark.usefixtures("content_capture")
@pytest.mark.parametrize(
    "kind, finish, make_invocation",
    [
        pytest.param(
            "execute_tool",
            "stop",
            lambda: ExecuteToolInvocation(
                tool_name="book", tool_call_arguments={"day": _DAY}
            ),
            id="stop_execute_tool-arguments",
        ),
        pytest.param(
            "execute_tool",
            "stop",
            lambda: ExecuteToolInvocation(
                tool_name="book", tool_call_result={"day": _DAY}
            ),
            id="stop_execute_tool-result",
        ),
        pytest.param(
            "execute_tool",
            "fail",
            lambda: ExecuteToolInvocation(
                tool_name="book", tool_call_arguments={"day": _DAY}
            ),
            id="fail_execute_tool-arguments",
        ),
        pytest.param(
            "invoke_agent",
            "stop",
            lambda: InvokeAgentInvocation(
                provider="provider",
                input_messages=[
                    InputMessage(
                        role="assistant",
                        parts=[
                            ToolCall(
                                id="call_1",
                                name="book",
                                arguments={"day": _DAY},
                            )
                        ],
                    )
                ],
            ),
            id="stop_invoke_agent-message",
        ),
    ],
)
def test_content_json_cannot_encode_still_detaches_and_ends_span(
    extended_handler, kind, finish, make_invocation
):
    handler, tracer, exporter = extended_handler
    invocation = make_invocation()

    with tracer.start_as_current_span("caller") as caller:
        getattr(handler, f"start_{kind}")(invocation)
        # As with stop_llm, the error still reaches the caller.
        with pytest.raises(TypeError, match="date is not JSON serializable"):
            _finalize(handler, kind, finish, invocation)
        current = trace.get_current_span()
        with tracer.start_as_current_span("later") as later:
            pass

    _assert_ended_and_detached(exporter, invocation, caller, current, later)


@pytest.mark.parametrize("finish", ["stop", "fail"])
@pytest.mark.parametrize("kind", list(_EXTENDED_INVOCATIONS))
def test_extended_finalize_failure_still_detaches_and_ends_span(
    extended_handler, kind, finish
):
    handler, tracer, exporter = extended_handler
    invocation = _EXTENDED_INVOCATIONS[kind]()

    with tracer.start_as_current_span("caller") as caller:
        getattr(handler, f"start_{kind}")(invocation)
        with patch(
            "opentelemetry.util.genai.extended_handler."
            f"_apply_{kind}_finish_attributes",
            side_effect=RuntimeError("probe attributes boom"),
        ):
            with pytest.raises(RuntimeError, match="probe attributes boom"):
                _finalize(handler, kind, finish, invocation)
        current = trace.get_current_span()
        with tracer.start_as_current_span("later") as later:
            pass

    _assert_ended_and_detached(exporter, invocation, caller, current, later)
