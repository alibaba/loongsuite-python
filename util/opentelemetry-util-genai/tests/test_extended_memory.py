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

import json
import os
from unittest.mock import patch

import pytest

from opentelemetry.instrumentation._semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_memory import MemoryInvocation
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_memory_attributes import (
    GEN_AI_MEMORY_QUERY_TEXT,
    GEN_AI_MEMORY_RECORD_COUNT,
    GEN_AI_MEMORY_RECORD_ID,
    GEN_AI_MEMORY_RECORDS,
    GEN_AI_MEMORY_STORE_ID,
    GenAiMemoryOperationNameValues,
)


@pytest.fixture(name="memory_telemetry")
def fixture_memory_telemetry():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = get_extended_telemetry_handler(tracer_provider=provider)
    yield handler, exporter
    exporter.clear()
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")


def _span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _enable_content_capture():
    return patch.dict(
        os.environ,
        {
            OTEL_SEMCONV_STABILITY_OPT_IN: "gen_ai_latest_experimental",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
        },
    )


def _reset_semconv_mode():
    _OpenTelemetrySemanticConventionStability._initialized = False
    _OpenTelemetrySemanticConventionStability._initialize()


def test_search_memory_uses_upstream_name_and_fields(memory_telemetry):
    handler, exporter = memory_telemetry
    invocation = MemoryInvocation(
        operation=GenAiMemoryOperationNameValues.SEARCH_MEMORY.value,
        provider="mem0",
        store_id="store-1",
        record_id="memory-1",
        record_count=2,
    )

    with handler.memory(invocation):
        pass

    span = _span(exporter)
    assert span.name == "search_memory"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes == {
        GenAI.GEN_AI_OPERATION_NAME: "search_memory",
        GenAI.GEN_AI_PROVIDER_NAME: "mem0",
        GEN_AI_MEMORY_STORE_ID: "store-1",
        GEN_AI_MEMORY_RECORD_ID: "memory-1",
        GEN_AI_MEMORY_RECORD_COUNT: 2,
        GEN_AI_SPAN_KIND: GenAiSpanKindValues.MEMORY.value,
    }


def test_local_memory_operation_can_be_internal(memory_telemetry):
    handler, exporter = memory_telemetry
    invocation = MemoryInvocation(
        operation=GenAiMemoryOperationNameValues.UPSERT_MEMORY.value,
        span_kind=SpanKind.INTERNAL,
    )

    with handler.memory(invocation):
        pass

    assert _span(exporter).kind is SpanKind.INTERNAL


def test_memory_content_is_opt_in(memory_telemetry):
    handler, exporter = memory_telemetry
    invocation = MemoryInvocation(
        operation=GenAiMemoryOperationNameValues.SEARCH_MEMORY.value,
        query_text="vegetarian preferences",
        records=[
            {
                "id": "memory-1",
                "content": "User prefers vegetarian meals",
                "score": 0.95,
            }
        ],
    )

    with _enable_content_capture():
        _reset_semconv_mode()
        with handler.memory(invocation):
            pass

    attributes = _span(exporter).attributes
    assert attributes[GEN_AI_MEMORY_QUERY_TEXT] == "vegetarian preferences"
    assert json.loads(attributes[GEN_AI_MEMORY_RECORDS]) == invocation.records


def test_memory_content_is_not_captured_by_default(memory_telemetry):
    handler, exporter = memory_telemetry
    invocation = MemoryInvocation(
        operation=GenAiMemoryOperationNameValues.SEARCH_MEMORY.value,
        query_text="sensitive query",
        records=[{"content": "sensitive memory"}],
    )

    with handler.memory(invocation):
        pass

    attributes = _span(exporter).attributes
    assert GEN_AI_MEMORY_QUERY_TEXT not in attributes
    assert GEN_AI_MEMORY_RECORDS not in attributes


def test_legacy_memory_fields_are_not_emitted(memory_telemetry):
    handler, exporter = memory_telemetry
    invocation = MemoryInvocation(
        operation=GenAiMemoryOperationNameValues.SEARCH_MEMORY.value,
        user_id="user-1",
        agent_id="agent-1",
        memory_id="old-memory-id",
        limit=10,
        top_k=5,
    )

    with handler.memory(invocation):
        pass

    attributes = _span(exporter).attributes
    assert not any(
        key.startswith("gen_ai.memory.user_")
        or key
        in {
            "gen_ai.memory.operation",
            "gen_ai.memory.agent_id",
            "gen_ai.memory.id",
            "gen_ai.memory.limit",
            "gen_ai.memory.top_k",
        }
        for key in attributes
    )


def test_memory_error_sets_error_type(memory_telemetry):
    handler, exporter = memory_telemetry

    with pytest.raises(RuntimeError):
        with handler.memory(
            MemoryInvocation(
                operation=GenAiMemoryOperationNameValues.DELETE_MEMORY.value
            )
        ):
            raise RuntimeError("delete failed")

    span = _span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
