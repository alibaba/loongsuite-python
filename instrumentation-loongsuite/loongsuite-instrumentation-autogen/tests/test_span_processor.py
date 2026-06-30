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

from __future__ import annotations

from opentelemetry.instrumentation.autogen.semantic_conventions import (
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GEN_AI_SYSTEM,
    GenAIOperation,
    GenAISpanKind,
)
from opentelemetry.instrumentation.autogen.span_processor import (
    AutoGenSemanticProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind


def _span_attributes(span):
    return dict(span.attributes or {})


def test_processor_normalizes_native_autogen_invoke_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "invoke_agent assistant",
        attributes={
            GEN_AI_SYSTEM: AUTOGEN_PROVIDER_NAME,
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            GEN_AI_AGENT_NAME: "assistant",
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == AUTOGEN_PROVIDER_NAME
    assert attributes[GEN_AI_SPAN_KIND] == GenAISpanKind.AGENT
    assert attributes[GEN_AI_OPERATION_NAME] == GenAIOperation.INVOKE_AGENT
    assert GEN_AI_SYSTEM not in attributes
    assert span.kind == SpanKind.INTERNAL


def test_processor_keeps_create_agent_as_lifecycle_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "create_agent assistant",
        kind=SpanKind.CLIENT,
        attributes={
            GEN_AI_SYSTEM: AUTOGEN_PROVIDER_NAME,
            GEN_AI_OPERATION_NAME: GenAIOperation.CREATE_AGENT,
            GEN_AI_AGENT_NAME: "assistant",
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == AUTOGEN_PROVIDER_NAME
    assert attributes[GEN_AI_OPERATION_NAME] == GenAIOperation.CREATE_AGENT
    assert GEN_AI_SPAN_KIND not in attributes
    assert GEN_AI_SYSTEM not in attributes
    assert span.kind == SpanKind.CLIENT


def test_processor_classifies_llm_span_from_provider_attributes():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "chat gpt-4o-mini",
        attributes={
            GEN_AI_PROVIDER_NAME: AUTOGEN_PROVIDER_NAME,
            GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_SPAN_KIND] == GenAISpanKind.LLM
    assert span.kind == SpanKind.CLIENT
