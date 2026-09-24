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

from opentelemetry.instrumentation.autogen.patch import _mark_autogen_live_span
from opentelemetry.instrumentation.autogen.semantic_conventions import (
    AUTOGEN_LIVE_SPAN_MARKER,
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_ID,
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


def test_processor_keeps_create_agent_as_lifecycle_span_without_agent_kind():
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

    assert GEN_AI_AGENT_NAME not in attributes
    assert GEN_AI_OPERATION_NAME not in attributes
    assert GEN_AI_PROVIDER_NAME not in attributes
    assert GEN_AI_SPAN_KIND not in attributes
    assert GEN_AI_SYSTEM not in attributes
    assert span.kind == SpanKind.CLIENT


def test_processor_removes_native_agent_kind_from_create_agent_span():
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
            GEN_AI_AGENT_DESCRIPTION: "test agent",
            GEN_AI_AGENT_ID: "agent-1",
            GEN_AI_OPERATION_NAME: GenAIOperation.CREATE_AGENT,
            GEN_AI_AGENT_NAME: "assistant",
            GEN_AI_PROVIDER_NAME: AUTOGEN_PROVIDER_NAME,
            GEN_AI_SPAN_KIND: GenAISpanKind.AGENT,
            "gen_ai.agent.version": "1.0.0",
            "gen_ai.system_instructions": "be helpful",
            "gen_ai.tool.definitions": ["search"],
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert GEN_AI_AGENT_DESCRIPTION not in attributes
    assert GEN_AI_AGENT_ID not in attributes
    assert GEN_AI_AGENT_NAME not in attributes
    assert GEN_AI_OPERATION_NAME not in attributes
    assert GEN_AI_PROVIDER_NAME not in attributes
    assert GEN_AI_SPAN_KIND not in attributes
    assert GEN_AI_SYSTEM not in attributes
    assert "gen_ai.agent.version" not in attributes
    assert "gen_ai.system_instructions" not in attributes
    assert "gen_ai.tool.definitions" not in attributes
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


def test_processor_classifies_llm_span_from_autogen_private_marker():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "chat gpt-4o-mini",
        attributes={
            GEN_AI_PROVIDER_NAME: "openai",
            GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
        },
    ) as span:
        setattr(span, AUTOGEN_LIVE_SPAN_MARKER, AUTOGEN_PROVIDER_NAME)
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == "openai"
    assert attributes[GEN_AI_SPAN_KIND] == GenAISpanKind.LLM
    assert AUTOGEN_LIVE_SPAN_MARKER not in attributes
    assert span.kind == SpanKind.CLIENT


def test_patch_marks_live_llm_span_without_exported_attribute():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "chat gpt-4o-mini",
        attributes={
            GEN_AI_PROVIDER_NAME: "openai",
            GEN_AI_OPERATION_NAME: GenAIOperation.CHAT,
        },
    ) as live_span:
        invocation = type("Invocation", (), {"span": live_span})()
        _mark_autogen_live_span(invocation)

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == "openai"
    assert attributes[GEN_AI_SPAN_KIND] == GenAISpanKind.LLM
    assert AUTOGEN_LIVE_SPAN_MARKER not in attributes
    assert span.kind == SpanKind.CLIENT


def test_processor_classifies_agent_span_from_known_private_attribute():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "invoke_agent assistant",
        attributes={
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            GEN_AI_AGENT_NAME: "assistant",
            "autogen.team.type": "RoundRobinGroupChat",
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == AUTOGEN_PROVIDER_NAME
    assert attributes[GEN_AI_SPAN_KIND] == GenAISpanKind.AGENT


def test_processor_ignores_maf_span_with_overlapping_agent_operation():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "invoke_agent my-agent",
        attributes={
            GEN_AI_PROVIDER_NAME: "microsoft.agent_framework",
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == "microsoft.agent_framework"
    assert GEN_AI_SPAN_KIND not in attributes


def test_processor_ignores_unmarked_overlapping_agent_operation():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "invoke_agent assistant",
        attributes={GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT},
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert GEN_AI_PROVIDER_NAME not in attributes
    assert GEN_AI_SPAN_KIND not in attributes


def test_processor_ignores_foreign_provider_agent_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    for provider_name in ("langchain", "agentscope"):
        with tracer.start_as_current_span(
            f"invoke_agent {provider_name}-agent",
            attributes={
                GEN_AI_PROVIDER_NAME: provider_name,
                GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            },
        ):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        attributes = _span_attributes(span)
        assert attributes[GEN_AI_PROVIDER_NAME] in {"langchain", "agentscope"}
        assert GEN_AI_SPAN_KIND not in attributes


def test_processor_ignores_unknown_autogen_prefixed_foreign_attribute():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AutoGenSemanticProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "invoke_agent langchain-agent",
        attributes={
            GEN_AI_PROVIDER_NAME: "langchain",
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            "autogen.unrelated": "foreign-custom-attribute",
        },
    ):
        pass

    [span] = exporter.get_finished_spans()
    attributes = _span_attributes(span)

    assert attributes[GEN_AI_PROVIDER_NAME] == "langchain"
    assert GEN_AI_SPAN_KIND not in attributes
