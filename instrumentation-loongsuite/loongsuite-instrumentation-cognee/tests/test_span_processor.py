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

"""Tests for ``CogneeAttributeSpanProcessor`` — rename + gen-ai attribute injection.

The processor rewrites spans in ``on_start`` (the SDK Span becomes read-only
after ``end()``). These tests emit spans via the SDK tracer and assert the
state captured by ``InMemorySpanExporter`` at end-of-span.

For attribute migration (``cognee.search.query`` → ``gen_ai.retrieval.query.text``),
we test ``install_attribute_migration_patch`` separately — it wraps Cognee's
``new_span`` to intercept ``span.set_attribute`` while the span is still mutable.
"""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.cognee.internal._span_processor import (
    _MIGRATION_MAP,
    CogneeAttributeSpanProcessor,
    _wrap_span_set_attribute,
)
from opentelemetry.instrumentation.cognee.semconv import (
    COGNEE_RESULT_SUMMARY,
    COGNEE_SEARCH_QUERY,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def processor_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(CogneeAttributeSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    yield tracer, exporter
    provider.shutdown()


def _emit(tracer, name, attrs=None):
    with tracer.start_as_current_span(name) as span:
        if attrs:
            for k, v in attrs.items():
                span.set_attribute(k, v)
    return span


def test_chain_workflow_cognify_rename(processor_tracer):
    tracer, exporter = processor_tracer
    _emit(tracer, "cognee.api.cognify")
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "workflow cognify"
    assert span.attributes["gen_ai.span.kind"] == "CHAIN"
    assert span.attributes["gen_ai.operation.name"] == "workflow"


def test_chain_workflow_search_rename(processor_tracer):
    tracer, exporter = processor_tracer
    _emit(tracer, "cognee.api.search")
    spans = exporter.get_finished_spans()
    assert spans[0].name == "workflow search"
    assert spans[0].attributes["gen_ai.span.kind"] == "CHAIN"


@pytest.mark.parametrize(
    "cognee_name,expected,expected_kind",
    [
        ("cognee.api.recall", "workflow recall", "CHAIN"),
        ("cognee.api.remember", "workflow remember", "CHAIN"),
        ("cognee.api.remember.import", "workflow remember_import", "CHAIN"),
        ("cognee.llm.completion", "task llm_completion", "CHAIN"),
        ("cognee.llm.summarize", "task llm_summarize", "CHAIN"),
    ],
)
def test_static_rename(processor_tracer, cognee_name, expected, expected_kind):
    tracer, exporter = processor_tracer
    _emit(tracer, cognee_name)
    spans = exporter.get_finished_spans()
    assert spans[0].name == expected
    assert spans[0].attributes["gen_ai.span.kind"] == expected_kind


def test_task_dynamic_rename(processor_tracer):
    tracer, exporter = processor_tracer
    _emit(tracer, "cognee.pipeline.task.extract_graph")
    spans = exporter.get_finished_spans()
    span = spans[0]
    assert span.name == "task extract_graph"
    assert span.attributes["gen_ai.span.kind"] == "TASK"
    assert span.attributes["gen_ai.operation.name"] == "run_task"


def test_retriever_dynamic_rename(processor_tracer):
    tracer, exporter = processor_tracer
    _emit(tracer, "cognee.retrieval.vector_search")
    spans = exporter.get_finished_spans()
    span = spans[0]
    assert span.name == "retrieval vector_search"
    assert span.attributes["gen_ai.span.kind"] == "RETRIEVER"
    assert span.attributes["gen_ai.operation.name"] == "retrieval"


def test_non_cognee_span_untouched(processor_tracer):
    tracer, exporter = processor_tracer
    _emit(tracer, "some.other.span")
    spans = exporter.get_finished_spans()
    span = spans[0]
    assert span.name == "some.other.span"
    assert "gen_ai.span.kind" not in span.attributes


class _RecordingSpan:
    """Fake span whose ``set_attribute`` records all writes (mutable post-end)."""

    def __init__(self):
        self.attributes = {}
        self.name = ""

    def update_name(self, name):
        self.name = name

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.mark.parametrize(
    "cognee_key,gen_ai_key",
    list(_MIGRATION_MAP.items()),
)
def test_attribute_migration_via_set_attribute_wrapper(cognee_key, gen_ai_key):
    """``_wrap_span_set_attribute`` must mirror cognee.* → gen_ai.* live."""
    span = _RecordingSpan()
    _wrap_span_set_attribute(span)
    span.set_attribute(cognee_key, "value")
    assert span.attributes.get(cognee_key) == "value"
    assert span.attributes.get(gen_ai_key) == "value"


def test_attribute_migration_truncates_long_values():
    span = _RecordingSpan()
    _wrap_span_set_attribute(span)
    long_value = "x" * 5000
    span.set_attribute(COGNEE_RESULT_SUMMARY, long_value)
    migrated = span.attributes.get("gen_ai.output.value")
    assert migrated.endswith("...")
    assert len(migrated) <= 4100  # MAX_PAYLOAD_BYTES=4096 + ellipsis


def test_attribute_migration_does_not_clobber_existing_genai_attr():
    span = _RecordingSpan()
    span.attributes["gen_ai.retrieval.query.text"] = "preset"
    _wrap_span_set_attribute(span)
    span.set_attribute(COGNEE_SEARCH_QUERY, "raw query")
    # Cognee key set, gen-ai key NOT overwritten by migration.
    assert span.attributes["gen_ai.retrieval.query.text"] == "preset"


def test_attribute_migration_skips_non_cognee_keys():
    span = _RecordingSpan()
    _wrap_span_set_attribute(span)
    span.set_attribute("some.other.key", "value")
    assert "gen_ai.retrieval.query.text" not in span.attributes
    assert span.attributes.get("some.other.key") == "value"


def test_attribute_migration_idempotent():
    span = _RecordingSpan()
    _wrap_span_set_attribute(span)
    original = span.set_attribute
    _wrap_span_set_attribute(span)
    assert span.set_attribute is original
