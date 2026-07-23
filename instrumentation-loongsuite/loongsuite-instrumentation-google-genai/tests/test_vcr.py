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

"""Recorded real-API coverage for every instrumented Google GenAI method."""

import os
from importlib.metadata import version

import pytest
from google import genai
from google.genai import types

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk import metrics, trace
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

_GENERATION_MODEL = "gemini-3.5-flash-lite"
_EMBEDDING_MODEL = "gemini-embedding-001"

pytestmark = pytest.mark.skipif(
    int(version("google-genai").split(".", maxsplit=1)[0]) < 2,
    reason="recorded SDK transport contract targets google-genai 2.x",
)


@pytest.fixture
def telemetry():
    span_exporter = InMemorySpanExporter()
    tracer_provider = trace.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = metrics.MeterProvider(metric_readers=[metric_reader])
    instrumentor = GoogleGenAiSdkInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    try:
        yield span_exporter, metric_reader
    finally:
        instrumentor.uninstrument()


@pytest.fixture
def client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv(
        "GOOGLE_API_KEY", "test-google-key"
    )
    value = genai.Client(api_key=api_key)
    try:
        yield value
    finally:
        value.close()


def _assert_generation_telemetry(span_exporter, metric_reader):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == f"generate_content {_GENERATION_MODEL}"
    assert span.attributes["gen_ai.response.id"]
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["gen_ai.usage.output_tokens"] > 0
    _assert_standard_metrics(metric_reader)


def _assert_embedding_telemetry(span_exporter, metric_reader):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == f"embeddings {_EMBEDDING_MODEL}"
    assert span.attributes["gen_ai.embeddings.dimension.count"] == 8
    # OSS extended metrics intentionally cover LLM calls only. Embedding
    # metrics remain an enterprise extension, while the OSS span is complete.
    assert metric_reader.get_metrics_data() is None


def _assert_standard_metrics(metric_reader):
    metric_names = {
        metric.name
        for resource_metrics in metric_reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert "gen_ai.client.operation.duration" in metric_names


@pytest.mark.vcr
def test_sync_generate_content(client, telemetry):
    response = client.models.generate_content(
        model=_GENERATION_MODEL,
        contents="Reply with exactly: sync ok",
    )
    assert response.text
    _assert_generation_telemetry(*telemetry)


@pytest.mark.vcr
def test_sync_generate_content_stream(client, telemetry):
    chunks = list(
        client.models.generate_content_stream(
            model=_GENERATION_MODEL,
            contents="Reply with exactly: sync stream ok",
        )
    )
    assert any(chunk.text for chunk in chunks)
    _assert_generation_telemetry(*telemetry)


@pytest.mark.vcr
def test_sync_embed_content(client, telemetry):
    response = client.models.embed_content(
        model=_EMBEDDING_MODEL,
        contents="LoongSuite sync embedding verification",
        config=types.EmbedContentConfig(output_dimensionality=8),
    )
    assert len(response.embeddings[0].values) == 8
    _assert_embedding_telemetry(*telemetry)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_generate_content(client, telemetry):
    response = await client.aio.models.generate_content(
        model=_GENERATION_MODEL,
        contents="Reply with exactly: async ok",
    )
    assert response.text
    _assert_generation_telemetry(*telemetry)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_generate_content_stream(client, telemetry):
    stream = await client.aio.models.generate_content_stream(
        model=_GENERATION_MODEL,
        contents="Reply with exactly: async stream ok",
    )
    chunks = [chunk async for chunk in stream]
    assert any(chunk.text for chunk in chunks)
    _assert_generation_telemetry(*telemetry)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_embed_content(client, telemetry):
    response = await client.aio.models.embed_content(
        model=_EMBEDDING_MODEL,
        contents="LoongSuite async embedding verification",
        config=types.EmbedContentConfig(output_dimensionality=8),
    )
    assert len(response.embeddings[0].values) == 8
    _assert_embedding_telemetry(*telemetry)
