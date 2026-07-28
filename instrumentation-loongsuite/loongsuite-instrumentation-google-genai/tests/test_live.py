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

"""Opt-in smoke test against the real Gemini Developer API."""

import asyncio
import os

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
from opentelemetry.trace import SpanKind

_LIVE_TEST_ENV = "RUN_GOOGLE_GENAI_LIVE_TESTS"
_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_EMBEDDING_MODEL = "gemini-embedding-001"
_INTERACTIONS_MODEL = _DEFAULT_MODEL


@pytest.mark.skipif(
    os.getenv(_LIVE_TEST_ENV) != "1",
    reason=f"set {_LIVE_TEST_ENV}=1 to call the real Gemini API",
)
def test_live_google_genai_all_instrumented_methods():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    assert api_key, "GEMINI_API_KEY or GOOGLE_API_KEY is required"
    model = os.getenv("GOOGLE_GENAI_LIVE_MODEL", _DEFAULT_MODEL)

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
    client = genai.Client(api_key=api_key)
    try:
        sync_response = client.models.generate_content(
            model=model,
            contents="Reply with the single word OK.",
        )
        sync_stream = list(
            client.models.generate_content_stream(
                model=model,
                contents="Reply with exactly: stream OK",
            )
        )
        sync_embedding = client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents="LoongSuite live sync embedding verification",
            config=types.EmbedContentConfig(output_dimensionality=8),
        )
        sync_interaction = client.interactions.create(
            model=_INTERACTIONS_MODEL,
            input="Reply with exactly: live interaction OK",
        )
        sync_interaction_stream = list(
            client.interactions.create(
                model=_INTERACTIONS_MODEL,
                input="Reply with exactly: live interaction stream OK",
                stream=True,
            )
        )

        def get_temperature(city: str) -> str:
            """Return the current temperature for a city."""

            assert city
            return "17 C"

        tool_response = client.models.generate_content(
            model=model,
            contents=(
                "You must call get_temperature for Paris, then reply with "
                "only the returned temperature."
            ),
            config=types.GenerateContentConfig(tools=[get_temperature]),
        )

        async def run_async_methods():
            try:
                async_response = await client.aio.models.generate_content(
                    model=model,
                    contents="Reply with exactly: async OK",
                )
                stream = await client.aio.models.generate_content_stream(
                    model=model,
                    contents="Reply with exactly: async stream OK",
                )
                async_stream = [chunk async for chunk in stream]
                async_embedding = await client.aio.models.embed_content(
                    model=_EMBEDDING_MODEL,
                    contents="LoongSuite live async embedding verification",
                    config=types.EmbedContentConfig(output_dimensionality=8),
                )
                async_interaction = await client.aio.interactions.create(
                    model=_INTERACTIONS_MODEL,
                    input="Reply with exactly: live async interaction OK",
                )
                interaction_stream = await client.aio.interactions.create(
                    model=_INTERACTIONS_MODEL,
                    input=(
                        "Reply with exactly: live async interaction stream OK"
                    ),
                    stream=True,
                )
                async_interaction_stream = [
                    event async for event in interaction_stream
                ]
                return (
                    async_response,
                    async_stream,
                    async_embedding,
                    async_interaction,
                    async_interaction_stream,
                )
            finally:
                await client.aio.aclose()

        (
            async_response,
            async_stream,
            async_embedding,
            async_interaction,
            async_interaction_stream,
        ) = asyncio.run(run_async_methods())
    finally:
        client.close()
        instrumentor.uninstrument()

    for response in (sync_response, async_response):
        assert response.text
        assert response.response_id
        assert response.usage_metadata is not None
        assert response.usage_metadata.prompt_token_count > 0
        assert response.usage_metadata.candidates_token_count > 0
    assert any(chunk.text for chunk in sync_stream)
    assert any(chunk.text for chunk in async_stream)
    assert len(sync_embedding.embeddings[0].values) == 8
    assert len(async_embedding.embeddings[0].values) == 8
    assert sync_interaction.id
    assert async_interaction.id
    assert sync_interaction_stream[-1].interaction.id
    assert async_interaction_stream[-1].interaction.id
    assert "17" in tool_response.text

    spans = span_exporter.get_finished_spans()
    generation_spans = [
        span for span in spans if span.name == f"generate_content {model}"
    ]
    embedding_spans = [
        span for span in spans if span.name == f"embeddings {_EMBEDDING_MODEL}"
    ]
    interaction_spans = [
        span
        for span in spans
        if span.name == f"interactions.create {_INTERACTIONS_MODEL}"
    ]
    tool_spans = [
        span for span in spans if span.name == "execute_tool get_temperature"
    ]
    assert len(generation_spans) >= 5
    assert len(embedding_spans) == 2
    assert len(interaction_spans) == 4
    assert len(tool_spans) == 1
    assert tool_spans[0].kind is SpanKind.INTERNAL
    assert all(
        span.attributes["gen_ai.response.id"] for span in generation_spans
    )
    assert all(
        span.attributes["gen_ai.usage.input_tokens"] > 0
        for span in generation_spans
    )
    assert all(
        span.attributes["gen_ai.usage.output_tokens"] > 0
        for span in generation_spans
    )
    assert all(
        span.attributes["gen_ai.embeddings.dimension.count"] == 8
        for span in embedding_spans
    )
    assert all(
        span.attributes["gen_ai.response.model"] == _INTERACTIONS_MODEL
        for span in interaction_spans
    )
    streaming_spans = [
        span
        for span in generation_spans + interaction_spans
        if "gen_ai.response.time_to_first_token" in span.attributes
    ]
    assert len(streaming_spans) == 4
    assert all(
        span.attributes["gen_ai.response.time_to_first_token"] > 0
        for span in streaming_spans
    )

    metric_names = {
        metric.name
        for resource_metrics in metric_reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert "gen_ai.client.operation.duration" in metric_names
    assert "gen_ai.client.token.usage" in metric_names
