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

from types import SimpleNamespace

import httpx
from google import genai
from google.genai import types
from google.genai.models import AsyncModels, Models

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai._wrappers import (
    create_sync_generate_wrapper,
)
from opentelemetry.sdk import metrics, trace
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


def test_instrumentor_patches_and_restores_supported_sdk_methods():
    originals = {
        (Models, "generate_content"): Models.generate_content,
        (Models, "generate_content_stream"): Models.generate_content_stream,
        (Models, "embed_content"): Models.embed_content,
        (AsyncModels, "generate_content"): AsyncModels.generate_content,
        (
            AsyncModels,
            "generate_content_stream",
        ): AsyncModels.generate_content_stream,
        (AsyncModels, "embed_content"): AsyncModels.embed_content,
    }
    instrumentor = GoogleGenAiSdkInstrumentor()

    instrumentor.instrument()
    try:
        for (target, name), original in originals.items():
            assert getattr(target, name) is not original
    finally:
        instrumentor.uninstrument()

    for (target, name), original in originals.items():
        assert getattr(target, name) is original


def test_extended_handler_emits_loongsuite_span_and_metrics():
    span_exporter = InMemorySpanExporter()
    tracer_provider = trace.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = metrics.MeterProvider(metric_readers=[metric_reader])
    handler = ExtendedTelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    response = types.GenerateContentResponse(
        response_id="response-123",
        model_version="gemini-2.5-flash-001",
        candidates=[
            types.Candidate(
                index=0,
                finish_reason=types.FinishReason.STOP,
                content=types.Content(
                    role="model", parts=[types.Part(text="answer")]
                ),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=7, candidates_token_count=4
        ),
    )

    def original(instance, *, model, contents, config=None):
        return response

    create_sync_generate_wrapper(original, handler, streaming=False)(
        SimpleNamespace(vertexai=False),
        model="gemini-2.5-flash",
        contents="question",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "generate_content gemini-2.5-flash"
    assert span.attributes["gen_ai.operation.name"] == "generate_content"
    assert span.attributes["gen_ai.span.kind"] == "LLM"
    assert span.attributes["gen_ai.provider.name"] == "gcp.gen_ai"
    assert span.attributes["gen_ai.response.id"] == "response-123"
    assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash-001"
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 4

    metric_names = {
        metric.name
        for resource_metrics in metric_reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert "gen_ai.client.operation.duration" in metric_names
    assert "gen_ai.client.token.usage" in metric_names


def test_instrumented_sdk_client_call_emits_response_telemetry():
    def handle_request(request):
        assert request.url.path.endswith("/models/gemini-test:generateContent")
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "responseId": "sdk-response-2x",
                "modelVersion": "gemini-test-002",
                "candidates": [
                    {
                        "index": 0,
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": "sdk answer"}],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            },
            request=request,
        )

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
    client = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            client_args={"transport": httpx.MockTransport(handle_request)}
        ),
    )
    try:
        response = client.models.generate_content(
            model="gemini-test",
            contents="sdk question",
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        instrumentor.uninstrument()

    assert response.text == "sdk answer"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.response.id"] == "sdk-response-2x"
    assert span.attributes["gen_ai.response.model"] == "gemini-test-002"
    assert span.attributes["gen_ai.usage.input_tokens"] == 5
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
