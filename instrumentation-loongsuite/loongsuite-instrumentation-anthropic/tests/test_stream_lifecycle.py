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

"""Exercise early stream closure through the public Anthropic SDK."""

import asyncio
import json

import pytest
from anthropic import Anthropic, AsyncAnthropic

from opentelemetry import trace
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


def _create(client):
    return client.messages.create(
        model="test-model",
        max_tokens=4,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )


def test_sync_stream_closes_response_on_early_exit(
    instrument_with_content,
    span_exporter,
    log_exporter,
    tracer_provider,
    vcr,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    with vcr.use_cassette("anthropic_stream.yaml"), Anthropic() as client:
        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span("parent") as parent:
            stream = _create(client)
            assert trace.get_current_span() is parent
            assert not stream.response.is_closed
            with stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        break
            assert stream.response.is_closed
            assert trace.get_current_span() is parent
    child, parent_span = span_exporter.get_finished_spans()
    assert child.parent.span_id == parent_span.context.span_id
    assert child.status.status_code is trace.StatusCode.UNSET
    assert "gen_ai.response.finish_reasons" not in child.attributes
    (message,) = json.loads(child.attributes["gen_ai.output.messages"])
    assert message["parts"][0]["content"] == "hello"
    assert message.get("finish_reason") is None
    (event,) = log_exporter.get_finished_logs()
    attrs = event.log_record.attributes
    assert "gen_ai.response.finish_reasons" not in attrs
    (event_message,) = attrs["gen_ai.output.messages"]
    assert event_message["parts"][0]["content"] == "hello"
    assert event_message.get("finish_reason") is None
    assert child.attributes["gen_ai.usage.cache_creation.input_tokens"] == 0
    assert child.attributes["gen_ai.usage.cache_read.input_tokens"] == 0


def test_unconsumed_stream_omits_usage(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette("anthropic_stream.yaml"), Anthropic() as client:
        _create(client).close()
    (span,) = span_exporter.get_finished_spans()
    assert "gen_ai.usage.cache_creation.input_tokens" not in span.attributes
    assert "gen_ai.usage.cache_read.input_tokens" not in span.attributes


def test_complete_stream_preserves_provider_reason(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette("anthropic_stream.yaml"), Anthropic() as client:
        with _create(client) as stream:
            list(stream)
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("end_turn",)
    (message,) = json.loads(span.attributes["gen_ai.output.messages"])
    assert message["finish_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_async_stream_closes_in_another_task_on_error(
    instrument_with_content, span_exporter, tracer_provider, vcr
):
    async def consume(stream):
        async with stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    raise ValueError("consumer failed")

    with vcr.use_cassette("anthropic_stream.yaml"):
        async with AsyncAnthropic() as client:
            tracer = tracer_provider.get_tracer(__name__)
            with tracer.start_as_current_span("parent") as parent:
                stream = await _create(client)
                assert trace.get_current_span() is parent
                assert not stream.response.is_closed
                with pytest.raises(ValueError, match="consumer failed"):
                    await asyncio.create_task(consume(stream))
                assert stream.response.is_closed
                assert trace.get_current_span() is parent
    child, parent_span = span_exporter.get_finished_spans()
    assert child.parent.span_id == parent_span.context.span_id
    assert child.status.status_code is trace.StatusCode.ERROR
    assert child.attributes["gen_ai.response.finish_reasons"] == ("error",)
    assert child.attributes["gen_ai.usage.cache_creation.input_tokens"] == 0
    assert child.attributes["gen_ai.usage.cache_read.input_tokens"] == 0
