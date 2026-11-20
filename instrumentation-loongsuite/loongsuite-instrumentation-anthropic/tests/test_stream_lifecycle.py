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

import pytest
from anthropic import Anthropic, AsyncAnthropic

from opentelemetry import trace


def _create(client):
    return client.messages.create(
        model="test-model",
        max_tokens=4,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )


def test_sync_stream_closes_response_on_early_exit(
    instrument_anthropic, span_exporter, tracer_provider, vcr
):
    with vcr.use_cassette("anthropic_stream.yaml"), Anthropic() as client:
        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span("parent") as parent:
            stream = _create(client)
            assert trace.get_current_span() is parent
            assert not stream.response.is_closed
            with stream:
                pass
            assert stream.response.is_closed
            assert trace.get_current_span() is parent
    child, parent_span = span_exporter.get_finished_spans()
    assert child.parent.span_id == parent_span.context.span_id


@pytest.mark.asyncio
async def test_async_stream_closes_in_another_task_on_error(
    instrument_anthropic, span_exporter, tracer_provider, vcr
):
    async def consume(stream):
        async with stream:
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
