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

import timeit
from types import SimpleNamespace

import pytest
from google.genai import types

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.google_genai._wrappers import (
    create_async_embedding_wrapper,
    create_async_generate_wrapper,
    create_sync_embedding_wrapper,
    create_sync_generate_wrapper,
)
from opentelemetry.util.genai.types import Text


def _response(text="ok"):
    return types.GenerateContentResponse(
        response_id="rid",
        model_version="gemini-response",
        candidates=[
            types.Candidate(
                index=0,
                finish_reason=types.FinishReason.STOP,
                content=types.Content(
                    role="model", parts=[types.Part(text=text)]
                ),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=2, candidates_token_count=1
        ),
    )


class RecordingHandler:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.failed = []
        self.embedding_started = []
        self.embedding_stopped = []
        self.embedding_failed = []

    def start_llm(self, invocation):
        invocation.monotonic_start_s = timeit.default_timer()
        self.started.append(invocation)

    def stop_llm(self, invocation):
        self.stopped.append(invocation)

    def fail_llm(self, invocation, error):
        self.failed.append((invocation, error))

    def start_embedding(self, invocation):
        self.embedding_started.append(invocation)

    def stop_embedding(self, invocation):
        self.embedding_stopped.append(invocation)

    def fail_embedding(self, invocation, error):
        self.embedding_failed.append((invocation, error))


def test_sync_non_streaming_wrapper_uses_shared_handler():
    handler = RecordingHandler()

    def original(instance, *, model, contents, config=None):
        return _response("answer")

    wrapped = create_sync_generate_wrapper(original, handler, streaming=False)
    response = wrapped(
        SimpleNamespace(vertexai=False), model="gemini", contents="question"
    )

    assert response.response_id == "rid"
    assert len(handler.started) == len(handler.stopped) == 1
    invocation = handler.stopped[0]
    assert invocation.response_id == "rid"
    assert invocation.output_messages[0].parts == [Text(content="answer")]


def test_sync_streaming_wrapper_tracks_ttft_and_merged_output():
    handler = RecordingHandler()

    def original(instance, *, model, contents, config=None):
        yield _response("he")
        yield _response("llo")

    wrapped = create_sync_generate_wrapper(original, handler, streaming=True)
    stream = wrapped(
        SimpleNamespace(vertexai=False), model="gemini", contents="question"
    )
    assert [chunk.text for chunk in stream] == ["he", "llo"]

    invocation = handler.stopped[0]
    assert invocation.monotonic_first_token_s is not None
    assert invocation.output_messages[0].parts == [Text(content="hello")]


def test_sync_stream_close_finalizes_partial_response():
    handler = RecordingHandler()

    def original(instance, *, model, contents, config=None):
        yield _response("partial")
        yield _response("unused")

    stream = create_sync_generate_wrapper(original, handler, streaming=True)(
        SimpleNamespace(vertexai=False), model="gemini", contents="question"
    )
    assert next(stream).text == "partial"
    stream.close()
    assert handler.stopped[0].output_messages[0].parts == [
        Text(content="partial")
    ]


def test_sync_stream_context_error_closes_underlying_and_reports_error():
    handler = RecordingHandler()

    class ClosableStream:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            self.closed = True

    underlying = ClosableStream()

    def original(instance, *, model, contents, config=None):
        return underlying

    stream = create_sync_generate_wrapper(original, handler, streaming=True)(
        SimpleNamespace(vertexai=False), model="gemini", contents="question"
    )

    with pytest.raises(ValueError, match="consumer failure"):
        with stream:
            raise ValueError("consumer failure")

    assert underlying.closed
    assert handler.stopped == []
    assert handler.failed[0][1].type is ValueError


def test_sync_business_error_is_reported_and_preserved():
    handler = RecordingHandler()

    def original(instance, *, model, contents, config=None):
        raise RuntimeError("business failure")

    wrapped = create_sync_generate_wrapper(original, handler, streaming=False)
    with pytest.raises(RuntimeError, match="business failure"):
        wrapped(
            SimpleNamespace(vertexai=False),
            model="gemini",
            contents="question",
        )
    assert handler.failed[0][1].type is RuntimeError


def test_standard_suppression_skips_provider_span():
    handler = RecordingHandler()

    def original(instance, *, model, contents, config=None):
        return _response()

    token = context.attach(
        context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        wrapped = create_sync_generate_wrapper(
            original, handler, streaming=False
        )
        wrapped(
            SimpleNamespace(vertexai=False),
            model="gemini",
            contents="question",
        )
    finally:
        context.detach(token)
    assert handler.started == []


@pytest.mark.asyncio
async def test_async_generation_and_streaming():
    handler = RecordingHandler()

    async def generate(instance, *, model, contents, config=None):
        return _response("answer")

    async def stream(instance, *, model, contents, config=None):
        async def responses():
            yield _response("a")
            yield _response("b")

        return responses()

    response = await create_async_generate_wrapper(
        generate, handler, streaming=False
    )(SimpleNamespace(vertexai=False), model="gemini", contents="question")
    assert response.text == "answer"

    response_stream = await create_async_generate_wrapper(
        stream, handler, streaming=True
    )(SimpleNamespace(vertexai=False), model="gemini", contents="question")
    chunks = [chunk.text async for chunk in response_stream]
    assert chunks == ["a", "b"]
    assert handler.stopped[-1].output_messages[0].parts == [Text(content="ab")]


@pytest.mark.asyncio
async def test_async_stream_context_error_closes_underlying_and_reports_error():
    handler = RecordingHandler()

    class ClosableAsyncStream:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    underlying = ClosableAsyncStream()

    async def original(instance, *, model, contents, config=None):
        return underlying

    stream = await create_async_generate_wrapper(
        original, handler, streaming=True
    )(SimpleNamespace(vertexai=False), model="gemini", contents="question")

    with pytest.raises(ValueError, match="consumer failure"):
        async with stream:
            raise ValueError("consumer failure")

    assert underlying.closed
    assert handler.stopped == []
    assert handler.failed[0][1].type is ValueError


def test_sync_embedding_records_dimension():
    handler = RecordingHandler()

    def original(instance, *, model, contents, config=None):
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.1, 0.2, 0.3])]
        )

    response = create_sync_embedding_wrapper(original, handler)(
        SimpleNamespace(vertexai=False), model="embedding-001", contents="x"
    )
    assert response.embeddings[0].values == [0.1, 0.2, 0.3]
    assert handler.embedding_stopped[0].dimension_count == 3


@pytest.mark.asyncio
async def test_async_embedding_records_dimension():
    handler = RecordingHandler()

    async def original(instance, *, model, contents, config=None):
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.1, 0.2])]
        )

    await create_async_embedding_wrapper(original, handler)(
        SimpleNamespace(vertexai=False), model="embedding-001", contents="x"
    )
    assert handler.embedding_stopped[0].dimension_count == 2
