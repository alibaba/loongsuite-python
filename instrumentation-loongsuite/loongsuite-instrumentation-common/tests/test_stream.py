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

import asyncio

import pytest

from opentelemetry.instrumentation.loongsuite import (
    IsolatedAsyncStream,
    IsolatedStream,
)


class FailingIterator:
    def __init__(self, exception):
        self.exception = exception
        self.returned = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.returned:
            self.returned = True
            return "chunk"
        raise self.exception


class ClosingIterator:
    def __init__(self, close_exception=None):
        self.close_exception = close_exception
        self.closed = 0

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def close(self):
        self.closed += 1
        if self.close_exception is not None:
            raise self.close_exception
        return "closed"


def test_stream_callback_failure_does_not_drop_chunks():
    chunks = [object(), object()]
    finished = []

    def fail_chunk(_chunk):
        raise RuntimeError("instrumentation boom")

    stream = IsolatedStream(
        iter(chunks),
        on_chunk=fail_chunk,
        on_finish=lambda: finished.append(True),
        instrumentation_name="test",
    )

    assert list(stream) == chunks
    assert finished == [True]
    assert stream.is_finalized


def test_stream_callback_base_exception_finalizes_and_propagates():
    expected = BaseException("instrumentation control flow")
    errors = []

    def fail_chunk(_chunk):
        raise expected

    stream = IsolatedStream(
        iter(["chunk"]),
        on_chunk=fail_chunk,
        on_error=errors.append,
        instrumentation_name="test",
    )

    with pytest.raises(BaseException) as caught:
        next(stream)
    assert caught.value is expected
    assert errors == [expected]
    assert stream.is_finalized


def test_stream_business_exception_identity_is_preserved():
    expected = RuntimeError("application boom")
    errors = []
    stream = IsolatedStream(
        FailingIterator(expected),
        on_error=errors.append,
        instrumentation_name="test",
    )

    assert next(stream) == "chunk"
    with pytest.raises(RuntimeError) as caught:
        next(stream)
    assert caught.value is expected
    assert errors == [expected]


def test_stream_error_callback_cannot_replace_business_exception():
    expected = RuntimeError("application boom")

    def fail_error_callback(_exception):
        raise ValueError("instrumentation boom")

    stream = IsolatedStream(
        FailingIterator(expected),
        on_error=fail_error_callback,
        instrumentation_name="test",
    )

    assert next(stream) == "chunk"
    with pytest.raises(RuntimeError) as caught:
        next(stream)
    assert caught.value is expected


def test_stream_close_preserves_underlying_result_and_error():
    successful = ClosingIterator()
    stream = IsolatedStream(successful)
    assert stream.close() == "closed"
    assert successful.closed == 1

    expected = RuntimeError("close boom")
    failing = ClosingIterator(expected)
    errors = []
    stream = IsolatedStream(failing, on_error=errors.append)
    with pytest.raises(RuntimeError) as caught:
        stream.close()
    assert caught.value is expected
    assert errors == [expected]


def test_stream_close_is_idempotent():
    underlying = ClosingIterator()
    finished = []
    stream = IsolatedStream(
        underlying,
        on_finish=lambda: finished.append(True),
    )

    assert stream.close() == "closed"
    assert stream.close() is None
    assert underlying.closed == 1
    assert finished == [True]


def test_stream_finalizer_runs_once():
    finished = []
    stream = IsolatedStream(
        iter(()),
        on_finish=lambda: finished.append(True),
    )

    with pytest.raises(StopIteration):
        next(stream)
    stream.close()
    assert finished == [True]


def test_stream_finish_advice_failure_preserves_stop_iteration():
    def fail_finish():
        raise RuntimeError("instrumentation boom")

    stream = IsolatedStream(iter(()), on_finish=fail_finish)

    with pytest.raises(StopIteration):
        next(stream)


def test_stream_send_preserves_generator_protocol():
    chunks = []

    def generator():
        value = yield "ready"
        yield value

    stream = IsolatedStream(generator(), on_chunk=chunks.append)

    assert next(stream) == "ready"
    assert stream.send("sent") == "sent"
    assert chunks == ["ready", "sent"]


def test_stream_preserves_context_manager_suppression_result():
    class SuppressingStream(ClosingIterator):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return True

    finished = []
    errors = []
    stream = IsolatedStream(
        SuppressingStream(),
        on_finish=lambda: finished.append(True),
        on_error=errors.append,
    )
    assert stream.__enter__() is stream
    assert stream.__exit__(RuntimeError, RuntimeError("boom"), None) is True
    assert finished == [True]
    assert errors == []


def test_stream_context_exit_closes_plain_iterator():
    underlying = ClosingIterator()
    stream = IsolatedStream(underlying)

    assert stream.__enter__() is stream
    assert stream.__exit__(None, None, None) is False
    assert underlying.closed == 1


class AsyncFailingIterator:
    def __init__(self, exception=None):
        self.exception = exception
        self.returned = False
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.returned:
            self.returned = True
            return "chunk"
        if self.exception is not None:
            raise self.exception
        raise StopAsyncIteration

    async def aclose(self):
        self.closed += 1
        return "closed"


@pytest.mark.asyncio
async def test_async_stream_callback_failure_does_not_drop_chunks():
    async def fail_chunk(_chunk):
        raise RuntimeError("instrumentation boom")

    stream = IsolatedAsyncStream(
        AsyncFailingIterator(),
        on_chunk=fail_chunk,
        instrumentation_name="test",
    )

    assert await stream.__anext__() == "chunk"
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    assert stream.is_finalized


@pytest.mark.asyncio
async def test_async_stream_business_exception_identity_is_preserved():
    expected = RuntimeError("application boom")
    errors = []
    stream = IsolatedAsyncStream(
        AsyncFailingIterator(expected),
        on_error=errors.append,
        instrumentation_name="test",
    )

    assert await stream.__anext__() == "chunk"
    with pytest.raises(RuntimeError) as caught:
        await stream.__anext__()
    assert caught.value is expected
    assert errors == [expected]


@pytest.mark.asyncio
async def test_async_stream_preserves_cancellation():
    expected = asyncio.CancelledError()
    errors = []
    stream = IsolatedAsyncStream(
        AsyncFailingIterator(expected),
        on_error=errors.append,
    )

    assert await stream.__anext__() == "chunk"
    with pytest.raises(asyncio.CancelledError) as caught:
        await stream.__anext__()
    assert caught.value is expected
    assert errors == [expected]


@pytest.mark.asyncio
async def test_async_stream_aclose_preserves_underlying_result():
    underlying = AsyncFailingIterator()
    finished = []
    stream = IsolatedAsyncStream(
        underlying,
        on_finish=lambda: finished.append(True),
    )

    assert await stream.aclose() == "closed"
    assert underlying.closed == 1
    assert finished == [True]

    assert await stream.aclose() is None
    assert underlying.closed == 1


@pytest.mark.asyncio
async def test_async_stream_finish_advice_failure_preserves_stop_iteration():
    async def fail_finish():
        raise RuntimeError("instrumentation boom")

    stream = IsolatedAsyncStream(
        AsyncFailingIterator(),
        on_finish=fail_finish,
    )

    assert await stream.__anext__() == "chunk"
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_async_stream_preserves_context_manager_suppression_result():
    class SuppressingAsyncStream(AsyncFailingIterator):
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return True

    finished = []
    errors = []
    stream = IsolatedAsyncStream(
        SuppressingAsyncStream(),
        on_finish=lambda: finished.append(True),
        on_error=errors.append,
    )
    assert await stream.__aenter__() is stream
    assert (
        await stream.__aexit__(
            RuntimeError,
            RuntimeError("application boom"),
            None,
        )
        is True
    )
    assert finished == [True]
    assert errors == []


@pytest.mark.asyncio
async def test_async_stream_context_exit_closes_plain_iterator():
    underlying = AsyncFailingIterator()
    stream = IsolatedAsyncStream(underlying)

    assert await stream.__aenter__() is stream
    assert await stream.__aexit__(None, None, None) is False
    assert underlying.closed == 1
