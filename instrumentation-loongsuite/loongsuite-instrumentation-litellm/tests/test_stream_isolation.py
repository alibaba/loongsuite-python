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

from opentelemetry.instrumentation.litellm._stream_wrapper import (
    AsyncStreamWrapper,
    StreamWrapper,
)
from opentelemetry.util.genai import extended_advice


def test_stream_chunk_advice_failure_preserves_chunk(monkeypatch):
    expected = object()
    completed = []
    stream = StreamWrapper(
        stream=iter([expected]),
        span=None,
        callback=lambda *_args: completed.append(True),
    )
    monkeypatch.setattr(
        stream._accumulator,
        "record_chunk",
        lambda _chunk: (_ for _ in ()).throw(
            RuntimeError("instrumentation boom")
        ),
    )

    assert next(stream) is expected
    stream.close()
    assert list(stream) == []
    assert completed == [True]


def test_stream_chunk_logging_failure_also_preserves_chunk(monkeypatch):
    expected = object()
    stream = StreamWrapper(
        stream=iter([expected]),
        span=None,
        callback=lambda *_args: None,
    )
    monkeypatch.setattr(
        stream._accumulator,
        "record_chunk",
        lambda _chunk: (_ for _ in ()).throw(
            RuntimeError("instrumentation boom")
        ),
    )
    monkeypatch.setattr(
        extended_advice._logger,
        "debug",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("logger boom")
        ),
    )

    assert next(stream) is expected
    stream.close()


def test_stream_preserves_business_exception_identity():
    expected = RuntimeError("business boom")
    completed = []

    def source():
        raise expected
        yield  # pragma: no cover

    stream = StreamWrapper(
        stream=source(),
        span=None,
        callback=lambda *_args: completed.append(_args),
    )

    with pytest.raises(RuntimeError) as caught:
        next(stream)

    assert caught.value is expected
    assert completed[0][2] is expected


def test_stream_callback_failure_does_not_escape_or_repeat():
    calls = []

    def callback(*_args):
        calls.append(True)
        raise RuntimeError("instrumentation boom")

    stream = StreamWrapper(
        stream=iter([]),
        span=None,
        callback=callback,
    )

    assert list(stream) == []
    stream.close()
    assert calls == [True]


def test_stream_close_base_exception_still_finalizes():
    calls = []
    expected = KeyboardInterrupt("business interruption")

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise expected

    stream = StreamWrapper(
        stream=Source(),
        span=None,
        callback=lambda *_args: calls.append(_args),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        stream.close()

    assert caught.value is expected
    assert len(calls) == 1
    assert stream._finalized is True


@pytest.mark.asyncio
async def test_async_stream_chunk_advice_failure_preserves_chunk(monkeypatch):
    expected = object()
    completed = []

    async def source():
        yield expected

    stream = AsyncStreamWrapper(
        stream=source(),
        span=None,
        callback=lambda *_args: completed.append(True),
    )
    monkeypatch.setattr(
        stream._accumulator,
        "record_chunk",
        lambda _chunk: (_ for _ in ()).throw(
            RuntimeError("instrumentation boom")
        ),
    )

    iterator = stream.__aiter__()
    assert await iterator.__anext__() is expected
    await iterator.aclose()
    assert completed == [True]


@pytest.mark.asyncio
async def test_async_stream_preserves_business_exception_identity():
    expected = RuntimeError("business boom")
    completed = []

    async def source():
        raise expected
        yield  # pragma: no cover

    stream = AsyncStreamWrapper(
        stream=source(),
        span=None,
        callback=lambda *_args: completed.append(_args),
    )

    with pytest.raises(RuntimeError) as caught:
        await stream.__aiter__().__anext__()

    assert caught.value is expected
    assert completed[0][2] is expected


@pytest.mark.asyncio
async def test_async_stream_callback_failure_does_not_escape_or_repeat():
    calls = []

    async def source():
        if False:  # pragma: no cover
            yield None

    def callback(*_args):
        calls.append(True)
        raise RuntimeError("instrumentation boom")

    stream = AsyncStreamWrapper(
        stream=source(),
        span=None,
        callback=callback,
    )

    assert [chunk async for chunk in stream] == []
    await stream.aclose()
    assert calls == [True]


@pytest.mark.asyncio
async def test_async_stream_cancellation_during_close_still_finalizes():
    calls = []

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise asyncio.CancelledError

    stream = AsyncStreamWrapper(
        stream=Source(),
        span=None,
        callback=lambda *_args: calls.append(_args),
    )

    with pytest.raises(asyncio.CancelledError):
        await stream.aclose()

    assert len(calls) == 1
    assert stream._finalized is True


@pytest.mark.asyncio
async def test_async_only_stream_can_aclose_after_sync_close():
    close_calls = []

    class Source:
        async def aclose(self):
            close_calls.append(True)

    stream = AsyncStreamWrapper(
        stream=Source(),
        span=None,
        callback=lambda *_args: None,
    )

    stream.close()
    assert close_calls == []
    assert stream._stream_closed is False

    await stream.aclose()
    assert close_calls == [True]
    assert stream._stream_closed is True


def test_async_stream_sync_close_base_exception_still_finalizes():
    calls = []
    expected = KeyboardInterrupt("business interruption")

    class Source:
        def close(self):
            raise expected

    stream = AsyncStreamWrapper(
        stream=Source(),
        span=None,
        callback=lambda *_args: calls.append(_args),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        stream.close()

    assert caught.value is expected
    assert len(calls) == 1
    assert stream._finalized is True
