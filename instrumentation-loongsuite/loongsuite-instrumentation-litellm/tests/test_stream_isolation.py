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

import pytest

from opentelemetry.instrumentation.litellm._stream_wrapper import (
    AsyncStreamWrapper,
    StreamWrapper,
)


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
    assert list(stream) == []
    assert completed == [True]


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
