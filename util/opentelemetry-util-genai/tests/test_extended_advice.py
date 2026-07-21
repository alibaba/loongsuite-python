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
from unittest.mock import patch

import pytest

from opentelemetry.util.genai.extended_advice import (
    async_hook_advice,
    hook_advice,
)


def test_hook_advice_preserves_success_identity():
    expected = object()

    @hook_advice("test", "success")
    def advice():
        return expected

    assert advice() is expected
    assert advice.__name__ == "advice"


def test_hook_advice_swallows_instrumentation_exception():
    @hook_advice("test", "failure")
    def advice():
        raise RuntimeError("instrumentation boom")

    assert advice() is None


def test_hook_advice_strict_mode_preserves_exception():
    expected = RuntimeError("instrumentation boom")

    @hook_advice("test", "failure", throw_exception=True)
    def advice():
        raise expected

    with pytest.raises(RuntimeError) as caught:
        advice()
    assert caught.value is expected


def test_hook_advice_failure_logging_is_also_isolated():
    @hook_advice("test", "failure")
    def advice():
        raise RuntimeError("instrumentation boom")

    with patch(
        "opentelemetry.util.genai.extended_advice._logger.debug",
        side_effect=RuntimeError("logger boom"),
    ):
        assert advice() is None


def test_hook_advice_rejects_generator_function():
    def generator():
        yield "chunk"

    with pytest.raises(TypeError, match="stream wrapper"):
        hook_advice("test", "stream")(generator)


def test_hook_advice_rejects_coroutine_function():
    async def advice():
        return None

    with pytest.raises(TypeError, match="async_hook_advice"):
        hook_advice("test", "async")(advice)


def test_hook_advice_does_not_swallow_base_exception():
    expected = KeyboardInterrupt()

    @hook_advice("test", "interrupt")
    def advice():
        raise expected

    with pytest.raises(KeyboardInterrupt) as caught:
        advice()
    assert caught.value is expected


@pytest.mark.asyncio
async def test_async_hook_advice_preserves_success_identity():
    expected = object()

    @async_hook_advice("test", "success")
    async def advice():
        return expected

    assert await advice() is expected
    assert advice.__name__ == "advice"


@pytest.mark.asyncio
async def test_async_hook_advice_swallows_instrumentation_exception():
    @async_hook_advice("test", "failure")
    async def advice():
        raise RuntimeError("instrumentation boom")

    assert await advice() is None


@pytest.mark.asyncio
async def test_async_hook_advice_strict_mode_preserves_exception():
    expected = RuntimeError("instrumentation boom")

    @async_hook_advice("test", "failure", throw_exception=True)
    async def advice():
        raise expected

    with pytest.raises(RuntimeError) as caught:
        await advice()
    assert caught.value is expected


def test_async_hook_advice_rejects_async_generator_function():
    async def generator():
        yield "chunk"

    with pytest.raises(TypeError, match="stream wrapper"):
        async_hook_advice("test", "stream")(generator)


def test_async_hook_advice_rejects_sync_function():
    def advice():
        return None

    with pytest.raises(TypeError, match="hook_advice"):
        async_hook_advice("test", "sync")(advice)


@pytest.mark.asyncio
async def test_async_hook_advice_does_not_swallow_cancellation():
    expected = asyncio.CancelledError()

    @async_hook_advice("test", "cancel")
    async def advice():
        raise expected

    with pytest.raises(asyncio.CancelledError) as caught:
        await advice()
    assert caught.value is expected
