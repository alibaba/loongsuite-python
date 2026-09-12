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
from types import SimpleNamespace

import pytest

from opentelemetry import trace
from opentelemetry.instrumentation.litellm import _wrapper
from opentelemetry.instrumentation.litellm._wrapper import (
    AsyncCompletionWrapper,
    CompletionWrapper,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import LLMInvocation


class _ProbeHandler:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def start_llm(self, _invocation):
        self.calls.append("start")
        if self.failure == "start":
            raise RuntimeError("probe start boom")

    def stop_llm(self, _invocation):
        self.calls.append("stop")
        if self.failure == "stop":
            raise RuntimeError("probe stop boom")

    def fail_llm(self, _invocation, _error):
        self.calls.append("fail")
        if self.failure == "fail":
            raise RuntimeError("probe fail boom")

    def detach_llm_context(self, _invocation):
        self.calls.append("detach")

    def abandon_llm(self, _invocation):
        self.calls.append("abandon")


@pytest.fixture
def completion_advice(monkeypatch):
    invocation = SimpleNamespace(
        span=None,
        output_messages=[],
        finish_reasons=None,
    )
    monkeypatch.setattr(
        _wrapper,
        "normalize_litellm_completion_kwargs",
        lambda _func, _args, kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        _wrapper,
        "create_llm_invocation_from_litellm",
        lambda **_kwargs: invocation,
    )
    return invocation


@pytest.mark.parametrize("failure", ["start", "stop"])
def test_probe_lifecycle_failure_preserves_business_response(
    completion_advice, monkeypatch, failure
):
    response = object()
    business_calls = []
    handler = _ProbeHandler(failure)
    if failure == "stop":
        monkeypatch.setattr(
            _wrapper,
            "apply_litellm_llm_response_to_invocation",
            lambda *_args, **_kwargs: None,
        )

    wrapped = CompletionWrapper(
        handler,
        lambda **_kwargs: business_calls.append(True) or response,
    )

    assert wrapped(model="model") is response
    assert business_calls == [True]


def test_response_mapping_failure_preserves_business_response(
    completion_advice, monkeypatch
):
    response = object()
    business_calls = []
    handler = _ProbeHandler()
    monkeypatch.setattr(
        _wrapper,
        "apply_litellm_llm_response_to_invocation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("probe mapping boom")
        ),
    )

    wrapped = CompletionWrapper(
        handler,
        lambda **_kwargs: business_calls.append(True) or response,
    )

    assert wrapped(model="model") is response
    assert business_calls == [True]
    assert handler.calls == ["start", "abandon"]


def test_successful_stop_is_not_abandoned(completion_advice, monkeypatch):
    handler = _ProbeHandler()
    monkeypatch.setattr(
        _wrapper,
        "apply_litellm_llm_response_to_invocation",
        lambda *_args, **_kwargs: None,
    )

    response = object()
    assert (
        CompletionWrapper(handler, lambda **_kwargs: response)(model="model")
        is response
    )
    assert handler.calls == ["start", "stop"]


def test_stream_detach_failure_returns_raw_business_stream(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = TelemetryHandler(tracer_provider=provider)
    before = trace.get_current_span()
    expected_chunk = object()
    raw_stream = iter([expected_chunk])

    def business(**_kwargs):
        return raw_stream

    monkeypatch.setattr(
        _wrapper,
        "normalize_litellm_completion_kwargs",
        lambda _func, _args, kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        _wrapper,
        "create_llm_invocation_from_litellm",
        lambda **_kwargs: LLMInvocation(request_model="model"),
    )
    original_detach = handler.detach_llm_context
    detach_calls = 0

    def flaky_detach(invocation):
        nonlocal detach_calls
        detach_calls += 1
        if detach_calls == 1:
            raise RuntimeError("probe detach boom")
        return original_detach(invocation)

    monkeypatch.setattr(handler, "detach_llm_context", flaky_detach)

    response = CompletionWrapper(handler, business)(model="model", stream=True)

    assert response is raw_stream
    assert trace.get_current_span() is before
    assert list(response) == [expected_chunk]
    assert detach_calls == 2
    assert len(exporter.get_finished_spans()) == 1


def test_failure_reporting_does_not_replace_business_exception(
    completion_advice,
):
    expected = RuntimeError("business boom")
    handler = _ProbeHandler("fail")

    def business(**_kwargs):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        CompletionWrapper(handler, business)(model="model")

    assert caught.value is expected
    assert handler.calls == ["start", "fail", "abandon"]


@pytest.mark.asyncio
async def test_async_cancellation_identity_survives_probe_failure(
    completion_advice,
):
    expected = asyncio.CancelledError("business cancellation")
    handler = _ProbeHandler("fail")

    async def business(**_kwargs):
        raise expected

    with pytest.raises(asyncio.CancelledError) as caught:
        await AsyncCompletionWrapper(handler, business)(model="model")

    assert caught.value is expected


@pytest.mark.asyncio
async def test_async_stream_detaches_before_cross_task_consumption(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = TelemetryHandler(tracer_provider=provider)
    before = trace.get_current_span()

    async def source():
        if False:  # pragma: no cover
            yield None

    async def business(**_kwargs):
        return source()

    monkeypatch.setattr(
        _wrapper,
        "normalize_litellm_completion_kwargs",
        lambda _func, _args, kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        _wrapper,
        "create_llm_invocation_from_litellm",
        lambda **_kwargs: LLMInvocation(request_model="model"),
    )

    response = await AsyncCompletionWrapper(handler, business)(
        model="model", stream=True
    )

    assert trace.get_current_span() is before

    async def consume_in_sibling_task():
        assert trace.get_current_span() is before
        assert [chunk async for chunk in response] == []
        assert trace.get_current_span() is before

    await asyncio.create_task(consume_in_sibling_task())

    assert trace.get_current_span() is before
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_stream_detach_failure_returns_raw_business_stream(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = TelemetryHandler(tracer_provider=provider)
    before = trace.get_current_span()
    expected_chunk = object()

    async def source():
        yield expected_chunk

    raw_stream = source()

    async def business(**_kwargs):
        return raw_stream

    monkeypatch.setattr(
        _wrapper,
        "normalize_litellm_completion_kwargs",
        lambda _func, _args, kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        _wrapper,
        "create_llm_invocation_from_litellm",
        lambda **_kwargs: LLMInvocation(request_model="model"),
    )
    original_detach = handler.detach_llm_context
    detach_calls = 0

    def flaky_detach(invocation):
        nonlocal detach_calls
        detach_calls += 1
        if detach_calls == 1:
            raise RuntimeError("probe detach boom")
        return original_detach(invocation)

    monkeypatch.setattr(handler, "detach_llm_context", flaky_detach)

    response = await AsyncCompletionWrapper(handler, business)(
        model="model", stream=True
    )

    assert response is raw_stream
    assert trace.get_current_span() is before
    assert [chunk async for chunk in response] == [expected_chunk]
    assert detach_calls == 2
    assert len(exporter.get_finished_spans()) == 1
