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
import inspect

import pytest
from agentuniverse.agent.agent import Agent
from agentuniverse.base.annotation import trace as trace_module
from agentuniverse.base.annotation.trace import trace_llm
from agentuniverse.llm.llm_output import LLMOutput, TokenUsage

from opentelemetry import context
from opentelemetry.instrumentation import agentuniverse as instrumentation
from opentelemetry.instrumentation.agentuniverse._hooks import (
    current_tool_call_id,
)
from opentelemetry.instrumentation.agentuniverse._usage import _current_call
from opentelemetry.trace import get_current_span, set_span_in_context

from .conftest import NATIVE_WRAPPER_GLOBALS, StreamingLLM, StubLLM
from .runtime_app import Workload
from .runtime_provider import USAGE


def test_name_proxy_preserves_otel_context_for_genai_events(
    configured_providers,
):
    from opentelemetry.instrumentation.agentuniverse._common import (
        NamePreservingSpan,
    )

    provider, _, _ = configured_providers
    span = provider.get_tracer("test").start_span("native-name")
    try:
        proxy = NamePreservingSpan(span)
        current = get_current_span(set_span_in_context(proxy))
        assert current.get_span_context() == span.get_span_context()
        assert current.get_span_context().is_valid
        proxy.update_name("canonical-name")
        assert span.name == "native-name"
    finally:
        span.end()


def test_auxiliary_install_failure_restores_every_hook_and_allows_retry(
    monkeypatch,
):
    original = {
        name: getattr(trace_module, name) for name in NATIVE_WRAPPER_GLOBALS
    }
    before = getattr(Agent, "on_before_llm_round", None)
    after = getattr(Agent, "on_after_llm_round", None)
    install = instrumentation.install_auxiliary_hooks

    def fail():
        raise RuntimeError("synthetic auxiliary hook install failure")

    monkeypatch.setattr(instrumentation, "install_auxiliary_hooks", fail)
    instrumentor = instrumentation.AgentUniverseInstrumentor()
    instrumentor.instrument()
    assert not instrumentor.is_instrumented_by_opentelemetry
    assert getattr(Agent, "on_before_llm_round", None) is before
    assert getattr(Agent, "on_after_llm_round", None) is after
    assert all(
        getattr(trace_module, name) is value
        for name, value in original.items()
    )
    monkeypatch.setattr(instrumentation, "install_auxiliary_hooks", install)
    instrumentor.instrument()
    assert instrumentor.is_instrumented_by_opentelemetry


def test_uninstrument_preserves_later_hook_owners():
    converter = inspect.getattr_static(TokenUsage, "from_openai")
    instrumentor = instrumentation.AgentUniverseInstrumentor()
    instrumentor.instrument()

    def third_party(*args, **kwargs):
        return None

    third_converter = classmethod(third_party)
    try:
        TokenUsage.from_openai = third_converter
        instrumentor.uninstrument()
        assert (
            inspect.getattr_static(TokenUsage, "from_openai")
            is third_converter
        )
    finally:
        TokenUsage.from_openai = converter


def test_provider_totals_do_not_mutate_business_usage(loongsuite_harness):
    original = TokenUsage.from_openai(USAGE)
    response = LLMOutput(text="answer", usage=original, raw={"usage": USAGE})

    class NativeLLM(StubLLM):
        @trace_llm
        def call(self, **kwargs):
            return response

    assert NativeLLM().call() is response
    assert response.usage is original
    assert original.total_tokens == 38
    assert (
        loongsuite_harness.spans()[0].attributes["au.llm.usage.total_tokens"]
        == 30
    )


def test_unread_sync_stream_close_finalizes_once(loongsuite_harness):
    before = context.get_current()
    stream = StreamingLLM().call(prompt="hello")
    assert context.get_current() is before
    stream.close()
    stream.close()
    assert len(loongsuite_harness.spans()) == 1
    assert context.get_current() is before
    assert _current_call.get() is None


def test_cross_task_cancel_preserves_business_exception_and_origin(
    loongsuite_harness,
):
    error = asyncio.CancelledError("synthetic cancellation")

    class CancelLLM(StubLLM):
        @trace_llm
        async def call(self, **kwargs):
            async def stream():
                yield LLMOutput(text="first")
                raise error

            return stream()

    async def run():
        before = context.get_current()
        stream = await CancelLLM().call()
        assert context.get_current() is before

        async def consume():
            assert (await stream.__anext__()).text == "first"
            with pytest.raises(asyncio.CancelledError) as raised:
                await stream.__anext__()
            assert raised.value is error
            assert _current_call.get() is None
            assert current_tool_call_id() is None

        await asyncio.create_task(consume())
        await stream.aclose()
        assert context.get_current() is before
        StubLLM().call(prompt="clean sibling")
        assert context.get_current() is before

    asyncio.run(run())
    spans = loongsuite_harness.spans()
    assert len(spans) == 2
    assert all(span.parent is None for span in spans)


def test_concurrent_collector_fault_keeps_business_and_clean_sibling(
    monkeypatch, configured_providers, session_exporter
):
    from opentelemetry.instrumentation.agentuniverse import _llm

    provider, meter_provider, _ = configured_providers
    instrumentor = instrumentation.AgentUniverseInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider, meter_provider=meter_provider
    )
    original = _llm.chunk_record
    failures = []

    def fail_once(chunk):
        if not failures:
            failures.append("synthetic collector fault")
            raise RuntimeError(failures[0])
        return original(chunk)

    monkeypatch.setattr(_llm, "chunk_record", fail_once)
    before = context.get_current()
    result = asyncio.run(Workload().run(provider))
    assert result["request_count"] == 24 and len(failures) == 1
    assert context.get_current() is before
    assert _current_call.get() is None
    assert current_tool_call_id() is None
    StubLLM().call(prompt="clean sibling")
    assert context.get_current() is before
    spans = session_exporter.get_finished_spans()
    assert len(spans) == 52
    assert spans[-1].parent is None
