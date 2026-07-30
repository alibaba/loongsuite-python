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

"""QwenPaw 2 ``Runtime.run`` Entry lifecycle and isolation tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from opentelemetry import trace
from opentelemetry.instrumentation.qwenpaw import QwenPawInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)


def _request(session_id: str, text: str = "hello"):
    return SimpleNamespace(
        session_id=session_id,
        user_id=f"user-{session_id}",
        channel="console",
        input=[
            SimpleNamespace(
                role="user",
                content=[SimpleNamespace(text=text)],
            )
        ],
    )


def _completed_message(text: str):
    return SimpleNamespace(
        object="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(text=text)],
    )


def _runtime(runtime_module, agent_id: str = "v2-agent"):
    return runtime_module.Runtime(
        workspace=SimpleNamespace(agent_id=agent_id),
        app_services=None,
    )


def _instrument(tracer_provider):
    instrumentor = QwenPawInstrumentor()
    instrumentor.instrument(
        skip_dep_check=True,
        tracer_provider=tracer_provider,
    )
    return instrumentor


async def _task_next(stream):
    async def advance():
        return await stream.__anext__()

    return await asyncio.create_task(advance())


def _entry_spans(span_exporter):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "enter_ai_application_system"
    ]


@pytest.mark.asyncio
async def test_runtime_entry_parents_heartbeat_child_across_tasks(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    child_tracer = trace.get_tracer(
        "qwenpaw-v2-test",
        tracer_provider=tracer_provider,
    )
    expected = _completed_message("done")

    async def fake_run(self, request):
        del self, request

        async def heartbeat_child():
            with child_tracer.start_as_current_span("invoke_agent"):
                return expected

        yield await asyncio.create_task(heartbeat_child())

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    try:
        stream = (
            _runtime(runtime_module).run(_request("session-1")).__aiter__()
        )
        assert await _task_next(stream) is expected
        with pytest.raises(StopAsyncIteration):
            await _task_next(stream)
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    [entry] = [
        span for span in spans if span.name == "enter_ai_application_system"
    ]
    [agent] = [span for span in spans if span.name == "invoke_agent"]
    assert agent.context.trace_id == entry.context.trace_id
    assert agent.parent.span_id == entry.context.span_id
    assert not trace.get_current_span().get_span_context().is_valid
    assert entry.attributes[GEN_AI_SESSION_ID] == "session-1"
    assert entry.attributes[GEN_AI_USER_ID] == "user-session-1"
    assert entry.attributes["qwenpaw.agent_id"] == "v2-agent"
    assert entry.attributes["qwenpaw.channel"] == "console"


@pytest.mark.asyncio
async def test_runtime_sequential_requests_finish_as_separate_traces(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    child_tracer = trace.get_tracer(
        "qwenpaw-v2-test",
        tracer_provider=tracer_provider,
    )

    async def fake_run(self, request):
        del self
        with child_tracer.start_as_current_span("invoke_agent"):
            yield _completed_message(request.session_id)

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    try:
        runtime = _runtime(runtime_module)
        for session_id in ("session-a", "session-b"):
            chunks = [item async for item in runtime.run(_request(session_id))]
            assert len(chunks) == 1
            assert len(_entry_spans(span_exporter)) == (
                1 if session_id == "session-a" else 2
            )
    finally:
        instrumentor.uninstrument()

    entries = _entry_spans(span_exporter)
    assert len(entries) == 2
    assert len({span.context.trace_id for span in entries}) == 2


@pytest.mark.asyncio
async def test_runtime_entry_captures_v2_input_and_output(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "SPAN_ONLY",
    )
    expected = _completed_message("v2-output")

    async def fake_run(self, request):
        del self, request
        yield expected

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    try:
        chunks = [
            item
            async for item in _runtime(runtime_module).run(
                _request("content-session", "v2-input")
            )
        ]
        assert chunks == [expected]
    finally:
        instrumentor.uninstrument()

    [entry] = _entry_spans(span_exporter)
    input_messages = json.loads(entry.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    output_messages = json.loads(
        entry.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    )
    assert input_messages[0]["parts"][0]["content"] == "v2-input"
    assert output_messages[0]["parts"][0]["content"] == "v2-output"


@pytest.mark.asyncio
async def test_runtime_preserves_business_error_when_fail_callback_fails(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    business_error = RuntimeError("business failure")

    async def fake_run(self, request):
        del self, request
        yield _completed_message("partial")
        raise business_error

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)

    def fail_entry(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe failure")

    monkeypatch.setattr(instrumentor._handler, "fail_entry", fail_entry)
    try:
        stream = _runtime(runtime_module).run(_request("error-session"))
        iterator = stream.__aiter__()
        await iterator.__anext__()
        try:
            await iterator.__anext__()
        except RuntimeError as exc:
            assert exc is business_error
        else:
            pytest.fail("business error was not raised")
    finally:
        instrumentor.uninstrument()

    assert len(_entry_spans(span_exporter)) == 1


@pytest.mark.asyncio
async def test_runtime_start_failure_preserves_business_chunks(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    expected = _completed_message("business-result")
    calls = 0

    async def fake_run(self, request):
        nonlocal calls
        del self, request
        calls += 1
        yield expected

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    original_start_entry = instrumentor._handler.start_entry

    def start_entry(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe start failure")

    monkeypatch.setattr(instrumentor._handler, "start_entry", start_entry)
    try:
        chunks = [
            item
            async for item in _runtime(runtime_module).run(
                _request("start-failure")
            )
        ]
        monkeypatch.setattr(
            instrumentor._handler,
            "start_entry",
            original_start_entry,
        )
        clean_chunks = [
            item
            async for item in _runtime(runtime_module).run(
                _request("clean-sibling")
            )
        ]
    finally:
        instrumentor.uninstrument()

    assert calls == 2
    assert chunks == [expected]
    assert clean_chunks == [expected]
    assert len(_entry_spans(span_exporter)) == 1


@pytest.mark.asyncio
async def test_runtime_mapping_and_stop_failures_preserve_business_chunks(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    expected = _completed_message("business-result")

    async def fake_run(self, request):
        del self, request
        yield expected

    def map_output(item):
        del item
        raise ValueError("probe mapping failure")

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    monkeypatch.setattr(
        "opentelemetry.instrumentation.qwenpaw.patch."
        "output_message_from_runtime_item",
        map_output,
    )
    instrumentor = _instrument(tracer_provider)

    def stop_entry(*args, **kwargs):
        del args, kwargs
        raise ValueError("probe stop failure")

    monkeypatch.setattr(instrumentor._handler, "stop_entry", stop_entry)
    try:
        chunks = [
            item
            async for item in _runtime(runtime_module).run(
                _request("stop-failure")
            )
        ]
    finally:
        instrumentor.uninstrument()

    assert chunks == [expected]
    assert len(_entry_spans(span_exporter)) == 1


@pytest.mark.asyncio
async def test_runtime_cancellation_finishes_entry(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    started = asyncio.Event()
    never = asyncio.Event()

    async def fake_run(self, request):
        del self, request
        started.set()
        await never.wait()
        yield None

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    try:
        stream = _runtime(runtime_module).run(_request("cancel-session"))
        task = asyncio.create_task(stream.__anext__())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        instrumentor.uninstrument()

    assert len(_entry_spans(span_exporter)) == 1


@pytest.mark.asyncio
async def test_runtime_aclose_closes_business_stream_once(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    expected = _completed_message("first")
    closed = 0

    async def fake_run(self, request):
        nonlocal closed
        del self, request
        try:
            yield expected
            yield _completed_message("second")
        finally:
            closed += 1

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    try:
        stream = _runtime(runtime_module).run(_request("close-session"))
        assert await stream.__anext__() is expected
        await stream.aclose()
    finally:
        instrumentor.uninstrument()

    assert closed == 1
    assert len(_entry_spans(span_exporter)) == 1


@pytest.mark.asyncio
async def test_runtime_aclose_error_still_finishes_entry(
    runtime_module,
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    close_error = ValueError("business close failure")

    async def fake_run(self, request):
        del self, request
        try:
            yield _completed_message("first")
        finally:
            raise close_error

    monkeypatch.setattr(runtime_module.Runtime, "run", fake_run)
    instrumentor = _instrument(tracer_provider)
    try:
        stream = _runtime(runtime_module).run(_request("close-error"))
        await stream.__anext__()
        try:
            await stream.aclose()
        except ValueError as exc:
            assert exc is close_error
        else:
            pytest.fail("business close error was not raised")
    finally:
        instrumentor.uninstrument()

    assert len(_entry_spans(span_exporter)) == 1
