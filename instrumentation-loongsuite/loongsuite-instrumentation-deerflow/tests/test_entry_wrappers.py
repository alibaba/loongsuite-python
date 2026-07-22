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

"""Focused lifecycle tests for DeerFlow application ENTRY wrappers."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context as ContextVarsContext
from contextvars import ContextVar
from threading import Barrier
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from opentelemetry import baggage, context, trace
from opentelemetry.instrumentation.deerflow.internal.constants import (
    DEERFLOW_RUN_STATUS,
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)
from opentelemetry.instrumentation.deerflow.internal.patch import (
    _ClientStreamWrapper,
    _GatewayRunAgentWrapper,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.extended_types import EntryInvocation


class _Event:
    def __init__(self, event_type: str, data: dict[str, Any]):
        self.type = event_type
        self.data = data


class _Client:
    _agent_name = "research-agent"


class _DefaultClient:
    _agent_name = None


def _entry_spans(span_exporter):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.span.kind") == "ENTRY"
    ]


def test_client_stream_uses_isolated_context_and_generated_thread_id(
    handler,
    tracer_provider,
    span_exporter,
    capture_content,
    monkeypatch,
):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.internal.patch._effective_user_id",
        lambda fallback=None: fallback or "deerflow-user",
    )
    observed: dict[str, Any] = {}

    def stream(message, *, thread_id=None, **_kwargs):
        observed["message"] = message
        observed["thread_id"] = thread_id
        observed["span_during_next"] = (
            trace.get_current_span().get_span_context().is_valid
        )
        yield _Event(
            "messages-tuple",
            {"type": "ai", "id": "answer", "content": "hello "},
        )
        observed["span_during_resume"] = (
            trace.get_current_span().get_span_context().is_valid
        )
        yield _Event(
            "messages-tuple",
            {"type": "ai", "id": "answer", "content": "world"},
        )

    wrapper = _ClientStreamWrapper(handler)
    result = wrapper(stream, _Client(), ("question",), {})

    assert not trace.get_current_span().get_span_context().is_valid
    first = next(result)
    assert first.data["content"] == "hello "
    assert not trace.get_current_span().get_span_context().is_valid
    with tracer_provider.get_tracer(__name__).start_as_current_span(
        "consumer-between-yields"
    ) as consumer_span:
        consumer_context = consumer_span.get_span_context()
    second = next(result)
    assert second.data["content"] == "world"
    assert list(result) == []
    assert not trace.get_current_span().get_span_context().is_valid

    assert observed["message"] == "question"
    uuid.UUID(observed["thread_id"])
    assert observed["span_during_next"] is True
    assert observed["span_during_resume"] is True

    entry = _entry_spans(span_exporter)[0]
    assert consumer_context.trace_id != entry.context.trace_id
    assert consumer_context.span_id != entry.context.span_id
    assert entry.attributes[GEN_AI_SESSION_ID] == observed["thread_id"]
    assert entry.attributes[GEN_AI_USER_ID] == "deerflow-user"
    assert entry.attributes["gen_ai.framework"] == "deerflow"
    assert entry.attributes["gen_ai.agent.name"] == "research-agent"
    assert entry.attributes["deerflow.assistant.id"] == "research-agent"
    assert entry.attributes[DEERFLOW_RUN_STATUS] == "success"
    assert entry.attributes["gen_ai.response.time_to_first_token"] >= 0
    assert (
        json.loads(entry.attributes["gen_ai.input.messages"])[0]["parts"][0][
            "content"
        ]
        == "question"
    )
    assert (
        json.loads(entry.attributes["gen_ai.output.messages"])[0]["parts"][0][
            "content"
        ]
        == "hello world"
    )


def test_client_stream_never_consumed_does_not_create_entry(
    handler,
    span_exporter,
):
    observed = {"started": False}

    def stream(_message, *, thread_id=None):
        del thread_id
        observed["started"] = True
        yield _Event("end", {})

    result = _ClientStreamWrapper(handler)(stream, _Client(), ("q",), {})

    assert observed["started"] is False
    assert _entry_spans(span_exporter) == []
    result.close()
    assert observed["started"] is False
    assert _entry_spans(span_exporter) == []


def test_client_stream_keeps_caller_parent_when_consumed_in_other_thread(
    handler,
    tracer_provider,
    span_exporter,
):
    tracer = tracer_provider.get_tracer(__name__)
    wrapper = _ClientStreamWrapper(handler)

    def stream(_message, *, thread_id=None):
        assert thread_id == "cross-thread"
        yield _Event("end", {})

    with tracer.start_as_current_span("producer-caller") as caller:
        caller_context = caller.get_span_context()
        result = wrapper(
            stream,
            _Client(),
            ("question",),
            {"thread_id": "cross-thread"},
        )

    def consume() -> tuple[int, int]:
        with tracer.start_as_current_span("consumer") as consumer:
            consumer_context = consumer.get_span_context()
            list(result)
            return consumer_context.trace_id, consumer_context.span_id

    with ThreadPoolExecutor(max_workers=1) as executor:
        consumer_trace_id, consumer_span_id = executor.submit(consume).result(
            timeout=10
        )

    entry = _entry_spans(span_exporter)[0]
    assert entry.context.trace_id == caller_context.trace_id
    assert entry.parent.span_id == caller_context.span_id
    assert entry.context.trace_id != consumer_trace_id
    assert entry.parent.span_id != consumer_span_id


def test_client_stream_baggage_identity_has_precedence(
    handler,
    span_exporter,
    monkeypatch,
):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.deerflow.internal.patch._effective_user_id",
        lambda fallback=None: fallback or "deerflow-user",
    )

    def stream(_message, *, thread_id=None):
        assert thread_id == "deerflow-thread"
        yield _Event("end", {})

    ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "host-session")
    ctx = baggage.set_baggage(GEN_AI_USER_ID, "host-user", ctx)
    token = context.attach(ctx)
    try:
        result = _ClientStreamWrapper(handler)(
            stream,
            _Client(),
            ("question",),
            {"thread_id": "deerflow-thread"},
        )
        list(result)
    finally:
        context.detach(token)

    entry = _entry_spans(span_exporter)[0]
    assert entry.attributes[GEN_AI_SESSION_ID] == "host-session"
    assert entry.attributes[GEN_AI_USER_ID] == "host-user"


def test_default_client_does_not_invent_assistant_id(handler, span_exporter):
    result = _ClientStreamWrapper(handler)(
        lambda *_args, **_kwargs: iter([_Event("end", {})]),
        _DefaultClient(),
        ("q",),
        {},
    )
    list(result)

    entry = _entry_spans(span_exporter)[0]
    assert entry.attributes["gen_ai.agent.name"] == "lead-agent"
    assert "deerflow.assistant.id" not in entry.attributes


def test_client_stream_close_interrupts_entry_without_error(
    handler,
    span_exporter,
):
    observed = {"closed": False}

    def stream(_message, *, thread_id=None):
        del thread_id
        try:
            yield _Event("messages-tuple", {"type": "ai", "content": "x"})
            yield _Event("end", {})
        finally:
            observed["closed"] = True

    result = _ClientStreamWrapper(handler)(stream, _Client(), ("q",), {})
    next(result)
    result.close()

    assert observed["closed"] is True
    assert not trace.get_current_span().get_span_context().is_valid
    entry = _entry_spans(span_exporter)[0]
    assert entry.status.status_code == StatusCode.UNSET
    assert "error.type" not in entry.attributes
    assert entry.attributes[DEERFLOW_RUN_STATUS] == "interrupted"


@pytest.mark.parametrize(
    ("exception_type", "expected_status"),
    [
        (ValueError, "error"),
        (TimeoutError, "timeout"),
    ],
)
def test_client_stream_exception_fails_entry_with_mapped_status(
    exception_type,
    expected_status,
    handler,
    span_exporter,
):
    def stream(_message, *, thread_id=None):
        del thread_id
        yield _Event("messages-tuple", {"type": "ai", "content": "x"})
        raise exception_type("stream failed")

    result = _ClientStreamWrapper(handler)(stream, _Client(), ("q",), {})
    next(result)
    with pytest.raises(exception_type, match="stream failed"):
        next(result)

    entry = _entry_spans(span_exporter)[0]
    assert entry.status.status_code == StatusCode.ERROR
    assert entry.attributes["error.type"] == exception_type.__name__
    assert entry.attributes[DEERFLOW_RUN_STATUS] == expected_status


def test_client_stream_reuses_active_host_entry(handler, span_exporter):
    host = EntryInvocation(session_id="host")
    handler.start_entry(host)
    try:
        result = _ClientStreamWrapper(handler)(
            lambda *_args, **_kwargs: iter([_Event("end", {})]),
            _Client(),
            ("q",),
            {},
        )
        list(result)
    finally:
        handler.stop_entry(host)

    assert len(_entry_spans(span_exporter)) == 1


def test_client_stream_reuses_host_entry_under_current_child_span(
    handler,
    tracer_provider,
    span_exporter,
):
    tracer = tracer_provider.get_tracer(__name__)
    observed = {}

    def stream(_message, *, thread_id=None):
        del thread_id
        observed["span_id"] = (
            trace.get_current_span().get_span_context().span_id
        )
        yield _Event("end", {})

    host = EntryInvocation(session_id="host")
    handler.start_entry(host)
    try:
        with tracer.start_as_current_span("host-child") as child:
            child_span_id = child.get_span_context().span_id
            result = _ClientStreamWrapper(handler)(
                stream,
                _Client(),
                ("q",),
                {},
            )
            list(result)
    finally:
        handler.stop_entry(host)

    assert observed["span_id"] == child_span_id
    assert len(_entry_spans(span_exporter)) == 1


def test_client_stream_reuses_host_entry_across_consumer_context(
    handler,
    tracer_provider,
    span_exporter,
):
    tracer = tracer_provider.get_tracer(__name__)
    observed = {}

    def stream(_message, *, thread_id=None):
        del thread_id
        observed["session_id"] = baggage.get_baggage(GEN_AI_SESSION_ID)
        observed["user_id"] = baggage.get_baggage(GEN_AI_USER_ID)
        with tracer.start_as_current_span("hosted-stream-work"):
            yield _Event("end", {})

    host = EntryInvocation(session_id="host-session", user_id="host-user")
    handler.start_entry(host)
    try:
        result = _ClientStreamWrapper(handler)(
            stream,
            _Client(),
            ("q",),
            {},
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                ContextVarsContext().run,
                lambda: list(result),
            ).result(timeout=10)
    finally:
        handler.stop_entry(host)

    entry = _entry_spans(span_exporter)[0]
    work = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "hosted-stream-work"
    )
    assert len(_entry_spans(span_exporter)) == 1
    assert work.context.trace_id == entry.context.trace_id
    assert work.parent.span_id == entry.context.span_id
    assert observed == {
        "session_id": "host-session",
        "user_id": "host-user",
    }


def test_client_chat_reuses_stream_and_creates_one_entry(
    handler,
    span_exporter,
):
    class ChatClient:
        _agent_name = "chat-agent"

        def __init__(self):
            self.stream_calls = 0
            self._stream_wrapper = _ClientStreamWrapper(handler)

        def _raw_stream(self, message, *, thread_id=None):
            self.stream_calls += 1
            assert message == "question"
            assert thread_id == "chat-thread"
            yield _Event(
                "messages-tuple",
                {"type": "ai", "id": "answer", "content": "answer"},
            )

        def stream(self, message, *, thread_id=None):
            return self._stream_wrapper(
                self._raw_stream,
                self,
                (message,),
                {"thread_id": thread_id},
            )

        def chat(self, message, *, thread_id=None):
            return "".join(
                event.data.get("content", "")
                for event in self.stream(message, thread_id=thread_id)
            )

    client = ChatClient()

    assert client.chat("question", thread_id="chat-thread") == "answer"
    assert client.stream_calls == 1
    entries = _entry_spans(span_exporter)
    assert len(entries) == 1
    assert entries[0].attributes[GEN_AI_SESSION_ID] == "chat-thread"
    assert entries[0].attributes[DEERFLOW_RUN_STATUS] == "success"


def test_client_stream_concurrent_threads_isolate_identity_and_trace(
    handler,
    tracer_provider,
    span_exporter,
):
    barrier = Barrier(2)
    wrapper = _ClientStreamWrapper(handler)
    tracer = tracer_provider.get_tracer(__name__)

    def consume(suffix: str) -> None:
        ctx = baggage.set_baggage(
            GEN_AI_SESSION_ID,
            f"baggage-session-{suffix}",
        )
        ctx = baggage.set_baggage(
            GEN_AI_USER_ID,
            f"baggage-user-{suffix}",
            ctx,
        )
        token = context.attach(ctx)
        try:

            def stream(_message, *, thread_id=None):
                assert thread_id == f"deerflow-thread-{suffix}"
                barrier.wait(timeout=5)
                with tracer.start_as_current_span(f"embedded-child-{suffix}"):
                    yield _Event("end", {})

            result = wrapper(
                stream,
                _Client(),
                (f"question-{suffix}",),
                {"thread_id": f"deerflow-thread-{suffix}"},
            )
            list(result)
            assert not trace.get_current_span().get_span_context().is_valid
        finally:
            context.detach(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(consume, suffix) for suffix in ("a", "b")]
        for future in futures:
            future.result(timeout=10)

    entries = {
        span.attributes[GEN_AI_SESSION_ID]: span
        for span in _entry_spans(span_exporter)
    }
    children = {
        span.name.removeprefix("embedded-child-"): span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("embedded-child-")
    }
    assert set(entries) == {"baggage-session-a", "baggage-session-b"}
    assert set(children) == {"a", "b"}
    assert len({entry.context.trace_id for entry in entries.values()}) == 2

    for suffix in ("a", "b"):
        entry = entries[f"baggage-session-{suffix}"]
        child = children[suffix]
        assert entry.attributes[GEN_AI_USER_ID] == f"baggage-user-{suffix}"
        assert child.context.trace_id == entry.context.trace_id
        assert child.parent.span_id == entry.context.span_id


def test_client_stream_records_ttft_for_empty_tool_call_event_without_capture(
    handler,
    span_exporter,
    monkeypatch,
):
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        raising=False,
    )

    def stream(_message, *, thread_id=None):
        del thread_id
        yield _Event(
            "messages-tuple",
            {
                "type": "ai",
                "id": "tool-decision",
                "content": "",
                "tool_calls": [{"name": "lookup", "id": "call-1"}],
            },
        )

    result = _ClientStreamWrapper(handler)(stream, _Client(), ("q",), {})
    list(result)

    entry = _entry_spans(span_exporter)[0]
    assert entry.attributes["gen_ai.response.time_to_first_token"] >= 0
    assert "gen_ai.output.messages" not in entry.attributes


def test_client_stream_generates_and_isolates_correlation_id(
    handler,
    span_exporter,
    monkeypatch,
):
    current_trace_id: ContextVar[str | None] = ContextVar(
        "fake_deerflow_trace_id",
        default=None,
    )
    generated_trace_id = "generated-correlation-id"

    trace_context_module = ModuleType("deerflow.trace_context")
    trace_context_module.get_current_trace_id = current_trace_id.get
    trace_context_module.generate_trace_id = lambda: generated_trace_id
    trace_context_module.set_current_trace_id = current_trace_id.set
    trace_context_module.reset_current_trace_id = current_trace_id.reset
    app_config_module = ModuleType("deerflow.config.app_config")
    app_config_module.is_trace_correlation_enabled = lambda _config: True
    monkeypatch.setitem(
        sys.modules,
        "deerflow.trace_context",
        trace_context_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "deerflow.config.app_config",
        app_config_module,
    )

    observed = {}

    def stream(_message, *, thread_id=None):
        del thread_id
        observed["inner_trace_id"] = current_trace_id.get()
        yield _Event("end", {})

    client = SimpleNamespace(
        _agent_name="research-agent",
        _app_config=object(),
    )
    assert current_trace_id.get() is None
    result = _ClientStreamWrapper(handler)(stream, client, ("q",), {})
    assert current_trace_id.get() is None
    list(result)

    entry = _entry_spans(span_exporter)[0]
    assert observed["inner_trace_id"] == generated_trace_id
    assert entry.attributes["deerflow.trace.id"] == generated_trace_id
    assert current_trace_id.get() is None


@pytest.mark.parametrize("failure", ["resolve", "bind", "reset"])
def test_client_trace_correlation_failure_does_not_break_stream(
    failure,
    handler,
    span_exporter,
    monkeypatch,
    caplog,
):
    caplog.set_level(
        logging.DEBUG,
        logger="opentelemetry.instrumentation.deerflow.internal.patch",
    )

    def fail() -> None:
        raise RuntimeError(f"{failure} failed")

    trace_context_module = ModuleType("deerflow.trace_context")
    trace_context_module.get_current_trace_id = lambda: None
    trace_context_module.generate_trace_id = (
        fail if failure == "resolve" else lambda: "correlation-id"
    )
    trace_context_module.set_current_trace_id = (
        (lambda _trace_id: fail())
        if failure == "bind"
        else lambda _trace_id: object()
    )
    trace_context_module.reset_current_trace_id = (
        (lambda _token: fail()) if failure == "reset" else lambda _token: None
    )
    app_config_module = ModuleType("deerflow.config.app_config")
    app_config_module.is_trace_correlation_enabled = lambda _config: True
    monkeypatch.setitem(
        sys.modules,
        "deerflow.trace_context",
        trace_context_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "deerflow.config.app_config",
        app_config_module,
    )

    def stream(_message, *, thread_id=None):
        del thread_id
        yield _Event("end", {})

    client = SimpleNamespace(
        _agent_name="research-agent",
        _app_config=object(),
    )
    list(_ClientStreamWrapper(handler)(stream, client, ("q",), {}))

    entry = _entry_spans(span_exporter)[0]
    assert entry.status.status_code is StatusCode.UNSET
    if failure == "resolve":
        assert "deerflow.trace.id" not in entry.attributes
    else:
        assert entry.attributes["deerflow.trace.id"] == "correlation-id"
    expected_log = {
        "resolve": "Failed to resolve DeerFlow trace correlation id",
        "bind": "Failed to bind DeerFlow trace correlation id",
        "reset": "Failed to reset DeerFlow trace context",
    }[failure]
    assert expected_log in caplog.messages


@pytest.mark.asyncio
async def test_gateway_entry_uses_mutated_success_status(
    handler,
    tracer_provider,
    span_exporter,
    capture_content,
    monkeypatch,
):
    def resolve_runtime_user_id(runtime):
        return runtime.context.get("user_id", "effective-user")

    deerflow_module = ModuleType("deerflow")
    runtime_module = ModuleType("deerflow.runtime")
    user_context_module = ModuleType("deerflow.runtime.user_context")
    user_context_module.resolve_runtime_user_id = resolve_runtime_user_id
    deerflow_module.runtime = runtime_module
    runtime_module.user_context = user_context_module
    monkeypatch.setitem(sys.modules, "deerflow", deerflow_module)
    monkeypatch.setitem(sys.modules, "deerflow.runtime", runtime_module)
    monkeypatch.setitem(
        sys.modules,
        "deerflow.runtime.user_context",
        user_context_module,
    )
    record = SimpleNamespace(
        run_id="run-1",
        thread_id="thread-1",
        assistant_id="assistant-1",
        user_id="stale-record-user",
        metadata={"deerflow_trace_id": "trace-1"},
        status="pending",
        error=None,
        last_ai_message=None,
    )

    async def run_agent(_bridge, _manager, mutable_record, **_kwargs):
        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span("gateway-child"):
            pass
        mutable_record.status = "success"
        mutable_record.last_ai_message = "finished"

    wrapper = _GatewayRunAgentWrapper(handler)
    await wrapper(
        run_agent,
        None,
        (None, None, record),
        {
            "graph_input": {"messages": [{"role": "user", "content": "go"}]},
            "config": {
                "context": {"user_id": "runtime-user"},
                "run_name": "configured-agent",
            },
        },
    )

    entry = _entry_spans(span_exporter)[0]
    child = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "gateway-child"
    )
    assert child.parent.span_id == entry.context.span_id
    assert entry.status.status_code != StatusCode.ERROR
    assert entry.attributes[DEERFLOW_RUN_STATUS] == "success"
    assert entry.attributes["deerflow.assistant.id"] == "assistant-1"
    assert entry.attributes["deerflow.run.id"] == "run-1"
    assert entry.attributes["deerflow.trace.id"] == "trace-1"
    assert entry.attributes[GEN_AI_SESSION_ID] == "thread-1"
    assert entry.attributes[GEN_AI_USER_ID] == "runtime-user"
    assert entry.attributes["gen_ai.agent.name"] == "configured-agent"
    assert (
        json.loads(entry.attributes["gen_ai.output.messages"])[0]["parts"][0][
            "content"
        ]
        == "finished"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        ("error", "DeerFlowRunError"),
        ("timeout", "DeerFlowRunTimeout"),
    ],
)
async def test_gateway_terminal_status_fails_entry(
    status,
    error_type,
    handler,
    span_exporter,
):
    record = SimpleNamespace(
        run_id="run-1",
        thread_id="thread-1",
        assistant_id=None,
        user_id=None,
        metadata={},
        status="running",
        error="terminal status",
    )

    async def run_agent(_bridge, _manager, mutable_record, **_kwargs):
        mutable_record.status = status

    await _GatewayRunAgentWrapper(handler)(
        run_agent,
        None,
        (None, None, record),
        {"graph_input": {}, "config": {}},
    )

    entry = _entry_spans(span_exporter)[0]
    assert entry.status.status_code == StatusCode.ERROR
    assert entry.attributes["error.type"] == error_type
    assert entry.attributes[DEERFLOW_RUN_STATUS] == status


@pytest.mark.asyncio
async def test_gateway_interrupted_status_stops_entry_without_error(
    handler,
    span_exporter,
):
    record = SimpleNamespace(
        run_id="run-interrupted",
        thread_id="thread-interrupted",
        assistant_id=None,
        user_id=None,
        metadata={},
        status="running",
        error=None,
    )

    async def run_agent(_bridge, _manager, mutable_record, **_kwargs):
        mutable_record.status = "interrupted"

    await _GatewayRunAgentWrapper(handler)(
        run_agent,
        None,
        (None, None, record),
        {"graph_input": {}, "config": {}},
    )

    entry = _entry_spans(span_exporter)[0]
    assert entry.status.status_code == StatusCode.UNSET
    assert "error.type" not in entry.attributes
    assert entry.attributes[DEERFLOW_RUN_STATUS] == "interrupted"


@pytest.mark.asyncio
async def test_gateway_cancellation_interrupts_entry_without_error(
    handler,
    span_exporter,
):
    record = SimpleNamespace(
        run_id="run-cancelled",
        thread_id="thread-cancelled",
        assistant_id=None,
        metadata={},
        status="running",
        error=None,
    )

    async def run_agent(_bridge, _manager, mutable_record, **_kwargs):
        mutable_record.status = "interrupted"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _GatewayRunAgentWrapper(handler)(
            run_agent,
            None,
            (None, None, record),
            {"graph_input": {}, "config": {}},
        )

    entry = _entry_spans(span_exporter)[0]
    assert entry.status.status_code == StatusCode.UNSET
    assert "error.type" not in entry.attributes
    assert entry.attributes[DEERFLOW_RUN_STATUS] == "interrupted"


@pytest.mark.asyncio
async def test_gateway_reuses_active_host_entry(handler, span_exporter):
    record = SimpleNamespace(
        run_id="run-hosted",
        thread_id="thread-hosted",
        assistant_id=None,
        metadata={},
        status="running",
        error=None,
    )
    observed = {}

    async def run_agent(_bridge, _manager, mutable_record, **_kwargs):
        observed["span_id"] = (
            trace.get_current_span().get_span_context().span_id
        )
        mutable_record.status = "success"

    host = EntryInvocation(session_id="host-session")
    handler.start_entry(host)
    try:
        host_span_id = trace.get_current_span().get_span_context().span_id
        await _GatewayRunAgentWrapper(handler)(
            run_agent,
            None,
            (None, None, record),
            {"graph_input": {}, "config": {}},
        )
    finally:
        handler.stop_entry(host)

    assert observed["span_id"] == host_span_id
    assert len(_entry_spans(span_exporter)) == 1
