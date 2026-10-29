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

"""Protocol semantics, lifecycle, and version-compatible real SDK contracts."""

import asyncio

import httpx
import pytest

from opentelemetry import context, trace
from opentelemetry.instrumentation.a2a import _telemetry
from opentelemetry.instrumentation.a2a._semconv import attributes
from opentelemetry.instrumentation.a2a._wrappers import wrap
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.trace import SpanKind, StatusCode


@pytest.mark.parametrize(
    "headers", [{}, {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}]
)
@pytest.mark.asyncio
async def test_client_real_sdk(telemetry, headers):
    provider, exporter, _ = telemetry
    observed = []
    try:
        from a2a.types.a2a_pb2 import Role

        modern = True
    except ImportError:
        modern = False
    from a2a.types import Message, SendMessageRequest

    async def respond(request):
        observed.append(dict(request.headers))
        message = {
            "messageId": "reply",
            "role": "ROLE_AGENT" if modern else "agent",
            "parts": [
                {"text": "hello", **({} if modern else {"kind": "text"})}
            ],
            "contextId": "session",
        }
        if not modern:
            message["kind"] = "message"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "test",
                "result": {"message": message} if modern else message,
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), headers=headers
    ) as http:
        if modern:
            from a2a.client.transports.jsonrpc import JsonRpcTransport
            from a2a.types import AgentCard, Part

            client = JsonRpcTransport(
                http,
                AgentCard(name="remote", version="1"),
                "http://agent.test/rpc",
            )
            request = SendMessageRequest(
                message=Message(
                    message_id="input",
                    context_id="session",
                    role=Role.ROLE_USER,
                    parts=[Part(text="hi")],
                )
            )
        else:
            from a2a.client import A2AClient

            client = A2AClient(http, url="http://agent.test/rpc")
            request = SendMessageRequest(
                id="test",
                params={
                    "message": {
                        "messageId": "input",
                        "contextId": "session",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                    }
                },
            )
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("caller") as parent:
            await client.send_message(request)
            assert trace.get_current_span() is parent
    spans = exporter.get_finished_spans()
    calls = [
        s
        for s in spans
        if s.attributes.get("a2a.method.name") == "SendMessage"
    ]
    assert len(calls) == 1
    span = calls[0]
    assert span.kind == SpanKind.CLIENT
    assert span.name == "SendMessage"
    assert span.attributes["a2a.message.id"] == "input"
    assert span.attributes["gen_ai.conversation.id"] == "session"
    assert span.parent.span_id == parent.get_span_context().span_id
    assert observed[0].get("traceparent") == headers.get("traceparent")
    assert not any("gen_ai.operation.name" in s.attributes for s in spans)


def test_instrumentation_does_not_patch_httpx(telemetry):
    provider, _, instrumentor = telemetry
    instrumentor.uninstrument()
    original = httpx.AsyncClient.send
    instrumentor.instrument(tracer_provider=provider)
    assert httpx.AsyncClient.send is original


@pytest.mark.asyncio
async def test_server_uses_current_context_not_request_headers(telemetry):
    from types import SimpleNamespace

    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")
    headers = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}
    call_context = SimpleNamespace(state={"headers": headers})

    async def business(*args, **kwargs):
        return None

    token = context.attach(context.Context())
    try:
        # The transport can establish context without a recording SERVER span.
        remote = trace.SpanContext(
            3, 4, is_remote=True, trace_flags=trace.TraceFlags(1)
        )
        with trace.use_span(trace.NonRecordingSpan(remote)):
            await wrap(tracer, "GetTask", True, False)(
                business, None, ({"id": "task"}, call_context), {}
            )
        # Raw trace headers alone must not establish a parent in the A2A plugin.
        await wrap(tracer, "GetTask", True, False)(
            business, None, ({"id": "task"}, call_context), {}
        )
    finally:
        context.detach(token)
    inherited, standalone = exporter.get_finished_spans()
    assert inherited.parent == remote
    assert inherited.context.trace_id == remote.trace_id
    assert standalone.parent is None
    assert headers["traceparent"] == "00-" + "1" * 32 + "-" + "2" * 16 + "-01"


@pytest.mark.parametrize("server", [False, True])
@pytest.mark.parametrize(
    "failure", ["mapping", "start", "attach", "finish-attributes", "export"]
)
@pytest.mark.asyncio
async def test_probe_failure_preserves_business_and_sibling(
    telemetry, monkeypatch, server, failure
):
    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")
    result = object()
    calls = []

    async def business(*args, **kwargs):
        calls.append(1)
        return result

    wrapper = wrap(tracer, "SendMessage", server, False)

    def fail(*args, **kwargs):
        raise RuntimeError("probe failure")

    with tracer.start_as_current_span("parent") as parent:
        with monkeypatch.context() as patch:
            if failure == "mapping":
                patch.setattr(_telemetry, "attributes", fail)
            elif failure == "start":
                patch.setattr(tracer, "start_span", fail)
            elif failure == "attach":
                patch.setattr(_telemetry.context, "attach", fail)
            elif failure == "finish-attributes":
                original = _telemetry.attributes
                patch.setattr(
                    _telemetry,
                    "attributes",
                    lambda value, **kw: (
                        original(value, **kw) if kw else fail()
                    ),
                )
            else:
                patch.setattr(exporter, "export", fail)
            assert await wrapper(business, None, (), {}) is result
        assert trace.get_current_span() is parent
        assert await wrapper(business, None, (), {}) is result
        assert trace.get_current_span() is parent
    assert calls == [1, 1]


@pytest.mark.parametrize(
    "exception", [ValueError("original"), asyncio.CancelledError()]
)
@pytest.mark.asyncio
async def test_original_exception_and_cleanup(
    telemetry, monkeypatch, exception
):
    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")
    count = 0

    async def business():
        nonlocal count
        count += 1
        raise exception

    with tracer.start_as_current_span("parent") as parent:
        with pytest.raises(type(exception)) as caught:
            await wrap(tracer, "GetTask", False, False)(business, None, (), {})
        assert caught.value is exception
        assert trace.get_current_span() is parent
    assert count == 1
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_borrowed_server_not_ended_or_duplicated(telemetry):
    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")

    async def business(*args):
        return {"id": "task", "status": {"state": "completed"}}

    with tracer.start_as_current_span(
        "POST /rpc",
        kind=SpanKind.SERVER,
        attributes={"http.request.method": "POST"},
    ) as parent:
        await wrap(tracer, "GetTask", True, False)(
            business, None, ({"id": "task"},), {}
        )
        assert trace.get_current_span() is parent
        assert parent.is_recording()
        assert not exporter.get_finished_spans()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "GetTask"
    assert spans[0].attributes["a2a.task.state"] == "TASK_STATE_COMPLETED"


@pytest.mark.asyncio
async def test_grpc_server_name_preserved(telemetry):
    provider, _, _ = telemetry
    tracer = provider.get_tracer("test")

    async def business():
        return None

    with tracer.start_as_current_span(
        "a2a.A2AService/GetTask",
        kind=SpanKind.SERVER,
        attributes={"rpc.system": "grpc"},
    ) as parent:
        await wrap(tracer, "GetTask", True, False)(business, None, (), {})
        assert parent.name == "a2a.A2AService/GetTask"
        assert parent.attributes["a2a.method.name"] == "GetTask"


@pytest.mark.asyncio
async def test_stream_terminal_context_cross_task(telemetry):
    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")
    chunks = [
        {"taskId": "t", "status": {"state": "working"}},
        {"messageId": "reply"},
    ]
    calls = []

    async def business():
        calls.append(1)
        for chunk in chunks:
            yield chunk

    with tracer.start_as_current_span("creator") as parent:
        stream = wrap(tracer, "SendStreamingMessage", False, True)(
            business, None, (), {}
        )
        assert trace.get_current_span() is parent

        async def consume():
            value = await stream.__anext__()
            assert trace.get_current_span() is parent
            return value

        assert await asyncio.create_task(consume()) is chunks[0]
        assert not exporter.get_finished_spans()
        assert await asyncio.create_task(consume()) is chunks[1]
        assert (
            len(exporter.get_finished_spans()) == 1
        )  # Before yielding terminal result.
        await asyncio.create_task(stream.aclose())
        await stream.aclose()
        assert trace.get_current_span() is parent
    assert calls == [1]
    assert len(exporter.get_finished_spans()) == 2


@pytest.mark.parametrize("mode", ["close", "cancel", "error", "exhaust"])
@pytest.mark.asyncio
async def test_stream_cleanup(telemetry, mode):
    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")
    error = (
        asyncio.CancelledError()
        if mode == "cancel"
        else ValueError("original")
    )
    closed = []

    async def business():
        try:
            yield {"taskId": "task"}
            if mode in {"cancel", "error"}:
                raise error
        finally:
            closed.append(1)

    with tracer.start_as_current_span("parent") as parent:
        stream = wrap(tracer, "SubscribeToTask", False, True)(
            business, None, (), {}
        )
        await stream.__anext__()
        if mode == "close":
            await asyncio.create_task(stream.aclose())
        elif mode == "exhaust":
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
        else:
            with pytest.raises(type(error)) as caught:
                await stream.__anext__()
            assert caught.value is error
        assert trace.get_current_span() is parent
        assert len(exporter.get_finished_spans()) == 1
    assert closed == [1]


@pytest.mark.asyncio
async def test_suppression(telemetry):
    provider, exporter, _ = telemetry

    async def business():
        return 7

    with suppress_instrumentation():
        assert (
            await wrap(provider.get_tracer("test"), "GetTask", False, False)(
                business, None, (), {}
            )
            == 7
        )
    assert not exporter.get_finished_spans()


def test_native_tracing_restored_and_user_override(telemetry, monkeypatch):
    import a2a.utils.telemetry as native

    provider, _, instrumentor = telemetry
    original = instrumentor._native[1]
    assert native.trace is not original
    instrumentor.uninstrument()
    assert native.trace is original
    monkeypatch.setenv("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "true")
    instrumentor.instrument(tracer_provider=provider)
    assert native.trace is original


@pytest.mark.asyncio
async def test_disabled_sdk_tracing_preserves_transport_span(telemetry):
    import a2a.utils.telemetry as native

    provider, _, _ = telemetry

    @native.trace_function
    async def internal():
        assert trace.get_current_span() is parent
        raise ValueError("handled by the application")

    with provider.get_tracer("test").start_as_current_span(
        "HTTP", kind=SpanKind.SERVER
    ) as parent:
        with pytest.raises(ValueError):
            await internal()
        assert trace.get_current_span() is parent
        assert parent.is_recording()
        assert parent.status.status_code == StatusCode.UNSET
        assert not parent.events


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            {
                "root": {
                    "result": {
                        "id": "task",
                        "contextId": "session",
                        "status": {"state": "input-required"},
                    }
                }
            },
            {
                "a2a.task.id": "task",
                "gen_ai.conversation.id": "session",
                "a2a.task.state": "TASK_STATE_INPUT_REQUIRED",
            },
        ),
        (
            {
                "message": {
                    "messageId": "m",
                    "parts": [{"text": "secret"}],
                    "referenceTaskIds": ["t"],
                }
            },
            {},
        ),
    ],
)
def test_semantic_mapping_no_content(value, expected):
    assert attributes(value) == expected


@pytest.mark.asyncio
async def test_future_handler_and_uninstrument(telemetry):
    from a2a.server.request_handlers import DefaultRequestHandler

    provider, exporter, instrumentor = telemetry

    class CustomHandler(DefaultRequestHandler):
        def __init__(self):
            self.calls = 0

        async def on_get_task(self, *args, **kwargs):
            self.calls += 1
            return {"id": "task", "status": {"state": "completed"}}

    handler = CustomHandler()
    first = await handler.on_get_task({"id": "task"})
    assert first["id"] == "task"
    assert len(exporter.get_finished_spans()) == 1
    instrumentor.uninstrument()
    await handler.on_get_task({"id": "task"})
    assert len(exporter.get_finished_spans()) == 1
    instrumentor.instrument(tracer_provider=provider)
    await handler.on_get_task({"id": "task"})
    assert len(exporter.get_finished_spans()) == 2
    assert handler.calls == 3


@pytest.mark.asyncio
async def test_unrelated_http_not_injected(telemetry):
    seen = []

    async def respond(request):
        seen.append(request.headers)
        return httpx.Response(200)

    with telemetry[0].get_tracer("test").start_as_current_span("unrelated"):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as http:
            await http.get("http://unrelated.test")
    assert "traceparent" not in seen[0]


@pytest.mark.asyncio
async def test_stream_original_athrow_and_exhaustion(telemetry):
    provider, exporter, _ = telemetry
    tracer = provider.get_tracer("test")
    exception = GeneratorExit()

    async def business():
        yield object()

    with tracer.start_as_current_span("parent") as parent:
        stream = wrap(tracer, "SendStreamingMessage", False, True)(
            business, None, (), {}
        )
        await stream.asend(None)
        with pytest.raises(GeneratorExit) as caught:
            await stream.athrow(exception)
        assert caught.value is exception
        assert trace.get_current_span() is parent
        assert len(exporter.get_finished_spans()) == 1


def test_unknown_numeric_state_keeps_task_metadata():
    assert attributes({"id": "task", "status": {"state": 999}}) == {
        "a2a.task.id": "task"
    }


def test_jsonrpc_request_id_is_not_task_id():
    assert (
        attributes(
            {
                "jsonrpc": "2.0",
                "id": "rpc-id",
                "method": "GetExtendedAgentCard",
            },
            request=True,
        )
        == {}
    )


def test_terminal_task_envelope():
    from opentelemetry.instrumentation.a2a._semconv import terminal

    assert terminal({"task": {"id": "task", "status": {"state": "completed"}}})
