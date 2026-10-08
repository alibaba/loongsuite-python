"""Real SDK client, dispatcher, executor and response serialization over ASGI HTTP."""

import asyncio
import inspect
import uuid

import httpx
import pytest

from opentelemetry import trace
from opentelemetry.trace import SpanKind

try:
    from a2a.types import a2a_pb2  # noqa: F401

    MODERN = True
except ImportError:
    MODERN = False


def make_card():
    from a2a.types import AgentCard

    if MODERN:
        from a2a.types import AgentInterface

        return AgentCard(
            name="Echo agent",
            description="Test echo",
            version="1.0",
            supported_interfaces=[
                AgentInterface(
                    url="http://agent.test/rpc",
                    protocol_binding="JSONRPC",
                    protocol_version="1.0",
                )
            ],
            capabilities={"streaming": True},
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
        )
    return AgentCard(
        name="Echo agent",
        description="Test echo",
        version="1.0",
        url="http://agent.test/rpc",
        capabilities={"streaming": True},
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[],
    )


def make_message(session, text, agent=False):
    from a2a.types import Message

    if MODERN:
        from a2a.types import Part, Role

        return Message(
            message_id=uuid.uuid4().hex,
            context_id=session,
            role=Role.ROLE_AGENT if agent else Role.ROLE_USER,
            parts=[Part(text=text)],
        )
    return Message(
        messageId=uuid.uuid4().hex,
        contextId=session,
        role="agent" if agent else "user",
        parts=[{"kind": "text", "text": text}],
    )


def make_request(session, streaming=False):
    from a2a.types import SendMessageRequest

    message = make_message(session, "hello")
    if MODERN:
        return SendMessageRequest(message=message)
    from a2a.types import SendStreamingMessageRequest

    cls = SendStreamingMessageRequest if streaming else SendMessageRequest
    return cls(id=uuid.uuid4().hex, params={"message": message})


def make_client(http):
    if MODERN:
        from a2a.client.transports.jsonrpc import JsonRpcTransport

        return JsonRpcTransport(http, make_card(), "http://agent.test/rpc")
    from a2a.client import A2AClient

    return A2AClient(http, agent_card=make_card())


def make_app(calls):
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore

    class EchoExecutor(AgentExecutor):
        async def execute(self, context, event_queue):
            calls.append(
                (
                    context.context_id,
                    trace.get_current_span().get_span_context(),
                )
            )
            pending = event_queue.enqueue_event(
                make_message(context.context_id, "echo", agent=True)
            )
            if inspect.isawaitable(pending):
                await pending

        async def cancel(self, context, event_queue):
            return None

    kwargs = {
        "agent_executor": EchoExecutor(),
        "task_store": InMemoryTaskStore(),
    }
    if MODERN:
        kwargs["agent_card"] = make_card()
    handler = DefaultRequestHandler(**kwargs)
    if MODERN:
        from a2a.server.routes import create_jsonrpc_routes
        from starlette.applications import Starlette

        return Starlette(
            routes=create_jsonrpc_routes(handler, "/rpc")
        ), handler
    from a2a.server.apps import A2AStarletteApplication

    return A2AStarletteApplication(make_card(), handler).build(
        rpc_url="/rpc"
    ), handler


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("http_instrumentation", [False, True])
@pytest.mark.asyncio
async def test_real_http_propagation(
    telemetry, streaming, http_instrumentation
):
    provider, exporter, _ = telemetry
    calls = []
    app, handler = make_app(calls)
    if http_instrumentation:
        asgi = pytest.importorskip("opentelemetry.instrumentation.asgi")
        http_probe = pytest.importorskip("opentelemetry.instrumentation.httpx")
        app = asgi.OpenTelemetryMiddleware(app, tracer_provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={"a2a-version": "1.0"} if MODERN else {},
    ) as http:
        if http_instrumentation:
            http_probe.HTTPXClientInstrumentor.instrument_client(
                http, tracer_provider=provider
            )
        client = make_client(http)
        with provider.get_tracer("test").start_as_current_span(
            "caller"
        ) as parent:
            request = make_request("conversation", streaming)
            if streaming:
                stream = client.send_message_streaming(request)
                chunks = [chunk async for chunk in stream]
                assert chunks
                await stream.aclose()
            else:
                result = await client.send_message(request)
                assert result is not None
            assert trace.get_current_span() is parent
        if http_instrumentation:
            http_probe.HTTPXClientInstrumentor.uninstrument_client(http)
    close = getattr(handler, "aclose", None)
    if close:
        await close()
    spans = exporter.get_finished_spans()
    method = "SendStreamingMessage" if streaming else "SendMessage"
    client_spans = [
        s
        for s in spans
        if s.attributes.get("a2a.method.name") == method
        and s.kind == SpanKind.CLIENT
    ]
    server_spans = [
        s
        for s in spans
        if s.attributes.get("a2a.method.name") == method
        and s.kind == SpanKind.SERVER
    ]
    assert len(client_spans) == len(server_spans) == 1
    client_span, server_span = client_spans[0], server_spans[0]
    assert (
        client_span.context.trace_id
        == server_span.context.trace_id
        == parent.get_span_context().trace_id
    )
    assert server_span.parent.is_remote
    transport_spans = [
        s
        for s in spans
        if s.kind == SpanKind.CLIENT and "a2a.method.name" not in s.attributes
    ]
    immediate_parent = (
        transport_spans[0] if http_instrumentation else client_span
    )
    assert server_span.parent.span_id == immediate_parent.context.span_id
    assert calls[0][1].span_id == server_span.context.span_id
    assert len(calls) == 1
    assert not any(
        s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        for s in spans
    )
    assert not any(
        s.instrumentation_scope.name == "a2a-python-sdk" for s in spans
    )
    assert server_span.attributes["a2a.message.id"]
    assert server_span.attributes["gen_ai.agent.name"] == "Echo agent"
    assert not any("gen_ai.input.messages" in s.attributes for s in spans)


@pytest.mark.asyncio
async def test_concurrent_sessions_and_two_turns(telemetry):
    provider, exporter, _ = telemetry
    calls = []
    app, handler = make_app(calls)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={"a2a-version": "1.0"} if MODERN else {},
    ) as http:
        client = make_client(http)

        async def worker(index):
            with provider.get_tracer("test").start_as_current_span(
                "worker"
            ) as parent:
                for _ in range(2):
                    await client.send_message(make_request(f"session-{index}"))
                    assert trace.get_current_span() is parent
                return parent.get_span_context().trace_id

        trace_ids = await asyncio.gather(*(worker(i) for i in range(3)))
    close = getattr(handler, "aclose", None)
    if close:
        await close()
    assert len(set(trace_ids)) == 3
    spans = [
        s
        for s in exporter.get_finished_spans()
        if "a2a.method.name" in s.attributes
    ]
    assert len(spans) == 12
    for index, trace_id in enumerate(trace_ids):
        group = [s for s in spans if s.context.trace_id == trace_id]
        assert len(group) == 4
        assert {s.attributes["gen_ai.conversation.id"] for s in group} == {
            f"session-{index}"
        }
    assert len(calls) == 6


@pytest.mark.skipif(not MODERN, reason="HTTP+JSON 1.x transport")
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_real_rest_transport(telemetry, streaming):
    from a2a.client.transports.rest import RestTransport
    from a2a.server.routes import create_rest_routes
    from starlette.applications import Starlette

    provider, exporter, _ = telemetry
    calls = []
    _, handler = make_app(calls)
    app = Starlette(routes=create_rest_routes(handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), headers={"a2a-version": "1.0"}
    ) as http:
        client = RestTransport(http, make_card(), "http://agent.test")
        with provider.get_tracer("test").start_as_current_span(
            "parent"
        ) as parent:
            if streaming:
                stream = client.send_message_streaming(make_request("rest"))
                results = [r async for r in stream]
                assert results
                await stream.aclose()
            else:
                await client.send_message(make_request("rest"))
            assert trace.get_current_span() is parent
    await handler.aclose()
    spans = [
        s
        for s in exporter.get_finished_spans()
        if "a2a.method.name" in s.attributes
    ]
    assert len(spans) == 2
    assert {s.kind for s in spans} == {SpanKind.CLIENT, SpanKind.SERVER}
    assert len({s.context.trace_id for s in spans}) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["get_task", "cancel_task"])
@pytest.mark.asyncio
async def test_real_task_error(telemetry, operation):
    provider, exporter, _ = telemetry
    app, handler = make_app([])
    from a2a.types import CancelTaskRequest, GetTaskRequest

    request_type = (
        GetTaskRequest if operation == "get_task" else CancelTaskRequest
    )
    request = (
        request_type(id="missing")
        if MODERN
        else request_type(id="rpc", params={"id": "missing"})
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={"a2a-version": "1.0"} if MODERN else {},
    ) as http:
        client = make_client(http)
        if MODERN:
            with pytest.raises(Exception):
                await getattr(client, operation)(request)
        else:
            result = await getattr(client, operation)(request)
            assert result.root.error.code == -32001
    close = getattr(handler, "aclose", None)
    if close:
        await close()
    spans = [
        s
        for s in exporter.get_finished_spans()
        if "a2a.method.name" in s.attributes
    ]
    assert len(spans) == 2
    assert all(s.attributes["a2a.task.id"] == "missing" for s in spans)
    assert all(s.status.status_code.name == "ERROR" for s in spans)
    assert len({s.context.trace_id for s in spans}) == 1
