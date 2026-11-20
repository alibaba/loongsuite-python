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

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp import JSONRPCRequest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
from pydantic import AnyUrl
from wrapt import FunctionWrapper

from opentelemetry import context, propagate
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.mcp import MCPInstrumentor, ServerMetrics
from opentelemetry.instrumentation.mcp.session_handler import _SessionContext
from opentelemetry.metrics import get_meter
from opentelemetry.trace import SpanKind

FASTMCP_SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "fastmcp_server.py"
)


@asynccontextmanager
async def mock_client(params):
    yield params


def test_session_context_detach_removes_attached_context():
    target = SimpleNamespace()
    session_context = _SessionContext({})
    context = (target,)
    session_context._safe_attach(context)
    assert hasattr(target, "_session_context")

    session_context._safe_detach(context)

    assert not hasattr(target, "_session_context")


def test_session_context_detach_without_attachment_does_not_raise():
    target = SimpleNamespace()
    session_context = _SessionContext({})
    session_context._safe_detach((target,))
    assert not hasattr(target, "_session_context")


@pytest.mark.asyncio
async def test_client_invalid(tracer_provider):
    import mcp.client.sse  # noqa: PLC0415
    import mcp.client.stdio  # noqa: PLC0415
    import mcp.client.streamable_http  # noqa: PLC0415
    import mcp.client.websocket  # noqa: PLC0415

    stdio_client_backup = mcp.client.stdio.stdio_client
    websocket_client_backup = mcp.client.websocket.websocket_client
    sse_client_backup = mcp.client.sse.sse_client
    streamablehttp_client_backup = (
        mcp.client.streamable_http.streamablehttp_client
    )

    mcp.client.stdio.stdio_client = mock_client
    mcp.client.websocket.websocket_client = mock_client
    mcp.client.sse.sse_client = mock_client
    mcp.client.streamable_http.streamablehttp_client = mock_client

    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    from mcp.client.sse import sse_client  # noqa: PLC0415
    from mcp.client.stdio import stdio_client  # noqa: PLC0415
    from mcp.client.streamable_http import (  # noqa: PLC0415
        streamablehttp_client,
    )
    from mcp.client.websocket import websocket_client  # noqa: PLC0415

    assert isinstance(stdio_client, FunctionWrapper)
    assert isinstance(websocket_client, FunctionWrapper)
    assert isinstance(sse_client, FunctionWrapper)
    assert isinstance(streamablehttp_client, FunctionWrapper)

    # ensure no exception is raised, and result is as expected
    for testure in [
        stdio_client,
        websocket_client,
        sse_client,
        streamablehttp_client,
    ]:
        async with testure(None) as result:
            assert result is None

        async with testure([]) as result:
            assert result == []

        async with testure([1, 2]) as result:
            assert result == [1, 2]

        async with testure([1, 2, "hello"]) as result:
            assert result == [1, 2, "hello"]

        async with testure([1, 2, "hello", "world"]) as result:
            assert result == [1, 2, "hello", "world"]
    mcp_instrumentor._uninstrument()
    mcp.client.stdio.stdio_client = stdio_client_backup
    mcp.client.websocket.websocket_client = websocket_client_backup
    mcp.client.sse.sse_client = sse_client_backup
    mcp.client.streamable_http.streamablehttp_client = (
        streamablehttp_client_backup
    )


class MockStream:
    def __init__(self):
        pass

    async def send(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Exception):
            raise args[0]


@pytest.mark.asyncio
async def test_send_wrapper(tracer_provider):
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    from mcp.client.stdio import (  # noqa: PLC0415
        StdioServerParameters,
        stdio_client,
    )

    from opentelemetry.instrumentation.mcp.session_handler import (  # noqa: PLC0415
        _writer_send_wrapper,
    )

    server_params = StdioServerParameters(
        command="python",
        args=[FASTMCP_SERVER_SCRIPT],
    )
    async with stdio_client(server_params) as (stdio, write):
        assert write.send._self_wrapper == _writer_send_wrapper

        mock_stream = MockStream()

        await _writer_send_wrapper(
            mock_stream.send, mock_stream, (), {}
        )  # len(args) == 0
        await _writer_send_wrapper(mock_stream.send, mock_stream, (1,), {})
        await _writer_send_wrapper(mock_stream.send, mock_stream, (1, 2), {})

        # wrapped func raises exception
        with pytest.raises(Exception):
            await _writer_send_wrapper(
                mock_stream.send, mock_stream, (Exception("test"),), {}
            )

        tracer = trace_api.get_tracer(
            __name__, None, tracer_provider=tracer_provider
        )
        with tracer.start_as_current_span(
            name="test_send_request_propagator",
            kind=SpanKind.CLIENT,
            # FIXME: ruff failed, why not end span?
        ) as span:  # noqa: F841
            # invalid carrier
            meta = set()
            with pytest.raises(TypeError):
                propagate.get_global_textmap().inject(meta)

            # here we dont raise exception if inject failed
            session_message = SessionMessage(
                message=JSONRPCMessage(
                    root=JSONRPCRequest(
                        jsonrpc="2.0",
                        method="test",
                        id=1,
                        params={"_meta": ""},
                    )
                )
            )
            await _writer_send_wrapper(
                mock_stream.send, mock_stream, (session_message,), {}
            )

    mcp_instrumentor._uninstrument()


@pytest.mark.asyncio
async def test_server_handle_request_wrapper(tracer_provider):
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    from mcp.shared.session import RequestResponder  # noqa: PLC0415
    from mcp.types import (  # noqa: PLC0415
        CallToolRequest,
        CallToolRequestParams,
        ClientRequest,
        GetPromptRequest,
        GetPromptRequestParams,
        ListToolsRequest,
        ReadResourceRequest,
        ReadResourceRequestParams,
        SubscribeRequest,
        SubscribeRequestParams,
    )

    from opentelemetry.instrumentation.mcp.semconv import (  # noqa: PLC0415
        MCPAttributes,
    )

    wrapper = _create_server_wrapper(mcp_instrumentor, tracer_provider)

    request = ClientRequest(root=ListToolsRequest(method="tools/list"))
    responder = RequestResponder(
        request_id=1,
        request_meta=None,
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    attributes, span_name = wrapper.extract_attributes((responder,))
    assert attributes[MCPAttributes.RPC_REQUEST_ID] == "1"
    assert attributes[MCPAttributes.MCP_METHOD_NAME] == "tools/list"
    assert span_name == "tools/list"

    attributes, span_name = wrapper.extract_attributes(())
    attributes, span_name = wrapper.extract_attributes((None,))
    attributes, span_name = wrapper.extract_attributes(("invalid str",))

    request = ClientRequest(
        root=CallToolRequest(
            method="tools/call", params=CallToolRequestParams(name="test")
        )
    )
    responder = RequestResponder(
        request_id=1,
        request_meta=None,
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    attributes, span_name = wrapper.extract_attributes((responder,))
    assert attributes[MCPAttributes.RPC_REQUEST_ID] == "1"
    assert attributes[MCPAttributes.MCP_METHOD_NAME] == "tools/call"
    assert span_name == "tools/call test"

    request = ClientRequest(
        root=GetPromptRequest(
            method="prompts/get",
            params=GetPromptRequestParams(name="prompt_name"),
        )
    )
    responder = RequestResponder(
        request_id=1,
        request_meta=None,
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    attributes, span_name = wrapper.extract_attributes((responder,))
    assert attributes[MCPAttributes.RPC_REQUEST_ID] == "1"
    assert attributes[MCPAttributes.MCP_METHOD_NAME] == "prompts/get"
    assert span_name == "prompts/get prompt_name"

    request = ClientRequest(
        root=ReadResourceRequest(
            method="resources/read",
            params=ReadResourceRequestParams(
                uri=AnyUrl("test://resource_uri")
            ),
        )
    )
    responder = RequestResponder(
        request_id=1,
        request_meta=None,
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    attributes, span_name = wrapper.extract_attributes((responder,))
    assert attributes[MCPAttributes.RPC_REQUEST_ID] == "1"
    assert attributes[MCPAttributes.MCP_METHOD_NAME] == "resources/read"
    assert span_name == "resources/read test://resource_uri"

    request = ClientRequest(
        root=SubscribeRequest(
            method="resources/subscribe",
            params=SubscribeRequestParams(uri=AnyUrl("test://resource_uri")),
        )
    )
    responder = RequestResponder(
        request_id=1,
        request_meta=None,
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    attributes, span_name = wrapper.extract_attributes((responder,))
    assert attributes[MCPAttributes.RPC_REQUEST_ID] == "1"
    assert attributes[MCPAttributes.MCP_METHOD_NAME] == "resources/subscribe"
    assert span_name == "resources/subscribe test://resource_uri"
    mcp_instrumentor._uninstrument()


def _create_server_wrapper(mcp_instrumentor, tracer_provider):
    from opentelemetry.instrumentation.mcp.session_handler import (  # noqa: PLC0415
        ServerHandleRequestWrapper,
    )

    tracer = trace_api.get_tracer(
        __name__, None, tracer_provider=tracer_provider
    )
    meter = get_meter(
        __name__,
        None,
        None,
        schema_url="https://opentelemetry.io/schemas/1.11.0",
    )
    metrics = ServerMetrics(meter)
    wrapper = ServerHandleRequestWrapper(tracer, metrics)
    return wrapper


@pytest.mark.asyncio
async def test_server_extract_parent_context(tracer_provider):
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    from mcp.shared.session import RequestResponder  # noqa: PLC0415
    from mcp.types import (  # noqa: PLC0415
        ClientRequest,
        ListToolsRequest,
        RequestParams,
    )

    wrapper = _create_server_wrapper(mcp_instrumentor, tracer_provider)

    assert wrapper.extract_parent_context(()) is None
    assert wrapper.extract_parent_context((None,)) is None
    assert wrapper.extract_parent_context(("invalid str",)) is None

    request = ClientRequest(root=ListToolsRequest(method="tools/list"))
    responder = RequestResponder(
        request_id=1,
        request_meta=None,
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    assert wrapper.extract_parent_context((responder,)) is None

    tracer = trace_api.get_tracer(
        __name__, None, tracer_provider=tracer_provider
    )
    with tracer.start_as_current_span(
        name="test_send_request_propagator",
        kind=SpanKind.CLIENT,
        # FIXME: ruff failed
    ) as span:  # noqa: F841
        meta = {}
        current_context = context.get_current()
        propagate.get_global_textmap().inject(meta, current_context)
        parent_trace_id = span.get_span_context().trace_id

    traceparent = meta.get("traceparent")
    print(traceparent)

    request = ClientRequest(root=ListToolsRequest(method="tools/list"))
    responder = RequestResponder(
        request_id=1,
        request_meta=RequestParams.Meta(traceparent=traceparent),
        request=request,
        session=None,
        on_complete=None,
        message_metadata=None,
    )
    parent_context = wrapper.extract_parent_context((responder,))

    assert parent_context is not None
    assert isinstance(parent_context, trace_api.Context)
    parent_span_context = trace_api.get_current_span(
        parent_context
    ).get_span_context()
    assert parent_span_context.is_valid

    with tracer.start_as_current_span(
        name="test_send_request_propagator_2",
        kind=SpanKind.SERVER,
        context=parent_context,
    ) as span:
        assert str(span.get_span_context().trace_id) == str(parent_trace_id)

    mcp_instrumentor._uninstrument()
