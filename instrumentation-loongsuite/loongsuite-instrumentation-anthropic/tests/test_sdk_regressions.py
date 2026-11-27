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

"""Regression checks through real SDK clients with an HTTP-only fake server."""

import asyncio
import json

import anthropic
import pytest
from anthropic import _base_client

from opentelemetry import trace
from opentelemetry.instrumentation.anthropic import patch
from opentelemetry.instrumentation.utils import suppress_instrumentation

# Anthropic 2.x uses httpx2; the supported oldest SDK uses httpx.
HTTP = getattr(_base_client, "httpx2", None) or _base_client.httpx
PARAMS = dict(
    model="test-model",
    max_tokens=1500,
    messages=[{"role": "user", "content": "test"}],
)
MARKER = "SYNTHETIC_PROMPT_MARKER"
USAGE = dict(
    input_tokens=7,
    cache_creation_input_tokens=40,
    cache_read_input_tokens=100,
    output_tokens=9,
)
MESSAGE = dict(
    type="message",
    id="msg_test",
    role="assistant",
    model="test-model",
    content=[{"type": "text", "text": "ok"}],
    usage=USAGE,
    stop_reason="end_turn",
    stop_sequence=None,
)


def sse(thinking=True):
    events = [
        dict(
            type="message_start",
            message={
                **MESSAGE,
                "content": [],
                "stop_reason": None,
                "usage": {**USAGE, "output_tokens": 1},
            },
        )
    ]
    blocks = [
        (dict(type="text", text=""), [dict(type="text_delta", text="before")]),
        (
            dict(type="tool_use", id="toolu_test", name="example", input={}),
            [dict(type="input_json_delta", partial_json='{ "value": 1 }')],
        ),
        (dict(type="text", text=""), [dict(type="text_delta", text="after")]),
    ]
    if not hasattr(anthropic.types, "ToolUseBlock"):
        blocks.pop(1)
    if thinking:
        blocks.insert(
            0,
            (
                dict(type="thinking", thinking="", signature=""),
                [
                    dict(type="thinking_delta", thinking="test reasoning"),
                    dict(type="signature_delta", signature="test signature"),
                ],
            ),
        )
    for i, (block, deltas) in enumerate(blocks):
        events.append(
            dict(type="content_block_start", index=i, content_block=block)
        )
        events.extend(
            dict(type="content_block_delta", index=i, delta=d) for d in deltas
        )
        events.append(dict(type="content_block_stop", index=i))
    events.extend(
        [
            dict(
                type="message_delta",
                usage={"output_tokens": 9},
                delta={"stop_reason": "end_turn", "stop_sequence": None},
            ),
            dict(type="message_stop"),
        ]
    )
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    )


@pytest.fixture
def server():
    sent = []

    def respond(request):
        body = json.loads(request.content)
        sent.append(body)
        messages = body["messages"]
        if not messages or messages[0]["content"] == MARKER:
            return HTTP.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": MARKER,
                    },
                },
            )
        if body.get("stream"):
            # Extended thinking was not part of Anthropic 0.16's event schema.
            return HTTP.Response(
                200,
                text=sse(hasattr(anthropic.types, "ThinkingBlock")),
                headers={"content-type": "text/event-stream"},
            )
        return HTTP.Response(200, json=MESSAGE)

    return respond, sent


def client_for(server, asynchronous=False):
    client_type = (
        anthropic.AsyncAnthropic if asynchronous else anthropic.Anthropic
    )
    http_type = HTTP.AsyncClient if asynchronous else HTTP.Client
    return client_type(
        api_key="synthetic",
        base_url="https://anthropic.test",
        max_retries=0,
        http_client=http_type(transport=HTTP.MockTransport(server[0])),
    )


def assert_usage(span_exporter, metric_reader):
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.usage.input_tokens"] == 147
    assert span.attributes["gen_ai.usage.output_tokens"] == 9
    assert span.attributes["gen_ai.usage.total_tokens"] == 156
    assert span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 40
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 100
    data = metric_reader.get_metrics_data()
    points = [
        p
        for r in data.resource_metrics
        for s in r.scope_metrics
        for m in s.metrics
        if m.name == "gen_ai.client.token.usage"
        for p in m.data.data_points
    ]
    assert {
        p.attributes["gen_ai.token.type"]: (p.sum, p.count) for p in points
    } == {"input": (147, 1), "output": (9, 1)}
    return span


def assert_parts(span):
    (output,) = json.loads(span.attributes["gen_ai.output.messages"])
    expected = (
        ["text", "tool_call", "text"]
        if hasattr(anthropic.types, "ToolUseBlock")
        else ["text", "text"]
    )
    if hasattr(anthropic.types, "ThinkingBlock"):
        expected.insert(0, "reasoning")
    assert [p["type"] for p in output["parts"]] == expected
    assert output["parts"][-1]["content"] == "after"
    if hasattr(anthropic.types, "ToolUseBlock"):
        assert output["parts"][-2]["arguments"] == {"value": 1}


@pytest.mark.parametrize(
    "mode", ["instrument_with_content", "instrument_no_content"]
)
def test_one_shot_messages_sync(mode, request, server, span_exporter):
    with client_for(server) as client:
        client.messages.create(
            **{**PARAMS, "messages": iter(PARAMS["messages"])}
        )
        request.getfixturevalue(mode)
        client.messages.create(
            **{**PARAMS, "messages": iter(PARAMS["messages"])}
        )
    assert [r["messages"] for r in server[1]] == [
        PARAMS["messages"],
        PARAMS["messages"],
    ]
    (span,) = span_exporter.get_finished_spans()
    assert "gen_ai.input.messages" not in span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["instrument_with_content", "instrument_no_content"]
)
async def test_one_shot_messages_async(mode, request, server):
    async with client_for(server, True) as client:
        await client.messages.create(
            **{**PARAMS, "messages": iter(PARAMS["messages"])}
        )
        request.getfixturevalue(mode)
        await client.messages.create(
            **{**PARAMS, "messages": iter(PARAMS["messages"])}
        )
    assert [r["messages"] for r in server[1]] == [
        PARAMS["messages"],
        PARAMS["messages"],
    ]


@pytest.mark.parametrize("api", ["create", "raw", "helper", "text"])
def test_sync_usage_and_stream_helpers(
    api,
    server,
    instrument_with_content,
    span_exporter,
    metric_reader,
    tracer_provider,
):
    with client_for(server) as client:
        with tracer_provider.get_tracer(__name__).start_as_current_span(
            "parent"
        ) as parent:
            if api == "create":
                assert client.messages.create(**PARAMS).content[0].text == "ok"
            else:
                manager = (
                    client.messages.create(**PARAMS, stream=True)
                    if api == "raw"
                    else client.messages.stream(**PARAMS)
                )
                with manager as stream:
                    assert trace.get_current_span() is parent
                    if api == "helper":
                        message = stream.get_final_message()
                        assert message.content[-1].text == "after"
                    elif api == "text":
                        assert "".join(stream.text_stream) == "beforeafter"
                    else:
                        assert list(stream)[-1].type == "message_stop"
                assert stream.response.is_closed
            span = assert_usage(span_exporter, metric_reader)
            assert span.parent.span_id == parent.get_span_context().span_id
            if api != "create":
                assert_parts(span)
    assert len(server[1]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["create", "raw", "helper", "text"])
async def test_async_usage_and_stream_helpers(
    api,
    server,
    instrument_with_content,
    span_exporter,
    metric_reader,
    tracer_provider,
):
    async with client_for(server, True) as client:
        with tracer_provider.get_tracer(__name__).start_as_current_span(
            "parent"
        ) as parent:
            if api == "create":
                assert (await client.messages.create(**PARAMS)).content[
                    0
                ].text == "ok"
            else:
                manager = (
                    await client.messages.create(**PARAMS, stream=True)
                    if api == "raw"
                    else client.messages.stream(**PARAMS)
                )
                async with manager as stream:
                    assert trace.get_current_span() is parent
                    if api == "helper":
                        assert (await stream.get_final_message()).content[
                            -1
                        ].text == "after"
                    elif api == "text":
                        assert (
                            "".join(
                                [text async for text in stream.text_stream]
                            )
                            == "beforeafter"
                        )
                    else:
                        assert [event async for event in stream][
                            -1
                        ].type == "message_stop"
                assert stream.response.is_closed
            span = assert_usage(span_exporter, metric_reader)
            assert span.parent.span_id == parent.get_span_context().span_id
            if api != "create":
                assert_parts(span)
    assert len(server[1]) == 1


@pytest.mark.parametrize("helper", [False, True])
def test_sync_error_privacy(
    helper, server, instrument_no_content, span_exporter, log_exporter
):
    with (
        client_for(server) as client,
        pytest.raises(anthropic.BadRequestError, match=MARKER),
    ):
        params = {**PARAMS, "messages": [{"role": "user", "content": MARKER}]}
        if helper:
            with client.messages.stream(**params):
                pytest.fail("Error response must reach the caller")
        else:
            client.messages.create(**params)
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes["error.type"] == "BadRequestError"
    assert MARKER not in str(span.attributes) + str(span.events) + str(
        span.status.description
    ) + str(log_exporter.get_finished_logs())


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [False, True])
async def test_async_error_privacy(
    helper, server, instrument_no_content, span_exporter
):
    async with client_for(server, True) as client:
        with pytest.raises(anthropic.BadRequestError, match=MARKER):
            params = {
                **PARAMS,
                "messages": [{"role": "user", "content": MARKER}],
            }
            if helper:
                async with client.messages.stream(**params):
                    pytest.fail("Error response must reach the caller")
            else:
                await client.messages.create(**params)
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert MARKER not in str(span.attributes) + str(span.events) + str(
        span.status.description
    )


def test_sync_suppression(server, instrument_with_content, span_exporter):
    with client_for(server) as client, suppress_instrumentation():
        client.messages.create(**PARAMS)
        with client.messages.stream(**PARAMS) as stream:
            stream.get_final_message()
    assert len(server[1]) == 2
    assert not span_exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_async_suppression(
    server, instrument_with_content, span_exporter
):
    async with client_for(server, True) as client:
        with suppress_instrumentation():
            await client.messages.create(**PARAMS)
            async with client.messages.stream(**PARAMS) as stream:
                await stream.get_final_message()
    assert len(server[1]) == 2
    assert not span_exporter.get_finished_spans()


def fail_wrapper(*args, **kwargs):
    raise RuntimeError("synthetic instrumentation failure")


@pytest.mark.parametrize("helper", [False, True])
def test_sync_wrapper_failure_is_fail_open(
    helper, server, instrument_with_content, span_exporter, monkeypatch
):
    monkeypatch.setattr(patch, "AnthropicStreamWrapper", fail_wrapper)
    with client_for(server) as client:
        manager = (
            client.messages.stream(**PARAMS)
            if helper
            else client.messages.create(**PARAMS, stream=True)
        )
        with manager as stream:
            assert list(stream)[-1].type == "message_stop"
    assert len(server[1]) == 1
    assert len(span_exporter.get_finished_spans()) == 1
    assert not trace.get_current_span().is_recording()


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [False, True])
async def test_async_wrapper_failure_is_fail_open(
    helper, server, instrument_with_content, span_exporter, monkeypatch
):
    monkeypatch.setattr(patch, "AsyncAnthropicStreamWrapper", fail_wrapper)
    async with client_for(server, True) as client:
        manager = (
            client.messages.stream(**PARAMS)
            if helper
            else await client.messages.create(**PARAMS, stream=True)
        )
        async with manager as stream:
            assert [e async for e in stream][-1].type == "message_stop"
    assert len(server[1]) == 1
    assert len(span_exporter.get_finished_spans()) == 1
    assert not trace.get_current_span().is_recording()


@pytest.mark.parametrize("explicit_close", [False, True])
def test_helper_early_close(
    explicit_close, server, instrument_with_content, span_exporter
):
    with client_for(server) as client:
        with client.messages.stream(**PARAMS) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    break
            if explicit_close:
                stream.close()
                assert len(span_exporter.get_finished_spans()) == 1
        assert stream.response.is_closed
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.UNSET
    assert "gen_ai.response.finish_reasons" not in span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_close", [False, True])
async def test_async_helper_early_close(
    explicit_close, server, instrument_with_content, span_exporter
):
    async with client_for(server, True) as client:
        async with client.messages.stream(**PARAMS) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    break
            if explicit_close:
                await stream.close()
                assert len(span_exporter.get_finished_spans()) == 1
        assert stream.response.is_closed
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.UNSET
    assert "gen_ai.response.finish_reasons" not in span.attributes


def test_uninstrument_restores_both_entry_points(
    server, instrument_with_content, span_exporter
):
    instrument_with_content.uninstrument()
    with client_for(server) as client:
        client.messages.create(**PARAMS)
        with client.messages.stream(**PARAMS) as stream:
            stream.get_final_message()
    assert not span_exporter.get_finished_spans()


@pytest.mark.parametrize("api", ["raw", "helper"])
def test_stream_read_failure_preserves_exception(
    api, instrument_no_content, span_exporter, log_exporter
):
    error = HTTP.ReadError(MARKER)

    class BrokenStream(HTTP.SyncByteStream):
        def __iter__(self):
            yield sse(False).split("event: message_delta")[0].encode()
            raise error

    def respond(request):
        return HTTP.Response(
            200,
            stream=BrokenStream(),
            headers={"content-type": "text/event-stream"},
        )

    with client_for((respond, [])) as client:
        with pytest.raises(HTTP.ReadError) as caught:
            manager = (
                client.messages.stream(**PARAMS)
                if api == "helper"
                else client.messages.create(**PARAMS, stream=True)
            )
            with manager as stream:
                list(stream)
        assert stream.response.is_closed
    assert caught.value is error
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert MARKER not in str(span.attributes) + str(span.events) + str(
        span.status.description
    ) + str(log_exporter.get_finished_logs())


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["raw", "helper"])
async def test_stream_cancellation_preserves_exception(
    api, instrument_no_content, span_exporter
):
    error = asyncio.CancelledError(MARKER)

    class CancelledStream(HTTP.AsyncByteStream):
        async def __aiter__(self):
            yield sse(False).split("event: message_delta")[0].encode()
            raise error

    def respond(request):
        return HTTP.Response(
            200,
            stream=CancelledStream(),
            headers={"content-type": "text/event-stream"},
        )

    async with client_for((respond, []), True) as client:
        with pytest.raises(asyncio.CancelledError) as caught:
            manager = (
                client.messages.stream(**PARAMS)
                if api == "helper"
                else await client.messages.create(**PARAMS, stream=True)
            )
            async with manager as stream:
                async for _ in stream:
                    pass
        assert stream.response.is_closed
    assert caught.value is error
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert MARKER not in str(span.attributes) + str(span.events) + str(
        span.status.description
    )
