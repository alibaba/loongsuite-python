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

import json
import os
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI, OpenAI

from opentelemetry.instrumentation._semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)

from .test_utils import DEFAULT_MODEL, assert_messages_attribute

responses_resources = pytest.importorskip("openai.resources.responses")
AsyncResponses = responses_resources.AsyncResponses
Responses = responses_resources.Responses


def _to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {
            key: _to_dict(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_dict(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_dict(item) for key, item in value.items()}
    return value


class _ResponseObject(SimpleNamespace):
    def to_dict(self):
        return _to_dict(self)


def _obj(**kwargs):
    return _ResponseObject(**kwargs)


def _response():
    return _obj(
        id="resp_123",
        model=DEFAULT_MODEL,
        status="completed",
        service_tier="default",
        output=[
            _obj(
                type="message",
                role="assistant",
                content=[
                    _obj(
                        type="output_text",
                        text="This is a Responses API test.",
                    )
                ],
            )
        ],
        usage=_obj(
            input_tokens=11,
            output_tokens=7,
            input_tokens_details=_obj(cached_tokens=3),
            output_tokens_details=_obj(reasoning_tokens=2),
        ),
    )


def _tool_response():
    return _obj(
        id="resp_123",
        model=DEFAULT_MODEL,
        status="completed",
        output=[
            _obj(type="reasoning", summary=[]),
            _obj(
                type="function_call",
                call_id="call_1",
                name="lookup_weather",
                arguments='{"city": "Seattle"}',
            ),
        ],
        usage=_obj(input_tokens=11, output_tokens=7),
    )


def _response_with_status(status, incomplete_reason=None):
    incomplete_details = None
    if incomplete_reason:
        incomplete_details = _obj(reason=incomplete_reason)
    return _obj(
        id="resp_123",
        model=DEFAULT_MODEL,
        status=status,
        output=[],
        incomplete_details=incomplete_details,
        usage=_obj(input_tokens=11, output_tokens=7),
    )


def _created_response():
    return _obj(
        id="resp_123",
        model=DEFAULT_MODEL,
        status="in_progress",
        output=[],
        usage=None,
    )


def _response_created_event(sequence_number=0):
    return SimpleNamespace(
        type="response.created",
        sequence_number=sequence_number,
        response=_created_response(),
    )


def _response_completed_event(sequence_number=1):
    return SimpleNamespace(
        type="response.completed",
        sequence_number=sequence_number,
        response=_response(),
    )


class _RawStreamResponse:
    def close(self):
        pass

    async def aclose(self):
        pass


class _RawResponse:
    def __init__(self, parsed_response):
        self._parsed_response = parsed_response

    def parse(self):
        return self._parsed_response


class _ResponseStream:
    def __init__(self, events):
        self._events = iter(events)
        self.closed = False
        self.response = _RawStreamResponse()

    def __iter__(self):
        return self

    def __next__(self):
        event = next(self._events)
        if isinstance(event, BaseException):
            raise event
        return event

    def close(self):
        self.closed = True


class _AsyncResponseStream:
    def __init__(self, events):
        self._events = iter(events)
        self.closed = False
        self.response = _RawStreamResponse()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = next(self._events)
        except StopIteration as error:
            raise StopAsyncIteration from error
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self):
        self.closed = True


def _instrument(
    tracer_provider,
    logger_provider,
    meter_provider,
    content_capture_mode="span_only",
):
    _OpenTelemetrySemanticConventionStability._initialized = False
    os.environ[OTEL_SEMCONV_STABILITY_OPT_IN] = "gen_ai_latest_experimental"
    if content_capture_mode is None:
        os.environ.pop(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None
        )
    else:
        os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
            content_capture_mode
        )
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    return instrumentor


def _cleanup(instrumentor):
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    os.environ.pop(OTEL_SEMCONV_STABILITY_OPT_IN, None)
    instrumentor.uninstrument()
    _OpenTelemetrySemanticConventionStability._initialized = False


def _assert_response_span(span):
    assert span.name == f"chat {DEFAULT_MODEL}"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "resp_123"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    assert span.attributes["gen_ai.openai.response.status"] == "completed"
    assert (
        span.attributes[OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER]
        == "default"
    )
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 3
    assert (
        span.attributes["gen_ai.usage.output_tokens_details.reasoning_tokens"]
        == 2
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "Say this is a test"}],
            }
        ],
    )
    assert json.loads(
        span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    ) == [{"type": "text", "content": "You are concise."}]
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "text",
                        "content": "This is a Responses API test.",
                    }
                ],
                "finish_reason": "stop",
            }
        ],
    )


def test_responses_create_with_content(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _response()

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Say this is a test",
                        }
                    ],
                }
            ],
            service_tier="default",
            reasoning={"effort": "low", "summary": "auto"},
        )
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)
    assert span.attributes["gen_ai.openai.request.reasoning.effort"] == "low"
    assert span.attributes["gen_ai.openai.request.reasoning.summary"] == "auto"


@pytest.mark.asyncio
async def test_async_responses_create_with_content(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    async def fake_create(self, **kwargs):
        return _response()

    monkeypatch.setattr(AsyncResponses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        await getattr(AsyncOpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
        )
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)


def test_responses_create_with_raw_response(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _RawResponse(_response())

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        raw_response = getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
        )
    finally:
        _cleanup(instrumentor)

    assert isinstance(raw_response, _RawResponse)
    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)


@pytest.mark.parametrize(
    ("status", "incomplete_reason", "finish_reason"),
    [
        ("incomplete", None, "length"),
        ("incomplete", "content_filter", "content_filter"),
        ("failed", None, "error"),
        ("cancelled", None, "error"),
    ],
)
def test_responses_create_finish_reason_status_mapping(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    status,
    incomplete_reason,
    finish_reason,
):
    def fake_create(self, **kwargs):
        return _response_with_status(status, incomplete_reason)

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            input="Say this is a test",
        )
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        finish_reason,
    )


@pytest.mark.asyncio
async def test_async_responses_create_streaming(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    async def fake_create(self, **kwargs):
        return _AsyncResponseStream(
            [
                _response_created_event(),
                _response_completed_event(),
            ]
        )

    monkeypatch.setattr(AsyncResponses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        stream = await getattr(AsyncOpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
            stream=True,
        )
        async for _ in stream:
            pass
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)


def test_responses_create_streaming_error(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _ResponseStream(
            [
                SimpleNamespace(
                    type="response.created",
                    response=SimpleNamespace(
                        id="resp_started",
                        model=DEFAULT_MODEL,
                        status="in_progress",
                    ),
                ),
                RuntimeError("stream failed"),
            ]
        )

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        stream = getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            input="Say this is a test",
            stream=True,
        )
        with pytest.raises(RuntimeError, match="stream failed"):
            for _ in stream:
                pass
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "resp_started"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == DEFAULT_MODEL
    )


def test_responses_create_error(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        raise RuntimeError("responses failed")

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        with pytest.raises(RuntimeError, match="responses failed"):
            getattr(OpenAI(), "responses").create(
                model=DEFAULT_MODEL,
                input="Say this is a test",
            )
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )


def test_responses_create_no_content(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _response()

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider,
        logger_provider,
        meter_provider,
        content_capture_mode=None,
    )
    try:
        getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
        )
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes


def test_responses_create_tool_call_skips_reasoning_items(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _tool_response()

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            input="Say this is a test",
            tools=[
                {
                    "type": "function",
                    "name": "lookup_weather",
                    "description": "Get weather.",
                    "parameters": {"type": "object"},
                }
            ],
        )
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_calls",
    )
    output_messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_messages) == 1
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "lookup_weather",
                        "arguments": {"city": "Seattle"},
                    }
                ],
                "finish_reason": "tool_calls",
            }
        ],
    )
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [
        {
            "name": "lookup_weather",
            "description": "Get weather.",
            "parameters": {"type": "object"},
            "type": "function",
        }
    ]


def test_responses_create_streaming(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _ResponseStream(
            [
                _response_created_event(),
                _response_completed_event(),
            ]
        )

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        stream = getattr(OpenAI(), "responses").create(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
            stream=True,
        )
        for _ in stream:
            pass
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)


def test_responses_stream_helper_filters_omit_sentinels(
    monkeypatch,
    caplog,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_create(self, **kwargs):
        return _ResponseStream(
            [
                _response_created_event(),
                _response_completed_event(),
            ]
        )

    monkeypatch.setattr(Responses, "create", fake_create)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        with caplog.at_level("WARNING"):
            with getattr(OpenAI(), "responses").stream(
                model=DEFAULT_MODEL,
                input="Say this is a test",
            ) as stream:
                for _ in stream:
                    pass
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert "gen_ai.request.temperature" not in span.attributes
    assert "gen_ai.request.top_p" not in span.attributes
    assert "gen_ai.openai.request.previous_response_id" not in span.attributes
    assert "gen_ai.openai.request.background" not in span.attributes
    assert "gen_ai.openai.request.store" not in span.attributes
    assert "gen_ai.openai.request.parallel_tool_calls" not in span.attributes
    assert not any(
        "Omit" in record.message or "Invalid type" in record.message
        for record in caplog.records
    )


def test_responses_stream_existing_response_uses_retrieve(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_retrieve(self, response_id, **kwargs):
        assert response_id == "resp_existing"
        assert kwargs["stream"] is True
        return _ResponseStream(
            [
                _response_created_event(),
                _response_completed_event(),
            ]
        )

    monkeypatch.setattr(Responses, "retrieve", fake_retrieve)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        with getattr(OpenAI(), "responses").stream(
            response_id="resp_existing"
        ) as stream:
            for _ in stream:
                pass
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"chat {DEFAULT_MODEL}"
    assert (
        span.attributes["gen_ai.openai.request.previous_response_id"]
        == "resp_existing"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7


def test_responses_retrieve_non_streaming_does_not_create_span(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_retrieve(self, response_id, **kwargs):
        assert response_id == "resp_existing"
        assert "stream" not in kwargs
        return _response()

    monkeypatch.setattr(Responses, "retrieve", fake_retrieve)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        result = getattr(OpenAI(), "responses").retrieve("resp_existing")
    finally:
        _cleanup(instrumentor)

    assert result.id == "resp_123"
    assert not span_exporter.get_finished_spans()


def test_responses_parse_with_content(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    def fake_parse(self, **kwargs):
        return _response()

    monkeypatch.setattr(Responses, "parse", fake_parse)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        parsed = getattr(OpenAI(), "responses").parse(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
        )
    finally:
        _cleanup(instrumentor)

    assert parsed.id == "resp_123"
    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)


@pytest.mark.asyncio
async def test_async_responses_parse_with_content(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    async def fake_parse(self, **kwargs):
        return _response()

    monkeypatch.setattr(AsyncResponses, "parse", fake_parse)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        parsed = await getattr(AsyncOpenAI(), "responses").parse(
            model=DEFAULT_MODEL,
            instructions="You are concise.",
            input="Say this is a test",
        )
    finally:
        _cleanup(instrumentor)

    assert parsed.id == "resp_123"
    (span,) = span_exporter.get_finished_spans()
    _assert_response_span(span)


@pytest.mark.asyncio
async def test_async_responses_stream_existing_response_uses_retrieve(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    async def fake_retrieve(self, response_id, **kwargs):
        assert response_id == "resp_existing"
        assert kwargs["stream"] is True
        return _AsyncResponseStream(
            [
                _response_created_event(),
                _response_completed_event(),
            ]
        )

    monkeypatch.setattr(AsyncResponses, "retrieve", fake_retrieve)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        async with getattr(AsyncOpenAI(), "responses").stream(
            response_id="resp_existing"
        ) as stream:
            async for _ in stream:
                pass
    finally:
        _cleanup(instrumentor)

    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"chat {DEFAULT_MODEL}"
    assert (
        span.attributes["gen_ai.openai.request.previous_response_id"]
        == "resp_existing"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7


@pytest.mark.asyncio
async def test_async_responses_retrieve_non_streaming_does_not_create_span(
    monkeypatch,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    async def fake_retrieve(self, response_id, **kwargs):
        assert response_id == "resp_existing"
        assert "stream" not in kwargs
        return _response()

    monkeypatch.setattr(AsyncResponses, "retrieve", fake_retrieve)
    instrumentor = _instrument(
        tracer_provider, logger_provider, meter_provider
    )
    try:
        result = await getattr(AsyncOpenAI(), "responses").retrieve(
            "resp_existing"
        )
    finally:
        _cleanup(instrumentor)

    assert result.id == "resp_123"
    assert not span_exporter.get_finished_spans()
