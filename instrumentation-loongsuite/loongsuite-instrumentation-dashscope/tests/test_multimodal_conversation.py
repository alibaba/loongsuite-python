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

"""Tests for MultiModalConversation instrumentation."""

import json
from types import SimpleNamespace
from typing import Optional

import pytest
from dashscope import MultiModalConversation

from opentelemetry.instrumentation._semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.dashscope.utils.multimodal import (
    _extract_multimodal_output_messages,
    _update_invocation_from_multimodal_response,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import LLMInvocation, Text, Uri


def _make_multimodal_response(content, finish_reason="stop"):
    return SimpleNamespace(
        output=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )
    )


@pytest.fixture(scope="function")
def content_capture_env(monkeypatch):
    _OpenTelemetrySemanticConventionStability._initialized = False
    monkeypatch.setenv(
        OTEL_SEMCONV_STABILITY_OPT_IN, "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY"
    )
    _OpenTelemetrySemanticConventionStability._initialize()
    yield
    _OpenTelemetrySemanticConventionStability._initialized = False


def _safe_getattr(obj, attr, default=None):
    """Safely get attribute from DashScope response objects."""
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default


def _assert_multimodal_span_attributes(
    span,
    request_model: str,
    response_model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    request_id: Optional[str] = None,
    expect_input_messages: bool = True,
    expect_output_messages: bool = True,
    expect_time_to_first_token: bool = False,
):
    """Assert MultiModalConversation span attributes."""
    # Span name format is "{operation_name} {model}"
    assert span.name == f"chat {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    # Optional attributes
    if response_model is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
            == response_model
        )

    if request_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == request_id
        )

    if input_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )

    if output_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
            == output_tokens
        )

    # Assert input/output messages based on expectation
    if expect_input_messages:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_INPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_INPUT_MESSAGES} should not be present"
        )

    if expect_output_messages:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_OUTPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_OUTPUT_MESSAGES} should not be present"
        )

    # Assert time to first token for streaming responses (in nanoseconds)
    if expect_time_to_first_token:
        assert "gen_ai.response.time_to_first_token" in span.attributes, (
            "Missing gen_ai.response.time_to_first_token"
        )
        ttft_ns = span.attributes["gen_ai.response.time_to_first_token"]
        assert isinstance(ttft_ns, int), (
            f"time_to_first_token should be an integer (nanoseconds), got {type(ttft_ns)}"
        )
        assert ttft_ns > 0, (
            f"time_to_first_token should be positive, got {ttft_ns}"
        )


@pytest.mark.parametrize(
    ("content_key", "url", "modality"),
    [
        ("image", "https://example.com/a.png", "image"),
        ("audio", "https://example.com/a.wav", "audio"),
        ("video", "https://example.com/a.mp4", "video"),
    ],
)
def test_extract_multimodal_output_messages_with_uri_content(
    content_key, url, modality
):
    """Test output message extraction for media URI content."""
    messages = _extract_multimodal_output_messages(
        _make_multimodal_response([{content_key: url}])
    )

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].finish_reason == "stop"
    assert len(messages[0].parts) == 1

    part = messages[0].parts[0]
    assert isinstance(part, Uri)
    assert part.uri == url
    assert part.modality == modality
    assert part.mime_type is None
    assert part.type == "uri"


def test_extract_multimodal_output_messages_with_text_and_image_content():
    """Test output message extraction preserves mixed text and image parts."""
    image_url = "https://example.com/generated.png"
    messages = _extract_multimodal_output_messages(
        _make_multimodal_response([{"text": "ok"}, {"image": image_url}])
    )

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].finish_reason == "stop"
    assert len(messages[0].parts) == 2

    text_part = messages[0].parts[0]
    assert isinstance(text_part, Text)
    assert text_part.content == "ok"
    assert text_part.type == "text"

    image_part = messages[0].parts[1]
    assert isinstance(image_part, Uri)
    assert image_part.uri == image_url
    assert image_part.modality == "image"
    assert image_part.mime_type is None
    assert image_part.type == "uri"


def test_multimodal_image_output_messages_written_to_span(
    content_capture_env, tracer_provider, span_exporter
):
    """Test image output URI is written to gen_ai.output.messages."""
    image_url = "https://example.com/generated.png"
    response = _make_multimodal_response([{"image": image_url}])
    invocation = LLMInvocation(request_model="wan2.7-image")
    invocation.provider = "dashscope"

    _update_invocation_from_multimodal_response(invocation, response)

    handler = TelemetryHandler(tracer_provider=tracer_provider)
    handler.start_llm(invocation)
    handler.stop_llm(invocation)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    output_messages = json.loads(
        spans[0].attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [
                {
                    "mime_type": None,
                    "modality": "image",
                    "uri": image_url,
                    "type": "uri",
                }
            ],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.vcr()
def test_multimodal_conversation_call_basic(
    instrument_with_content, span_exporter
):
    """Test synchronous MultiModalConversation.call can be instrumented."""
    messages = [
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}],
        }
    ]

    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
    )
    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)
    request_id = _safe_getattr(response, "request_id", None)

    _assert_multimodal_span_attributes(
        span,
        request_model="qwen-vl-plus",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        request_id=request_id,
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ MultiModalConversation.call (basic) completed successfully")


@pytest.mark.vcr()
def test_multimodal_conversation_call_with_image(
    instrument_with_content, span_exporter
):
    """Test MultiModalConversation.call with image input."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
                },
                {"text": "What do you see in this image?"},
            ],
        }
    ]

    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
    )
    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)
    request_id = _safe_getattr(response, "request_id", None)

    _assert_multimodal_span_attributes(
        span,
        request_model="qwen-vl-plus",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        request_id=request_id,
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ MultiModalConversation.call (with image) completed successfully")


@pytest.mark.vcr()
def test_multimodal_conversation_call_streaming(
    instrument_with_content, span_exporter
):
    """Test MultiModalConversation.call with streaming response."""
    messages = [
        {
            "role": "user",
            "content": [{"text": "Tell me a short story."}],
        }
    ]

    responses = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
        stream=True,
    )
    assert responses is not None

    # Consume the generator
    last_response = None
    for response in responses:
        last_response = response

    assert last_response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    usage = _safe_getattr(last_response, "usage", None)
    response_model = _safe_getattr(last_response, "model", None)
    request_id = _safe_getattr(last_response, "request_id", None)

    _assert_multimodal_span_attributes(
        span,
        request_model="qwen-vl-plus",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        request_id=request_id,
        expect_input_messages=True,
        expect_output_messages=True,
        expect_time_to_first_token=True,
    )

    print("✓ MultiModalConversation.call (streaming) completed successfully")
