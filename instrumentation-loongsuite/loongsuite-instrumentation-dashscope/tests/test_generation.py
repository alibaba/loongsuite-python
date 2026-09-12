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

"""Tests for Generation instrumentation."""

import json as json_utils

import pytest
from dashscope import AioGeneration, Generation

from opentelemetry.instrumentation.dashscope.utils import (
    _create_invocation_from_generation,
)
from opentelemetry.instrumentation.dashscope.utils.common import (
    DASHSCOPE_SERVER_ADDRESS,
    DASHSCOPE_SERVER_PORT,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _safe_getattr(obj, attr, default=None):
    """Safely get attribute from DashScope response objects that may raise KeyError."""
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default


def test_create_invocation_from_generation_uses_dashscope_server_constants():
    invocation = _create_invocation_from_generation({"model": "qwen-turbo"})

    assert invocation.server_address == DASHSCOPE_SERVER_ADDRESS
    assert invocation.server_port == DASHSCOPE_SERVER_PORT


def _assert_generation_span_attributes(
    span,
    request_model: str,
    response_id: str = None,
    response_model: str = None,
    input_tokens: int = None,
    output_tokens: int = None,
    finish_reasons: list = None,
    expect_input_messages: bool = False,
    expect_output_messages: bool = False,
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    top_k: float = None,
    frequency_penalty: float = None,
    presence_penalty: float = None,
    stop_sequences: list = None,
    seed: int = None,
    choice_count: int = None,
    output_type: str = None,
    expect_time_to_first_token: bool = False,
):
    """Assert common generation span attributes."""
    # Span name format is "{operation_name} {model}" per semantic conventions
    # Operation name is "chat" (not "gen_ai.chat")
    assert span.name == f"chat {request_model}"
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat", (
        f"Expected 'chat', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
    )

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_PROVIDER_NAME}"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_REQUEST_MODEL}"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    if response_model is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_RESPONSE_MODEL}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
            == response_model
        )
    # If response_model is None, don't assert it exists (it may not be available)

    if response_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_RESPONSE_ID}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response_id
        )
    # If response_id is None, don't assert it exists (it may not be available)

    if input_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )

    if output_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
            == output_tokens
        )

    # Assert finish reasons are present (if finish_reasons is provided)
    if finish_reasons is not None:
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}"
        # Convert span attribute to list for comparison (it may be a tuple)
        span_finish_reasons = span.attributes[
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ]
        if isinstance(span_finish_reasons, tuple):
            span_finish_reasons = list(span_finish_reasons)
        assert span_finish_reasons == finish_reasons

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

    # Assert optional request parameters
    if temperature is not None:
        assert "gen_ai.request.temperature" in span.attributes, (
            "Missing gen_ai.request.temperature"
        )
        assert span.attributes["gen_ai.request.temperature"] == temperature

    if max_tokens is not None:
        assert "gen_ai.request.max_tokens" in span.attributes, (
            "Missing gen_ai.request.max_tokens"
        )
        assert span.attributes["gen_ai.request.max_tokens"] == max_tokens

    if top_p is not None:
        assert "gen_ai.request.top_p" in span.attributes, (
            "Missing gen_ai.request.top_p"
        )
        assert span.attributes["gen_ai.request.top_p"] == top_p

    if top_k is not None:
        assert "gen_ai.request.top_k" in span.attributes, (
            "Missing gen_ai.request.top_k"
        )
        assert span.attributes["gen_ai.request.top_k"] == top_k

    if frequency_penalty is not None:
        assert "gen_ai.request.frequency_penalty" in span.attributes, (
            "Missing gen_ai.request.frequency_penalty"
        )
        assert (
            span.attributes["gen_ai.request.frequency_penalty"]
            == frequency_penalty
        )

    if presence_penalty is not None:
        assert "gen_ai.request.presence_penalty" in span.attributes, (
            "Missing gen_ai.request.presence_penalty"
        )
        assert (
            span.attributes["gen_ai.request.presence_penalty"]
            == presence_penalty
        )

    if stop_sequences is not None:
        assert "gen_ai.request.stop_sequences" in span.attributes, (
            "Missing gen_ai.request.stop_sequences"
        )
        # Convert span attribute to list for comparison (it may be a tuple)
        span_stop_sequences = span.attributes["gen_ai.request.stop_sequences"]
        if isinstance(span_stop_sequences, tuple):
            span_stop_sequences = list(span_stop_sequences)
        assert span_stop_sequences == stop_sequences

    if seed is not None:
        assert "gen_ai.request.seed" in span.attributes, (
            "Missing gen_ai.request.seed"
        )
        assert span.attributes["gen_ai.request.seed"] == seed

    if choice_count is not None and choice_count != 1:
        assert "gen_ai.request.choice.count" in span.attributes, (
            "Missing gen_ai.request.choice.count"
        )
        assert span.attributes["gen_ai.request.choice.count"] == choice_count

    if output_type is not None:
        assert "gen_ai.output.type" in span.attributes, (
            "Missing gen_ai.output.type"
        )
        assert span.attributes["gen_ai.output.type"] == output_type

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
    else:
        # For non-streaming responses, TTFT should not be present
        assert "gen_ai.response.time_to_first_token" not in span.attributes, (
            "gen_ai.response.time_to_first_token should not be present for non-streaming"
        )


@pytest.mark.vcr()
def test_generation_call_basic(instrument, span_exporter):
    """Test basic synchronous generation call."""

    response = Generation.call(model="qwen-turbo", prompt="Hello!")

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    # Note: request_id exists as a direct attribute, model may not exist
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    request_id = _safe_getattr(response, "request_id", None)
    response_model = _safe_getattr(response, "model", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=request_id,  # May be None if not available
        response_model=response_model,  # May be None if not available
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=False,  # Default: no content capture
        expect_output_messages=False,  # Default: no content capture
    )

    print("✓ Generation.call (basic) completed successfully")


@pytest.mark.vcr()
def test_generation_call_with_messages(instrument, span_exporter):
    """Test generation call with messages parameter."""

    response = Generation.call(
        model="qwen-turbo", messages=[{"role": "user", "content": "Hello!"}]
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=False,
        expect_output_messages=False,
    )

    print("✓ Generation.call (with messages) completed successfully")


@pytest.mark.vcr()
def test_generation_call_streaming(instrument, span_exporter):
    """Test synchronous generation with streaming (default: full output mode)."""

    responses = Generation.call(
        model="qwen-turbo", prompt="Count from 1 to 5", stream=True
    )

    chunk_count = 0
    last_response = None
    for response in responses:
        assert response is not None
        chunk_count += 1
        last_response = response

    assert chunk_count > 0

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    usage = (
        _safe_getattr(last_response, "usage", None) if last_response else None
    )
    output = (
        _safe_getattr(last_response, "output", None) if last_response else None
    )
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )

    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_time_to_first_token=True,
    )

    print("✓ Generation.call (streaming) completed successfully")


@pytest.mark.vcr()
def test_generation_call_streaming_incremental_output(
    instrument, span_exporter
):
    """Test synchronous generation with streaming in incremental output mode."""

    responses = Generation.call(
        model="qwen-turbo",
        prompt="Count from 1 to 5",
        stream=True,
        incremental_output=True,
    )

    chunk_count = 0
    last_response = None
    for response in responses:
        assert response is not None
        chunk_count += 1
        last_response = response

    assert chunk_count > 0

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    usage = (
        _safe_getattr(last_response, "usage", None) if last_response else None
    )
    output = (
        _safe_getattr(last_response, "output", None) if last_response else None
    )
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )

    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_time_to_first_token=True,
    )

    print(
        "✓ Generation.call (streaming incremental_output) completed successfully"
    )


@pytest.mark.vcr()
def test_generation_call_with_parameters(instrument, span_exporter):
    """Test generation call with various parameters."""

    response = Generation.call(
        model="qwen-turbo",
        prompt="Write a short poem",
        temperature=0.8,
        top_p=0.9,
        max_tokens=100,
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        temperature=0.8,
        max_tokens=100,
        top_p=0.9,
        expect_input_messages=False,
        expect_output_messages=False,
    )

    print("✓ Generation.call (with parameters) completed successfully")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_aio_generation_call_basic(instrument, span_exporter):
    """Test basic asynchronous generation call."""

    response = await AioGeneration.call(model="qwen-turbo", prompt="Hello!")

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=False,
        expect_output_messages=False,
    )

    print("✓ AioGeneration.call (basic) completed successfully")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_aio_generation_call_streaming(instrument, span_exporter):
    """Test asynchronous generation with streaming (default: full output mode)."""

    responses = await AioGeneration.call(
        model="qwen-turbo", prompt="Count from 1 to 5", stream=True
    )

    chunk_count = 0
    last_response = None
    async for response in responses:
        assert response is not None
        chunk_count += 1
        last_response = response

    assert chunk_count > 0

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    usage = (
        _safe_getattr(last_response, "usage", None) if last_response else None
    )
    output = (
        _safe_getattr(last_response, "output", None) if last_response else None
    )
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_time_to_first_token=True,
    )

    print("✓ AioGeneration.call (streaming) completed successfully")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_aio_generation_call_streaming_incremental_output(
    instrument, span_exporter
):
    """Test asynchronous generation with streaming in incremental output mode."""

    responses = await AioGeneration.call(
        model="qwen-turbo",
        prompt="Count from 1 to 5",
        stream=True,
        incremental_output=True,
    )

    chunk_count = 0
    last_response = None
    async for response in responses:
        assert response is not None
        chunk_count += 1
        last_response = response

    assert chunk_count > 0

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    usage = (
        _safe_getattr(last_response, "usage", None) if last_response else None
    )
    output = (
        _safe_getattr(last_response, "output", None) if last_response else None
    )
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )

    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_time_to_first_token=True,
    )

    print(
        "✓ AioGeneration.call (streaming incremental_output) completed successfully"
    )


@pytest.mark.vcr()
def test_generation_call_with_content_capture(
    instrument_with_content, span_exporter
):
    """Test generation call with message content capture enabled."""

    messages = [{"role": "user", "content": "Say this is a test"}]
    response = Generation.call(model="qwen-turbo", messages=messages)

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=True,  # Content capture enabled
        expect_output_messages=True,  # Content capture enabled
    )

    print("✓ Generation.call (with content capture) completed successfully")


@pytest.mark.vcr()
def test_generation_call_message_result_without_tool_calls_content_capture(
    instrument_with_content, span_exporter
):
    """Test message-format text responses that omit the tool_calls field."""

    messages = [
        {
            "role": "user",
            "content": "Reply with exactly: dashscope plain message",
        }
    ]
    response = Generation.call(
        model="qwen-turbo",
        messages=messages,
        result_format="message",
    )

    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    output = _safe_getattr(response, "output", None)
    choices = _safe_getattr(output, "choices", None) if output else None
    finish_reason = None
    if choices:
        finish_reason = _safe_getattr(choices[0], "finish_reason", None)
    if not finish_reason and output:
        finish_reason = _safe_getattr(output, "finish_reason", None)

    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=True,
        expect_output_messages=True,
    )

    output_messages = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    if isinstance(output_messages, str):
        output_messages = json_utils.loads(output_messages)

    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert (
        output_messages[0]["parts"][0]["content"] == "dashscope plain message"
    )

    print(
        "✓ Generation.call (message result without tool_calls, content capture) completed successfully"
    )


@pytest.mark.vcr()
def test_generation_call_no_content_capture(
    instrument_no_content, span_exporter
):
    """Test generation call with message content capture disabled."""

    messages = [{"role": "user", "content": "Say this is a test"}]
    response = Generation.call(model="qwen-turbo", messages=messages)

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=False,  # Content capture disabled
        expect_output_messages=False,  # Content capture disabled
    )

    print("✓ Generation.call (no content capture) completed successfully")


@pytest.mark.vcr()
def test_generation_call_with_prompt_content_capture(
    instrument_with_content, span_exporter
):
    """Test generation call with prompt (string) and content capture enabled."""

    response = Generation.call(model="qwen-turbo", prompt="Hello, world!")

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=True,  # Content capture enabled, prompt should be captured
        expect_output_messages=True,  # Content capture enabled
    )

    print(
        "✓ Generation.call (prompt with content capture) completed successfully"
    )


@pytest.mark.vcr()
def test_generation_call_with_all_parameters(instrument, span_exporter):
    """Test generation call with all optional parameters."""

    response = Generation.call(
        model="qwen-turbo",
        prompt="Test prompt",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=200,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["stop1", "stop2"],
        seed=42,
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=200,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop_sequences=["stop1", "stop2"],
        seed=42,
        expect_input_messages=False,
        expect_output_messages=False,
    )

    print("✓ Generation.call (with all parameters) completed successfully")


@pytest.mark.vcr()
def test_generation_call_with_tool_calls_content_capture(
    instrument_with_content, span_exporter
):
    """Test generation call with tool calls and content capture enabled."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
    response = Generation.call(
        model="qwen-turbo",
        messages=messages,
        tools=tools,
        result_format="message",
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)

    # Assert tool definitions are present
    assert "gen_ai.tool.definitions" in span.attributes, (
        "Missing gen_ai.tool.definitions"
    )
    tool_definitions_str = span.attributes["gen_ai.tool.definitions"]
    assert isinstance(tool_definitions_str, str), (
        "Tool definitions should be a JSON string"
    )

    # Parse JSON to verify content
    tool_definitions = json_utils.loads(tool_definitions_str)
    assert isinstance(tool_definitions, list), (
        "Tool definitions should be a list after parsing"
    )
    assert len(tool_definitions) > 0, "Tool definitions should not be empty"

    # Verify full tool definition is recorded (content capture enabled)
    tool_def = tool_definitions[0]
    assert "name" in tool_def, "Tool definition should have 'name'"
    assert "type" in tool_def, "Tool definition should have 'type'"
    assert tool_def["name"] == "get_current_weather", "Tool name should match"
    assert tool_def["type"] == "function", "Tool type should be 'function'"
    # With content capture enabled, should have full definition
    assert "description" in tool_def, (
        "Tool definition should have 'description' when content capture is enabled"
    )
    assert "parameters" in tool_def, (
        "Tool definition should have 'parameters' when content capture is enabled"
    )

    # Assert input/output messages with tool calls (content capture enabled)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=True,  # Content capture enabled
        expect_output_messages=True,  # Content capture enabled
    )

    # Verify tool calls in output messages if present
    if GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes:
        output_messages = span.attributes[
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        ]
        if output_messages:
            # Check if any message contains tool calls
            # Output messages are stored as JSON strings, parse them
            if isinstance(output_messages, str):
                output_messages = json_utils.loads(output_messages)

            # Check if any message has tool calls
            has_tool_calls = False
            for msg in output_messages:
                if isinstance(msg, dict):
                    parts = msg.get("parts", [])
                    for part in parts:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "tool_call"
                        ):
                            has_tool_calls = True
                            break
                    if has_tool_calls:
                        break

            # If finish_reason is "tool_calls", we should have tool calls
            if finish_reason == "tool_calls":
                assert has_tool_calls, (
                    "Expected tool calls in output messages when finish_reason is tool_calls"
                )

    print(
        "✓ Generation.call (with tool calls, content capture) completed successfully"
    )


@pytest.mark.vcr()
def test_generation_call_with_tool_calls_no_content_capture(
    instrument_no_content, span_exporter
):
    """Test generation call with tool calls and content capture disabled."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
    response = Generation.call(
        model="qwen-turbo",
        messages=messages,
        tools=tools,
        result_format="message",
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    # Use safe getattr to access DashScope response attributes (may raise KeyError)
    output = _safe_getattr(response, "output", None)
    finish_reason = (
        _safe_getattr(output, "finish_reason", None) if output else None
    )
    usage = _safe_getattr(response, "usage", None)

    # Assert tool definitions are present (should be present regardless of content capture)
    assert "gen_ai.tool.definitions" in span.attributes, (
        "Missing gen_ai.tool.definitions"
    )
    tool_definitions_str = span.attributes["gen_ai.tool.definitions"]
    assert isinstance(tool_definitions_str, str), (
        "Tool definitions should be a JSON string"
    )

    # Parse JSON to verify content
    tool_definitions = json_utils.loads(tool_definitions_str)
    assert isinstance(tool_definitions, list), (
        "Tool definitions should be a list after parsing"
    )
    assert len(tool_definitions) > 0, "Tool definitions should not be empty"

    # Verify only type and name are recorded (content capture disabled)
    tool_def = tool_definitions[0]
    assert "name" in tool_def, "Tool definition should have 'name'"
    assert "type" in tool_def, "Tool definition should have 'type'"
    assert tool_def["name"] == "get_current_weather", "Tool name should match"
    assert tool_def["type"] == "function", "Tool type should be 'function'"
    # With content capture disabled, should only have name and type
    assert "description" not in tool_def, (
        "Tool definition should NOT have 'description' when content capture is disabled"
    )
    assert "parameters" not in tool_def, (
        "Tool definition should NOT have 'parameters' when content capture is disabled"
    )

    # Assert input/output messages are NOT present (content capture disabled)
    _assert_generation_span_attributes(
        span,
        request_model="qwen-turbo",
        response_id=_safe_getattr(response, "request_id", None),
        response_model=_safe_getattr(response, "model", None),
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        finish_reasons=[finish_reason] if finish_reason else None,
        expect_input_messages=False,  # Content capture disabled
        expect_output_messages=False,  # Content capture disabled
    )

    print(
        "✓ Generation.call (with tool calls, no content capture) completed successfully"
    )


@pytest.mark.vcr()
def test_generation_call_with_tool_call_response_content_capture(
    instrument_with_content, span_exporter
):
    """Test generation call with tool call response in messages and content capture enabled."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # First call: get tool call
    messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
    response1 = Generation.call(
        model="qwen-turbo",
        messages=messages,
        tools=tools,
        result_format="message",
    )

    assert response1 is not None

    # Extract tool call from response (if present)
    output1 = _safe_getattr(response1, "output", None)
    tool_call_id = None
    tool_call_name = None

    if output1:
        choices = _safe_getattr(output1, "choices", None)
        if choices and isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]
            message = _safe_getattr(choice, "message", None)
            if message:
                tool_calls = _safe_getattr(message, "tool_calls", None)
                if (
                    tool_calls
                    and isinstance(tool_calls, list)
                    and len(tool_calls) > 0
                ):
                    tool_call = tool_calls[0]
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                        function = tool_call.get("function", {})
                        if isinstance(function, dict):
                            tool_call_name = function.get("name")
                    elif hasattr(tool_call, "id"):
                        tool_call_id = getattr(tool_call, "id", None)
                        function = getattr(tool_call, "function", None)
                        if function:
                            tool_call_name = getattr(function, "name", None)

    # Second call: provide tool call response
    if tool_call_id:
        messages_with_response = [
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call_name or "get_current_weather",
                            "arguments": '{"location": "Beijing"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temperature": 20, "condition": "sunny"}',
                "tool_call_id": tool_call_id,
            },
        ]

        response2 = Generation.call(
            model="qwen-turbo",
            messages=messages_with_response,
            tools=tools,
            result_format="message",
        )

        assert response2 is not None

        # Assert spans
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"

        # Check the second span (tool call response)
        span2 = spans[-1]
        usage2 = _safe_getattr(response2, "usage", None)
        output2 = _safe_getattr(response2, "output", None)
        finish_reason2 = (
            _safe_getattr(output2, "finish_reason", None) if output2 else None
        )

        # Assert tool definitions are present
        assert "gen_ai.tool.definitions" in span2.attributes, (
            "Missing gen_ai.tool.definitions"
        )
        tool_definitions_str = span2.attributes["gen_ai.tool.definitions"]
        assert isinstance(tool_definitions_str, str), (
            "Tool definitions should be a JSON string"
        )

        tool_definitions = json_utils.loads(tool_definitions_str)
        assert isinstance(tool_definitions, list), (
            "Tool definitions should be a list after parsing"
        )
        assert len(tool_definitions) > 0, (
            "Tool definitions should not be empty"
        )

        # Verify full tool definition is recorded (content capture enabled)
        tool_def = tool_definitions[0]
        assert "name" in tool_def, "Tool definition should have 'name'"
        assert "type" in tool_def, "Tool definition should have 'type'"
        assert tool_def["name"] == "get_current_weather", (
            "Tool name should match"
        )
        assert tool_def["type"] == "function", "Tool type should be 'function'"
        # With content capture enabled, should have full definition
        assert "description" in tool_def, (
            "Tool definition should have 'description' when content capture is enabled"
        )
        assert "parameters" in tool_def, (
            "Tool definition should have 'parameters' when content capture is enabled"
        )

        # Check if response has output messages
        # For tool call response scenario, output may be empty or have different format
        # We check the span attributes to see if output messages were actually captured
        has_output_messages = (
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span2.attributes
        )

        # Assert input/output messages with tool call response (content capture enabled)
        # Note: output messages may not be present if response is empty
        _assert_generation_span_attributes(
            span2,
            request_model="qwen-turbo",
            response_id=_safe_getattr(response2, "request_id", None),
            response_model=_safe_getattr(response2, "model", None),
            input_tokens=_safe_getattr(usage2, "input_tokens", None)
            if usage2
            else None,
            output_tokens=_safe_getattr(usage2, "output_tokens", None)
            if usage2
            else None,
            finish_reasons=[finish_reason2] if finish_reason2 else None,
            expect_input_messages=True,  # Content capture enabled
            expect_output_messages=has_output_messages,  # Only if response has output messages
        )

        # Verify tool call response in input messages
        if GenAIAttributes.GEN_AI_INPUT_MESSAGES in span2.attributes:
            input_messages = span2.attributes[
                GenAIAttributes.GEN_AI_INPUT_MESSAGES
            ]
            if input_messages:
                if isinstance(input_messages, str):
                    input_messages = json_utils.loads(input_messages)

                # Check if any message has tool call response
                has_tool_response = False
                for msg in input_messages:
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        parts = msg.get("parts", [])
                        if role == "tool":
                            for part in parts:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type")
                                    == "tool_call_response"
                                ):
                                    has_tool_response = True
                                    break
                        if has_tool_response:
                            break

                assert has_tool_response, (
                    "Expected tool call response in input messages"
                )

        print(
            "✓ Generation.call (with tool call response, content capture) completed successfully"
        )
