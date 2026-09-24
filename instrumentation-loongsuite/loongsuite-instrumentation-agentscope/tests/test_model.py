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

# -*- coding: utf-8 -*-
"""AgentScope Model Tests - Following dashscope test_generation.py structure"""

import asyncio

import agentscope
import pytest
from agentscope.model import DashScopeChatModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _safe_getattr(obj, attr, default=None):
    """Safely get attributes from AgentScope response objects"""
    try:
        return getattr(obj, attr, default)
    except (KeyError, AttributeError):
        return default


def _assert_chat_span_attributes(
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
    expect_time_to_first_token: bool = False,
):
    """Assert common chat model span attributes"""
    # Span name format: "chat {model}"
    assert span.name.startswith("chat "), f"Unexpected span name: {span.name}"
    assert request_model in span.name, (
        f"Model {request_model} not in span name: {span.name}"
    )

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat", (
        f"Expected 'chat', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
    )

    assert "gen_ai.provider.name" in span.attributes, (
        "Missing gen_ai.provider.name"
    )
    assert span.attributes["gen_ai.provider.name"] == "dashscope"

    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_REQUEST_MODEL}"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    # Optional attributes
    if response_model is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_RESPONSE_MODEL}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
            == response_model
        )

    if response_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_RESPONSE_ID}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response_id
        )

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

    if finish_reasons is not None:
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}"
        span_finish_reasons = span.attributes[
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ]
        if isinstance(span_finish_reasons, tuple):
            span_finish_reasons = list(span_finish_reasons)
        assert span_finish_reasons == finish_reasons

    # Message content assertions
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

    # Optional request parameters
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


@pytest.mark.vcr(record_mode="none")
def test_model_call_basic(instrument_no_content, span_exporter, request):
    """Test basic model call"""
    # Initialize agentscope
    agentscope.init(project="test_basic")

    # Create model
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    # Prepare messages
    messages = [{"role": "user", "content": "Hello!"}]

    # Call model
    async def call_model():
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    response = asyncio.run(call_model())
    assert response is not None

    # Verify spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"

    # Find chat model span
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1, (
        f"No chat spans found. Available spans: {[s.name for s in spans]}"
    )

    # Verify span attributes
    chat_span = chat_spans[0]
    # Older ChatUsage versions may omit counters; do not fabricate a miss.
    if "cache_input_tokens" in response.usage:
        assert (
            chat_span.attributes["gen_ai.usage.cache_read.input_tokens"]
            == response.usage["cache_input_tokens"]
        )
    else:
        assert (
            "gen_ai.usage.cache_read.input_tokens" not in chat_span.attributes
        )
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=False,  # Do not capture content by default
        expect_output_messages=False,  # Do not capture content by default
        expect_time_to_first_token=True,
    )

    print("✓ Model call (basic) completed successfully")


@pytest.mark.vcr()
def test_model_call_with_messages(
    instrument_no_content, span_exporter, request
):
    """Test model call with multiple messages"""
    agentscope.init(project="test_messages")

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1+1?"},
    ]

    async def call_model():
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    response = asyncio.run(call_model())
    assert response is not None

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    chat_span = chat_spans[0]
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=False,
        expect_output_messages=False,
        expect_time_to_first_token=True,
    )

    print("✓ Model call (with messages) completed successfully")


@pytest.mark.vcr()
async def test_model_call_async(instrument_no_content, span_exporter, request):
    """Test async model call"""
    agentscope.init(project="test_async")

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    messages = [{"role": "user", "content": "Hello from async!"}]

    response = await model(messages)
    if hasattr(response, "__aiter__"):
        result = []
        async for chunk in response:
            result.append(chunk)
        response = result[-1] if result else response

    assert response is not None

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    chat_span = chat_spans[0]
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=False,
        expect_output_messages=False,
        expect_time_to_first_token=True,
    )

    print("✓ Model call (async) completed successfully")


@pytest.mark.vcr()
def test_model_call_streaming(instrument, span_exporter, request):
    """Test streaming model call"""
    agentscope.init(project="test_streaming")

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
        stream=True,
    )

    messages = [{"role": "user", "content": "Count from 1 to 5"}]

    async def call_model():
        response = await model(messages)
        chunk_count = 0
        last_response = response
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                chunk_count += 1
                last_response = chunk
            return last_response, chunk_count
        else:
            return response, 1

    last_chunk, chunk_count = asyncio.run(call_model())
    assert last_chunk is not None, (
        f"last_chunk is None, chunk_count={chunk_count}"
    )
    print(f"Received {chunk_count} chunks")

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    chat_span = chat_spans[0]
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=False,
        expect_output_messages=False,
        expect_time_to_first_token=True,
    )

    print("✓ Model call (streaming) completed successfully")


@pytest.mark.vcr()
def test_model_call_with_parameters(instrument, span_exporter, request):
    """Test model call with parameters (passed to call method)"""
    agentscope.init(project="test_parameters")

    # DashScopeChatModel parameters passed via kwargs
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    messages = [{"role": "user", "content": "Write a short poem"}]

    async def call_model():
        # Parameters passed at call time
        response = await model(
            messages,
            temperature=0.8,
            top_p=0.9,
            max_tokens=100,
        )
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    response = asyncio.run(call_model())
    assert response is not None

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    chat_span = chat_spans[0]
    # Verify basic attributes
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=False,
        expect_output_messages=False,
        expect_time_to_first_token=True,
    )

    print("✓ Model call (with parameters) completed successfully")


@pytest.mark.vcr()
def test_model_call_with_content_capture(
    instrument_with_content, span_exporter, request
):
    """Test model call with content capture enabled"""
    agentscope.init(project="test_content_capture")

    # Note: Disable streaming to avoid complexity with content capture
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
        stream=False,
    )

    messages = [{"role": "user", "content": "Say this is a test"}]

    async def call_model():
        response = await model(messages)
        # Use try-except to safely check async generator, avoid KeyError from DashScope response object
        try:
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
        except (KeyError, AttributeError):
            pass
        return response

    response = asyncio.run(call_model())
    assert response is not None

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    chat_span = chat_spans[0]
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=True,  # Content capture enabled
        expect_output_messages=True,  # Content capture enabled
        expect_time_to_first_token=False,
    )

    print("✓ Model call (with content capture) completed successfully")


@pytest.mark.vcr()
def test_model_call_no_content_capture(
    instrument_no_content, span_exporter, request
):
    """Test model call with content capture disabled"""
    agentscope.init(project="test_no_content_capture")

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    messages = [{"role": "user", "content": "Say this is a test"}]

    async def call_model():
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    response = asyncio.run(call_model())
    assert response is not None

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    chat_span = chat_spans[0]
    _assert_chat_span_attributes(
        chat_span,
        request_model="qwen-max",
        expect_input_messages=False,  # Content capture disabled
        expect_output_messages=False,  # Content capture disabled
        expect_time_to_first_token=True,
    )

    print("✓ Model call (no content capture) completed successfully")


@pytest.mark.vcr()
def test_model_call_multiple_sequential(instrument, span_exporter, request):
    """Test multiple sequential model calls"""
    agentscope.init(project="test_multiple")

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    async def call_model(content: str):
        messages = [{"role": "user", "content": content}]
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    # Make multiple calls
    asyncio.run(call_model("First call"))
    asyncio.run(call_model("Second call"))
    asyncio.run(call_model("Third call"))

    # Verify spans
    spans = span_exporter.get_finished_spans()
    chat_spans = [span for span in spans if span.name.startswith("chat ")]

    # Should have at least 3  chat span
    assert len(chat_spans) >= 3, (
        f"Expected at least 3 chat spans, got {len(chat_spans)}"
    )

    # Verify each span
    for chat_span in chat_spans[:3]:
        _assert_chat_span_attributes(
            chat_span,
            request_model="qwen-max",
            expect_input_messages=False,
            expect_output_messages=False,
            expect_time_to_first_token=True,
        )

    print("✓ Model call (multiple sequential) completed successfully")
