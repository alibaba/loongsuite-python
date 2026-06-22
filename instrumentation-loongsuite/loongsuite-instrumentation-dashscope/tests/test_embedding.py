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

"""Tests for TextEmbedding instrumentation."""

from typing import Optional

import pytest
from dashscope import TextEmbedding

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)


def _assert_embedding_span_attributes(
    span,
    request_model: str,
    response=None,
    input_tokens: Optional[int] = None,
    dimension_count: Optional[int] = None,
    server_address: Optional[str] = None,
    server_port: Optional[int] = None,
):
    """Assert embedding span attributes according to GenAI semantic conventions.

    Args:
        span: The span to assert
        request_model: Expected model name
        response: Optional response object to extract attributes from
        input_tokens: Expected input token count (if available)
        dimension_count: Expected embedding dimension count (if available)
        server_address: Expected server address (if available)
        server_port: Expected server port (if available)
    """
    # Span name format is "{operation_name} {model}" per semantic conventions
    # Operation name is "embeddings" (plural, not "embedding")
    assert span.name == f"embeddings {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "embeddings"
    ), (
        f"Expected 'embeddings', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
    )

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_PROVIDER_NAME}"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    # Conditionally required attributes
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_REQUEST_MODEL}"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    # Recommended attributes - check if available
    if input_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )
    elif response:
        # Try to extract from response if not provided
        # DashScope embedding uses total_tokens instead of input_tokens
        try:
            usage = getattr(response, "usage", None)
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                if total_tokens is not None:
                    assert (
                        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
                        in span.attributes
                    ), f"Missing {GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS}"
                    assert (
                        span.attributes[
                            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
                        ]
                        == total_tokens
                    )
        except (KeyError, AttributeError):
            pass

    # Optional attributes - check if available
    # Dimension count should only be set if it was specified in the request
    if dimension_count is not None:
        # If dimension_count was explicitly provided in the request, it must be set in span
        assert "gen_ai.embeddings.dimension.count" in span.attributes, (
            "Missing gen_ai.embeddings.dimension.count"
        )
        assert (
            span.attributes["gen_ai.embeddings.dimension.count"]
            == dimension_count
        )
    else:
        # If dimension_count was not provided in the request, it should not be set in span
        assert "gen_ai.embeddings.dimension.count" not in span.attributes, (
            "gen_ai.embeddings.dimension.count should not be set when not specified in request"
        )

    if server_address is not None:
        assert ServerAttributes.SERVER_ADDRESS in span.attributes, (
            f"Missing {ServerAttributes.SERVER_ADDRESS}"
        )
        assert (
            span.attributes[ServerAttributes.SERVER_ADDRESS] == server_address
        )

    if server_port is not None:
        assert ServerAttributes.SERVER_PORT in span.attributes, (
            f"Missing {ServerAttributes.SERVER_PORT}"
        )
        assert span.attributes[ServerAttributes.SERVER_PORT] == server_port


@pytest.mark.vcr()
def test_text_embedding_basic(instrument, span_exporter):
    """Test basic text embedding call."""

    response = TextEmbedding.call(
        model="text-embedding-v4", input="Hello, world!"
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]

    # Extract input tokens from response if available
    # DashScope embedding uses total_tokens instead of input_tokens
    input_tokens = None
    try:
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "total_tokens", None)
    except (KeyError, AttributeError):
        pass

    # Assert all span attributes
    _assert_embedding_span_attributes(
        span,
        request_model="text-embedding-v4",
        response=response,
        input_tokens=input_tokens,
    )

    print("✓ TextEmbedding.call completed successfully")


@pytest.mark.vcr()
def test_text_embedding_batch(instrument, span_exporter):
    """Test text embedding with batch input."""

    response = TextEmbedding.call(
        model="text-embedding-v4", input=["Hello", "World"]
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]

    # Extract input tokens from response if available
    # DashScope embedding uses total_tokens instead of input_tokens
    input_tokens = None
    try:
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "total_tokens", None)
    except (KeyError, AttributeError):
        pass

    # Assert all span attributes
    _assert_embedding_span_attributes(
        span,
        request_model="text-embedding-v4",
        response=response,
        input_tokens=input_tokens,
    )

    print("✓ TextEmbedding.call (batch) completed successfully")


@pytest.mark.vcr()
def test_text_embedding_with_text_type(instrument, span_exporter):
    """Test text embedding with text_type parameter."""

    response = TextEmbedding.call(
        model="text-embedding-v4",
        input="What is machine learning?",
        text_type="query",
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]

    # Extract input tokens from response if available
    # DashScope embedding uses total_tokens instead of input_tokens
    input_tokens = None
    try:
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "total_tokens", None)
    except (KeyError, AttributeError):
        pass

    # Assert all span attributes
    _assert_embedding_span_attributes(
        span,
        request_model="text-embedding-v4",
        response=response,
        input_tokens=input_tokens,
    )

    print("✓ TextEmbedding.call (with text_type) completed successfully")


@pytest.mark.vcr()
def test_text_embedding_with_dimension(instrument, span_exporter):
    """Test text embedding with dimension parameter."""

    response = TextEmbedding.call(
        model="text-embedding-v4",
        input="What is machine learning?",
        dimension=512,
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]

    # Extract input tokens from response if available
    # DashScope embedding uses total_tokens instead of input_tokens
    input_tokens = None
    try:
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "total_tokens", None)
    except (KeyError, AttributeError):
        pass

    # Assert all span attributes including dimension_count
    _assert_embedding_span_attributes(
        span,
        request_model="text-embedding-v4",
        response=response,
        input_tokens=input_tokens,
        dimension_count=512,  # Should be captured from request
    )

    print("✓ TextEmbedding.call (with dimension) completed successfully")
