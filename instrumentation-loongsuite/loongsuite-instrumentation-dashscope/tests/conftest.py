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

"""Unit tests configuration module."""

import json
import os

import pytest
import yaml

# Set up DASHSCOPE_API_KEY environment variable BEFORE any dashscope modules are imported
# This is critical because dashscope SDK reads environment variables at module import time
# and caches them in module-level variables
# Only set default if not already set (allows override from environment)
if "DASHSCOPE_API_KEY" not in os.environ:
    # For VCR recording, use a placeholder that will be filtered
    # The actual API key should be set via environment variable before running tests
    os.environ["DASHSCOPE_API_KEY"] = "test_dashscope_api_key"

from opentelemetry.instrumentation.dashscope import DashScopeInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai._configuration import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    is_experimental_mode,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    """Create an in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider):
    """Instrument DashScope SDK for testing."""
    instrumentor = DashScopeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_no_content(tracer_provider):
    """Instrument DashScope SDK with message content capture disabled."""
    # Reset global state to allow environment variable changes to take effect
    is_experimental_mode.cache_clear()

    os.environ.update(
        {
            OTEL_SEMCONV_STABILITY_OPT_IN: "gen_ai_latest_experimental",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "NO_CONTENT",
        }
    )

    instrumentor = DashScopeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    yield instrumentor

    os.environ.pop(OTEL_SEMCONV_STABILITY_OPT_IN, None)
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()
    # Reset global state after test
    is_experimental_mode.cache_clear()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider):
    """Instrument DashScope SDK with message content capture enabled."""
    # Reset global state to allow environment variable changes to take effect
    is_experimental_mode.cache_clear()

    os.environ.update(
        {
            OTEL_SEMCONV_STABILITY_OPT_IN: "gen_ai_latest_experimental",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
        }
    )

    instrumentor = DashScopeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    yield instrumentor

    os.environ.pop(OTEL_SEMCONV_STABILITY_OPT_IN, None)
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()
    # Reset global state after test
    is_experimental_mode.cache_clear()


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_dashscope_api_key"),
            ("x-dashscope-api-key", "test_dashscope_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
    }


class LiteralBlockScalar(str):
    """Formats the string as a literal block scalar."""


def literal_block_scalar_presenter(dumper, data):
    """Represents a scalar string as a literal block."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    """Pretty-prints JSON or returns long strings as a LiteralBlockScalar."""
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    """Searches the data for body strings, attempting to pretty-print JSON."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)
            else:
                convert_body_to_literal(value)
    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = convert_body_to_literal(choice)
    return data


class PrettyPrintJSONBody:
    """Makes request and response body recordings more readable."""

    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(scope="module", autouse=True)
def fixture_vcr(vcr):
    """Register custom VCR serializer."""
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def scrub_response_headers(response):
    """Scrubs sensitive response headers."""
    # Add any sensitive headers to scrub from responses
    if "x-dashscope-request-id" in response.get("headers", {}):
        response["headers"]["x-dashscope-request-id"] = "test_request_id"
    return response
