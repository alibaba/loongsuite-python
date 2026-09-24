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
"""Test Configuration"""

import asyncio
import inspect
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import yaml

# Set up DASHSCOPE_API_KEY environment variable BEFORE any dashscope modules are imported
# This is critical because dashscope SDK reads environment variables at module import time
# and caches them in module-level variables
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_api_key"

# vcrpy's aiohttp stub still references a mixin removed by newer aiohttp
# releases. AgentScope tests use VCR for replay and should not fail during
# pytest marker setup just because aiohttp is importable in the environment.
try:
    import aiohttp.streams  # type: ignore[import-not-found]

    if not hasattr(aiohttp.streams, "AsyncStreamReaderMixin"):
        aiohttp.streams.AsyncStreamReaderMixin = object
except ImportError:
    pass

try:
    import aiohttp  # type: ignore[import-not-found]
    import vcr.stubs.aiohttp_stubs as aiohttp_stubs

    if (
        "stream_writer"
        in inspect.signature(aiohttp.ClientResponse.__init__).parameters
    ):

        class _CompatStreamWriter:
            output_size = 0

        class _CompatMockClientResponse(aiohttp_stubs.MockClientResponse):
            def __init__(self, method, url, request_info=None):
                aiohttp.ClientResponse.__init__(
                    self,
                    method=method,
                    url=url,
                    writer=None,
                    continue100=None,
                    timer=None,
                    request_info=request_info,
                    traces=None,
                    loop=asyncio.get_event_loop(),
                    session=None,
                    stream_writer=_CompatStreamWriter(),
                )

        aiohttp_stubs.MockClientResponse = _CompatMockClientResponse
except ImportError:
    pass

from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)

_V2_TEST_FILE = "test_v2_instrumentation.py"


def _agentscope_major() -> int:
    try:
        installed_version = version("agentscope")
    except PackageNotFoundError:
        return 1
    try:
        return int(installed_version.split(".", 1)[0])
    except ValueError:
        return 1


def pytest_configure(config: pytest.Config):
    # Configure pytest-asyncio to auto-detect async test functions
    config.option.asyncio_mode = "auto"

    # Set necessary environment variables
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

    # Set GenAI semantic conventions to latest experimental version
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if api_key is None:
        pytest.exit(
            "Environment variable 'DASHSCOPE_API_KEY' is not set. Aborting tests."
        )
    else:
        # Save environment variable to global config for use in subsequent tests
        config.option.api_key = api_key


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    path = Path(str(collection_path))
    if not path.name.startswith("test_") or path.suffix != ".py":
        return None

    major = _agentscope_major()
    if major >= 2:
        return path.name != _V2_TEST_FILE
    if path.name == _V2_TEST_FILE:
        return True
    return None


# ==================== Exporters and Readers ====================


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    """Create in-memory span exporter"""
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    """Create in-memory log exporter"""
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    """Create in-memory metric reader"""
    reader = InMemoryMetricReader()
    yield reader


# ==================== Providers ====================


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    """Create tracer provider"""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    """Create logger provider"""
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    """Create meter provider"""
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
    )
    return meter_provider


# ==================== Instrumentation Fixtures ====================


@pytest.fixture(scope="function")
def dashscope_model(request):
    """Create a DashScopeChatModel for testing"""
    from agentscope.model import DashScopeChatModel  # noqa: PLC0415

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
        stream=False,  # Default to non-streaming
    )
    return model


@pytest.fixture(scope="function")
def instrument(tracer_provider, logger_provider, meter_provider):
    """Instrument AgentScope with default settings"""
    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument without capturing message content"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "NO_CONTENT"}
    )

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument with capturing message content in spans only"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"}
    )

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content_and_events(
    tracer_provider, logger_provider, meter_provider
):
    """Instrument with capturing message content in both spans and events"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_AND_EVENT"}
    )

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content_unsampled(
    span_exporter, logger_provider, meter_provider
):
    """Instrument with content but unsampled spans"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"}
    )

    tracer_provider = TracerProvider(sampler=ALWAYS_OFF)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


# ==================== VCR Configuration ====================


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording and replaying HTTP requests"""
    return {
        "filter_headers": [
            ("authorization", "<redacted>"),
            ("api-key", "<redacted>"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
    }


class LiteralBlockScalar(str):
    """Format string as literal block scalar, preserving whitespace and not interpreting escape characters"""


def literal_block_scalar_presenter(dumper, data):
    """Represent scalar string as literal block using '|' syntax"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    """Format JSON or return long string as LiteralBlockScalar"""
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    """Search for body strings in data and attempt to format JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle response body case (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])

            # Handle request body case (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)

            else:
                convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = convert_body_to_literal(choice)

    return data


class PrettyPrintJSONBody:
    """Make request and response body recordings more readable"""

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
    """Register VCR serializer"""
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def scrub_response_headers(response):
    """
    Scrub sensitive response headers. Note they are case-sensitive!
    """
    # Clean response headers as needed
    if "Set-Cookie" in response["headers"]:
        response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response
