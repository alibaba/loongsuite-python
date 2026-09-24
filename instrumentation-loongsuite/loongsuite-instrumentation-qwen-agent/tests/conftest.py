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

"""Test configuration for Qwen-Agent instrumentation tests."""

import json
import os

import pytest
import yaml
from vcr.stubs import VCRHTTPResponse

# Set DASHSCOPE_API_KEY before any dashscope/qwen-agent imports
# dashscope SDK reads environment variables at module import time
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_dashscope_api_key"

from opentelemetry.instrumentation.qwen_agent import QwenAgentInstrumentor
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


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# ==================== Exporters ====================


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    reader = InMemoryMetricReader()
    yield reader


# ==================== Providers ====================


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
    )
    return meter_provider


# ==================== Instrumentation ====================


@pytest.fixture(scope="function")
def instrument(tracer_provider, logger_provider, meter_provider):
    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()


# ==================== VCR Support ====================


def _patch_vcr_response():
    """Patch VCRHTTPResponse to add missing version_string attribute.

    Newer urllib3 requires version_string on HTTP responses, but VCR.py's
    VCRHTTPResponse stub does not set it, causing AttributeError when dashscope
    SDK streams SSE responses.
    """
    if not hasattr(VCRHTTPResponse, "version_string"):
        VCRHTTPResponse.version_string = "HTTP/1.1"


_patch_vcr_response()


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


def scrub_response_headers(response):
    """Scrubs sensitive response headers."""
    if "x-dashscope-request-id" in response.get("headers", {}):
        response["headers"]["x-dashscope-request-id"] = "test_request_id"
    return response


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


@pytest.fixture(scope="module", autouse=True)
def fixture_vcr(vcr):
    """Register custom VCR serializer."""
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr
