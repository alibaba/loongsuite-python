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
"""
Shared test fixtures for LiteLLM instrumentation tests (pytest + VCR).
Provides OTel exporters, VCR configuration, and convenient instrumentor fixtures.
"""

from __future__ import annotations

import json
import os
import re

import litellm
import pytest
import yaml

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    return meter_provider


@pytest.fixture(autouse=True)
def environment():
    """Set up OTel specific environment variables and LiteLLM settings."""
    # Enable GenAI experimental semantic conventions
    os.environ.setdefault(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    # Allow capturing message content
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "True"
    )

    litellm.telemetry = False
    litellm.add_callback = []  # clear any default callbacks

    yield


@pytest.fixture(scope="function")
def instrumentor(tracer_provider, meter_provider):
    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    yield instrumentor
    instrumentor.uninstrument()


def scrub_request(request):
    """Clean up sensitive information from request before recording it."""
    # 1. Mask Authorization header
    if "authorization" in request.headers:
        # Standard Bearer token masking
        request.headers["authorization"] = "Bearer <masked>"

    # Mask common API Key headers
    for key in ["api-key", "x-api-key", "auth-token", "cookie"]:
        if key in request.headers:
            request.headers[key] = "<masked>"

    # 2. Mask query parameters in URI
    # Supports ?api_key=xxx or &api_key=xxx
    request.uri = re.sub(r"([?&]api_key=)[^&]+", r"\1<masked>", request.uri)
    request.uri = re.sub(r"([?&]api-key=)[^&]+", r"\1<masked>", request.uri)

    # 3. Mask sensitive info in body (JSON)
    if request.body:
        try:
            body_str = request.body.decode("utf-8")
            # Mask JSON values for key keys
            for key in ["api_key", "api-key", "token", "password", "secret"]:
                # Matches "key": "value"
                pattern = rf'"{key}"\s*:\s*"[^"]+"'
                replacement = f'"{key}": "<masked>"'
                body_str = re.sub(
                    pattern, replacement, body_str, flags=re.IGNORECASE
                )
            request.body = body_str.encode("utf-8")
        except Exception:
            pass

    return request


def scrub_response(response):
    """Clean up common sensitive response headers before recording."""
    headers = response.get("headers", {})
    # Mask common sensitive headers
    for secret_header in [
        "Set-Cookie",
        "x-request-id",
        "x-amz-request-id",
        "x-goog-hash",
        "WWW-Authenticate",
    ]:
        if secret_header in headers:
            headers[secret_header] = ["<masked>"]

    # Mask OpenAI/Provider specific headers that might leak info
    for key in list(headers.keys()):
        if any(
            prefix in key.lower()
            for prefix in ["x-openai-", "x-ratelimit-", "cf-"]
        ):
            headers[key] = ["<masked>"]

    return response


@pytest.fixture(scope="module")
def vcr_config(request):
    """VCR configuration to mask API keys and sensitive information."""
    record_mode = "none"
    try:
        record_mode = request.config.getoption("--record-mode") or "none"
    except Exception:
        pass

    def _flatten_path(path: str) -> str:
        base = os.path.basename(path)
        if not base.endswith(".yaml"):
            base = f"{base}.yaml"
        return base

    return {
        "cassette_library_dir": os.path.join(
            os.path.dirname(__file__), "cassettes"
        ),
        "path_transformer": _flatten_path,
        "decode_compressed_response": True,
        # IMPORTANT: Use these hooks to mask data WITHOUT affecting live requests
        "before_record_request": scrub_request,
        "before_record_response": scrub_response,
        "record_mode": record_mode,
        # Ignore telemetry/analytics to avoid slow recordings or playback failures
        "ignore_hosts": [
            "us.i.posthog.com",
            "app.posthog.com",
            "o1127038.ingest.sentry.io",
            "api.litellm.ai",
        ],
    }


class LiteralBlockScalar(str):
    """YAML literal block to keep long bodies readable."""


def literal_block_scalar_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def _process_string_value(string_value):
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(
            json.dumps(json_data, indent=2, ensure_ascii=False)
        )
    except (ValueError, TypeError):
        if isinstance(string_value, str) and len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def _convert_body_to_literal(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = _process_string_value(value["string"])
            elif key == "body" and isinstance(value, str):
                data[key] = _process_string_value(value)
            else:
                _convert_body_to_literal(value)
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            data[idx] = _convert_body_to_literal(v)
    return data


class PrettyPrintJSONBody:
    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = _convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(scope="function")
def fixture_vcr(vcr):
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr
