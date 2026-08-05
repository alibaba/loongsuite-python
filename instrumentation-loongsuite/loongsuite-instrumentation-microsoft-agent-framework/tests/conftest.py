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

"""Real Microsoft Agent Framework/VCR test configuration.

Committed cassettes are replay-only by default. The non-streaming and streaming
provider baselines can be deliberately re-recorded with a real provider key and
``--vcr-record=all``. The deterministic tool and provider-error scenarios use
the local OpenAI-compatible fixture server identified by the cassette URI;
start an equivalent server at ``127.0.0.1:18765`` before re-recording them. The
sanitizers below run before a cassette is written.
"""

from __future__ import annotations

import json

import pytest

from opentelemetry.instrumentation.microsoft_agent_framework import (
    MicrosoftAgentFrameworkInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def _scrub_request(request):
    """Retain only stable request data needed for semantic matching."""
    content_type = request.headers.get("content-type")
    request.headers.clear()
    if content_type:
        request.headers["content-type"] = content_type
    return request


def _scrub_response(response):
    """Remove provider/machine identifiers and normalize response IDs."""
    headers = response.get("headers", {})
    content_type = headers.get("content-type") or headers.get("Content-Type")
    response["headers"] = (
        {"content-type": content_type} if content_type else {}
    )

    body = response.get("body", {}).get("string")
    was_bytes = isinstance(body, bytes)
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if isinstance(body, str):
        normalized = _normalize_response_body(body)
        response["body"]["string"] = (
            normalized.encode("utf-8") if was_bytes else normalized
        )
    return response


def _normalize_response_body(body):
    """Normalize only provider response IDs, preserving tool-call IDs."""

    def normalize_json(payload):
        if isinstance(payload, dict):
            if "id" in payload:
                payload["id"] = "maf-vcr-response-id"
            if "created" in payload:
                payload["created"] = 1700000000
        return json.dumps(payload, separators=(",", ":"))

    if not body.startswith("data:"):
        try:
            return normalize_json(json.loads(body))
        except json.JSONDecodeError:
            return body

    normalized_lines = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            normalized_lines.append(line)
            continue
        try:
            line = f"data: {normalize_json(json.loads(line[6:]))}"
        except json.JSONDecodeError:
            pass
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


@pytest.fixture(scope="module")
def vcr_config():
    """Use strict, offline replay unless recording is explicitly requested."""
    return {
        "record_mode": "none",
        "filter_headers": [
            ("authorization", "Bearer test-api-key"),
            ("api-key", "test-api-key"),
            ("x-api-key", "test-api-key"),
        ],
        "decode_compressed_response": True,
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
        ],
        "before_record_request": _scrub_request,
        "before_record_response": _scrub_response,
    }


@pytest.fixture
def vcr_cassette_name(request):
    """Keep cassette names stable when a fixture is indirectly parametrized."""
    return getattr(request.node, "originalname", None) or request.node.name


@pytest.fixture
def maf_runtime(request):
    """Instrument the real installed MAF package with an isolated exporter."""
    capture_sensitive_data = getattr(request, "param", True)
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = MicrosoftAgentFrameworkInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        enable_sensitive_data=capture_sensitive_data,
        react_step_enabled=False,
        skip_dep_check=True,
    )
    yield exporter
    instrumentor.uninstrument()
    tracer_provider.shutdown()


def pytest_collection_modifyitems(items):
    """Make the replay contract easy to select in CI and locally."""
    for item in items:
        if item.get_closest_marker("vcr"):
            item.add_marker(pytest.mark.maf_vcr)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "maf_vcr: real-framework replay-only contract"
    )
