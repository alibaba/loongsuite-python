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

"""Shared test fixtures for Cognee instrumentation tests.

Tests do NOT require cognee to be installed — when a test needs the cognee
module path it registers a fake module in ``sys.modules`` before the
instrumentor runs.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

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


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="event_logger_provider")
def fixture_event_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    return meter_provider


@pytest.fixture(autouse=True)
def environment():
    os.environ.setdefault(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    yield


@pytest.fixture
def capture_content_enabled(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
    )
    yield


@pytest.fixture
def capture_content_disabled(monkeypatch):
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    monkeypatch.delenv("COGNEE_CAPTURE_MESSAGE_CONTENT", raising=False)
    yield


def _ensure_fake_module(dotted_path: str, attrs: dict) -> types.ModuleType:
    """Ensure ``dotted_path`` exists in ``sys.modules`` as a real (fake) module.

    Creates parent packages on the fly. ``attrs`` becomes the module's public
    attributes. Existing modules are left intact if they already exist.
    """
    parts = dotted_path.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub in sys.modules:
            continue
        mod = types.ModuleType(sub)
        sys.modules[sub] = mod
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            setattr(parent, parts[i - 1], mod)
    mod = sys.modules[dotted_path]
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def fake_cognee_v1_api():
    """Install fake ``cognee.api.v1.{add,cognify,search,recall,remember}`` modules."""
    async def add(data, *args, **kwargs):
        return {"added": data}

    async def cognify(datasets=None, *args, **kwargs):
        return {"cognified": datasets}

    async def search(query_text, *args, **kwargs):
        return [{"content": f"answer to {query_text}"}]

    async def recall(query_text, *args, **kwargs):
        return [{"recall": query_text}]

    async def remember(data, *args, **kwargs):
        return {"remembered": data}

    _ensure_fake_module("cognee", {})
    _ensure_fake_module("cognee.api", {})
    _ensure_fake_module("cognee.api.v1", {})
    _ensure_fake_module("cognee.api.v1.add", {"add": add})
    _ensure_fake_module("cognee.api.v1.cognify", {"cognify": cognify})
    _ensure_fake_module("cognee.api.v1.search", {"search": search})
    _ensure_fake_module("cognee.api.v1.recall", {"recall": recall})
    _ensure_fake_module("cognee.api.v1.remember", {"remember": remember})
    yield


@pytest.fixture
def fake_cognee_tool():
    async def execute_tool(user, dataset_id, tool_name, args=None, allowed_tools=None):
        return {"tool": tool_name, "args": args}

    _ensure_fake_module("cognee", {})
    _ensure_fake_module("cognee.modules", {})
    _ensure_fake_module("cognee.modules.tools", {})
    _ensure_fake_module(
        "cognee.modules.tools.execute_tool", {"execute_tool": execute_tool}
    )
    yield


@pytest.fixture
def fake_cognee_completion():
    async def generate_completion(
        query,
        context,
        user_prompt_path,
        system_prompt_path,
        system_prompt=None,
        conversation_history=None,
        response_model=str,
    ):
        return {"q": query, "p": user_prompt_path}

    _ensure_fake_module("cognee", {})
    _ensure_fake_module("cognee.modules", {})
    _ensure_fake_module("cognee.modules.retrieval", {})
    _ensure_fake_module("cognee.modules.retrieval.utils", {})
    _ensure_fake_module(
        "cognee.modules.retrieval.utils.completion",
        {"generate_completion": generate_completion},
    )
    yield
