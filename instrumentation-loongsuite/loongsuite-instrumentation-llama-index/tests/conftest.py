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

"""Test configuration for LlamaIndex instrumentation tests.

These tests exercise the instrumentation against the *real*
``llama-index-core`` dispatcher and real in-process LLM/embedding stubs
(``MockLLM`` / ``MockEmbedding``), asserting on the OTel spans exported to an
``InMemorySpanExporter`` (and events to an ``InMemoryLogExporter``). No
network access or provider credentials are required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure workspace src is importable when running from the package dir.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_configure(config: pytest.Config):
    # The shared GenAI util only honors content capture / event switches in
    # experimental semconv mode, exactly like the agno/hermes test suites.
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


from opentelemetry._logs import set_logger_provider  # noqa: E402
from opentelemetry.instrumentation.llama_index import (  # noqa: E402
    LlamaIndexInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider  # noqa: E402
from opentelemetry.sdk._logs.export import (  # noqa: E402
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    return InMemoryMetricReader()


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    return MeterProvider(metric_readers=[metric_reader])


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    set_logger_provider(provider)
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider, meter_provider, logger_provider):
    """Instrument LlamaIndex, yield the instrumentor, then uninstrument.

    Uninstrument matters for the RED tests: it must leave the dispatcher with
    no LoongSuite handler so a subsequent call produces zero spans.
    """
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
