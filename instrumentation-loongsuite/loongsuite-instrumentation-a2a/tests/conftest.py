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

"""Test configuration for A2A instrumentation tests.

Tests exercise the instrumentation against the *real* ``a2a-sdk``
``AgentExecutor`` ABC / ``RequestContext`` and real in-process executor
subclasses, asserting on OTel spans exported to an
``InMemorySpanExporter`` (and gen-ai details events exported to an
in-memory log exporter). No network access is required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


from opentelemetry.instrumentation.a2a import A2AInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)


def _reset_handler_singleton() -> None:
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogRecordExporter()
    yield exporter
    exporter.clear()


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


@pytest.fixture(scope="function", name="handler")
def fixture_handler(tracer_provider, logger_provider):
    """The shared util singleton, re-bound to the test providers."""
    _reset_handler_singleton()
    handler = get_extended_telemetry_handler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )
    yield handler
    _reset_handler_singleton()


@pytest.fixture(scope="function")
def instrument(tracer_provider, logger_provider, handler):
    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
    _reset_handler_singleton()


class _RaisingSpan:
    """Delegating SDK span whose attribute/error recording explodes.

    The instance forwards every attribute access to the real SDK span
    except for the recording primitives under fault injection, so a
    test can prove telemetry failures cannot change business behavior.
    """

    def __init__(self, real, *, setattr_raises, error_raises):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_setattr_raises", setattr_raises)
        object.__setattr__(self, "_error_raises", error_raises)

    def set_attribute(self, key, value):
        if object.__getattribute__(self, "_setattr_raises"):
            raise RuntimeError("span.set_attribute exploded")
        return self._real.set_attribute(key, value)

    def set_attributes(self, attributes):
        if object.__getattribute__(self, "_setattr_raises"):
            raise RuntimeError("span.set_attributes exploded")
        return self._real.set_attributes(attributes)

    def record_exception(self, *args, **kwargs):
        if object.__getattribute__(self, "_error_raises"):
            raise RuntimeError("span.record_exception exploded")
        return self._real.record_exception(*args, **kwargs)

    def set_status(self, status):
        if object.__getattribute__(self, "_error_raises"):
            raise RuntimeError("span.set_status exploded")
        return self._real.set_status(status)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


class _RaisingTracer:
    def __init__(self, real, *, setattr_raises=False, error_raises=False):
        self._real = real
        self._setattr_raises = setattr_raises
        self._error_raises = error_raises

    def start_span(self, *args, **kwargs):
        real = self._real.start_span(*args, **kwargs)
        return _RaisingSpan(
            real,
            setattr_raises=self._setattr_raises,
            error_raises=self._error_raises,
        )

    def start_as_current_span(self, *args, **kwargs):
        return self._real.start_as_current_span(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _RaisingTracerProvider(TracerProvider):
    def __init__(self, exporter, *, setattr_raises, error_raises):
        super().__init__()
        self.add_span_processor(SimpleSpanProcessor(exporter))
        self._setattr_raises = setattr_raises
        self._error_raises = error_raises

    def get_tracer(self, *args, **kwargs):
        real = super().get_tracer(*args, **kwargs)
        return _RaisingTracer(
            real,
            setattr_raises=self._setattr_raises,
            error_raises=self._error_raises,
        )


@pytest.fixture(scope="function")
def raising_tracer_provider(span_exporter):
    def _make(*, setattr_raises=False, error_raises=False):
        return _RaisingTracerProvider(
            span_exporter,
            setattr_raises=setattr_raises,
            error_raises=error_raises,
        )

    return _make
