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
"""QwenPaw instrumentation test fixtures."""

from __future__ import annotations

import importlib
import os

import pytest

from opentelemetry.instrumentation.qwenpaw import QwenPawInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(name="span_exporter")
def fixture_span_exporter():
    return InMemorySpanExporter()


@pytest.fixture(name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture
def instrument(tracer_provider):
    """Enable QwenPaw instrumentation for one test."""
    os.environ.setdefault(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    inst = QwenPawInstrumentor()
    inst.instrument(skip_dep_check=True, tracer_provider=tracer_provider)
    yield inst
    inst.uninstrument()


def _import_runner_module():
    for module_name in (
        "qwenpaw.app.runner.runner",
        "copaw.app.runner.runner",
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    pytest.skip("No supported QwenPaw runtime package is installed")


@pytest.fixture(name="runner_module")
def fixture_runner_module():
    return _import_runner_module()


@pytest.fixture(name="runtime_module")
def fixture_runtime_module():
    try:
        return importlib.import_module("qwenpaw.runtime.runtime")
    except ImportError:
        pytest.skip("QwenPaw 2 runtime is not installed")
