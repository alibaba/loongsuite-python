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

"""Smoke tests for the ``CogneeInstrumentor`` install/uninstall lifecycle."""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.cognee import CogneeInstrumentor
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture
def instrumented_no_cognee(monkeypatch):
    """Install instrumentor without cognee installed — wrappers should fail
    gracefully via try/except in ``install_*`` functions.
    """
    import opentelemetry.instrumentation.cognee as cognee_inst

    monkeypatch.setattr(
        cognee_inst, "_enable_cognee_tracing", lambda _provider: None
    )
    provider = TracerProvider()
    instrumentor = CogneeInstrumentor()
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    yield instrumentor
    instrumentor.uninstrument()


def test_instrument_idempotent(instrumented_no_cognee):
    """Calling instrument() twice is a no-op."""
    instrumentor = instrumented_no_cognee
    # Second call should not raise even though _is_instrumented is True.
    instrumentor.instrument()
    assert instrumentor._is_instrumented is True


def test_uninstrument_idempotent(instrumented_no_cognee):
    instrumentor = instrumented_no_cognee
    instrumentor.uninstrument()
    assert instrumentor._is_instrumented is False
    # Second call should not raise.
    instrumentor.uninstrument()


def test_instrumentation_dependencies():
    inst = CogneeInstrumentor()
    deps = inst.instrumentation_dependencies()
    assert any("cognee" in d for d in deps)
