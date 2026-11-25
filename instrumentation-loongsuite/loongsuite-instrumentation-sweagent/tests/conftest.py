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

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from opentelemetry.instrumentation.sweagent import SweagentInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def _ensure_sweagent_packaged_layout() -> None:
    """Editable/git installs ship ``config``/``tools``/``trajectories``; pip git often does not.

    ``sweagent`` asserts these directories exist at import time.
    """
    if os.environ.get("SWE_AGENT_CONFIG_DIR"):
        return
    root = Path(tempfile.mkdtemp(prefix="sweagent-instr-test-"))
    for name in ("config", "tools", "trajectories"):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ["SWE_AGENT_CONFIG_DIR"] = str(root / "config")
    os.environ["SWE_AGENT_TOOLS_DIR"] = str(root / "tools")
    os.environ["SWE_AGENT_TRAJECTORY_DIR"] = str(root / "trajectories")


def pytest_configure(config):
    del config
    _ensure_sweagent_packaged_layout()
    os.environ.setdefault(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
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
def instrumented_sweagent(tracer_provider):
    inst = SweagentInstrumentor()
    inst.instrument(tracer_provider=tracer_provider)
    yield inst
    inst.uninstrument()
