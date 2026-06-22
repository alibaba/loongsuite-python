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

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider, export
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Provide a fresh in-memory span exporter on a new TracerProvider.

    ``trace_api.set_tracer_provider`` only accepts the first provider in a
    process, so subsequent tests would silently write to the original
    provider's exporter. To avoid that cross-test pollution we attach the
    new exporter as a *new* span processor on the currently-active provider
    and remember the old processors so we can restore them on teardown.
    """
    exporter = InMemorySpanExporter()
    processor = export.SimpleSpanProcessor(exporter)
    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # First-test bootstrap: nothing active yet, install our own.
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    # Snapshot existing processors so we can remove them for the duration
    # of this test (their exporters would otherwise also receive spans).
    original_processors = list(provider._active_span_processor._span_processors)  # type: ignore[attr-defined]
    for p in original_processors:
        try:
            provider._active_span_processor._span_processors.remove(p)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    provider.add_span_processor(processor)
    try:
        yield exporter
    finally:
        try:
            provider._active_span_processor._span_processors.remove(processor)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        for p in original_processors:
            try:
                provider._active_span_processor._span_processors.append(p)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        try:
            processor.shutdown()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture(autouse=True)
def clear_capture_env(monkeypatch):
    """Default: content capture OFF. Tests that need it set the env var
    explicitly via monkeypatch."""
    for key in (
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "OTEL_SEMCONV_STABILITY_OPT_IN",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def enable_capture(monkeypatch):
    # util-genai gates content capture behind experimental mode AND the
    # capture-message-content flag. Both must be set, or _get_tool_call_data_attributes
    # short-circuits to an empty dict.
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    # Re-initialize the stability opt-in (cached at import time) so the
    # new env var takes effect for the current process.
    try:
        from opentelemetry.util.genai.utils import (  # noqa: PLC0415
            _OpenTelemetrySemanticConventionStability,
        )

        _OpenTelemetrySemanticConventionStability._initialized = False
        _OpenTelemetrySemanticConventionStability._initialize()
    except Exception:  # noqa: BLE001
        pass
    yield
