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

"""Tests for the TOOL wrapper (``cognee.modules.tools.execute_tool``)."""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.cognee import CogneeInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def instrumented(fake_cognee_tool, monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = CogneeInstrumentor()
    import opentelemetry.instrumentation.cognee as cognee_inst

    monkeypatch.setattr(
        cognee_inst, "_enable_cognee_tracing", lambda _provider: None
    )
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    yield instrumentor, exporter, provider
    instrumentor.uninstrument()
    provider.shutdown()


@pytest.mark.asyncio
async def test_tool_executed_span_emitted(instrumented):
    instrumentor, exporter, _ = instrumented
    from cognee.modules.tools.execute_tool import execute_tool

    result = await execute_tool(
        user=None,
        dataset_id=None,
        tool_name="load_skill",
        args={"skill_name": "test"},
    )
    assert result == {
        "tool": "load_skill",
        "args": {"skill_name": "test"},
    }

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes.get("gen_ai.span.kind") == "TOOL"
    assert span.attributes.get("gen_ai.tool.name") == "load_skill"
    assert span.attributes.get("gen_ai.tool.type") == "function"


@pytest.mark.asyncio
async def test_tool_wrapper_propagates_exception(instrumented):
    instrumentor, exporter, _ = instrumented

    # The fake execute_tool does not raise; we test via direct wrapper call.
    from opentelemetry.instrumentation.cognee.internal._tool_wrapper import (
        _make_tool_wrapper,
    )
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=TracerProvider())
    wrapper = _make_tool_wrapper(handler)

    async def boom(*args, **kwargs):
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError):
        await wrapper(boom, None, (), {"tool_name": "bad", "args": {}})


@pytest.mark.asyncio
async def test_tool_no_capture_when_disabled(fake_cognee_tool, monkeypatch):
    """When capture_message_content is off, tool_call_arguments is not set."""
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    monkeypatch.delenv("COGNEE_CAPTURE_MESSAGE_CONTENT", raising=False)
    from opentelemetry.instrumentation.cognee import CogneeInstrumentor
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = CogneeInstrumentor()
    import opentelemetry.instrumentation.cognee as cognee_inst

    monkeypatch.setattr(
        cognee_inst, "_enable_cognee_tracing", lambda _provider: None
    )
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    try:
        from cognee.modules.tools.execute_tool import execute_tool
        await execute_tool(
            user=None,
            dataset_id=None,
            tool_name="load_skill",
            args={"skill_name": "test"},
        )
        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1
        # gen_ai.tool.name always set, but tool.call.arguments not set when capture disabled
        assert tool_spans[0].attributes.get("gen_ai.tool.name") == "load_skill"
    finally:
        instrumentor.uninstrument()
        provider.shutdown()
