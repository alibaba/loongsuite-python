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

"""Tests for OTelTracingProcessor spanning all supported SDK span types."""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
)

from opentelemetry.instrumentation.openai_agents._processor import (
    OTelTracingProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)

# -----------------------------------------------------------
# Helpers: lightweight fakes for the SDK's Trace / Span ABCs
# -----------------------------------------------------------


class FakeTrace:
    def __init__(self, trace_id="trace_001", name="Test Workflow"):
        self.trace_id = trace_id
        self.name = name


class FakeSpan:
    def __init__(
        self,
        span_data,
        span_id="span_001",
        trace_id="trace_001",
        parent_id=None,
        error=None,
    ):
        self._span_data = span_data
        self._span_id = span_id
        self._trace_id = trace_id
        self._parent_id = parent_id
        self._error = error

    @property
    def span_data(self):
        return self._span_data

    @property
    def span_id(self):
        return self._span_id

    @property
    def trace_id(self):
        return self._trace_id

    @property
    def parent_id(self):
        return self._parent_id

    @property
    def error(self):
        return self._error


# -----------------------------------------------------------
# Fixtures
# -----------------------------------------------------------


@pytest.fixture()
def setup():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = ExtendedTelemetryHandler(tracer_provider=provider)
    processor = OTelTracingProcessor(handler=handler, capture_content=True)
    return processor, exporter


# -----------------------------------------------------------
# Tests
# -----------------------------------------------------------


class TestTraceLifecycle:
    def test_trace_creates_workflow_span(self, setup):
        processor, exporter = setup
        trace = FakeTrace(name="My Workflow")

        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "invoke_workflow My Workflow"
        attrs = dict(span.attributes)
        assert attrs["gen_ai.operation.name"] == "invoke_workflow"
        assert attrs["gen_ai.system"] == "openai_agents"


class TestAgentSpan:
    def test_agent_span_attributes(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        agent_data = AgentSpanData(
            name="weather_agent",
            tools=["get_weather", "get_forecast"],
            handoffs=["travel_agent"],
            output_type="str",
        )
        sdk_span = FakeSpan(agent_data, span_id="agent_001")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == "invoke_agent weather_agent"
        ]
        assert len(agent_spans) == 1
        attrs = dict(agent_spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "invoke_agent"
        assert attrs["gen_ai.agent.name"] == "weather_agent"
        assert attrs["gen_ai.output.type"] == "str"
        assert list(attrs["gen_ai.openai.agents.agent.tools"]) == [
            "get_weather",
            "get_forecast",
        ]
        assert list(attrs["gen_ai.openai.agents.agent.handoffs"]) == [
            "travel_agent"
        ]


class TestGenerationSpan:
    def test_generation_span_with_usage(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        gen_data = GenerationSpanData(
            model="gpt-4o",
            model_config={"temperature": 0.7, "max_tokens": 1024},
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
            },
            input=[{"role": "user", "content": "Hello"}],
            output=[{"role": "assistant", "content": "Hi there!"}],
        )
        sdk_span = FakeSpan(gen_data, span_id="gen_001")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        gen_spans = [s for s in spans if s.name == "chat gpt-4o"]
        assert len(gen_spans) == 1
        attrs = dict(gen_spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["gen_ai.response.model"] == "gpt-4o"
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.request.max_tokens"] == 1024
        assert "Hello" in attrs["gen_ai.input.messages"]
        assert "Hi there!" in attrs["gen_ai.output.messages"]

    def test_generation_span_no_content(self, setup):
        processor, exporter = setup
        processor._capture_content = False
        trace = FakeTrace()
        processor.on_trace_start(trace)

        gen_data = GenerationSpanData(
            model="gpt-4o",
            input=[{"role": "user", "content": "secret"}],
            output=[{"role": "assistant", "content": "classified"}],
        )
        sdk_span = FakeSpan(gen_data, span_id="gen_002")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        gen_spans = [s for s in spans if s.name == "chat gpt-4o"]
        attrs = dict(gen_spans[0].attributes)
        assert "gen_ai.input.messages" not in attrs
        assert "gen_ai.output.messages" not in attrs


class TestFunctionSpan:
    def test_tool_execution_span(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        func_data = FunctionSpanData(
            name="get_weather",
            input='{"city": "Tokyo"}',
            output='{"temp": 22}',
        )
        sdk_span = FakeSpan(func_data, span_id="func_001")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == "execute_tool get_weather"]
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.tool.name"] == "get_weather"
        assert attrs["gen_ai.tool.type"] == "function"
        assert "Tokyo" in attrs["gen_ai.tool.call.arguments"]
        assert "22" in attrs["gen_ai.tool.call.result"]


class TestHandoffSpan:
    def test_handoff_attributes(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        handoff_data = HandoffSpanData(
            from_agent="triage", to_agent="specialist"
        )
        sdk_span = FakeSpan(handoff_data, span_id="handoff_001")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        handoff_spans = [s for s in spans if s.name == "triage -> specialist"]
        assert len(handoff_spans) == 1
        attrs = dict(handoff_spans[0].attributes)
        assert attrs["gen_ai.openai.agents.handoff.from_agent"] == "triage"
        assert attrs["gen_ai.openai.agents.handoff.to_agent"] == "specialist"


class TestGuardrailSpan:
    def test_guardrail_triggered(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        guard_data = GuardrailSpanData(name="content_filter", triggered=True)
        sdk_span = FakeSpan(guard_data, span_id="guard_001")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        guard_spans = [
            s for s in spans if s.name == "guardrail content_filter"
        ]
        assert len(guard_spans) == 1
        attrs = dict(guard_spans[0].attributes)
        assert attrs["gen_ai.openai.agents.guardrail.name"] == "content_filter"
        assert attrs["gen_ai.openai.agents.guardrail.triggered"] is True

    def test_guardrail_not_triggered(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        guard_data = GuardrailSpanData(name="safety_check", triggered=False)
        sdk_span = FakeSpan(guard_data, span_id="guard_002")

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        guard_spans = [s for s in spans if s.name == "guardrail safety_check"]
        attrs = dict(guard_spans[0].attributes)
        assert attrs["gen_ai.openai.agents.guardrail.triggered"] is False


class TestSpanHierarchy:
    def test_nested_agent_tool_spans(self, setup):
        processor, exporter = setup

        trace = FakeTrace()
        processor.on_trace_start(trace)

        agent_data = AgentSpanData(name="assistant")
        agent_span = FakeSpan(agent_data, span_id="agent_001")
        processor.on_span_start(agent_span)

        func_data = FunctionSpanData(
            name="search", input="query", output="result"
        )
        func_span = FakeSpan(
            func_data,
            span_id="func_001",
            parent_id="agent_001",
        )
        processor.on_span_start(func_span)
        processor.on_span_end(func_span)

        processor.on_span_end(agent_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        tool_span = next(s for s in spans if s.name == "execute_tool search")
        otel_agent_span = next(
            s for s in spans if s.name == "invoke_agent assistant"
        )
        assert tool_span.parent.span_id == otel_agent_span.context.span_id


class TestErrorHandling:
    def test_span_error_sets_status(self, setup):
        processor, exporter = setup
        trace = FakeTrace()
        processor.on_trace_start(trace)

        agent_data = AgentSpanData(name="failing_agent")
        sdk_span = FakeSpan(
            agent_data,
            span_id="err_001",
            error={
                "message": "Tool execution failed",
                "data": None,
            },
        )

        processor.on_span_start(sdk_span)
        processor.on_span_end(sdk_span)
        processor.on_trace_end(trace)

        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == "invoke_agent failing_agent"
        ]
        assert len(agent_spans) == 1
        assert agent_spans[0].status.is_ok is False

    def test_processor_does_not_throw(self, setup):
        processor, _ = setup
        processor.on_span_start(None)
        processor.on_span_end(None)
        processor.on_trace_start(None)
        processor.on_trace_end(None)
