import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

from opentelemetry.instrumentation.strands import StrandsInstrumentor
from opentelemetry.instrumentation.strands._hooks import LoongsuiteHook
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


@dataclass
class BeforeInvocationEvent:
    agent: Any = None
    invocation_state: Any = None
    messages: list = field(default_factory=list)


@dataclass
class AfterInvocationEvent:
    agent: Any = None
    invocation_state: Any = None
    result: Any = None


@dataclass
class BeforeModelCallEvent:
    agent: Any = None
    invocation_state: Any = None


@dataclass
class AfterModelCallEvent:
    invocation_state: Any = None
    stop_response: Any = None
    exception: Any = None


@dataclass
class BeforeToolCallEvent:
    tool_use: Any = None
    selected_tool: Any = None
    invocation_state: Any = None


@dataclass
class AfterToolCallEvent:
    result: Any = None
    tool_use: Any = None


class FakeModel:
    def __init__(self, model_id):
        self.model_id = model_id


class FakeAgent:
    def __init__(self, name, model_id):
        self.name = name
        self.agent_id = f"{name}-id"
        self.model = FakeModel(model_id)


class FakeInvocationState:
    def __init__(self, messages=None):
        self.messages = messages or []


class FakeStopResponse:
    def __init__(self, content, stop_reason="end_turn", input_tokens=None, output_tokens=None):
        self.message = MagicMock()
        self.message.content = content
        self.stop_reason = stop_reason
        self.usage = MagicMock()
        self.usage.input_tokens = input_tokens
        self.usage.output_tokens = output_tokens


class FakeToolUse:
    def __init__(self, name, tool_use_id, input_data=None):
        self.name = name
        self.tool_use_id = tool_use_id
        self.input = input_data


@pytest.fixture
def span_exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture
def metric_reader():
    return MeterProvider(metric_readers=[InMemoryMetricReader()])


@pytest.fixture
def instrumented(tracer_provider, metric_reader):
    instrumentor = StrandsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=metric_reader,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture
def hook(tracer_provider):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    return LoongsuiteHook(handler)


class TestStrandsInstrumentor:
    def test_instrument_uninstrument(self, instrumented):
        assert instrumented is not None

    def test_instrumentation_dependencies(self):
        instrumentor = StrandsInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "strands-agents >= 0.1.0" in deps


class TestHookSpans:
    def test_agent_invocation_creates_span(self, hook, span_exporter):
        agent = FakeAgent("test_agent", "anthropic.claude-3")
        state = FakeInvocationState([{"role": "user", "content": "Hello"}])

        hook(BeforeInvocationEvent(agent=agent, invocation_state=state))
        hook(AfterInvocationEvent(
            agent=agent,
            invocation_state=state,
            result={"content": "Hi there!", "stop_reason": "end_turn"},
        ))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1
        agent_span = agent_spans[0]
        assert "test_agent" in agent_span.name
        assert agent_span.attributes.get("gen_ai.agent.name") == "test_agent"

    def test_model_call_creates_chat_span(self, hook, span_exporter):
        agent = FakeAgent("test_agent", "anthropic.claude-3")
        state = FakeInvocationState([{"role": "user", "content": "What is 2+2?"}])

        hook(BeforeModelCallEvent(agent=agent, invocation_state=state))
        hook(AfterModelCallEvent(
            invocation_state=state,
            stop_response=FakeStopResponse("4", input_tokens=10, output_tokens=5),
            exception=None,
        ))

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if "chat" in s.name]
        assert len(chat_spans) == 1
        chat_span = chat_spans[0]
        assert "anthropic.claude-3" in chat_span.name
        assert chat_span.attributes.get("gen_ai.request.model") == "anthropic.claude-3"
        assert chat_span.attributes.get("gen_ai.usage.input_tokens") == 10
        assert chat_span.attributes.get("gen_ai.usage.output_tokens") == 5

    def test_tool_call_creates_span(self, hook, span_exporter):
        tool_use = FakeToolUse("calculator", "tool-001", {"expression": "2+2"})

        hook(BeforeToolCallEvent(tool_use=tool_use, selected_tool=MagicMock(name="calculator")))
        hook(AfterToolCallEvent(result={"output": "4"}))

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        tool_span = tool_spans[0]
        assert "calculator" in tool_span.name
        assert tool_span.attributes.get("gen_ai.tool.name") == "calculator"
        assert tool_span.attributes.get("gen_ai.tool.call.id") == "tool-001"

    def test_model_call_error_creates_failed_span(self, hook, span_exporter):
        agent = FakeAgent("test_agent", "test-model")
        state = FakeInvocationState([])

        hook(BeforeModelCallEvent(agent=agent, invocation_state=state))
        hook(AfterModelCallEvent(
            invocation_state=state,
            exception=RuntimeError("API timeout"),
            stop_response=None,
        ))

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if "chat" in s.name]
        assert len(chat_spans) == 1
        from opentelemetry.trace import StatusCode

        assert chat_spans[0].status.status_code == StatusCode.ERROR

    def test_react_step_span_created(self, hook, span_exporter):
        agent = FakeAgent("test_agent", "test-model")
        state = FakeInvocationState([])

        hook(BeforeModelCallEvent(agent=agent, invocation_state=state))
        hook(AfterModelCallEvent(
            invocation_state=state,
            stop_response=FakeStopResponse("done", stop_reason="end_turn"),
            exception=None,
        ))

        spans = span_exporter.get_finished_spans()
        react_spans = [s for s in spans if "react" in s.name]
        assert len(react_spans) == 1
        assert react_spans[0].attributes.get("gen_ai.react.round") == 1

    def test_full_agent_lifecycle_span_hierarchy(self, hook, span_exporter):
        agent = FakeAgent("my_agent", "gpt-4")
        state = FakeInvocationState([{"role": "user", "content": "Calculate 2+2"}])

        hook(BeforeInvocationEvent(agent=agent, invocation_state=state))
        hook(BeforeModelCallEvent(agent=agent, invocation_state=state))
        hook(AfterModelCallEvent(
            invocation_state=state,
            stop_response=FakeStopResponse("Let me calculate", stop_reason="tool_use", input_tokens=20, output_tokens=10),
            exception=None,
        ))

        tool_use = FakeToolUse("calculator", "tool-002", {"expr": "2+2"})
        hook(BeforeToolCallEvent(tool_use=tool_use))
        hook(AfterToolCallEvent(result="4"))

        hook(BeforeModelCallEvent(agent=agent, invocation_state=state))
        hook(AfterModelCallEvent(
            invocation_state=state,
            stop_response=FakeStopResponse("The answer is 4", stop_reason="end_turn", input_tokens=30, output_tokens=8),
            exception=None,
        ))
        hook(AfterInvocationEvent(agent=agent, invocation_state=state, result={"content": "The answer is 4"}))

        spans = span_exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        assert any("invoke_agent" in n for n in span_names)
        assert sum(1 for n in span_names if "chat" in n) == 2
        assert sum(1 for n in span_names if "execute_tool" in n) == 1
        assert sum(1 for n in span_names if "react" in n) == 2
