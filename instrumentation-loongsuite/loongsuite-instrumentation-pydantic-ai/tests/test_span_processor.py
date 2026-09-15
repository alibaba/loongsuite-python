import asyncio

from opentelemetry.trace import StatusCode

from opentelemetry.instrumentation.pydantic_ai.capability import (
    LoongSuiteInstrumentationCapability,
)
from opentelemetry.instrumentation.pydantic_ai.span_processor import (
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    LoongSuiteSpanProcessor,
    normalize_genai_attributes,
)


def test_normalize_genai_attributes_adds_framework_and_span_kind():
    attrs = normalize_genai_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.usage.input_tokens": 3,
            "gen_ai.usage.output_tokens": 5,
        }
    )

    assert attrs[GEN_AI_FRAMEWORK] == "pydantic-ai"
    assert attrs[GEN_AI_SPAN_KIND] == "LLM"
    assert attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 8


def test_normalize_genai_attributes_preserves_existing_values():
    attrs = normalize_genai_attributes(
        {
            "gen_ai.operation.name": "execute_tool",
            GEN_AI_FRAMEWORK: "custom",
            GEN_AI_SPAN_KIND: "RETRIEVER",
            GEN_AI_USAGE_TOTAL_TOKENS: 10,
            "gen_ai.usage.input_tokens": 1,
            "gen_ai.usage.output_tokens": 2,
        }
    )

    assert attrs[GEN_AI_FRAMEWORK] == "custom"
    assert attrs[GEN_AI_SPAN_KIND] == "RETRIEVER"
    assert attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 10


def test_on_end_records_genai_and_arms_metrics():
    meter = FakeMeterProvider().meter
    processor = LoongSuiteSpanProcessor(
        meter_provider=FakeMeterProvider(meter),
        slow_call_threshold_seconds=0.5,
    )
    span = FakeReadableSpan(
        {
            "gen_ai.operation.name": "execute_tool",
            GEN_AI_FRAMEWORK: "pydantic-ai",
        },
        duration_seconds=1.25,
        status_code=StatusCode.ERROR,
    )

    processor.on_end(span)

    assert span.attributes[GEN_AI_SPAN_KIND] == "TOOL"
    metric_attributes = {"modelName": "unknown", "spanKind": "TOOL"}
    assert meter.instruments["genai_calls_count"].measurements == [
        (1, metric_attributes)
    ]
    assert meter.instruments["genai_calls_duration_seconds"].measurements == [
        (1.25, metric_attributes)
    ]
    assert meter.instruments["genai_calls_error_count"].measurements == [
        (1, metric_attributes)
    ]
    assert meter.instruments["genai_calls_slow_count"].measurements == [
        (1, metric_attributes)
    ]
    assert meter.instruments["arms_tool_requests_count"].measurements == [(1, None)]
    assert meter.instruments["arms_tool_requests_seconds"].measurements == [
        (1.25, None)
    ]
    assert meter.instruments["arms_tool_requests_error_count"].measurements == [
        (1, None)
    ]
    assert meter.instruments["arms_tool_requests_slow_count"].measurements == [
        (1, None)
    ]


def test_provider_llm_span_suppresses_framework_llm_metrics():
    meter = FakeMeterProvider().meter
    processor = LoongSuiteSpanProcessor(meter_provider=FakeMeterProvider(meter))
    trace_id = 123

    processor.on_end(
        FakeReadableSpan(
            {"gen_ai.operation.name": "chat"},
            scope_name="opentelemetry.instrumentation.openai_v2",
            trace_id=trace_id,
        )
    )
    processor.on_end(
        FakeReadableSpan(
            {
                "gen_ai.operation.name": "chat",
                GEN_AI_FRAMEWORK: "pydantic-ai",
            },
            scope_name="pydantic-ai",
            trace_id=trace_id,
        )
    )

    assert meter.instruments["genai_calls_count"].measurements == []
    assert "arms_chat_requests_count" not in meter.instruments


def test_wrap_node_run_creates_step_span(monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr(
        "opentelemetry.instrumentation.pydantic_ai.capability.trace.get_tracer",
        lambda _name: tracer,
    )
    capability = LoongSuiteInstrumentationCapability()
    node = type("ModelRequestNode", (), {})()
    ctx = type("Ctx", (), {"run_step": 2})()

    async def handler(received_node):
        assert received_node is node
        return type("End", (), {})()

    result = asyncio.run(
        capability.wrap_node_run(
            ctx,
            node=node,
            handler=handler,
        )
    )

    assert type(result).__name__ == "End"
    assert tracer.span.name == "react step"
    assert tracer.span.attributes["gen_ai.operation.name"] == "react"
    assert tracer.span.attributes[GEN_AI_SPAN_KIND] == "STEP"
    assert tracer.span.attributes["gen_ai.react.round"] == 2
    assert tracer.span.attributes["gen_ai.react.finish_reason"] == "stop"


class FakeMeterProvider:
    def __init__(self, meter=None):
        self.meter = meter or FakeMeter()

    def get_meter(self, _name):
        return self.meter


class FakeMeter:
    def __init__(self):
        self.instruments = {}

    def create_counter(self, name, unit=None):
        return self._instrument(name)

    def create_histogram(self, name, unit=None):
        return self._instrument(name)

    def _instrument(self, name):
        instrument = FakeInstrument()
        self.instruments[name] = instrument
        return instrument


class FakeInstrument:
    def __init__(self):
        self.measurements = []

    def add(self, value, attributes=None):
        self.measurements.append((value, attributes))

    def record(self, value, attributes=None):
        self.measurements.append((value, attributes))


class FakeReadableSpan:
    def __init__(
        self,
        attributes,
        *,
        duration_seconds=0.25,
        status_code=StatusCode.UNSET,
        scope_name="pydantic-ai",
        trace_id=1,
    ):
        self._attributes = dict(attributes)
        self.start_time = 0
        self.end_time = int(duration_seconds * 1_000_000_000)
        self.status = type("Status", (), {"status_code": status_code})()
        self.instrumentation_scope = type("Scope", (), {"name": scope_name})()
        self.context = type("Context", (), {"trace_id": trace_id})()

    @property
    def attributes(self):
        return self._attributes


class FakeSpan:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = dict(attributes)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc, escaped=False):
        self.exception = exc

    def set_status(self, status):
        self.status = status


class FakeSpanContextManager:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeTracer:
    def start_as_current_span(
        self,
        name,
        attributes=None,
        record_exception=None,
        set_status_on_exception=None,
    ):
        self.span = FakeSpan(name, attributes or {})
        return FakeSpanContextManager(self.span)
