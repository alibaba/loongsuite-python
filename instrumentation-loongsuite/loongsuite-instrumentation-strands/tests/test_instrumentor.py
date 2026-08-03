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

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel
from strands import Agent, tool
from strands.models import Model
from strands.telemetry.tracer import get_tracer

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

from opentelemetry.instrumentation.strands import StrandsInstrumentor
from opentelemetry.instrumentation.strands._hooks import _provider_name


class DeterministicModel(Model):
    def __init__(self, responses: Sequence[dict[str, Any]]):
        self.responses = list(responses)
        self.index = 0
        self.config = {"model_id": "deterministic-v1"}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    async def structured_output(
        self,
        output_model: type[BaseModel],
        prompt: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        if False:
            yield None

    async def stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        message = self.responses[self.index]
        self.index += 1
        stop_reason = "end_turn"
        yield {"messageStart": {"role": "assistant"}}
        for content in message["content"]:
            if "text" in content:
                yield {"contentBlockStart": {"start": {}}}
                yield {
                    "contentBlockDelta": {"delta": {"text": content["text"]}}
                }
                yield {"contentBlockStop": {}}
            elif "toolUse" in content:
                stop_reason = "tool_use"
                tool_use = content["toolUse"]
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": tool_use["name"],
                                "toolUseId": tool_use["toolUseId"],
                            }
                        }
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {
                            "toolUse": {
                                "input": json.dumps(tool_use["input"])[:10]
                            }
                        }
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {
                            "toolUse": {
                                "input": json.dumps(tool_use["input"])[10:]
                            }
                        }
                    }
                }
                yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop_reason}}
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 6,
                    "totalTokens": 18,
                    "cacheReadInputTokens": 2,
                    "cacheWriteInputTokens": 1,
                },
                "metrics": {"latencyMs": 7, "timeToFirstByteMs": 3},
            }
        }


class FailingModel(DeterministicModel):
    def __init__(self, error: Exception):
        super().__init__([])
        self.error = error

    async def stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        if False:
            yield {}
        raise self.error


class SlowModel(DeterministicModel):
    def __init__(self):
        super().__init__([])

    async def stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        await asyncio.Event().wait()
        if False:
            yield {}


@tool
def calculator(expression: str) -> str:
    """Evaluate the bounded test expression."""
    assert expression == "2+2"
    return "4"


@pytest.fixture
def telemetry():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    native = get_tracer()
    original_native_tracer = native.tracer
    native.tracer = provider.get_tracer(native.service_name)

    instrumentor = StrandsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    yield instrumentor, exporter, original_native_tracer
    instrumentor.uninstrument()
    native.tracer = original_native_tracer


def _agent() -> Agent:
    return Agent(
        model=DeterministicModel(
            [
                {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool-1",
                                "name": "calculator",
                                "input": {"expression": "2+2"},
                            }
                        }
                    ]
                },
                {"content": [{"text": "The answer is 4."}]},
            ]
        ),
        tools=[calculator],
        name="test_agent",
        retry_strategy=None,
    )


@pytest.mark.asyncio
async def test_real_strands_lifecycle_and_hierarchy(telemetry):
    _, exporter, _ = telemetry
    result = await _agent().invoke_async("Calculate 2+2")
    assert str(result).strip() == "The answer is 4."

    spans = exporter.get_finished_spans()
    assert [span.attributes["gen_ai.span.kind"] for span in spans] == [
        "LLM",
        "TOOL",
        "STEP",
        "LLM",
        "STEP",
        "AGENT",
    ]
    by_id = {span.context.span_id: span for span in spans}
    agent = next(
        span
        for span in spans
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    )
    steps = [
        span for span in spans if span.attributes["gen_ai.span.kind"] == "STEP"
    ]
    llms = [
        span for span in spans if span.attributes["gen_ai.span.kind"] == "LLM"
    ]
    tool_span = next(
        span for span in spans if span.attributes["gen_ai.span.kind"] == "TOOL"
    )

    assert all(by_id[step.parent.span_id] is agent for step in steps)
    assert by_id[llms[0].parent.span_id] is steps[0]
    assert by_id[tool_span.parent.span_id] is steps[0]
    assert by_id[llms[1].parent.span_id] is steps[1]
    assert [step.attributes["gen_ai.react.round"] for step in steps] == [1, 2]
    assert llms[0].attributes["gen_ai.response.finish_reasons"] == (
        "tool_calls",
    )
    assert llms[1].attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert llms[0].attributes["gen_ai.usage.cache_read.input_tokens"] == 2
    assert llms[0].attributes["gen_ai.usage.cache_creation.input_tokens"] == 1
    assert llms[0].attributes[
        "gen_ai.response.time_to_first_token"
    ] == pytest.approx(3_000_000, abs=1)
    assert tool_span.attributes["gen_ai.tool.call.id"] == "tool-1"
    assert all("gen_ai.framework" not in span.attributes for span in spans)


@pytest.mark.asyncio
async def test_agent_usage_is_per_invocation_across_multiple_turns(telemetry):
    _, exporter, _ = telemetry
    agent = Agent(
        model=DeterministicModel(
            [
                {"content": [{"text": "turn one"}]},
                {"content": [{"text": "turn two"}]},
            ]
        ),
        name="multi_turn_agent",
        retry_strategy=None,
    )

    await agent.invoke_async("first")
    await agent.invoke_async("second")

    agent_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ]
    assert [
        span.attributes["gen_ai.usage.input_tokens"] for span in agent_spans
    ] == [12, 12]
    assert [
        span.attributes["gen_ai.usage.output_tokens"] for span in agent_spans
    ] == [6, 6]
    assert [
        span.attributes["gen_ai.usage.total_tokens"] for span in agent_spans
    ] == [18, 18]


@pytest.mark.asyncio
async def test_capture_content_is_disabled_by_default(telemetry):
    _, exporter, _ = telemetry
    await _agent().invoke_async("Calculate 2+2")
    for span in exporter.get_finished_spans():
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.output.messages" not in span.attributes


@pytest.mark.asyncio
async def test_llm_span_can_be_disabled(monkeypatch, telemetry):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_STRANDS_LLM_SPAN_MODE", "never")
    _, exporter, _ = telemetry
    await _agent().invoke_async("Calculate 2+2")
    kinds = [
        span.attributes["gen_ai.span.kind"]
        for span in exporter.get_finished_spans()
    ]
    assert "LLM" not in kinds
    assert kinds.count("STEP") == 2


@pytest.mark.asyncio
async def test_concurrent_agents_keep_isolated_trace_trees(telemetry):
    _, exporter, _ = telemetry
    results = await asyncio.gather(
        *[
            _agent().invoke_async(f"Calculate 2+2 request {index}")
            for index in range(4)
        ]
    )
    assert [str(result).strip() for result in results] == [
        "The answer is 4."
    ] * 4

    spans = exporter.get_finished_spans()
    agents = [
        span
        for span in spans
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ]
    assert len(agents) == 4
    assert len({agent.context.trace_id for agent in agents}) == 4
    for agent in agents:
        trace_spans = [
            span
            for span in spans
            if span.context.trace_id == agent.context.trace_id
        ]
        assert len(trace_spans) == 6
        assert {
            span.attributes["gen_ai.span.kind"] for span in trace_spans
        } == {
            "AGENT",
            "STEP",
            "LLM",
            "TOOL",
        }


@pytest.mark.asyncio
async def test_outer_application_parent_is_preserved(telemetry):
    instrumentor, exporter, _ = telemetry
    tracer = instrumentor._hook._handler._tracer
    with tracer.start_as_current_span("outer") as outer:
        outer_context = outer.get_span_context()
        await _agent().invoke_async("Calculate 2+2")

    spans = exporter.get_finished_spans()
    agent = next(
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "AGENT"
    )
    assert agent.context.trace_id == outer_context.trace_id
    assert agent.parent.span_id == outer_context.span_id
    assert len(spans) == 7


@pytest.mark.asyncio
async def test_probe_callback_failure_does_not_change_business_result(
    telemetry,
):
    instrumentor, exporter, _ = telemetry
    instrumentor._hook._handler.start_llm = lambda invocation, context=None: (
        _ for _ in ()
    ).throw(RuntimeError("injected probe failure"))
    result = await _agent().invoke_async("Calculate 2+2")
    assert str(result).strip() == "The answer is 4."
    kinds = [
        span.attributes["gen_ai.span.kind"]
        for span in exporter.get_finished_spans()
    ]
    assert "AGENT" in kinds
    assert "LLM" not in kinds


@pytest.mark.asyncio
async def test_one_probe_fault_does_not_contaminate_concurrent_siblings(
    telemetry,
):
    instrumentor, exporter, _ = telemetry
    original_start_llm = instrumentor._hook._handler.start_llm

    def selective_start_llm(invocation, context=None):
        if "fault-request" in str(invocation.input_messages):
            raise RuntimeError("injected concurrent probe failure")
        return original_start_llm(invocation, context=context)

    instrumentor._hook._handler.start_llm = selective_start_llm
    results = await asyncio.gather(
        _agent().invoke_async("clean-request-0"),
        _agent().invoke_async("fault-request"),
        _agent().invoke_async("clean-request-1"),
    )
    assert [str(result).strip() for result in results] == [
        "The answer is 4."
    ] * 3
    assert instrumentor._hook._states == {}

    spans = exporter.get_finished_spans()
    agents = [
        span
        for span in spans
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ]
    span_counts = sorted(
        sum(span.context.trace_id == agent.context.trace_id for span in spans)
        for agent in agents
    )
    assert span_counts == [5, 6, 6]


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_name", ["stop_llm", "stop_invoke_agent"])
async def test_finalize_callback_failure_preserves_business_and_next_sibling(
    callback_name,
    telemetry,
):
    instrumentor, exporter, _ = telemetry
    handler = instrumentor._hook._handler
    original_callback = getattr(handler, callback_name)

    def failing_callback(*args, **kwargs):
        raise RuntimeError(f"injected {callback_name} failure")

    setattr(handler, callback_name, failing_callback)
    result = await _agent().invoke_async("Calculate 2+2 with probe fault")
    assert str(result).strip() == "The answer is 4."
    assert instrumentor._hook._states == {}

    setattr(handler, callback_name, original_callback)
    clean_result = await _agent().invoke_async("Calculate 2+2 clean sibling")
    assert str(clean_result).strip() == "The answer is 4."
    clean_agent = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ][-1]
    clean_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.context.trace_id == clean_agent.context.trace_id
    ]
    assert clean_agent.parent is None
    assert len(clean_spans) == 6


@pytest.mark.asyncio
async def test_failure_reporter_failure_preserves_exception_and_next_sibling(
    telemetry,
):
    instrumentor, exporter, _ = telemetry
    handler = instrumentor._hook._handler
    original_callback = handler.fail_llm

    def failing_callback(*args, **kwargs):
        raise RuntimeError("injected fail_llm failure")

    handler.fail_llm = failing_callback
    original = ValueError("provider unavailable")
    with pytest.raises(ValueError) as raised:
        await Agent(
            model=FailingModel(original), retry_strategy=None
        ).invoke_async("hello")
    assert raised.value is original
    assert instrumentor._hook._states == {}

    handler.fail_llm = original_callback
    clean_result = await _agent().invoke_async("Calculate 2+2 clean sibling")
    assert str(clean_result).strip() == "The answer is 4."
    clean_agent = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ][-1]
    clean_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.context.trace_id == clean_agent.context.trace_id
    ]
    assert clean_agent.parent is None
    assert len(clean_spans) == 6


@pytest.mark.asyncio
async def test_model_error_is_reraised_and_spans_fail(telemetry):
    _, exporter, _ = telemetry
    original = ValueError("provider unavailable")
    agent = Agent(model=FailingModel(original), retry_strategy=None)
    with pytest.raises(ValueError) as raised:
        await agent.invoke_async("hello")
    assert raised.value is original
    spans = exporter.get_finished_spans()
    assert {span.attributes["gen_ai.span.kind"] for span in spans} == {
        "LLM",
        "STEP",
        "AGENT",
    }
    assert all(span.status.status_code.name == "ERROR" for span in spans)
    assert all(span.attributes["error.type"] == "ValueError" for span in spans)
    agent_span = next(
        span
        for span in spans
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    )
    assert agent_span.status.description == "provider unavailable"


@pytest.mark.asyncio
async def test_cancellation_cleans_state_and_spans(telemetry):
    instrumentor, exporter, _ = telemetry
    agent = Agent(model=SlowModel(), retry_strategy=None)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agent.invoke_async("wait"), timeout=0.01)

    assert instrumentor._hook._states == {}
    spans = exporter.get_finished_spans()
    assert {span.attributes["gen_ai.span.kind"] for span in spans} == {
        "LLM",
        "STEP",
        "AGENT",
    }
    assert all(span.status.status_code.name == "ERROR" for span in spans)


@pytest.mark.asyncio
async def test_early_break_cross_task_stream_close_cleans_state_and_next_sibling(
    telemetry, caplog
):
    instrumentor, exporter, _ = telemetry
    stream = _agent().stream_async("Calculate 2+2")
    async for _ in stream:
        if instrumentor._hook._states:
            break
    await asyncio.create_task(stream.aclose())

    assert instrumentor._hook._states == {}
    clean_result = await _agent().invoke_async("Calculate 2+2 clean sibling")
    assert str(clean_result).strip() == "The answer is 4."
    agents = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ]
    closed_agent, clean_agent = agents
    assert closed_agent.status.status_code.name == "ERROR"
    assert (
        closed_agent.status.description
        == "Strands stream closed before invocation completed"
    )
    clean_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.context.trace_id == clean_agent.context.trace_id
    ]
    assert clean_agent.parent is None
    assert len(clean_spans) == 6
    assert not any(
        "different Context" in record.getMessage()
        or "Failed to detach context" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_name", ["attach_stream_contexts", "detach_stream_contexts"]
)
async def test_detach_stream_context_failure_preserves_chunks_and_next_sibling(
    callback_name, telemetry, monkeypatch, caplog
):
    instrumentor, exporter, _ = telemetry

    async def consume(agent: Agent) -> tuple[int, str]:
        event_count = 0
        text = ""
        async for event in agent.stream_async("Calculate 2+2"):
            event_count += 1
            if isinstance(event, dict) and isinstance(event.get("data"), str):
                text += event["data"]
        return event_count, text.strip()

    baseline = await consume(_agent())
    original_callback = getattr(instrumentor._hook, callback_name)
    calls = 0

    def fail_once(state_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(f"injected {callback_name} failure")
        return original_callback(state_key)

    monkeypatch.setattr(instrumentor._hook, callback_name, fail_once)
    fault = await consume(_agent())
    assert fault == baseline
    assert instrumentor._hook._states == {}

    clean_result = await _agent().invoke_async("Calculate 2+2 clean sibling")
    assert str(clean_result).strip() == "The answer is 4."
    agents = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ]
    assert len(agents) == 3
    assert all(span.parent is None for span in agents)
    assert all(
        len(
            [
                child
                for child in exporter.get_finished_spans()
                if child.context.trace_id == span.context.trace_id
            ]
        )
        == 6
        for span in agents
    )
    assert not any(
        "different Context" in record.getMessage()
        or "Failed to detach context" in record.getMessage()
        for record in caplog.records
    )


def test_native_tracer_is_restored_on_uninstrument(telemetry):
    instrumentor, _, _ = telemetry
    native = get_tracer()
    replacement = native.tracer
    instrumentor.uninstrument()
    assert native.tracer is not replacement


def test_instrumentation_dependencies():
    assert StrandsInstrumentor().instrumentation_dependencies() == (
        "strands-agents >= 1.50.2, < 2.0.0",
    )


def test_openai_compatible_dashscope_provider_is_detected():
    model = DeterministicModel([])
    model.client_args = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    agent = type("Agent", (), {"model": model})()
    assert _provider_name(agent) == "dashscope"


def test_dashscope_text_in_url_path_does_not_spoof_provider():
    model = DeterministicModel([])
    model.client_args = {
        "base_url": "https://example.invalid/dashscope.aliyuncs.com/v1"
    }
    agent = type("Agent", (), {"model": model})()
    assert _provider_name(agent) != "dashscope"
