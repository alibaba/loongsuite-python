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
import builtins
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

from opentelemetry.instrumentation.strands import (
    StrandsInstrumentor,
)
from opentelemetry.instrumentation.strands import (
    _hooks as hooks_module,
)
from opentelemetry.instrumentation.strands import (
    _instrumentor as instrumentor_module,
)
from opentelemetry.instrumentation.strands._hooks import (
    _agent_invocation_usage,
    _model_provider,
)
from opentelemetry.instrumentation.strands._native_telemetry import (
    NativeTelemetrySuppression,
)


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


class FailingCloseStream:
    def __init__(self, stream: Any):
        self.stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.stream.__anext__()

    async def aclose(self):
        await self.stream.aclose()
        raise RuntimeError("injected aclose failure")


class ProviderInstrumentedModel(DeterministicModel):
    _is_instrumented_by_opentelemetry = True

    def __init__(self, responses: Sequence[dict[str, Any]], tracer: Any):
        super().__init__(responses)
        self.tracer = tracer

    async def stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        with self.tracer.start_as_current_span("provider chat"):
            async for event in super().stream(*args, **kwargs):
                yield event


class OllamaQwenModel(DeterministicModel):
    pass


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


@pytest.fixture
def content_telemetry(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
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
    agent_instance = _agent()
    agent_instance.model.client_args = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    result = await agent_instance.invoke_async("Calculate 2+2")
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
    assert agent.attributes["gen_ai.provider.name"] == "strands-agents"
    assert all(
        span.attributes["gen_ai.provider.name"] == "dashscope" for span in llms
    )
    assert all(
        "gen_ai.tool.json_schema" not in span.attributes for span in spans
    )
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
async def test_capture_content_includes_every_model_round(content_telemetry):
    _, exporter, _ = content_telemetry
    await _agent().invoke_async("Calculate 2+2")

    spans = exporter.get_finished_spans()
    llms = [
        span for span in spans if span.attributes["gen_ai.span.kind"] == "LLM"
    ]
    agent = next(
        span
        for span in spans
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    )
    tool_span = next(
        span for span in spans if span.attributes["gen_ai.span.kind"] == "TOOL"
    )

    first_input = json.loads(llms[0].attributes["gen_ai.input.messages"])
    second_input = json.loads(llms[1].attributes["gen_ai.input.messages"])
    assert first_input == [
        {
            "role": "user",
            "parts": [{"content": "Calculate 2+2", "type": "text"}],
        }
    ]
    assert [message["role"] for message in second_input] == [
        "user",
        "assistant",
        "user",
    ]
    assert all("gen_ai.output.messages" in span.attributes for span in llms)
    assert json.loads(agent.attributes["gen_ai.input.messages"]) == first_input
    assert (
        json.loads(agent.attributes["gen_ai.output.messages"])[0]["parts"][0][
            "content"
        ]
        == "The answer is 4."
    )
    assert json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]) == {
        "expression": "2+2"
    }
    assert json.loads(tool_span.attributes["gen_ai.tool.call.result"])[
        "content"
    ] == [{"text": "4"}]


@pytest.mark.asyncio
async def test_framework_and_provider_llm_spans_can_coexist(telemetry):
    instrumentor, exporter, _ = telemetry
    model = ProviderInstrumentedModel(
        [{"content": [{"text": "provider result"}]}],
        instrumentor._hook._handler._tracer,
    )
    result = await Agent(model=model, retry_strategy=None).invoke_async(
        "hello"
    )
    assert str(result).strip() == "provider result"

    spans = exporter.get_finished_spans()
    framework_llm = next(
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "LLM"
    )
    provider_llm = next(span for span in spans if span.name == "provider chat")
    assert provider_llm.parent.span_id == framework_llm.context.span_id


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
    spans = exporter.get_finished_spans()
    kinds = [span.attributes["gen_ai.span.kind"] for span in spans]
    assert "AGENT" in kinds
    assert "LLM" not in kinds
    degraded_step = next(
        span for span in spans if span.attributes["gen_ai.span.kind"] == "STEP"
    )
    assert degraded_step.status.status_code.name == "ERROR"
    assert degraded_step.attributes["error.type"] == "RuntimeError"
    assert degraded_step.status.description == "injected probe failure"


@pytest.mark.asyncio
async def test_react_step_start_failure_does_not_leave_partial_state(
    telemetry, monkeypatch
):
    instrumentor, exporter, _ = telemetry
    handler = instrumentor._hook._handler
    original_start_react_step = handler.start_react_step
    monkeypatch.setattr(
        handler,
        "start_react_step",
        lambda invocation, context=None: (_ for _ in ()).throw(
            RuntimeError("injected start_react_step failure")
        ),
    )

    result = await _agent().invoke_async("Calculate 2+2 with step fault")
    assert str(result).strip() == "The answer is 4."
    assert instrumentor._hook._states == {}

    monkeypatch.setattr(handler, "start_react_step", original_start_react_step)
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
    assert len(clean_spans) == 6


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
    # The injected failure now matches the first model round because its input
    # snapshot is available. That round loses both the LLM span and the empty
    # standalone STEP that the previous late match left behind.
    assert span_counts == [4, 6, 6]


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
    fault_spans = exporter.get_finished_spans()
    assert all(
        "gen_ai.operation.name" in span.attributes for span in fault_spans
    )
    fallback_span = next(
        span
        for span in fault_spans
        if span.status.description == f"injected {callback_name} failure"
    )
    assert fallback_span.attributes["error.type"] == "RuntimeError"
    assert "gen_ai.provider.name" in fallback_span.attributes

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
async def test_response_mapping_failure_cleans_spans_and_next_sibling(
    telemetry, monkeypatch
):
    instrumentor, exporter, _ = telemetry
    original_mapping = hooks_module._convert_output_message

    def fail_mapping(*args, **kwargs):
        raise RuntimeError("injected response mapping failure")

    monkeypatch.setattr(hooks_module, "_convert_output_message", fail_mapping)
    result = await _agent().invoke_async("Calculate 2+2 with mapping fault")
    assert str(result).strip() == "The answer is 4."
    assert instrumentor._hook._states == {}
    fault_spans = exporter.get_finished_spans()
    assert any(
        span.status.description == "injected response mapping failure"
        for span in fault_spans
    )
    assert all(span.end_time is not None for span in fault_spans)

    monkeypatch.setattr(
        hooks_module, "_convert_output_message", original_mapping
    )
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
    fault_llm = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "LLM"
    )
    assert fault_llm.attributes["error.type"] == "ValueError"
    assert fault_llm.status.description == "provider unavailable"

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


def test_unfinished_llm_reporter_failure_still_finalizes_step_and_agent(
    telemetry,
):
    instrumentor, exporter, _ = telemetry
    hook = instrumentor._hook
    agent = _agent()
    invocation_state = {
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "event_loop_cycle_id": "unfinished-cycle",
    }
    hook._before_invocation(
        hooks_module.BeforeInvocationEvent(
            agent=agent,
            invocation_state=invocation_state,
            messages=invocation_state["messages"],
        )
    )
    hook._before_model(
        hooks_module.BeforeModelCallEvent(
            agent=agent,
            invocation_state=invocation_state,
        )
    )

    original_callback = hook._handler.fail_llm

    def failing_callback(*args, **kwargs):
        raise RuntimeError("injected unfinished fail_llm failure")

    hook._handler.fail_llm = failing_callback
    hook._after_invocation(
        hooks_module.AfterInvocationEvent(
            agent=agent,
            invocation_state=invocation_state,
            result=None,
        )
    )

    assert hook._states == {}
    spans = exporter.get_finished_spans()
    assert {span.attributes["gen_ai.span.kind"] for span in spans} == {
        "LLM",
        "STEP",
        "AGENT",
    }
    assert all(span.end_time is not None for span in spans)
    assert all(span.status.status_code.name == "ERROR" for span in spans)
    llm = next(
        span for span in spans if span.attributes["gen_ai.span.kind"] == "LLM"
    )
    assert llm.status.description == "model call did not finish"
    assert llm.attributes["error.type"] == "RuntimeError"
    hook._handler.fail_llm = original_callback


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
    assert all(
        span.attributes["error.type"] == "CancelledError" for span in spans
    )


@pytest.mark.asyncio
async def test_generator_exit_from_early_break_cleans_state_and_next_sibling(
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
    assert closed_agent.attributes["error.type"] == "GeneratorExit"
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
async def test_aclose_failure_is_isolated_and_next_sibling_is_clean(
    telemetry,
):
    instrumentor, exporter, _ = telemetry
    stream = instrumentor._forward_stream(
        FailingCloseStream(_agent().stream_async("Calculate 2+2"))
    )
    async for _ in stream:
        if instrumentor._hook._states:
            break
    await stream.aclose()

    assert instrumentor._hook._states == {}
    closed_agents = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["gen_ai.span.kind"] == "AGENT"
    ]
    assert len(closed_agents) == 1
    assert closed_agents[0].status.status_code.name == "ERROR"

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


def test_missing_per_invocation_usage_does_not_use_lifetime_totals():
    metrics = type(
        "Metrics",
        (),
        {"accumulated_usage": {"inputTokens": 999}},
    )()
    result = type("Result", (), {"metrics": metrics})()
    assert _agent_invocation_usage(result) == {}


def test_native_suppression_install_failure_is_fail_open(monkeypatch, caplog):
    wrap_calls = []

    def fail_install(self):
        raise RuntimeError("private Strands telemetry API moved")

    monkeypatch.setattr(NativeTelemetrySuppression, "install", fail_install)
    monkeypatch.setattr(
        instrumentor_module,
        "wrap_function_wrapper",
        lambda *args, **kwargs: wrap_calls.append((args, kwargs)),
    )

    instrumentor = StrandsInstrumentor()
    instrumentor._instrument(tracer_provider=TracerProvider())
    assert len(wrap_calls) == 2
    assert "LoongSuite instrumentation will continue" in caplog.text


def test_agent_wrapper_install_failure_restores_native_telemetry(
    monkeypatch, caplog
):
    wrap_calls = 0
    restore_calls = []
    unwrap_calls = []

    def fail_second_wrap(*args, **kwargs):
        nonlocal wrap_calls
        wrap_calls += 1
        if wrap_calls == 2:
            raise RuntimeError("injected stream wrapper failure")

    monkeypatch.setattr(
        NativeTelemetrySuppression, "install", lambda self: None
    )
    monkeypatch.setattr(
        NativeTelemetrySuppression,
        "restore",
        lambda self: restore_calls.append(True),
    )
    monkeypatch.setattr(
        instrumentor_module, "wrap_function_wrapper", fail_second_wrap
    )
    monkeypatch.setattr(
        StrandsInstrumentor,
        "_unwrap_agent_lifecycle",
        staticmethod(lambda: unwrap_calls.append(True)),
    )

    instrumentor = StrandsInstrumentor()
    instrumentor._instrument(tracer_provider=TracerProvider())
    assert wrap_calls == 2
    assert unwrap_calls == [True]
    assert restore_calls == [True]
    assert "Failed to wrap strands Agent lifecycle" in caplog.text


def test_agent_hook_registration_failure_preserves_construction(telemetry):
    instrumentor, _, _ = telemetry
    wrapped_calls = []

    class FailingHooks:
        def add_hook(self, hook):
            raise RuntimeError("injected hook registration failure")

    instance = type("AgentLike", (), {"hooks": FailingHooks()})()

    def wrapped(*args, **kwargs):
        wrapped_calls.append((args, kwargs))

    instrumentor._agent_init_wrapper(wrapped, instance, (), {})
    assert len(wrapped_calls) == 1
    assert not hasattr(instance, "_loongsuite_strands_hook")


def test_partial_native_suppression_install_can_be_restored(monkeypatch):
    native = get_tracer()
    original_tracer = native.tracer
    original_import = builtins.__import__

    def fail_event_loop_import(name, *args, **kwargs):
        if name == "strands.event_loop":
            raise ImportError("injected private module move")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_event_loop_import)
    suppression = NativeTelemetrySuppression()
    with pytest.raises(ImportError, match="private module move"):
        suppression.install()
    suppression.restore()
    assert native.tracer is original_tracer


def test_openai_compatible_dashscope_provider_is_detected():
    model = DeterministicModel([])
    model.client_args = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    agent = type("Agent", (), {"model": model})()
    assert _model_provider(agent) == "dashscope"


def test_dashscope_text_in_url_path_does_not_spoof_provider():
    model = DeterministicModel([])
    model.client_args = {
        "base_url": "https://example.invalid/dashscope.aliyuncs.com/v1"
    }
    agent = type("Agent", (), {"model": model})()
    assert _model_provider(agent) != "dashscope"


def test_qwen_model_served_by_ollama_keeps_ollama_provider():
    model = OllamaQwenModel([])
    model.config["model_id"] = "qwen3:8b"
    agent = type("Agent", (), {"model": model})()
    assert _model_provider(agent) == "ollama"


def test_qwen_model_without_provider_evidence_is_not_dashscope():
    model = DeterministicModel([])
    model.config["model_id"] = "qwen3:8b"
    agent = type("Agent", (), {"model": model})()
    assert _model_provider(agent) != "dashscope"
