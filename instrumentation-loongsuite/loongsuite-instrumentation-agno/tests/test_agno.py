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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from agno.agent import Agent
from agno.metrics import MessageMetrics
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.tools.function import Function, FunctionCall

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agno import AgnoInstrumentor
from opentelemetry.instrumentation.agno._wrapper import (
    AgnoFunctionCallWrapper,
    AgnoModelWrapper,
)
from opentelemetry.instrumentation.agno.utils import (
    convert_agent_input,
    create_agent_invocation,
    create_llm_invocation,
    create_tool_invocation,
    update_agent_invocation_from_response,
    update_llm_invocation_from_response,
    update_tool_invocation_from_response,
)
from opentelemetry.sdk.trace import Resource, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)


class EchoModel(Model):
    def __init__(self):
        super().__init__(id="echo-model", name="echo", provider="test")

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            role="assistant",
            content="hello",
            response_usage=MessageMetrics(input_tokens=2, output_tokens=3),
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            role="assistant",
            content="hello async",
            response_usage=MessageMetrics(input_tokens=2, output_tokens=4),
        )

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        yield ModelResponse(
            role="assistant",
            content="he",
            response_usage=MessageMetrics(input_tokens=2, output_tokens=1),
        )
        yield ModelResponse(
            role="assistant",
            content="llo",
            response_usage=MessageMetrics(output_tokens=2),
        )

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator:
        yield ModelResponse(
            role="assistant",
            content="he",
            response_usage=MessageMetrics(input_tokens=2, output_tokens=1),
        )
        yield ModelResponse(
            role="assistant",
            content="llo",
            response_usage=MessageMetrics(output_tokens=2),
        )

    def _parse_provider_response(
        self, response: Any, **kwargs: Any
    ) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    provider = TracerProvider(resource=Resource(attributes={}))
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(autouse=True)
def instrument(tracer_provider: trace_api.TracerProvider):
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgnoInstrumentor().uninstrument()
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")


def _spans_by_name(span_exporter: InMemorySpanExporter):
    return {span.name: span for span in span_exporter.get_finished_spans()}


class RecordingHandler:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.failed = []

    def start_llm(self, invocation, context=None):
        self.started.append((invocation, context))
        return invocation

    def stop_llm(self, invocation):
        self.stopped.append(invocation)
        return invocation

    def fail_llm(self, invocation, error):
        self.failed.append((invocation, error))
        return invocation

    def start_execute_tool(self, invocation, context=None):
        self.started.append((invocation, context))
        return invocation

    def stop_execute_tool(self, invocation):
        self.stopped.append(invocation)
        return invocation

    def fail_execute_tool(self, invocation, error):
        self.failed.append((invocation, error))
        return invocation


def test_agent_model_and_tool_spans_use_genai_util(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(
        name="EchoAgent",
        model=EchoModel(),
        tools=[],
        instructions=["Always answer tersely."],
    )

    response = agent.run("Say hello", user_id="u1", session_id="s1")
    assert response.content == "hello"

    fn = Function.from_callable(lambda city: f"sunny in {city}")
    fn.name = "get_weather"
    function_call = FunctionCall(
        function=fn,
        arguments={"city": "Hangzhou"},
        call_id="call_1",
    )
    function_call.execute()

    spans = _spans_by_name(span_exporter)
    assert "invoke_agent EchoAgent" in spans
    assert "chat echo-model" in spans
    assert "execute_tool get_weather" in spans

    agent_attrs = spans["invoke_agent EchoAgent"].attributes
    model_attrs = spans["chat echo-model"].attributes
    tool_attrs = spans["execute_tool get_weather"].attributes

    assert agent_attrs["gen_ai.span.kind"] == "AGENT"
    assert agent_attrs["gen_ai.operation.name"] == "invoke_agent"
    assert agent_attrs["gen_ai.agent.name"] == "EchoAgent"
    assert agent_attrs["gen_ai.session.id"] == "s1"
    assert agent_attrs["gen_ai.user.id"] == "u1"
    assert "Say hello" in agent_attrs["gen_ai.input.messages"]
    assert "hello" in agent_attrs["gen_ai.output.messages"]
    assert "Always answer tersely" in agent_attrs["gen_ai.system_instructions"]

    assert model_attrs["gen_ai.span.kind"] == "LLM"
    assert model_attrs["gen_ai.operation.name"] == "chat"
    assert model_attrs["gen_ai.request.model"] == "echo-model"
    assert model_attrs["gen_ai.usage.input_tokens"] == 2
    assert model_attrs["gen_ai.usage.output_tokens"] == 3

    assert tool_attrs["gen_ai.span.kind"] == "TOOL"
    assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
    assert tool_attrs["gen_ai.tool.name"] == "get_weather"
    assert tool_attrs["gen_ai.tool.call.id"] == "call_1"
    assert "Hangzhou" in tool_attrs["gen_ai.tool.call.arguments"]
    assert "sunny in Hangzhou" in tool_attrs["gen_ai.tool.call.result"]


def test_agno_skill_tool_invocation_captures_skill_attributes():
    fn = Function.from_callable(lambda skill_name: f"loaded {skill_name}")
    fn.name = "get_skill_instructions"
    fn.description = "Load full instructions for a skill."
    function_call = FunctionCall(
        function=fn,
        arguments={"skill_name": "code-review"},
        call_id="call_skill_1",
    )

    invocation = create_tool_invocation(function_call)

    assert invocation.tool_name == "get_skill_instructions"
    assert invocation.skill_name == "code-review"
    assert invocation.skill_id == "code-review"

    update_tool_invocation_from_response(
        invocation,
        SimpleNamespace(
            result={
                "skill_name": "code-review",
                "frontmatter": {
                    "description": "Review code for quality and security.",
                    "metadata": {"version": "1.2.3"},
                },
            }
        ),
    )

    assert invocation.skill_name == "code-review"
    assert invocation.skill_id == "code-review"
    assert (
        invocation.skill_description == "Review code for quality and security."
    )
    assert invocation.skill_version == "1.2.3"


def test_streaming_agent_finishes_agent_and_model_spans(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(name="StreamAgent", model=EchoModel(), tools=[])

    chunks = list(agent.run("stream please", stream=True))
    assert [chunk.content for chunk in chunks] == ["he", "llo"]

    spans = _spans_by_name(span_exporter)
    agent_attrs = spans["invoke_agent StreamAgent"].attributes
    model_attrs = spans["chat echo-model"].attributes

    assert agent_attrs["gen_ai.span.kind"] == "AGENT"
    assert "hello" in agent_attrs["gen_ai.output.messages"]
    assert "gen_ai.response.time_to_first_token" in agent_attrs
    assert model_attrs["gen_ai.span.kind"] == "LLM"
    assert "hello" in model_attrs["gen_ai.output.messages"]
    assert "gen_ai.response.time_to_first_token" in model_attrs
    assert model_attrs["gen_ai.usage.input_tokens"] == 2
    assert model_attrs["gen_ai.usage.output_tokens"] == 3


def test_streaming_agent_span_finishes_when_consumer_breaks(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(name="BreakAgent", model=EchoModel(), tools=[])

    stream = agent.run("stream please", stream=True)
    first_chunk = next(stream)
    assert first_chunk.content == "he"
    stream.close()

    spans = _spans_by_name(span_exporter)
    assert "invoke_agent BreakAgent" in spans
    agent_attrs = spans["invoke_agent BreakAgent"].attributes
    assert agent_attrs["gen_ai.span.kind"] == "AGENT"
    assert "he" in agent_attrs["gen_ai.output.messages"]


def test_async_agent_run_finishes_agent_and_model_spans(
    span_exporter: InMemorySpanExporter,
):
    async def run_agent():
        agent = Agent(name="AsyncAgent", model=EchoModel(), tools=[])
        return await agent.arun(
            "Say hello async", user_id="u2", session_id="s2"
        )

    response = asyncio.run(run_agent())
    assert response.content == "hello async"

    spans = _spans_by_name(span_exporter)
    agent_attrs = spans["invoke_agent AsyncAgent"].attributes
    model_attrs = spans["chat echo-model"].attributes

    assert agent_attrs["gen_ai.span.kind"] == "AGENT"
    assert agent_attrs["gen_ai.user.id"] == "u2"
    assert agent_attrs["gen_ai.session.id"] == "s2"
    assert model_attrs["gen_ai.usage.input_tokens"] == 2
    assert model_attrs["gen_ai.usage.output_tokens"] == 4


def test_async_streaming_agent_finishes_spans(
    span_exporter: InMemorySpanExporter,
):
    async def run_agent():
        agent = Agent(name="AsyncStreamAgent", model=EchoModel(), tools=[])
        chunks = []
        async for chunk in agent.arun("stream please", stream=True):
            chunks.append(chunk.content)
        return chunks

    assert asyncio.run(run_agent()) == ["he", "llo"]

    spans = _spans_by_name(span_exporter)
    assert "invoke_agent AsyncStreamAgent" in spans
    assert (
        spans["invoke_agent AsyncStreamAgent"].attributes["gen_ai.span.kind"]
        == "AGENT"
    )
    assert "chat echo-model" in spans


def test_concurrent_runs_do_not_drop_spans(
    span_exporter: InMemorySpanExporter,
):
    def run_once(index: int):
        agent = Agent(name=f"Agent{index}", model=EchoModel(), tools=[])
        return agent.run(f"hello {index}").content

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_once, range(3)))

    assert results == ["hello", "hello", "hello"]
    spans = span_exporter.get_finished_spans()
    agent_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "AGENT"
    ]
    model_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.span.kind") == "LLM"
    ]
    assert len(agent_spans) == 3
    assert len(model_spans) == 3


@pytest.mark.parametrize("content_capture_mode", [None, "NO_CONTENT"])
def test_content_capture_mode_does_not_gate_span_creation(
    monkeypatch,
    span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    content_capture_mode: str | None,
):
    AgnoInstrumentor().uninstrument()
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")
    span_exporter.clear()

    env_var = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    if content_capture_mode is None:
        monkeypatch.delenv(env_var, raising=False)
    else:
        monkeypatch.setenv(env_var, content_capture_mode)

    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)

    agent = Agent(name="NoContentAgent", model=EchoModel(), tools=[])
    response = agent.run("Say hello without content")

    assert response.content == "hello"
    spans = _spans_by_name(span_exporter)
    assert "invoke_agent NoContentAgent" in spans
    assert "chat echo-model" in spans

    agent_attrs = spans["invoke_agent NoContentAgent"].attributes
    model_attrs = spans["chat echo-model"].attributes
    assert agent_attrs["gen_ai.span.kind"] == "AGENT"
    assert model_attrs["gen_ai.span.kind"] == "LLM"
    assert "gen_ai.input.messages" not in agent_attrs
    assert "gen_ai.output.messages" not in agent_attrs
    assert "gen_ai.output.messages" not in model_attrs


def test_async_function_call_emits_tool_span(
    span_exporter: InMemorySpanExporter,
):
    async def async_weather(city: str) -> str:
        return f"rain in {city}"

    fn = Function.from_callable(async_weather)
    fn.name = "async_weather"
    function_call = FunctionCall(
        function=fn,
        arguments={"city": "Hangzhou"},
        call_id="call_async",
    )

    asyncio.run(function_call.aexecute())

    spans = _spans_by_name(span_exporter)
    tool_attrs = spans["execute_tool async_weather"].attributes
    assert tool_attrs["gen_ai.span.kind"] == "TOOL"
    assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
    assert tool_attrs["gen_ai.tool.call.id"] == "call_async"
    assert "Hangzhou" in tool_attrs["gen_ai.tool.call.arguments"]
    assert "rain in Hangzhou" in tool_attrs["gen_ai.tool.call.result"]


def test_function_call_wrapper_failure_calls_fail_handler():
    handler = RecordingHandler()
    wrapper = AgnoFunctionCallWrapper(handler)
    function_call = SimpleNamespace(
        function=SimpleNamespace(name="failing_tool", description=None),
        arguments={},
        call_id="call_fail",
    )

    def wrapped(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("tool boom")

    with pytest.raises(RuntimeError, match="tool boom"):
        wrapper.execute(wrapped, function_call, (), {})

    assert len(handler.started) == 1
    assert len(handler.stopped) == 0
    assert len(handler.failed) == 1
    assert handler.failed[0][0].tool_name == "failing_tool"


def test_aresponse_returns_result_not_coroutine():
    handler = RecordingHandler()
    wrapper = AgnoModelWrapper(handler)
    model = EchoModel()

    async def wrapped(*args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(role="assistant", content="expected")

    async def run_test():
        return await wrapper.aresponse(wrapped, model, (), {"messages": []})

    result = asyncio.run(run_test())

    assert not asyncio.iscoroutine(result)
    assert result.content == "expected"
    assert len(handler.started) == 1
    assert len(handler.stopped) == 1


def test_response_stream_calls_wrapped_once():
    handler = RecordingHandler()
    wrapper = AgnoModelWrapper(handler)
    model = EchoModel()
    wrapped = MagicMock(
        return_value=iter(
            [
                ModelResponse(role="assistant", content="chunk1"),
                ModelResponse(role="assistant", content="chunk2"),
            ]
        )
    )

    results = list(
        wrapper.response_stream(wrapped, model, (), {"messages": []})
    )

    assert wrapped.call_count == 1
    assert [result.content for result in results] == ["chunk1", "chunk2"]
    assert len(handler.started) == 1
    assert len(handler.stopped) == 1


def test_response_stream_merges_tool_calls_from_chunks():
    handler = RecordingHandler()
    wrapper = AgnoModelWrapper(handler)
    model = EchoModel()
    wrapped = MagicMock(
        return_value=iter(
            [
                ModelResponse(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Hangzhou"}',
                            },
                        }
                    ],
                ),
                ModelResponse(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"city":"Hangzhou"}',
                            },
                        }
                    ],
                ),
            ]
        )
    )

    list(wrapper.response_stream(wrapped, model, (), {"messages": []}))

    invocation = handler.stopped[0]
    parts = invocation.output_messages[0].parts
    tool_calls = [
        part for part in parts if getattr(part, "type", None) == "tool_call"
    ]
    assert [tool_call.name for tool_call in tool_calls] == [
        "get_weather",
        "get_time",
    ]


def test_agent_response_preserves_tool_call_parts():
    agent = Agent(name="ToolCallAgent", model=EchoModel(), tools=[])
    invocation = create_agent_invocation(
        agent, {"input": "call the weather tool"}
    )
    response = SimpleNamespace(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Hangzhou"}',
                },
            }
        ],
    )

    update_agent_invocation_from_response(invocation, response)

    parts = invocation.output_messages[0].parts
    tool_calls = [
        part for part in parts if getattr(part, "type", None) == "tool_call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"city": "Hangzhou"}


def test_tool_result_messages_do_not_duplicate_text_parts():
    messages = convert_agent_input(
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": {"temperature": 21},
            }
        ]
    )

    parts = messages[0].parts

    assert len(parts) == 1
    assert parts[0].type == "tool_call_response"
    assert parts[0].id == "call_1"
    assert parts[0].response == {"temperature": 21}


def test_model_dump_objects_are_serialized_without_pydantic_base_class():
    class ModelDumpToolCall:
        def model_dump(self, mode="json"):
            assert mode == "json"
            return {
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Hangzhou"}',
                },
            }

    messages = convert_agent_input(
        [
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[ModelDumpToolCall()],
            )
        ]
    )

    parts = messages[0].parts
    tool_calls = [
        part for part in parts if getattr(part, "type", None) == "tool_call"
    ]

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"city": "Hangzhou"}


def test_missing_finish_reason_is_not_reported():
    agent = Agent(name="NoFinishReasonAgent", model=EchoModel(), tools=[])
    invocation = create_agent_invocation(agent, {"input": "hello"})
    response = SimpleNamespace(content="hello")

    update_agent_invocation_from_response(invocation, response)

    assert invocation.finish_reasons is None
    assert invocation.output_messages[0].finish_reason is None


def test_llm_response_uses_provider_and_prompt_completion_tokens():
    model = SimpleNamespace(id="request-model", provider=None)
    invocation = create_llm_invocation(model, {})
    response = SimpleNamespace(
        role="assistant",
        content="hello",
        model="response-model",
        model_provider="dashscope",
        prompt_tokens=7,
        completion_tokens=11,
    )

    update_llm_invocation_from_response(invocation, response)

    assert invocation.provider == "dashscope"
    assert invocation.input_tokens == 7
    assert invocation.output_tokens == 11


def test_aresponse_stream_calls_wrapped_once():
    handler = RecordingHandler()
    wrapper = AgnoModelWrapper(handler)
    model = EchoModel()
    call_count = 0

    async def stream():
        yield ModelResponse(role="assistant", content="async_chunk1")
        yield ModelResponse(role="assistant", content="async_chunk2")

    async def wrapped(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        return stream()

    async def run_test():
        results = []
        async for chunk in wrapper.aresponse_stream(
            wrapped, model, (), {"messages": []}
        ):
            results.append(chunk.content)
        return results

    results = asyncio.run(run_test())

    assert call_count == 1
    assert results == ["async_chunk1", "async_chunk2"]
    assert len(handler.started) == 1
    assert len(handler.stopped) == 1


def test_model_response_failure_calls_fail_handler():
    handler = RecordingHandler()
    wrapper = AgnoModelWrapper(handler)
    model = EchoModel()

    def wrapped(*args: Any, **kwargs: Any) -> ModelResponse:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        wrapper.response(wrapped, model, (), {"messages": []})

    assert len(handler.started) == 1
    assert len(handler.stopped) == 0
    assert len(handler.failed) == 1
