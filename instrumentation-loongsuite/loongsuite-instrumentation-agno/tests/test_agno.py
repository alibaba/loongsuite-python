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
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from agno.agent import Agent
from agno.metrics import MessageMetrics
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.tools.function import Function, FunctionCall

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agno import AgnoInstrumentor
from opentelemetry.instrumentation.agno._wrapper import (
    AgnoAgentWrapper,
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
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
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


class ToolLoopModel(Model):
    def __init__(self):
        super().__init__(
            id="tool-loop-model",
            name="tool-loop",
            provider="test",
        )
        self.calls = 0

    def _next_response(self) -> ModelResponse:
        self.calls += 1
        if self.calls % 2 == 1:
            return ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Hangzhou"}',
                        },
                    }
                ],
                response_usage=MessageMetrics(
                    input_tokens=10,
                    output_tokens=2,
                    cache_read_tokens=4,
                ),
            )
        return ModelResponse(
            role="assistant",
            content="sunny in Hangzhou",
            response_usage=MessageMetrics(input_tokens=20, output_tokens=3),
        )

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next_response()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next_response()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        response = self._next_response()
        if response.tool_calls:
            yield ModelResponse(
                role="assistant",
                tool_calls=response.tool_calls,
                response_usage=response.response_usage,
            )
            return
        yield ModelResponse(
            role="assistant",
            content="sunny ",
            response_usage=MessageMetrics(input_tokens=20, output_tokens=1),
        )
        yield ModelResponse(
            role="assistant",
            content="in Hangzhou",
            response_usage=MessageMetrics(output_tokens=2),
        )

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator:
        for response in self.invoke_stream(*args, **kwargs):
            yield response

    def _parse_provider_response(
        self, response: Any, **kwargs: Any
    ) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


class LatencyToolLoopModel(ToolLoopModel):
    def __init__(self, delay_s: float = 0.03):
        super().__init__()
        self.delay_s = delay_s

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        time.sleep(self.delay_s)
        return self._next_response()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(self.delay_s)
        return self._next_response()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        response = self._next_response()
        time.sleep(self.delay_s)
        if response.tool_calls:
            yield ModelResponse(
                role="assistant",
                tool_calls=response.tool_calls,
                response_usage=response.response_usage,
            )
            return
        yield ModelResponse(
            role="assistant",
            content="sunny ",
            response_usage=MessageMetrics(input_tokens=20, output_tokens=1),
        )
        time.sleep(self.delay_s)
        yield ModelResponse(
            role="assistant",
            content="in Hangzhou",
            response_usage=MessageMetrics(output_tokens=2),
        )

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator:
        response = self._next_response()
        await asyncio.sleep(self.delay_s)
        if response.tool_calls:
            yield ModelResponse(
                role="assistant",
                tool_calls=response.tool_calls,
                response_usage=response.response_usage,
            )
            return
        yield ModelResponse(
            role="assistant",
            content="sunny ",
            response_usage=MessageMetrics(input_tokens=20, output_tokens=1),
        )
        await asyncio.sleep(self.delay_s)
        yield ModelResponse(
            role="assistant",
            content="in Hangzhou",
            response_usage=MessageMetrics(output_tokens=2),
        )


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


def _spans_by_kind(
    span_exporter: InMemorySpanExporter, span_kind: str
) -> list[Any]:
    return sorted(
        [
            span
            for span in span_exporter.get_finished_spans()
            if span.attributes.get("gen_ai.span.kind") == span_kind
        ],
        key=lambda span: span.start_time,
    )


def _duration_ms(span: Any) -> float:
    return (span.end_time - span.start_time) / 1_000_000


def _weather_tool() -> Function:
    fn = Function.from_callable(lambda city: f"weather for {city}: sunny")
    fn.name = "get_weather"
    return fn


def _assert_tool_loop_tree(
    span_exporter: InMemorySpanExporter,
    agent_name: str,
) -> None:
    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    step_spans = _spans_by_kind(span_exporter, "STEP")
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    agent_span = next(
        span
        for span in agent_spans
        if span.name == f"invoke_agent {agent_name}"
    )
    assert len(step_spans) == 2
    assert len(llm_spans) == 2
    assert len(tool_spans) == 1

    for step_span in step_spans:
        assert step_span.parent is not None
        assert step_span.parent.span_id == agent_span.context.span_id

    assert llm_spans[0].parent is not None
    assert llm_spans[0].parent.span_id == step_spans[0].context.span_id
    assert tool_spans[0].parent is not None
    assert tool_spans[0].parent.span_id == step_spans[0].context.span_id
    assert llm_spans[1].parent is not None
    assert llm_spans[1].parent.span_id == step_spans[1].context.span_id

    assert step_spans[0].attributes["gen_ai.react.round"] == 1
    assert step_spans[0].attributes["gen_ai.react.finish_reason"] == (
        "tool_calls"
    )
    assert step_spans[1].attributes["gen_ai.react.round"] == 2
    assert step_spans[1].attributes["gen_ai.react.finish_reason"] == "stop"

    assert [
        span.attributes["gen_ai.usage.total_tokens"] for span in llm_spans
    ] == [
        12,
        23,
    ]
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 30
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 5
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 35
    assert agent_span.attributes["gen_ai.usage.cache_read.input_tokens"] == 4


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

    # This standalone tool call happens after the agent run completes; run-local
    # identity must not leak to tool spans outside the active Agno run.
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
    assert model_attrs["gen_ai.session.id"] == "s1"
    assert model_attrs["gen_ai.user.id"] == "u1"
    assert model_attrs["gen_ai.conversation.id"] == "s1"
    assert model_attrs["gen_ai.agent.name"] == "EchoAgent"
    assert model_attrs["gen_ai.usage.input_tokens"] == 2
    assert model_attrs["gen_ai.usage.output_tokens"] == 3

    assert tool_attrs["gen_ai.span.kind"] == "TOOL"
    assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
    assert tool_attrs["gen_ai.tool.name"] == "get_weather"
    assert tool_attrs["gen_ai.tool.call.id"] == "call_1"
    assert "gen_ai.session.id" not in tool_attrs
    assert "gen_ai.user.id" not in tool_attrs
    assert "Hangzhou" in tool_attrs["gen_ai.tool.call.arguments"]
    assert "sunny in Hangzhou" in tool_attrs["gen_ai.tool.call.result"]


def test_agent_invocation_prefers_entry_baggage_identity():
    agent = SimpleNamespace(
        name="EntryAgent",
        id="agent-id",
        model=EchoModel(),
        user_id="agno-user",
        session_id="agno-session",
        tools=[],
        instructions=[],
    )
    ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
    ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user", ctx)
    token = otel_context.attach(ctx)
    try:
        invocation = create_agent_invocation(
            agent,
            {
                "input": "hello",
                "user_id": "run-user",
                "session_id": "run-session",
            },
        )
    finally:
        otel_context.detach(token)

    assert invocation.conversation_id == "entry-session"
    assert invocation.attributes[GEN_AI_SESSION_ID] == "entry-session"
    assert invocation.attributes[GEN_AI_USER_ID] == "entry-user"


def test_agent_invocation_applies_entry_identity_per_key():
    agent = SimpleNamespace(
        name="PartialEntryAgent",
        id="agent-id",
        model=EchoModel(),
        user_id="agent-user",
        session_id="agent-session",
        tools=[],
        instructions=[],
    )

    ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
    token = otel_context.attach(ctx)
    try:
        session_invocation = create_agent_invocation(
            agent,
            {
                "input": "hello",
                "user_id": "run-user",
                "session_id": "run-session",
            },
        )
    finally:
        otel_context.detach(token)

    ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user")
    token = otel_context.attach(ctx)
    try:
        user_invocation = create_agent_invocation(
            agent,
            {
                "input": "hello",
                "user_id": "run-user",
                "session_id": "run-session",
            },
        )
    finally:
        otel_context.detach(token)

    assert session_invocation.conversation_id == "entry-session"
    assert session_invocation.attributes[GEN_AI_SESSION_ID] == "entry-session"
    assert session_invocation.attributes[GEN_AI_USER_ID] == "run-user"
    assert user_invocation.conversation_id == "run-session"
    assert user_invocation.attributes[GEN_AI_SESSION_ID] == "run-session"
    assert user_invocation.attributes[GEN_AI_USER_ID] == "entry-user"


def test_agent_invocation_falls_back_to_agent_identity():
    agent = SimpleNamespace(
        name="FallbackAgent",
        id="agent-id",
        model=EchoModel(),
        user_id="agent-user",
        session_id="agent-session",
        tools=[],
        instructions=[],
    )

    invocation = create_agent_invocation(agent, {"input": "hello"})

    assert invocation.conversation_id == "agent-session"
    assert invocation.attributes[GEN_AI_SESSION_ID] == "agent-session"
    assert invocation.attributes[GEN_AI_USER_ID] == "agent-user"


def test_agent_run_identity_propagates_to_tool_span(
    span_exporter: InMemorySpanExporter,
):
    fn = Function.from_callable(lambda city: f"sunny in {city}")
    fn.name = "get_weather"
    function_call = FunctionCall(
        function=fn,
        arguments={"city": "Hangzhou"},
        call_id="call_in_agent",
    )
    agent = SimpleNamespace(
        name="ToolAgent",
        id="agent-id",
        model=EchoModel(),
        tools=[],
        instructions=[],
    )
    wrapper = AgnoAgentWrapper(get_extended_telemetry_handler())

    def wrapped(
        prompt: str, user_id: str | None = None, session_id: str | None = None
    ):
        assert prompt == "call the tool"
        assert user_id == "tool-user"
        assert session_id == "tool-session"
        function_call.execute()
        return SimpleNamespace(role="assistant", content="done")

    response = wrapper._run(
        wrapped,
        agent,
        ("call the tool",),
        {"user_id": "tool-user", "session_id": "tool-session"},
        {
            "input": "call the tool",
            "user_id": "tool-user",
            "session_id": "tool-session",
        },
    )

    assert response.content == "done"
    spans = _spans_by_name(span_exporter)
    tool_attrs = spans["execute_tool get_weather"].attributes
    assert tool_attrs["gen_ai.session.id"] == "tool-session"
    assert tool_attrs["gen_ai.user.id"] == "tool-user"
    assert tool_attrs["gen_ai.agent.name"] == "ToolAgent"


def test_nested_agent_run_restores_outer_identity(
    span_exporter: InMemorySpanExporter,
):
    handler = get_extended_telemetry_handler()
    agent_wrapper = AgnoAgentWrapper(handler)
    model_wrapper = AgnoModelWrapper(handler)
    outer_agent = SimpleNamespace(
        name="OuterAgent",
        id="outer-agent-id",
        model=EchoModel(),
        tools=[],
        instructions=[],
    )
    inner_agent = SimpleNamespace(
        name="InnerAgent",
        id="inner-agent-id",
        model=EchoModel(),
        tools=[],
        instructions=[],
    )

    def model_call(name: str):
        model = SimpleNamespace(id=f"{name}-model", provider="test")

        def wrapped(messages=None):
            return SimpleNamespace(role="assistant", content=name)

        return model_wrapper.response(wrapped, model, (), {"messages": []})

    def inner_wrapped(prompt, user_id=None, session_id=None):
        assert prompt == "inner"
        assert user_id == "inner-user"
        assert session_id == "inner-session"
        model_call("inner")
        return SimpleNamespace(role="assistant", content="inner done")

    def outer_wrapped(prompt, user_id=None, session_id=None):
        assert prompt == "outer"
        assert user_id == "outer-user"
        assert session_id == "outer-session"
        model_call("outer-before")
        agent_wrapper._run(
            inner_wrapped,
            inner_agent,
            ("inner",),
            {"user_id": "inner-user", "session_id": "inner-session"},
            {
                "input": "inner",
                "user_id": "inner-user",
                "session_id": "inner-session",
            },
        )
        model_call("outer-after")
        return SimpleNamespace(role="assistant", content="outer done")

    agent_wrapper._run(
        outer_wrapped,
        outer_agent,
        ("outer",),
        {"user_id": "outer-user", "session_id": "outer-session"},
        {
            "input": "outer",
            "user_id": "outer-user",
            "session_id": "outer-session",
        },
    )

    spans = _spans_by_name(span_exporter)
    outer_before_attrs = spans["chat outer-before-model"].attributes
    inner_attrs = spans["chat inner-model"].attributes
    outer_after_attrs = spans["chat outer-after-model"].attributes

    assert outer_before_attrs["gen_ai.user.id"] == "outer-user"
    assert outer_before_attrs["gen_ai.session.id"] == "outer-session"
    assert outer_before_attrs["gen_ai.agent.name"] == "OuterAgent"
    assert inner_attrs["gen_ai.user.id"] == "inner-user"
    assert inner_attrs["gen_ai.session.id"] == "inner-session"
    assert inner_attrs["gen_ai.agent.name"] == "InnerAgent"
    assert outer_after_attrs["gen_ai.user.id"] == "outer-user"
    assert outer_after_attrs["gen_ai.session.id"] == "outer-session"
    assert outer_after_attrs["gen_ai.agent.name"] == "OuterAgent"


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


def test_agno_non_skill_tool_invocation_ignores_skill_like_payload():
    fn = Function.from_callable(lambda skill_name: f"loaded {skill_name}")
    fn.name = "get_weather"
    function_call = FunctionCall(
        function=fn,
        arguments={"skill_name": "code-review"},
        call_id="call_regular_1",
    )

    invocation = create_tool_invocation(function_call)
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

    assert invocation.tool_name == "get_weather"
    assert invocation.skill_name is None
    assert invocation.skill_id is None
    assert invocation.skill_description is None
    assert invocation.skill_version is None


def test_streaming_agent_finishes_agent_and_model_spans(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(name="StreamAgent", model=EchoModel(), tools=[])

    chunks = list(
        agent.run(
            "stream please",
            stream=True,
            user_id="stream-user",
            session_id="stream-session",
        )
    )
    assert [chunk.content for chunk in chunks] == ["he", "llo"]

    spans = _spans_by_name(span_exporter)
    agent_attrs = spans["invoke_agent StreamAgent"].attributes
    model_attrs = spans["chat echo-model"].attributes

    assert agent_attrs["gen_ai.span.kind"] == "AGENT"
    assert agent_attrs["gen_ai.user.id"] == "stream-user"
    assert agent_attrs["gen_ai.session.id"] == "stream-session"
    assert "hello" in agent_attrs["gen_ai.output.messages"]
    assert "gen_ai.response.time_to_first_token" in agent_attrs
    assert model_attrs["gen_ai.span.kind"] == "LLM"
    assert model_attrs["gen_ai.user.id"] == "stream-user"
    assert model_attrs["gen_ai.session.id"] == "stream-session"
    assert model_attrs["gen_ai.conversation.id"] == "stream-session"
    assert "hello" in model_attrs["gen_ai.output.messages"]
    assert "gen_ai.response.time_to_first_token" in model_attrs
    assert model_attrs["gen_ai.usage.input_tokens"] == 2
    assert model_attrs["gen_ai.usage.output_tokens"] == 3


def test_streaming_agent_identity_is_scoped_between_yields(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(name="ScopedStreamAgent", model=EchoModel(), tools=[])
    fn = Function.from_callable(lambda city: f"sunny in {city}")
    fn.name = "get_weather"
    function_call = FunctionCall(
        function=fn,
        arguments={"city": "Hangzhou"},
        call_id="call_between_stream_chunks",
    )

    stream = agent.run(
        "stream please",
        stream=True,
        user_id="scoped-stream-user",
        session_id="scoped-stream-session",
    )
    first_chunk = next(stream)
    function_call.execute()
    remaining_chunks = list(stream)
    chunk_contents = [
        first_chunk.content,
        *[chunk.content for chunk in remaining_chunks],
    ]

    assert chunk_contents == ["he", "llo"]

    spans = _spans_by_name(span_exporter)
    model_attrs = spans["chat echo-model"].attributes
    tool_attrs = spans["execute_tool get_weather"].attributes

    assert model_attrs["gen_ai.user.id"] == "scoped-stream-user"
    assert model_attrs["gen_ai.session.id"] == "scoped-stream-session"
    assert "gen_ai.user.id" not in tool_attrs
    assert "gen_ai.session.id" not in tool_attrs


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
    assert model_attrs["gen_ai.user.id"] == "u2"
    assert model_attrs["gen_ai.session.id"] == "s2"
    assert model_attrs["gen_ai.conversation.id"] == "s2"
    assert model_attrs["gen_ai.usage.input_tokens"] == 2
    assert model_attrs["gen_ai.usage.output_tokens"] == 4


def test_async_streaming_agent_finishes_spans(
    span_exporter: InMemorySpanExporter,
):
    async def run_agent():
        agent = Agent(name="AsyncStreamAgent", model=EchoModel(), tools=[])
        chunks = []
        async for chunk in agent.arun(
            "stream please",
            stream=True,
            user_id="async-stream-user",
            session_id="async-stream-session",
        ):
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
    model_attrs = spans["chat echo-model"].attributes
    assert model_attrs["gen_ai.user.id"] == "async-stream-user"
    assert model_attrs["gen_ai.session.id"] == "async-stream-session"
    assert model_attrs["gen_ai.conversation.id"] == "async-stream-session"


def test_async_streaming_agent_identity_is_scoped_between_yields(
    span_exporter: InMemorySpanExporter,
):
    fn = Function.from_callable(lambda city: f"sunny in {city}")
    fn.name = "get_weather"
    function_call = FunctionCall(
        function=fn,
        arguments={"city": "Hangzhou"},
        call_id="call_between_async_stream_chunks",
    )

    async def run_agent():
        agent = Agent(
            name="AsyncScopedStreamAgent", model=EchoModel(), tools=[]
        )
        chunks = []
        async for chunk in agent.arun(
            "stream please",
            stream=True,
            user_id="async-scoped-stream-user",
            session_id="async-scoped-stream-session",
        ):
            chunks.append(chunk.content)
            if len(chunks) == 1:
                function_call.execute()
        return chunks

    assert asyncio.run(run_agent()) == ["he", "llo"]

    spans = _spans_by_name(span_exporter)
    model_attrs = spans["chat echo-model"].attributes
    tool_attrs = spans["execute_tool get_weather"].attributes

    assert model_attrs["gen_ai.user.id"] == "async-scoped-stream-user"
    assert model_attrs["gen_ai.session.id"] == "async-scoped-stream-session"
    assert "gen_ai.user.id" not in tool_attrs
    assert "gen_ai.session.id" not in tool_attrs


def test_tool_call_loop_emits_react_steps_and_split_llm_spans(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(
        name="ToolLoopAgent",
        model=ToolLoopModel(),
        tools=[_weather_tool()],
    )

    response = agent.run("what is the weather")

    assert response.content == "sunny in Hangzhou"
    _assert_tool_loop_tree(span_exporter, "ToolLoopAgent")


def test_async_tool_call_loop_emits_react_steps_and_split_llm_spans(
    span_exporter: InMemorySpanExporter,
):
    async def run_agent():
        agent = Agent(
            name="AsyncToolLoopAgent",
            model=ToolLoopModel(),
            tools=[_weather_tool()],
        )
        return await agent.arun("what is the weather")

    response = asyncio.run(run_agent())

    assert response.content == "sunny in Hangzhou"
    _assert_tool_loop_tree(span_exporter, "AsyncToolLoopAgent")


def test_stream_tool_call_loop_emits_agent_tokens_and_split_llm_spans(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(
        name="StreamToolLoopAgent",
        model=ToolLoopModel(),
        tools=[_weather_tool()],
    )

    list(agent.run("what is the weather", stream=True))

    _assert_tool_loop_tree(span_exporter, "StreamToolLoopAgent")


def test_tool_call_loop_llm_spans_cover_provider_latency(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(
        name="LatencyToolLoopAgent",
        model=LatencyToolLoopModel(),
        tools=[_weather_tool()],
        telemetry=False,
    )

    agent.run("what is the weather")

    llm_spans = _spans_by_kind(span_exporter, "LLM")
    assert len(llm_spans) == 2
    assert all(_duration_ms(span) >= 25 for span in llm_spans)


def test_stream_tool_call_loop_llm_spans_cover_provider_latency(
    span_exporter: InMemorySpanExporter,
):
    agent = Agent(
        name="LatencyStreamToolLoopAgent",
        model=LatencyToolLoopModel(),
        tools=[_weather_tool()],
        telemetry=False,
    )

    list(agent.run("what is the weather", stream=True))

    llm_spans = _spans_by_kind(span_exporter, "LLM")
    assert len(llm_spans) == 2
    assert all(_duration_ms(span) >= 25 for span in llm_spans)


def test_async_stream_tool_call_loop_emits_agent_tokens_and_split_llm_spans(
    span_exporter: InMemorySpanExporter,
):
    async def run_agent():
        agent = Agent(
            name="AsyncStreamToolLoopAgent",
            model=ToolLoopModel(),
            tools=[_weather_tool()],
        )
        async for _event in agent.arun("what is the weather", stream=True):
            pass

    asyncio.run(run_agent())

    _assert_tool_loop_tree(span_exporter, "AsyncStreamToolLoopAgent")


def test_direct_model_call_without_agent_does_not_emit_react_step(
    span_exporter: InMemorySpanExporter,
):
    model = EchoModel()

    response = model.response(
        messages=[Message(role="user", content="say hello")]
    )

    assert response.content == "hello"
    assert len(_spans_by_kind(span_exporter, "LLM")) == 1
    assert len(_spans_by_kind(span_exporter, "STEP")) == 0


def test_concurrent_runs_do_not_drop_spans(
    span_exporter: InMemorySpanExporter,
):
    def run_once(index: int):
        agent = Agent(name=f"Agent{index}", model=EchoModel(), tools=[])
        return agent.run(
            f"hello {index}",
            user_id=f"user-{index}",
            session_id=f"session-{index}",
        ).content

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
    assert {
        (
            span.attributes.get("gen_ai.user.id"),
            span.attributes.get("gen_ai.session.id"),
        )
        for span in agent_spans
    } == {
        ("user-0", "session-0"),
        ("user-1", "session-1"),
        ("user-2", "session-2"),
    }
    assert {
        (
            span.attributes.get("gen_ai.user.id"),
            span.attributes.get("gen_ai.session.id"),
        )
        for span in model_spans
    } == {
        ("user-0", "session-0"),
        ("user-1", "session-1"),
        ("user-2", "session-2"),
    }


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
