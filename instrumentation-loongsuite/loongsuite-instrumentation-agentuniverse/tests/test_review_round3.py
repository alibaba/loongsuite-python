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

"""Review follow-ups for the third review round.

One test per finding: a stable framework without a usable
``TokenUsage.from_anthropic``, a streamed chunk whose telemetry read fails, the
``LLMStreamEvent.text_delta`` payload of 0.1.0a2, the output messages of a
tool-planning round, the agent output boundary, the per-call model override and
the ReAct STEP spans of an agent's rounds.

Every stub is driven through the real ``@trace_llm``/``@trace_agent`` seams, so
what is under test is the framework's own wrapper resolution rather than a
stand-in for it.
"""

import asyncio
import json
from typing import Any, AsyncIterator, List, Optional

from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.base.annotation.trace import trace_llm
from agentuniverse.llm.llm_output import (
    LLMOutput,
    LLMStreamEvent,
    StreamReducer,
    TokenUsage,
    ToolCallDelta,
)

from opentelemetry.instrumentation.agentuniverse import _llm as llm_layer

from .conftest import StubAgent, StubLLM, build_agent
from .test_instrumentor import (
    AGENT_SPAN,
    capture_on,
    one_layer_record,
    run_agent,
)

STEP_SPAN_NAME = "react step"
LLM_SPAN = "au.llm."


def span_names(harness: Any) -> List[str]:
    return [record.name for record in harness.records()]


def step_records(harness: Any) -> List[Any]:
    return [
        record for record in harness.records() if record.name == STEP_SPAN_NAME
    ]


def output_messages(record: Any) -> List[Any]:
    """The ``gen_ai.output.messages`` payload of a span, decoded."""
    payload = record.gen_ai.get("gen_ai.output.messages")
    assert payload is not None, sorted(record.gen_ai)
    return json.loads(payload)


def message_text(messages: List[Any]) -> str:
    return "".join(
        part.get("content", "")
        for message in messages
        for part in message.get("parts", [])
    )


# ---------------------------------------------------------------------------
# P1: a stable framework has no ``TokenUsage.from_anthropic``
# ---------------------------------------------------------------------------


class RawProviderUsageLLM(StubLLM):
    """A call whose usage only arrives in the provider payload."""

    name = "raw_provider_usage_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any) -> LLMOutput:
        return LLMOutput(
            text=f"raw:{prompt}",
            raw={
                "id": "resp-1",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            },
            finish_reason="stop",
        )


class StableLikeTokenUsage:
    """The stable release's converter surface: only ``from_openai``."""

    from_openai = staticmethod(TokenUsage.from_openai)


class TestStableFrameworkUsageConversion:
    def test_usage_is_read_without_the_anthropic_converter(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        """0.0.19.1 only has ``from_openai``; its payload must still convert.

        The installed build here is 0.1.0a2, which has both converters, so the
        stable release's converter surface is stood in for instead of the
        installed one.
        """
        monkeypatch.setattr(llm_layer, "TokenUsage", StableLikeTokenUsage)
        RawProviderUsageLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.usage.total_tokens"] == 25
        assert record.au["au.llm.usage.prompt_tokens"] == 20
        assert record.au["au.llm.usage.completion_tokens"] == 5

    def test_usage_is_skipped_without_any_converter(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        """No converter at all is a missing value, not a failed call."""

        class NoConverters:
            """A release that offers no converter at all."""

        monkeypatch.setattr(llm_layer, "TokenUsage", NoConverters)
        result = RawProviderUsageLLM().call(prompt="hello")
        assert result.text == "raw:hello"
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert "au.llm.usage.total_tokens" not in record.au


# ---------------------------------------------------------------------------
# P1: a failing collector must not disturb the stream
# ---------------------------------------------------------------------------


class ThreeChunkLLM(StubLLM):
    """A stream of three chunks the caller reads in order."""

    name = "three_chunk_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any):
        yield LLMOutput(text="one ")
        yield LLMOutput(text="two ")
        yield LLMOutput(text="three", usage=TokenUsage(text_in=1, text_out=3))


class AsyncThreeChunkLLM(ThreeChunkLLM):
    name = "async_three_chunk_llm"

    @trace_llm
    async def call(self, prompt: str, **kwargs: Any):
        async def stream() -> AsyncIterator[LLMOutput]:
            yield LLMOutput(text="one ")
            yield LLMOutput(text="two ")
            yield LLMOutput(
                text="three", usage=TokenUsage(text_in=1, text_out=3)
            )

        return stream()


class TestStreamChunkIsolation:
    def test_a_failing_collector_leaves_the_sync_stream_untouched(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("collector exploded")

        monkeypatch.setattr(llm_layer, "chunk_record", boom)
        chunks = list(ThreeChunkLLM().call(prompt="hello"))
        assert [chunk.text for chunk in chunks] == ["one ", "two ", "three"]
        # The span still closes, with the output the collector could not read.
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.status"] == "success"

    def test_a_failing_collector_leaves_the_async_stream_untouched(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("collector exploded")

        monkeypatch.setattr(llm_layer, "chunk_record", boom)

        async def drive() -> List[str]:
            stream = await AsyncThreeChunkLLM().call(prompt="hello")
            return [chunk.text async for chunk in stream]

        assert asyncio.run(drive()) == ["one ", "two ", "three"]
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.status"] == "success"

    def test_a_failing_tool_call_collector_leaves_the_stream_untouched(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        def boom(self: Any, chunk: Any) -> None:
            raise RuntimeError("tool collector exploded")

        monkeypatch.setattr(llm_layer.ToolCallAccumulator, "feed", boom)
        chunks = list(ThreeChunkLLM().call(prompt="hello"))
        assert [chunk.text for chunk in chunks] == ["one ", "two ", "three"]
        # The usage the chunk carried still reached the span.
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.usage.total_tokens"] == 4


class HostileChunk:
    """A chunk whose own shape breaks the collector reading it."""

    @property
    def text(self) -> str:
        raise RuntimeError("chunk in trouble")


class HostileChunkLLM(StubLLM):
    """A stream whose middle chunk cannot be read at all."""

    name = "hostile_chunk_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any):
        yield LLMOutput(text="one ")
        yield HostileChunk()
        yield LLMOutput(text="three", usage=TokenUsage(text_in=1, text_out=2))


class AsyncHostileChunkLLM(HostileChunkLLM):
    name = "async_hostile_chunk_llm"

    @trace_llm
    async def call(self, prompt: str, **kwargs: Any):
        async def stream() -> AsyncIterator[Any]:
            yield LLMOutput(text="one ")
            yield HostileChunk()
            yield LLMOutput(
                text="three", usage=TokenUsage(text_in=1, text_out=2)
            )

        return stream()


class TestStreamChunkIsolation:
    def test_a_chunk_that_breaks_the_collector_is_still_delivered(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            chunks = list(HostileChunkLLM().call(prompt="hello"))
        assert len(chunks) == 3, chunks
        assert chunks[1].__class__ is HostileChunk
        assert [type(chunk).__name__ for chunk in chunks] == [
            "LLMOutput",
            "HostileChunk",
            "LLMOutput",
        ]
        assert chunks[0].text == "one "
        assert chunks[2].text == "three"
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.status"] == "success"
        assert record.au["au.llm.output"] == "one three"

    def test_an_async_chunk_that_breaks_the_collector_is_still_delivered(
        self, loongsuite_harness: Any
    ) -> None:
        async def drive() -> List[Any]:
            stream = await AsyncHostileChunkLLM().call(prompt="hello")
            return [chunk async for chunk in stream]

        with capture_on():
            chunks = asyncio.run(drive())
        assert len(chunks) == 3, chunks
        assert chunks[1].__class__ is HostileChunk
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.status"] == "success"


# ---------------------------------------------------------------------------
# P2: a 0.1.0a2 stream carries its text in ``text_delta``
# ---------------------------------------------------------------------------


class StreamEventLLM(StubLLM):
    """A 0.1.0a2 stream: ``LLMStreamEvent``s instead of ``LLMOutput`` chunks."""

    name = "stream_event_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any):
        yield LLMStreamEvent.text("part-1")
        yield LLMStreamEvent.text("part-2")
        yield LLMStreamEvent.done(
            finish_reason="stop", usage=TokenUsage(text_in=2, text_out=5)
        )


class TestStreamEventTextDelta:
    def test_text_delta_reaches_the_span_output(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            events = list(StreamEventLLM().call(prompt="hello"))
        assert [event.text_delta for event in events] == [
            "part-1",
            "part-2",
            None,
        ]
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.output"] == "part-1part-2"
        assert message_text(output_messages(record)) == "part-1part-2"

    def test_the_frameworks_own_reducer_sees_the_same_text(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            reducer = StreamReducer()
            for event in StreamEventLLM().call(prompt="hello"):
                reducer.feed(event)
            output = reducer.build()
        assert output.text == "part-1part-2"
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.output"] == "part-1part-2"


# ---------------------------------------------------------------------------
# The tool-planning round's output messages
# ---------------------------------------------------------------------------


class ToolPlanningLLM(StubLLM):
    """A round that answers with a tool call, the way the framework reduces it."""

    name = "tool_planning_llm"
    #: ``Agent.judge_stream`` reads both of these to decide the call shape.
    streaming = False
    _channel_instance = None

    def __init__(self) -> None:
        self.calls = 0

    @trace_llm
    def call(
        self, prompt: str = "", messages: Any = None, **kwargs: Any
    ) -> LLMOutput:
        self.calls += 1
        if self.calls > 1:
            return LLMOutput(
                text="answered",
                usage=TokenUsage(text_in=2, text_out=3),
                finish_reason="stop",
            )
        reducer = StreamReducer()
        reducer.feed(
            LLMStreamEvent.tool_call(
                ToolCallDelta(
                    index=0,
                    id="call-1",
                    name="stub_tool",
                    arguments_delta='{"query": ',
                )
            )
        )
        reducer.feed(
            LLMStreamEvent.tool_call(
                ToolCallDelta(index=0, arguments_delta='"hello"}')
            )
        )
        reducer.feed(LLMStreamEvent.done(finish_reason="tool_calls"))
        return reducer.build()


class TestToolPlanningOutputMessages:
    def test_a_tool_call_round_reports_its_calls(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            ToolPlanningLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        messages = output_messages(record)
        assert messages, "the tool-planning round reported no output messages"
        assert messages[0]["finish_reason"] == "tool_calls"
        serialised = json.dumps(messages)
        assert "stub_tool" in serialised
        assert "hello" in serialised


# ---------------------------------------------------------------------------
# The agent output boundary
# ---------------------------------------------------------------------------


class EnvelopeAgent(StubAgent):
    """The envelope ``AgentTemplate.customized_execute`` really returns."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        return {
            **agent_input,
            "output": f"echo:{input_object.get_data('input')}",
        }


class TestAgentOutputBoundary:
    def test_the_span_reports_the_answer_not_the_envelope(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            run_agent(build_agent("envelope_agent", EnvelopeAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert json.loads(record.au["au.agent.output"]) == "echo:hello"
        assert message_text(output_messages(record)) == "echo:hello"

    def test_the_span_never_echoes_the_agent_input(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            run_agent(build_agent("envelope_agent", EnvelopeAgent))
        record = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert "kwargs" not in record.au["au.agent.output"]
        # The input keeps its own carrier.
        assert record.au["au.agent.input"]


# ---------------------------------------------------------------------------
# The per-call model override
# ---------------------------------------------------------------------------


class ConfiguredModelLLM(StubLLM):
    """An LLM configured with one model, callable with another."""

    name = "configured_model_llm"
    model_name = "qwen-turbo"


class TestPerCallModelOverride:
    def test_a_call_that_names_a_model_reports_that_model(
        self, loongsuite_harness: Any
    ) -> None:
        ConfiguredModelLLM().call(prompt="hello", model="qwen-plus")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.gen_ai["gen_ai.request.model"] == "qwen-plus"
        # The component's own name stays where it was.
        assert record.au["au.llm.name"] == "configured_model_llm"

    def test_a_call_without_an_override_reports_the_configured_model(
        self, loongsuite_harness: Any
    ) -> None:
        ConfiguredModelLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.gen_ai["gen_ai.request.model"] == "qwen-turbo"


# ---------------------------------------------------------------------------
# ReAct STEP spans
# ---------------------------------------------------------------------------


class RoundDrivingAgent(StubAgent):
    """An agent whose body drives the framework's own round hooks."""

    rounds: int = 2
    explode_at: Optional[int] = None

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        llm = StubLLM()
        for index in range(self.rounds):
            self.on_before_llm_round(None, round_index=index)
            if self.explode_at == index:
                # The round's LLM call never returns, so the matching
                # ``on_after_llm_round`` never runs and the step stays open.
                raise RuntimeError("round exploded")
            output = llm.call(prompt="hello")
            self.on_after_llm_round(None, output, round_index=index)
        return {"output": "done"}


class ToolCallingLoopAgent(StubAgent):
    """An agent whose body runs the framework's real tool-calling loop."""

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        llm = ToolPlanningLLM()
        output = self.run_tool_calling_loop(
            llm, LoopContext(), input_object, max_iterations=3
        )
        return {"output": output.text}


class LoopContext:
    """The slice of ``AgentContext`` the tool-calling loop uses."""

    def __init__(self) -> None:
        self.tools_schema: list = []
        self.messages: list = [{"role": "user", "content": "hello"}]

    def build_messages(self) -> list:
        return list(self.messages)

    def append_message(self, message: Any) -> None:
        self.messages.append(message)


def build_round_agent(
    name: str, rounds: int = 2, explode_at: Optional[int] = None
) -> Any:
    agent = RoundDrivingAgent()
    agent.rounds = rounds
    agent.explode_at = explode_at
    agent.agent_model = AgentModel(info={"name": name})
    return agent


def build_loop_agent(name: str) -> Any:
    agent = ToolCallingLoopAgent()
    agent.agent_model = AgentModel(info={"name": name})
    return agent


class TestReactSteps:
    def test_each_round_gets_its_own_step_span(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_round_agent("round_agent"))
        steps = step_records(loongsuite_harness)
        assert len(steps) == 2, span_names(loongsuite_harness)
        assert [step.gen_ai["gen_ai.react.round"] for step in steps] == [1, 2]
        for step in steps:
            assert step.kind == "INTERNAL"
            assert step.gen_ai["gen_ai.span.kind"] == "STEP"
            assert step.gen_ai["gen_ai.operation.name"] == "react"
            assert step.au["au.span.kind"] == "step"
            assert step.parent == "au.agent.round_agent"

    def test_the_step_closes_when_the_round_aborts(
        self, loongsuite_harness: Any
    ) -> None:
        try:
            run_agent(build_round_agent("aborting_agent", explode_at=0))
        except RuntimeError:
            pass
        steps = step_records(loongsuite_harness)
        assert len(steps) == 1, span_names(loongsuite_harness)
        assert steps[0].status == "ERROR"
        agent = one_layer_record(loongsuite_harness, AGENT_SPAN)
        assert agent.au["au.agent.status"] == "error"

    def test_the_real_loop_records_one_step_per_round(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_loop_agent("loop_agent"))
        steps = step_records(loongsuite_harness)
        assert len(steps) == 2, span_names(loongsuite_harness)
        assert [step.gen_ai["gen_ai.react.round"] for step in steps] == [1, 2]
        assert steps[0].gen_ai["gen_ai.react.finish_reason"] == "tool_calls"
        assert steps[1].gen_ai["gen_ai.react.finish_reason"] == "stop"

    def test_the_round_usage_still_reaches_the_agent(
        self, loongsuite_harness: Any
    ) -> None:
        run_agent(build_loop_agent("loop_agent"))
        agent = one_layer_record(loongsuite_harness, AGENT_SPAN)
        # Only the answering round reported usage (2 in, 3 out).
        assert agent.au["au.agent.usage.total_tokens"] == 5
