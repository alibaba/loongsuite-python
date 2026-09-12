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

"""Integration tests for span generation from the qwen-agent instrumentation plugin.

Tests verify that the instrumented methods produce correct OpenTelemetry spans
with the expected names, kinds, and attributes.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from qwen_agent.agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import ContentItem, FunctionCall, Message

from opentelemetry.instrumentation.qwen_agent import QwenAgentInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode

# ---------------------------------------------------------------------------
# Helpers: minimal concrete subclasses of the abstract qwen-agent classes
# ---------------------------------------------------------------------------


class _StubChatModel(BaseChatModel):
    """A minimal BaseChatModel subclass for testing.

    The real BaseChatModel.__init__ reads from a cfg dict; we bypass most of
    that logic by calling super().__init__ with a minimal config and then
    overriding the attributes we care about.
    """

    def __init__(self, model="test-model", model_type="qwen_dashscope"):
        cfg = {"model": model, "model_type": model_type}
        super().__init__(cfg)
        # Disable raw_api mode which requires stream-only and API key
        self.use_raw_api = False

    # Abstract methods required by BaseChatModel
    def _chat_no_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_with_functions(self, messages, functions, **kwargs):
        raise NotImplementedError


class _StubAgent(Agent):
    """A minimal Agent subclass for testing.

    Agent.__init__ normally requires complex setup; we use ``__new__`` and
    manually assign the attributes the instrumentation reads so that we can
    test the wrapped methods without triggering real agent initialization.
    """

    @classmethod
    def create(cls, name="TestAgent", llm=None):
        """Factory that skips the heavy __init__."""
        obj = cls.__new__(cls)
        obj.name = name
        obj.description = "A test agent"
        obj.system_message = "You are a helpful assistant."
        obj.llm = llm
        obj.function_map = {}
        obj.extra_generate_cfg = {}
        return obj

    # Abstract method required by Agent
    def _run(self, messages, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LLM chat span tests
# ---------------------------------------------------------------------------


class TestLLMChatSpan:
    """Verify that BaseChatModel.chat() produces a correct LLM (chat) span."""

    def test_non_stream_chat_creates_span(self, span_exporter, instrument):
        """Non-streaming chat() should create a single chat span with model info."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")

        fake_response = [Message(role="assistant", content="Hello there!")]

        with patch.object(
            _StubChatModel,
            "_chat_no_stream",
            return_value=fake_response,
        ):
            result = model.chat(
                messages=[Message(role="user", content="Hi")],
                stream=False,
            )

        assert result is not None

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1, (
            f"Expected a chat span, got: {[s.name for s in spans]}"
        )

        span = chat_spans[0]
        assert span.name == "chat qwen-max"
        assert span.kind == SpanKind.CLIENT
        attrs = dict(span.attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "qwen-max"
        # Provider name is stored as "gen_ai.provider.name" in newer semconv
        provider = attrs.get("gen_ai.provider.name") or attrs.get(
            GenAIAttributes.GEN_AI_SYSTEM
        )
        assert provider == "dashscope", (
            f"Expected 'dashscope', got attrs: {attrs}"
        )

    def test_non_stream_chat_records_token_usage(
        self, span_exporter, instrument
    ):
        """Non-streaming chat() should record token usage from Message.extra."""
        model = _StubChatModel(
            model="deepseek-v3", model_type="qwen_dashscope"
        )

        fake_response = [
            Message(
                role="assistant",
                content="4",
                extra={
                    "model_service_info": SimpleNamespace(
                        usage={
                            "input_tokens": 21,
                            "output_tokens": 1,
                            "total_tokens": 22,
                            "prompt_tokens_details": {"cached_tokens": 4},
                        }
                    )
                },
            )
        ]

        with patch.object(
            _StubChatModel,
            "_chat_no_stream",
            return_value=fake_response,
        ):
            model.chat(
                messages=[Message(role="user", content="What is 2+2?")],
                stream=False,
            )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 21
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 1
        assert attrs.get("gen_ai.usage.total_tokens") == 22
        assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 4

    def test_stream_chat_creates_span(self, span_exporter, instrument):
        """Streaming chat() should create a chat span after the iterator is consumed."""
        model = _StubChatModel(model="qwen-turbo", model_type="qwen_dashscope")

        chunk1 = [Message(role="assistant", content="Hello")]
        chunk2 = [Message(role="assistant", content="Hello world")]

        def fake_stream(messages, **kwargs):
            yield chunk1
            yield chunk2

        with patch.object(
            _StubChatModel,
            "_chat_stream",
            side_effect=fake_stream,
        ):
            response_iter = model.chat(
                messages=[Message(role="user", content="Hi")],
                stream=True,
            )
            # Consume the iterator to trigger span completion
            responses = list(response_iter)

        assert len(responses) == 2

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        span = chat_spans[0]
        assert span.name == "chat qwen-turbo"
        assert span.kind == SpanKind.CLIENT

    def test_stream_chat_records_token_usage(self, span_exporter, instrument):
        """Streaming chat() should record final cumulative token usage."""
        model = _StubChatModel(
            model="deepseek-v3", model_type="qwen_dashscope"
        )

        chunk1 = [
            Message(
                role="assistant",
                content="The",
                extra={
                    "model_service_info": {
                        "usage": {
                            "input_tokens": 18,
                            "output_tokens": 1,
                            "total_tokens": 19,
                        }
                    }
                },
            )
        ]
        chunk2 = [
            Message(
                role="assistant",
                content="The answer is 4.",
                extra={
                    "model_service_info": {
                        "usage": {
                            "input_tokens": 18,
                            "output_tokens": 5,
                            "total_tokens": 23,
                        }
                    }
                },
            )
        ]

        def fake_stream(messages, **kwargs):
            yield chunk1
            yield chunk2

        with patch.object(
            _StubChatModel,
            "_chat_stream",
            side_effect=fake_stream,
        ):
            response_iter = model.chat(
                messages=[Message(role="user", content="What is 2+2?")],
                stream=True,
            )
            list(response_iter)

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5
        assert attrs.get("gen_ai.usage.total_tokens") == 23

    def test_stream_chat_keeps_most_complete_usage(
        self, span_exporter, instrument
    ):
        """Streaming chat() should keep the largest cumulative usage seen."""
        model = _StubChatModel(
            model="deepseek-v3", model_type="qwen_dashscope"
        )

        chunk1 = [
            Message(
                role="assistant",
                content="The",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 1}
                    }
                },
            )
        ]
        chunk2 = [
            Message(
                role="assistant",
                content="The answer",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 5}
                    }
                },
            )
        ]
        chunk3 = [Message(role="assistant", content="The answer is 4.")]

        def fake_stream(messages, **kwargs):
            yield chunk1
            yield chunk2
            yield chunk3

        with patch.object(
            _StubChatModel,
            "_chat_stream",
            side_effect=fake_stream,
        ):
            response_iter = model.chat(
                messages=[Message(role="user", content="What is 2+2?")],
                stream=True,
            )
            list(response_iter)

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5
        assert attrs.get("gen_ai.usage.total_tokens") == 23

    def test_chat_with_function_call_response(self, span_exporter, instrument):
        """Chat response containing a function_call should still produce a valid span."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")

        fake_response = [
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="get_weather",
                    arguments='{"city": "Beijing"}',
                ),
            )
        ]

        with patch.object(
            _StubChatModel,
            "_chat_no_stream",
            return_value=fake_response,
        ):
            model.chat(
                messages=[
                    Message(role="user", content="What is the weather?")
                ],
                stream=False,
            )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1

    def test_chat_error_creates_error_span(self, span_exporter, instrument):
        """An exception during chat() should still produce a span with error status."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")

        with patch.object(
            _StubChatModel,
            "_chat_no_stream",
            side_effect=RuntimeError("API timeout"),
        ):
            with pytest.raises(RuntimeError, match="API timeout"):
                model.chat(
                    messages=[Message(role="user", content="Hi")],
                    stream=False,
                )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        span = chat_spans[0]
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Agent run span tests
# ---------------------------------------------------------------------------


class TestAgentRunSpan:
    """Verify that Agent.run() and run_nonstream() produce invoke_agent spans."""

    def test_agent_run_creates_invoke_agent_span(
        self, span_exporter, instrument
    ):
        """Agent.run() (generator) should create an invoke_agent span."""
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"

        agent = _StubAgent.create(name="WeatherBot", llm=llm)

        response_msgs = [Message(role="assistant", content="It is sunny.")]

        def fake_run(messages, **kwargs):
            yield response_msgs

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            results = list(
                agent.run([Message(role="user", content="Weather?")])
            )

        assert len(results) >= 1

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) >= 1, (
            f"Expected invoke_agent span, got: {[s.name for s in spans]}"
        )

        span = agent_spans[0]
        assert span.name == "invoke_agent WeatherBot"
        assert span.kind == SpanKind.INTERNAL

    def test_agent_run_nonstream_creates_invoke_agent_span(
        self, span_exporter, instrument
    ):
        """Agent.run_nonstream() should produce exactly one invoke_agent span.

        run_nonstream is NOT wrapped separately — it calls self.run() internally,
        so the single invoke_agent span comes from the run() wrapper.
        """
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"

        agent = _StubAgent.create(name="ChatBot", llm=llm)

        response_msgs = [Message(role="assistant", content="Hello!")]

        def fake_run(messages, **kwargs):
            yield response_msgs

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            result = agent.run_nonstream([Message(role="user", content="Hi")])

        assert result is not None

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        # run_nonstream is not wrapped; only run() creates the invoke_agent span.
        # Exactly one span should be produced (no duplication).
        assert len(agent_spans) == 1
        span_names = [s.name for s in agent_spans]
        assert any("ChatBot" in n for n in span_names)

    def test_agent_run_error_creates_error_span(
        self, span_exporter, instrument
    ):
        """An exception during Agent.run() should produce an error invoke_agent span."""
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"

        agent = _StubAgent.create(name="FailBot", llm=llm)

        def fake_run(messages, **kwargs):
            if False:
                yield  # make it a generator
            raise ValueError("Agent processing failed")

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            with pytest.raises(ValueError, match="Agent processing failed"):
                list(agent.run([Message(role="user", content="Go")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) >= 1
        span = agent_spans[0]
        assert span.status.status_code == StatusCode.ERROR

    def test_agent_run_multiple_yields(self, span_exporter, instrument):
        """Agent.run() yielding multiple times should produce one invoke_agent span."""
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"

        agent = _StubAgent.create(name="MultiYieldBot", llm=llm)

        def fake_run(messages, **kwargs):
            yield [Message(role="assistant", content="Thinking...")]
            yield [Message(role="assistant", content="Done!")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            results = list(agent.run([Message(role="user", content="Go")]))

        assert len(results) == 2

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        # The wrapper should produce exactly one span per run() call
        assert len(agent_spans) == 1

    def test_agent_run_records_only_final_output_message(
        self,
        span_exporter,
        tracer_provider,
        logger_provider,
        meter_provider,
        monkeypatch,
    ):
        """Agent output should keep the final answer, not tool calls/results."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        instrumentor = QwenAgentInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )

        llm = MagicMock()
        llm.model = "qwen-plus"
        llm.model_type = "qwen_dashscope"

        agent = _StubAgent.create(name="OnePilotBot", llm=llm)

        response_msgs = [
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="get_operational_snapshot",
                    arguments='{"incident_id": "INC-1"}',
                ),
            ),
            Message(
                role="function",
                name="get_operational_snapshot",
                content='{"p95_latency_ms": 1840}',
            ),
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="score_bundle_plan",
                    arguments='{"plan_name": "gray-rollout"}',
                ),
            ),
            Message(
                role="function",
                name="score_bundle_plan",
                content='{"score": 80}',
            ),
            Message(role="assistant", content="Final verdict: continue."),
        ]

        def fake_run(messages, **kwargs):
            yield response_msgs

        try:
            with patch.object(_StubAgent, "_run", side_effect=fake_run):
                list(agent.run([Message(role="user", content="diagnose")]))
        finally:
            instrumentor.uninstrument()

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1

        attrs = dict(agent_spans[0].attributes or {})
        output = json.loads(attrs["gen_ai.output.messages"])
        assert len(output) == 1
        assert output[0]["role"] == "assistant"
        assert output[0]["finish_reason"] == "stop"
        assert output[0]["parts"] == [
            {"content": "Final verdict: continue.", "type": "text"}
        ]

    def test_agent_run_without_final_answer_skips_tool_output_messages(
        self,
        span_exporter,
        tracer_provider,
        logger_provider,
        meter_provider,
        monkeypatch,
    ):
        """Agent output should not fall back to intermediate tool messages."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        instrumentor = QwenAgentInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )

        llm = MagicMock()
        llm.model = "qwen-plus"
        llm.model_type = "qwen_dashscope"

        agent = _StubAgent.create(name="ToolOnlyBot", llm=llm)

        response_msgs = [
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="get_operational_snapshot",
                    arguments='{"incident_id": "INC-1"}',
                ),
            ),
            Message(
                role="function",
                name="get_operational_snapshot",
                content='{"p95_latency_ms": 1840}',
            ),
        ]

        def fake_run(messages, **kwargs):
            yield response_msgs

        try:
            with patch.object(_StubAgent, "_run", side_effect=fake_run):
                list(agent.run([Message(role="user", content="diagnose")]))
        finally:
            instrumentor.uninstrument()

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1

        attrs = dict(agent_spans[0].attributes or {})
        assert "gen_ai.output.messages" not in attrs

    def test_nested_agent_run_creates_child_invoke_agent_span(
        self, span_exporter, instrument
    ):
        """Nested runs on different agent instances should not be suppressed."""
        parent_llm = MagicMock()
        parent_llm.model = "qwen-turbo"
        parent_llm.model_type = "qwen_dashscope"
        child_llm = MagicMock()
        child_llm.model = "qwen-plus"
        child_llm.model_type = "qwen_dashscope"

        parent_agent = _StubAgent.create(name="ParentBot", llm=parent_llm)
        child_agent = _StubAgent.create(name="ChildBot", llm=child_llm)

        def fake_run(self, messages, **kwargs):
            if self is parent_agent:
                yield from child_agent.run(
                    [Message(role="user", content="child task")]
                )
                yield [Message(role="assistant", content="parent final")]
            elif self is child_agent:
                yield [Message(role="assistant", content="child final")]
            else:
                yield [Message(role="assistant", content="unexpected")]

        with patch.object(
            _StubAgent, "_run", autospec=True, side_effect=fake_run
        ):
            results = list(
                parent_agent.run([Message(role="user", content="parent task")])
            )

        assert len(results) == 2

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        span_by_name = {s.name: s for s in agent_spans}
        assert set(span_by_name) == {
            "invoke_agent ParentBot",
            "invoke_agent ChildBot",
        }

        parent_span = span_by_name["invoke_agent ParentBot"]
        child_span = span_by_name["invoke_agent ChildBot"]
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id


# ---------------------------------------------------------------------------
# Tool call span tests
# ---------------------------------------------------------------------------


class TestToolCallSpan:
    """Verify that Agent._call_tool() produces an execute_tool span."""

    def _make_agent_with_tool(self, tool_name="get_weather"):
        """Create a stub agent with a tool in its function_map."""
        agent = _StubAgent.create(name="ToolAgent")

        mock_tool = MagicMock()
        mock_tool.description = "Get weather information"
        mock_tool.call = MagicMock(return_value="Sunny, 25 degrees")
        agent.function_map = {tool_name: mock_tool}
        return agent, mock_tool

    def test_call_tool_creates_execute_tool_span(
        self, span_exporter, instrument
    ):
        """_call_tool() should create an execute_tool span with tool name."""
        agent, mock_tool = self._make_agent_with_tool("get_weather")

        result = agent._call_tool("get_weather", '{"city": "Beijing"}')

        assert result == "Sunny, 25 degrees"

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1, (
            f"Expected execute_tool span, got: {[s.name for s in spans]}"
        )

        span = tool_spans[0]
        assert span.name == "execute_tool get_weather"
        assert span.kind == SpanKind.INTERNAL

    def test_call_tool_with_dict_args(self, span_exporter, instrument):
        """_call_tool() should handle dict arguments."""
        agent, mock_tool = self._make_agent_with_tool("search")
        mock_tool.call = MagicMock(return_value="Found 3 results")

        result = agent._call_tool("search", {"query": "OpenTelemetry"})

        assert result == "Found 3 results"

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        assert tool_spans[0].name == "execute_tool search"

    def test_call_tool_error_creates_span(self, span_exporter, instrument):
        """An exception in _call_tool() should still produce an execute_tool span.

        Note: qwen-agent's Agent._call_tool() catches tool exceptions internally
        and returns an error string rather than re-raising.
        """
        agent, mock_tool = self._make_agent_with_tool("broken_tool")
        mock_tool.call = MagicMock(side_effect=RuntimeError("Tool crashed"))

        # qwen-agent catches the exception and returns an error string
        result = agent._call_tool("broken_tool", "{}")
        assert isinstance(result, str)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        assert tool_spans[0].name == "execute_tool broken_tool"

    def test_call_tool_returns_content_items(self, span_exporter, instrument):
        """_call_tool() returning List[ContentItem] should still create a valid span."""
        agent, mock_tool = self._make_agent_with_tool("image_gen")
        mock_tool.call = MagicMock(
            return_value=[ContentItem(text="Generated image description")]
        )

        result = agent._call_tool("image_gen", '{"prompt": "a cat"}')

        assert isinstance(result, list)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        assert tool_spans[0].name == "execute_tool image_gen"

    def test_call_unknown_tool_no_crash(self, span_exporter, instrument):
        """Calling a tool not in function_map should still produce a span without crashing."""
        agent = _StubAgent.create(name="ToolAgent")
        agent.function_map = {}

        # The real Agent._call_tool returns an error string for unknown tools
        result = agent._call_tool("nonexistent", "{}")

        assert "does not exist" in result.lower() or isinstance(result, str)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1


# ---------------------------------------------------------------------------
# Span hierarchy / parent-child tests
# ---------------------------------------------------------------------------


class TestSpanHierarchy:
    """Verify that spans are correctly nested when operations are composed."""

    def test_agent_run_with_llm_call_produces_nested_spans(
        self, span_exporter, instrument
    ):
        """When an agent run internally calls LLM chat, spans should be nested."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        agent = _StubAgent.create(name="NestBot", llm=model)

        llm_response = [
            Message(
                role="assistant",
                content="The answer is 42.",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 21, "output_tokens": 7}
                    }
                },
            )
        ]

        def fake_run(messages, **kwargs):
            # Simulate the agent calling LLM internally
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=llm_response,
            ):
                agent.llm.chat(messages=messages, stream=False)
            yield [Message(role="assistant", content="The answer is 42.")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="What is 6*7?")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        chat_spans = [s for s in spans if s.name.startswith("chat")]

        assert len(agent_spans) >= 1
        assert len(chat_spans) >= 1

        # The chat span should be a child of the agent span
        agent_span = agent_spans[0]
        chat_span = chat_spans[0]
        assert chat_span.context.trace_id == agent_span.context.trace_id
        assert chat_span.parent is not None
        assert chat_span.parent.span_id == agent_span.context.span_id

        agent_attrs = dict(agent_span.attributes or {})
        assert agent_attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 21
        assert agent_attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 7
        assert agent_attrs.get("gen_ai.usage.total_tokens") == 28

    def test_agent_run_with_tool_call_produces_nested_spans(
        self, span_exporter, instrument
    ):
        """When an agent run internally calls a tool, spans should be nested."""
        agent = _StubAgent.create(name="ToolNestBot")

        mock_tool = MagicMock()
        mock_tool.description = "Calculator tool"
        mock_tool.call = MagicMock(return_value="42")
        agent.function_map = {"calculator": mock_tool}

        def fake_run(messages, **kwargs):
            # Simulate the agent calling a tool internally
            agent._call_tool("calculator", '{"expr": "6*7"}')
            yield [Message(role="assistant", content="The result is 42.")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="Calculate 6*7")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        assert len(agent_spans) >= 1
        assert len(tool_spans) >= 1

        agent_span = agent_spans[0]
        tool_span = tool_spans[0]
        assert tool_span.context.trace_id == agent_span.context.trace_id
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == agent_span.context.span_id


# ---------------------------------------------------------------------------
# ReAct Step span tests
# ---------------------------------------------------------------------------


class TestReactStepSpan:
    """Verify that react_step spans are created for ReAct agents with tools,
    and NOT created for agents without tools."""

    def test_react_agent_with_tools_creates_react_step_spans(
        self, span_exporter, instrument
    ):
        """FnCallAgent-like agent with tools: each _call_llm round should
        produce a react_step span."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        agent = _StubAgent.create(name="ReactBot", llm=model)

        # Give the agent a tool -> react mode ON
        mock_tool = MagicMock()
        mock_tool.description = "Calculator"
        mock_tool.call = MagicMock(return_value="42")
        agent.function_map = {"calculator": mock_tool}

        # Simulate 2-round ReAct: LLM -> tool -> LLM -> done
        call_count = [0]

        def fake_run(messages, **kwargs):
            # Round 1: LLM decides to call tool
            call_count[0] += 1
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=[
                    Message(
                        role="assistant",
                        content="",
                        function_call=FunctionCall(
                            name="calculator",
                            arguments='{"expr": "6*7"}',
                        ),
                    )
                ],
            ):
                # This calls _call_llm internally -> react_step 1
                agent._call_llm(messages=messages, functions=[], stream=False)

            # Tool call (inside react_step 1 context)
            agent._call_tool("calculator", '{"expr": "6*7"}')

            # Round 2: LLM summarizes (react_step 2)
            call_count[0] += 1
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=[
                    Message(role="assistant", content="The result is 42.")
                ],
            ):
                agent._call_llm(messages=messages, functions=[], stream=False)

            yield [Message(role="assistant", content="The result is 42.")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="Calculate 6*7")]))

        spans = span_exporter.get_finished_spans()
        react_spans = [s for s in spans if "react step" in s.name]
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        # Should have exactly 2 react_step spans
        assert len(react_spans) == 2, (
            f"Expected 2 react_step spans, got {len(react_spans)}: "
            f"{[s.name for s in spans]}"
        )

        # react_step spans should be children of invoke_agent
        agent_span = agent_spans[0]
        for rs in react_spans:
            assert rs.parent is not None
            assert rs.parent.span_id == agent_span.context.span_id

        # chat spans should be children of react_step (not directly of invoke_agent)
        for cs in chat_spans:
            assert cs.parent is not None
            parent_is_react = any(
                cs.parent.span_id == rs.context.span_id for rs in react_spans
            )
            assert parent_is_react, (
                f"chat span parent should be a react_step, "
                f"not {cs.parent.span_id}"
            )

        # tool span should be child of react_step 1 (the first one)
        assert len(tool_spans) >= 1
        tool_span = tool_spans[0]
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == react_spans[0].context.span_id

    def test_agent_without_tools_no_react_step_spans(
        self, span_exporter, instrument
    ):
        """Agent WITHOUT tools: no react_step spans should be created,
        even though _call_llm is wrapped."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        agent = _StubAgent.create(name="SimpleBot", llm=model)
        agent.function_map = {}  # No tools -> react mode OFF

        def fake_run(messages, **kwargs):
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=[Message(role="assistant", content="Hello!")],
            ):
                agent._call_llm(messages=messages, stream=False)
            yield [Message(role="assistant", content="Hello!")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="Hi")]))

        spans = span_exporter.get_finished_spans()
        react_spans = [s for s in spans if "react step" in s.name]
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        chat_spans = [s for s in spans if s.name.startswith("chat")]

        # NO react_step spans
        assert len(react_spans) == 0, (
            f"Expected 0 react_step spans for no-tool agent, "
            f"got {len(react_spans)}: {[s.name for s in spans]}"
        )

        # invoke_agent and chat should still work normally
        assert len(agent_spans) >= 1
        assert len(chat_spans) >= 1

        # chat should be direct child of invoke_agent (no react_step in between)
        chat_span = chat_spans[0]
        assert chat_span.parent is not None
        assert chat_span.parent.span_id == agent_spans[0].context.span_id

    def test_single_round_react_agent_creates_one_step(
        self, span_exporter, instrument
    ):
        """Agent WITH tools but LLM answers directly (no tool call):
        should create exactly 1 react_step span (step=1)."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        agent = _StubAgent.create(name="DirectBot", llm=model)

        mock_tool = MagicMock()
        mock_tool.description = "Unused tool"
        agent.function_map = {"unused_tool": mock_tool}

        def fake_run(messages, **kwargs):
            # LLM answers directly, no tool call
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=[
                    Message(role="assistant", content="I know the answer: 42.")
                ],
            ):
                agent._call_llm(messages=messages, functions=[], stream=False)
            yield [Message(role="assistant", content="I know the answer: 42.")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="What is 42?")]))

        spans = span_exporter.get_finished_spans()
        react_spans = [s for s in spans if "react step" in s.name]

        # Exactly 1 react_step (agent had tools, so react mode was on)
        assert len(react_spans) == 1

    def test_react_step_does_not_leak_across_runs(
        self, span_exporter, instrument
    ):
        """Running an agent WITH tools, then an agent WITHOUT tools:
        the second run should NOT have react_step spans."""
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")

        # Run 1: agent with tools
        agent1 = _StubAgent.create(name="ToolBot", llm=model)
        mock_tool = MagicMock()
        mock_tool.description = "Tool"
        mock_tool.call = MagicMock(return_value="ok")
        agent1.function_map = {"tool1": mock_tool}

        def fake_run_1(messages, **kwargs):
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=[Message(role="assistant", content="Done")],
            ):
                agent1._call_llm(messages=messages, functions=[], stream=False)
            yield [Message(role="assistant", content="Done")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run_1):
            list(agent1.run([Message(role="user", content="Go")]))

        spans_after_run1 = span_exporter.get_finished_spans()
        react_spans_1 = [s for s in spans_after_run1 if "react step" in s.name]
        assert len(react_spans_1) == 1  # Tool agent had react_step

        span_exporter.clear()

        # Run 2: agent WITHOUT tools
        agent2 = _StubAgent.create(name="PlainBot", llm=model)
        agent2.function_map = {}

        def fake_run_2(messages, **kwargs):
            with patch.object(
                _StubChatModel,
                "_chat_no_stream",
                return_value=[Message(role="assistant", content="Hi")],
            ):
                agent2._call_llm(messages=messages, stream=False)
            yield [Message(role="assistant", content="Hi")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run_2):
            list(agent2.run([Message(role="user", content="Hello")]))

        spans_after_run2 = span_exporter.get_finished_spans()
        react_spans_2 = [s for s in spans_after_run2 if "react step" in s.name]
        assert len(react_spans_2) == 0  # No tool -> no react_step
