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

"""
Test cases for _FlowKickoffAsyncWrapper in CrewAI instrumentation.

This test suite validates the _FlowKickoffAsyncWrapper functionality including:
- CHAIN span creation for flow workflows
- Proper attribute setting (gen_ai.system, gen_ai.operation.name, gen_ai.input/output.messages)
- Error handling and exception recording
- Flow name extraction from instance
"""

import os

# Set environment variables for content capture
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_ONLY"

# Forcefully enable experimental mode in OpenTelemetry's internal mapping
try:
    from opentelemetry.instrumentation._semconv import (
        _OpenTelemetrySemanticConventionStability,
        _OpenTelemetryStabilitySignalType,
        _StabilityMode,
    )

    _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING[
        _OpenTelemetryStabilitySignalType.GEN_AI
    ] = _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL
except (ImportError, AttributeError):
    pass

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry.instrumentation.crewai import (
    _AgentExecuteTaskWrapper,
    _CrewKickoffAsyncWrapper,
    _CrewKickoffWrapper,
    _FlowKickoffAsyncWrapper,
    _FlowKickoffWrapper,
    _TaskExecuteSyncWrapper,
    _ToolUseWrapper,
)
from opentelemetry.instrumentation.crewai.utils import (
    extract_tool_inputs,
    gen_ai_json_dumps,
)
from opentelemetry.sdk.trace import TracerProvider

# Use SDK tracer for testing
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


class TestFlowKickoffAsyncWrapper(unittest.IsolatedAsyncioTestCase):
    """Test _FlowKickoffAsyncWrapper class."""

    def setUp(self):
        """Setup test resources."""
        # Create tracer provider with in-memory exporter
        self.memory_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.memory_exporter)
        )
        # Create wrapper instance
        self.handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider
        )
        self.wrapper = _FlowKickoffAsyncWrapper(self.handler)

    def tearDown(self):
        """Cleanup test resources."""
        self.memory_exporter.clear()

    def test_wrapper_init(self):
        """Test wrapper initialization."""
        wrapper = _FlowKickoffAsyncWrapper(self.handler)
        self.assertEqual(wrapper._handler, self.handler)

    async def test_basic_flow_kickoff(self):
        """
        Test basic flow kickoff creates CHAIN span with correct attributes.

        Verification:
        - CHAIN span is created
        - gen_ai.system = "crewai"
        - gen_ai.operation.name is set to flow name
        - gen_ai.input.messages and gen_ai.output.messages are captured (JSON)
        - Status is OK
        """
        # Create mock wrapped function
        mock_wrapped = AsyncMock(return_value="flow result")

        # Create mock flow instance with name
        mock_instance = MagicMock()
        mock_instance.name = "test_flow"

        # Call wrapper
        result = await self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify wrapped function was called
        mock_wrapped.assert_called_once_with()
        self.assertEqual(result, "flow result")

        # Verify span was created
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "enter_ai_application_system")
        self.assertEqual(span.attributes.get("gen_ai.operation.name"), "enter")
        self.assertEqual(
            span.attributes.get("gen_ai.crewai.operation"), "flow.kickoff"
        )
        self.assertEqual(span.attributes.get("gen_ai.agent.name"), "test_flow")
        self.assertEqual(span.attributes.get("gen_ai.provider.name"), "crewai")

        output_messages_json = span.attributes.get("gen_ai.output.messages")
        self.assertIsNotNone(output_messages_json)
        output_messages = json.loads(output_messages_json)
        self.assertGreater(len(output_messages), 0)
        self.assertIn("flow result", output_messages[0]["parts"][0]["content"])
        self.assertEqual(span.status.status_code, StatusCode.OK)

    async def test_flow_kickoff_without_name(self):
        """
        Test flow kickoff when instance has no name attribute.

        Verification:
        - Uses default name "flow.kickoff"
        - Span is created with default name
        """
        # Create mock wrapped function
        mock_wrapped = AsyncMock(return_value="result")

        # Create mock flow instance without name
        mock_instance = MagicMock(spec=[])  # No name attribute

        # Call wrapper
        await self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify span was created with default name
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "enter_ai_application_system")
        self.assertEqual(span.attributes.get("gen_ai.operation.name"), "enter")
        self.assertEqual(
            span.attributes.get("gen_ai.crewai.operation"), "flow.kickoff"
        )

    async def test_flow_kickoff_with_inputs(self):
        """
        Test flow kickoff with input parameters.

        Verification:
        - Inputs are captured in input.value attribute
        - Inputs are properly serialized to JSON
        """
        # Create mock wrapped function
        mock_wrapped = AsyncMock(return_value="processed result")

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "input_flow"

        # Call wrapper with inputs
        inputs = {"query": "test query", "count": 10}
        await self.wrapper(mock_wrapped, mock_instance, (), {"inputs": inputs})

        # Verify wrapped function was called with correct kwargs
        mock_wrapped.assert_called_once_with(inputs=inputs)

        # Verify span captures inputs
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]

        input_messages_json = span.attributes.get("gen_ai.input.messages")
        self.assertIsNotNone(input_messages_json)
        input_messages = json.loads(input_messages_json)
        self.assertGreater(len(input_messages), 0)
        content = input_messages[0]["parts"][0]["content"]
        self.assertIn("test query", content)
        self.assertIn("10", content)

    async def test_flow_kickoff_with_args(self):
        """
        Test flow kickoff with positional arguments.

        Verification:
        - Args are passed to wrapped function
        """
        # Create mock wrapped function
        mock_wrapped = AsyncMock(return_value="result with args")

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "args_flow"

        # Call wrapper with args
        result = await self.wrapper(
            mock_wrapped, mock_instance, ("arg1", "arg2"), {}
        )

        # Verify wrapped function was called with args
        mock_wrapped.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "result with args")

    async def test_flow_kickoff_exception_handling(self):
        """
        Test flow kickoff exception handling.

        Verification:
        - Exception is recorded in span
        - Exception is re-raised
        - Span still has CHAIN kind
        """
        # Create mock wrapped function that raises exception
        test_exception = ValueError("Test error in flow")
        mock_wrapped = AsyncMock(side_effect=test_exception)

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "error_flow"

        # Call wrapper and expect exception
        with self.assertRaises(ValueError) as context:
            await self.wrapper(mock_wrapped, mock_instance, (), {})

        self.assertEqual(str(context.exception), "Test error in flow")

        # Verify span was created with exception recorded
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "enter_ai_application_system")
        self.assertEqual(span.attributes.get("gen_ai.provider.name"), "crewai")

        # Verify exception was recorded in events
        self.assertGreater(len(span.events), 0)
        exception_event = span.events[0]
        self.assertEqual(exception_event.name, "exception")

    async def test_flow_kickoff_with_none_name(self):
        """
        Test flow kickoff when instance.name is None.

        Verification:
        - Uses default name "flow.kickoff" when name is None
        """
        # Create mock wrapped function
        mock_wrapped = AsyncMock(return_value="result")

        # Create mock flow instance with None name
        mock_instance = MagicMock()
        mock_instance.name = None

        # Call wrapper
        await self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify span was created with default name
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "enter_ai_application_system")
        self.assertEqual(span.attributes.get("gen_ai.operation.name"), "enter")
        self.assertEqual(
            span.attributes.get("gen_ai.crewai.operation"), "flow.kickoff"
        )

    async def test_flow_kickoff_with_complex_result(self):
        """
        Test flow kickoff with complex result object.

        Verification:
        - Complex result is properly serialized
        - output.value contains serialized result
        """
        # Create mock wrapped function with complex result
        complex_result = {
            "status": "success",
            "data": {"items": [1, 2, 3]},
            "message": "Flow completed",
        }
        mock_wrapped = AsyncMock(return_value=complex_result)

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "complex_flow"

        # Call wrapper
        result = await self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify result is returned correctly
        self.assertEqual(result, complex_result)

        # Verify span output contains serialized result
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]

        output_messages_json = span.attributes.get("gen_ai.output.messages")
        self.assertIsNotNone(output_messages_json)
        output_messages = json.loads(output_messages_json)
        content = output_messages[0]["parts"][0]["content"]
        self.assertIn("success", content)
        self.assertIn("Flow completed", content)

    async def test_flow_kickoff_with_none_result(self):
        """
        Test flow kickoff when wrapped function returns None.

        Verification:
        - None result is handled gracefully
        - output.value is set appropriately
        """
        # Create mock wrapped function returning None
        mock_wrapped = AsyncMock(return_value=None)

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "none_result_flow"

        # Call wrapper
        result = await self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify result is None
        self.assertIsNone(result)

        # Verify span was created
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.status.status_code, StatusCode.OK)

    async def test_flow_kickoff_span_kind(self):
        """
        Test that flow kickoff span has correct SpanKind.

        Verification:
        - Span kind is INTERNAL
        """
        # Create mock wrapped function
        mock_wrapped = AsyncMock(return_value="result")

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "kind_test_flow"

        # Call wrapper
        await self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify span kind
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.kind, SpanKind.INTERNAL)


class TestCrewKickoffWrapper(unittest.TestCase):
    """Test _CrewKickoffWrapper class."""

    def setUp(self):
        """Setup test resources."""
        self.memory_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.memory_exporter)
        )
        self.handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider
        )
        self.wrapper = _CrewKickoffWrapper(self.handler)

    def tearDown(self):
        """Cleanup test resources."""
        self.memory_exporter.clear()

    def test_crew_kickoff_records_token_usage(self):
        """Test Crew.kickoff maps CrewAI token_usage onto the entry span."""
        result = SimpleNamespace(
            raw="final crew result",
            token_usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=7,
                total_tokens=19,
                successful_requests=1,
            ),
        )
        mock_wrapped = MagicMock(return_value=result)
        crew = SimpleNamespace(
            name="usage_crew",
            process=SimpleNamespace(value="sequential"),
            tasks=[object()],
            agents=[object()],
        )

        returned = self.wrapper(
            mock_wrapped,
            crew,
            (),
            {"inputs": {"session_id": "sess-1", "user_id": "user-1"}},
        )

        self.assertIs(returned, result)
        mock_wrapped.assert_called_once_with(
            inputs={"session_id": "sess-1", "user_id": "user-1"}
        )

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.name, "enter_ai_application_system")
        self.assertEqual(span.attributes["gen_ai.operation.name"], "enter")
        self.assertEqual(
            span.attributes["gen_ai.crewai.operation"], "crew.kickoff"
        )
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 12)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 7)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 19)
        self.assertEqual(
            span.attributes["gen_ai.crewai.usage.successful_requests"], 1
        )
        self.assertEqual(span.status.status_code, StatusCode.OK)

    def test_crew_kickoff_uses_next_nonzero_token_usage(self):
        """Test token usage falls back when the first candidate is empty."""
        result = SimpleNamespace(
            raw="final crew result",
            token_usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )
        crew = SimpleNamespace(
            name="usage_crew",
            process=SimpleNamespace(value="sequential"),
            tasks=[object()],
            agents=[object()],
            token_usage=SimpleNamespace(
                prompt_tokens=21,
                completion_tokens=9,
                total_tokens=30,
            ),
        )

        self.wrapper(MagicMock(return_value=result), crew, (), {})

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 21)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 9)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 30)

    def test_crew_kickoff_records_zero_token_usage(self):
        """Test zero token usage is preserved when no nonzero candidate exists."""
        result = SimpleNamespace(
            raw="final crew result",
            token_usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                successful_requests=1,
            ),
        )
        crew = SimpleNamespace(name="usage_crew", tasks=[], agents=[])

        self.wrapper(MagicMock(return_value=result), crew, (), {})

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 0)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 0)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 0)
        self.assertEqual(
            span.attributes["gen_ai.crewai.usage.successful_requests"], 1
        )

    def test_streaming_crew_kickoff_defers_to_inner_execution(self):
        """Test stream=True kickoff traces CrewAI's inner real execution."""
        crew = SimpleNamespace(
            name="streaming_crew",
            stream=True,
            tasks=[],
            agents=[],
        )

        def wrapped_stream(*args, **kwargs):
            crew.stream = False
            return self.wrapper(
                MagicMock(return_value="inner result"), crew, args, kwargs
            )

        result = self.wrapper(wrapped_stream, crew, (), {})

        self.assertEqual(result, "inner result")
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes["gen_ai.crewai.operation"], "crew.kickoff"
        )

    def test_success_post_processing_failure_does_not_escape(self):
        """Test telemetry post-processing failures do not fail user calls."""
        crew = SimpleNamespace(name="safe_crew", tasks=[], agents=[])

        with patch(
            "opentelemetry.instrumentation.crewai.to_output_messages",
            side_effect=RuntimeError("post boom"),
        ):
            result = self.wrapper(MagicMock(return_value="ok"), crew, (), {})

        self.assertEqual(result, "ok")
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertNotEqual(spans[0].status.status_code, StatusCode.ERROR)

    def test_entry_invocation_build_failure_preserves_user_call(self):
        """Test invocation construction failures do not block user code."""
        wrapped = MagicMock(return_value="ok")
        crew = SimpleNamespace(name="safe_crew", tasks=[], agents=[])

        with patch(
            "opentelemetry.instrumentation.crewai.create_entry_invocation",
            side_effect=RuntimeError("build boom"),
        ):
            result = self.wrapper(wrapped, crew, (), {})

        self.assertEqual(result, "ok")
        wrapped.assert_called_once_with()
        self.assertEqual(len(self.memory_exporter.get_finished_spans()), 0)


class TestEntryNestingGuards(unittest.IsolatedAsyncioTestCase):
    """Test nested CrewAI entry wrappers emit only one entry span."""

    def setUp(self):
        self.memory_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.memory_exporter)
        )
        self.handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider
        )

    async def test_crew_kickoff_async_does_not_double_wrap_sync_kickoff(self):
        """Test kickoff_async -> kickoff produces one entry span."""
        sync_wrapper = _CrewKickoffWrapper(self.handler)
        async_wrapper = _CrewKickoffAsyncWrapper(self.handler)

        async def wrapped_async(*args, **kwargs):
            return sync_wrapper(
                MagicMock(return_value="inner result"),
                SimpleNamespace(name="inner_crew", stream=False),
                args,
                kwargs,
            )

        result = await async_wrapper(
            wrapped_async,
            SimpleNamespace(name="outer_crew", stream=False),
            (),
            {"inputs": {"session_id": "nested-session"}},
        )

        self.assertEqual(result, "inner result")
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes["gen_ai.agent.name"], "outer_crew"
        )
        self.assertEqual(
            spans[0].attributes["gen_ai.crewai.operation"], "crew.kickoff"
        )

    def test_flow_kickoff_sync_does_not_double_wrap_async_kickoff(self):
        """Test Flow.kickoff -> kickoff_async produces one entry span."""
        sync_wrapper = _FlowKickoffWrapper(self.handler)
        async_wrapper = _FlowKickoffAsyncWrapper(self.handler)

        def wrapped_sync(*args, **kwargs):
            async def run_inner():
                return await async_wrapper(
                    AsyncMock(return_value="flow result"),
                    SimpleNamespace(name="inner_flow", stream=False),
                    args,
                    kwargs,
                )

            return asyncio.run(run_inner())

        result = sync_wrapper(
            wrapped_sync,
            SimpleNamespace(name="outer_flow", stream=False),
            (),
            {"inputs": {"session_id": "flow-session"}},
        )

        self.assertEqual(result, "flow result")
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes["gen_ai.agent.name"], "outer_flow"
        )
        self.assertEqual(
            spans[0].attributes["gen_ai.crewai.operation"], "flow.kickoff"
        )


class TestTaskAndToolWrappers(unittest.TestCase):
    """Test task and tool wrapper compatibility behavior."""

    def setUp(self):
        self.memory_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.memory_exporter)
        )
        self.handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider
        )

    def test_task_execute_uses_positional_agent_role_as_agent_name(self):
        """Test Task.execute_sync(agent=...) uses the runtime agent role."""
        wrapper = _TaskExecuteSyncWrapper(self.handler)
        token_process = SimpleNamespace(
            get_summary=MagicMock(
                return_value=SimpleNamespace(
                    prompt_tokens=17,
                    completion_tokens=5,
                    total_tokens=22,
                    successful_requests=1,
                )
            )
        )
        agent = SimpleNamespace(role="Runtime Agent", llm="qwen-turbo")
        agent._token_process = token_process
        task = SimpleNamespace(
            description="Write a short summary.",
            expected_output="A short summary.",
            agent=None,
        )

        wrapper(MagicMock(return_value="done"), task, (agent,), {})

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["gen_ai.agent.name"], "Runtime Agent")
        self.assertEqual(
            span.attributes["gen_ai.crewai.operation"], "task.execute"
        )
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 17)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 5)
        self.assertEqual(span.attributes["gen_ai.usage.total_tokens"], 22)

    def test_agent_execute_preserves_user_call_when_handler_start_fails(self):
        """Test instrumentation failures do not block user code."""
        handler = SimpleNamespace(
            start_invoke_agent=MagicMock(side_effect=RuntimeError("boom")),
            stop_invoke_agent=MagicMock(),
            fail_invoke_agent=MagicMock(),
        )
        wrapper = _AgentExecuteTaskWrapper(handler)
        wrapped = MagicMock(return_value="agent result")
        agent = SimpleNamespace(role="Runtime Agent", goal="Help", tools=[])
        task = SimpleNamespace(description="Write a short summary.")

        result = wrapper(wrapped, agent, (task,), {})

        self.assertEqual(result, "agent result")
        wrapped.assert_called_once_with(task)
        handler.stop_invoke_agent.assert_not_called()
        handler.fail_invoke_agent.assert_not_called()

    def test_agent_execute_skips_inside_task_span(self):
        """Test task execution does not create a duplicate agent span."""
        task_wrapper = _TaskExecuteSyncWrapper(self.handler)
        agent_wrapper = _AgentExecuteTaskWrapper(self.handler)
        agent = SimpleNamespace(role="Runtime Agent", llm="qwen-turbo")
        task = SimpleNamespace(
            description="Write a short summary.",
            expected_output="A short summary.",
            agent=agent,
        )

        def wrapped_task(*args, **kwargs):
            return agent_wrapper(
                MagicMock(return_value="agent result"),
                agent,
                (task,),
                {},
            )

        result = task_wrapper(wrapped_task, task, (agent,), {})

        self.assertEqual(result, "agent result")
        spans = self.memory_exporter.get_finished_spans()
        task_spans = [
            span
            for span in spans
            if span.attributes.get("gen_ai.crewai.operation") == "task.execute"
        ]
        agent_spans = [
            span
            for span in spans
            if span.attributes.get("gen_ai.crewai.operation")
            == "agent.execute"
        ]
        self.assertEqual(len(task_spans), 1)
        self.assertEqual(len(agent_spans), 0)

    def test_content_capture_disabled_drops_sensitive_task_content(self):
        """Test content-like fields honor util-genai capture controls."""
        wrapper = _TaskExecuteSyncWrapper(self.handler)
        agent = SimpleNamespace(role="Runtime Agent", llm="qwen-turbo")
        task = SimpleNamespace(
            description="PII: customer id 123",
            expected_output="PII: private answer",
            agent=agent,
        )

        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT"
            },
        ):
            wrapper(MagicMock(return_value="PII: result"), task, (agent,), {})

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertNotIn("gen_ai.input.messages", span.attributes)
        self.assertNotIn("gen_ai.output.messages", span.attributes)
        self.assertNotIn("gen_ai.agent.description", span.attributes)
        self.assertNotIn("gen_ai.crewai.task.description", span.attributes)
        self.assertNotIn("gen_ai.crewai.task.expected_output", span.attributes)

    def test_tool_wrapper_marks_parsing_attempt_error(self):
        """Test parsing-attempt overflow records a failed tool span."""
        wrapper = _ToolUseWrapper(self.handler)
        tool = SimpleNamespace(
            name="failing_tool", description="A deterministic failing tool."
        )
        tool_usage = SimpleNamespace(_run_attempts=3, _max_parsing_attempts=2)

        result = wrapper(
            MagicMock(return_value="partial result"),
            tool_usage,
            (tool, SimpleNamespace(arguments={"x": "y"})),
            {},
        )

        self.assertEqual(result, "partial result")
        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertEqual(
            span.attributes["gen_ai.crewai.operation"], "tool.execute"
        )
        self.assertIn(
            "failing_tool", span.events[0].attributes["exception.message"]
        )

    def test_tool_wrapper_reads_crewai_use_signature_arguments(self):
        """Test ToolUsage._use(tool_string, tool, calling) captures arguments."""
        wrapper = _ToolUseWrapper(self.handler)
        tool = SimpleNamespace(
            name="word_count", description="A deterministic tool."
        )
        calling = SimpleNamespace(
            id="call-1",
            tool_name="word_count",
            arguments={"text": "CrewAI telemetry"},
        )

        wrapper(
            MagicMock(return_value="word_count=2"),
            SimpleNamespace(),
            ("word_count({})", tool, calling),
            {},
        )

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["gen_ai.tool.name"], "word_count")
        self.assertIn(
            "CrewAI telemetry",
            span.attributes["gen_ai.tool.call.arguments"],
        )

    def test_extract_tool_inputs_keeps_json_string_arguments(self):
        """Test legacy helper keeps JSON-stringified tool arguments."""
        messages = extract_tool_inputs("tool_name", {"location": "Hangzhou"})

        tool_call = messages[0].parts[0]
        self.assertEqual(tool_call.name, "tool_name")
        self.assertEqual(tool_call.arguments, '{"location":"Hangzhou"}')


class TestGenAiJsonDumps(unittest.TestCase):
    """Test gen_ai_json_dumps utility function."""

    def test_simple_string(self):
        """Test with simple string input."""
        result = gen_ai_json_dumps("hello")
        self.assertEqual(result, '"hello"')

    def test_simple_int(self):
        """Test with integer input."""
        result = gen_ai_json_dumps(42)
        self.assertEqual(result, "42")

    def test_simple_float(self):
        """Test with float input."""
        result = gen_ai_json_dumps(3.14)
        self.assertEqual(result, "3.14")

    def test_simple_bool(self):
        """Test with boolean input."""
        result = gen_ai_json_dumps(True)
        self.assertEqual(result, "true")

    def test_dict_serialization(self):
        """Test dictionary serialization."""
        data = {"key": "value", "number": 123}
        result = gen_ai_json_dumps(data)
        self.assertIn('"key":"value"', result)
        self.assertIn('"number":123', result)

    def test_list_serialization(self):
        """Test list serialization."""
        data = [1, 2, 3, "four"]
        result = gen_ai_json_dumps(data)
        self.assertEqual(result, '[1,2,3,"four"]')

    def test_non_serializable_object(self):
        """Test handling of non-serializable objects."""

        class CustomObject:
            def __str__(self):
                return "CustomObject instance"

        # gen_ai_json_dumps uses standard json or partial json.dump
        # which might raise TypeError if not supported.
        # But our partial uses _GenAiJsonEncoder.
        with self.assertRaises(TypeError):
            gen_ai_json_dumps(CustomObject())


if __name__ == "__main__":
    unittest.main()
