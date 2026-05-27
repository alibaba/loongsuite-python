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
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "span_only"

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

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.instrumentation.crewai import (
    GenAIHookHelper,
    _FlowKickoffAsyncWrapper,
)
from opentelemetry.instrumentation.crewai.utils import gen_ai_json_dumps
from opentelemetry.sdk.trace import TracerProvider

# Use SDK tracer for testing
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode


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
        self.tracer = self.tracer_provider.get_tracer(__name__)

        # Create wrapper instance
        self.helper = GenAIHookHelper()
        self.wrapper = _FlowKickoffAsyncWrapper(self.tracer, self.helper)

    def tearDown(self):
        """Cleanup test resources."""
        self.memory_exporter.clear()

    def test_wrapper_init(self):
        """Test wrapper initialization."""
        wrapper = _FlowKickoffAsyncWrapper(self.tracer, self.helper)
        self.assertEqual(wrapper._tracer, self.tracer)
        self.assertEqual(wrapper._helper, self.helper)

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
        self.assertEqual(span.name, "test_flow")
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "crew.kickoff"
        )
        self.assertEqual(span.attributes.get("gen_ai.system"), "crewai")

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
        self.assertEqual(span.name, "flow.kickoff")
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "crew.kickoff"
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
        self.assertEqual(span.name, "error_flow")
        self.assertEqual(span.attributes.get("gen_ai.system"), "crewai")

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
        self.assertEqual(span.name, "flow.kickoff")
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "crew.kickoff"
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
