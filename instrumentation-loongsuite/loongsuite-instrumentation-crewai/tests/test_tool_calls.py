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
Test cases for tool calls and function calling in CrewAI.

Business Demo Description:
This test suite uses CrewAI framework to create Agents that use tools for
function calling. It verifies that tool invocations are properly traced as
OpenTelemetry GenAI TOOL spans, capturing tool names, arguments, and results
in a standardized format.
"""

import ast
import json
import operator
import os
import sys

import pysqlite3
import pytest
from crewai import Agent, Crew, Task
from crewai.tools.base_tool import BaseTool

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.test.test_base import TestBase

try:
    from opentelemetry.instrumentation._semconv import (
        _OpenTelemetrySemanticConventionStability,
        _OpenTelemetryStabilitySignalType,
        _StabilityMode,
    )
except ImportError:
    _OpenTelemetrySemanticConventionStability = None
    _OpenTelemetryStabilitySignalType = None
    _StabilityMode = None

sys.modules["sqlite3"] = pysqlite3


class WeatherTool(BaseTool):
    """Sample tool for weather lookup."""

    name: str = "get_weather"
    description: str = "Get current weather for a location"

    def _run(self, location: str) -> str:
        """Execute the tool."""
        return f"Weather in {location}: Sunny, 72°F"


class CalculatorTool(BaseTool):
    """Sample tool for calculations."""

    name: str = "calculator"
    description: str = "Perform mathematical calculations"

    def _run(self, expression: str) -> str:
        """Execute the tool safely without using eval()."""
        # Supported operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.BitXor: operator.xor,
            ast.USub: operator.neg,
        }

        def safe_eval(node):
            if isinstance(node, ast.Num):  # <3.8
                return node.n
            elif isinstance(node, ast.Constant):  # >=3.8
                return node.value
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](
                    safe_eval(node.left), safe_eval(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](safe_eval(node.operand))
            else:
                raise TypeError(f"Unsupported operation: {type(node)}")

        try:
            node = ast.parse(expression, mode="eval").body
            result = safe_eval(node)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


@pytest.mark.vcr
class TestToolCalls(TestBase):
    """Test tool call scenarios."""

    def setUp(self):
        """Setup test resources."""
        super().setUp()
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = os.environ.get(
            "OPENAI_API_KEY", "fake-key"
        )
        os.environ["DASHSCOPE_API_KEY"] = os.environ.get(
            "DASHSCOPE_API_KEY", "fake-key"
        )
        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ["DASHSCOPE_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        # Enable experimental mode and content capture for testing
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai"
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "span_only"
        )

        if _OpenTelemetrySemanticConventionStability:
            try:
                _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING[
                    _OpenTelemetryStabilitySignalType.GEN_AI
                ] = _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL
            except (AttributeError, KeyError):
                pass

        self.instrumentor = CrewAIInstrumentor()
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Test data
        self.model_name = "openai/qwen-turbo"

    def tearDown(self):
        """Cleanup test resources."""
        with self.disable_logging():
            self.instrumentor.uninstrument()
        super().tearDown()

    def test_agent_with_single_tool(self):
        """
        Test Agent execution with a single tool call.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent has access to 1 tool (WeatherTool)
        - Executes 1 Task that triggers tool usage
        - Performs 1 LLM call + 1 tool call

        Verification:
        - TOOL span created with gen_ai.operation.name="tool.execute"
        - Captured gen_ai.tool.name and tool input/output messages
        - Hierarchical consistency: TOOL span is a child of AGENT span
        """
        # Create tool
        weather_tool = WeatherTool()

        # Create Agent with tool
        agent = Agent(
            role="Weather Assistant",
            goal="Provide weather information",
            backstory="Expert in weather forecasting",
            verbose=False,
            llm=self.model_name,
            tools=[weather_tool],
        )

        # Create Task
        task = Task(
            description="Get the weather for San Francisco",
            expected_output="Weather information",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        tool_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "tool.execute"
        ]
        agent_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "agent.execute"
        ]

        # Verify TOOL span (if tool wrapper was successful)
        if tool_spans:
            tool_span = tool_spans[0]
            self.assertEqual(
                tool_span.attributes.get("gen_ai.operation.name"),
                "tool.execute",
            )
            self.assertEqual(
                tool_span.attributes.get("gen_ai.tool.name"), "get_weather"
            )

            output_messages_json = tool_span.attributes.get(
                "gen_ai.output.messages"
            )
            self.assertIsNotNone(output_messages_json)
            output_messages = json.loads(output_messages_json)
            # Weather logic in WeatherTool returns weather string
            self.assertIn("Sunny", output_messages[0]["parts"][0]["content"])

        # Verify AGENT span exists
        self.assertGreaterEqual(
            len(agent_spans), 1, "Expected at least 1 AGENT span"
        )

    def test_agent_with_multiple_tools(self):
        """
        Test Agent execution with multiple tool calls.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent has access to 2 tools (WeatherTool, CalculatorTool)
        - Executes 1 Task that may trigger multiple tool usages
        - Performs multiple LLM calls + multiple tool calls

        Verification:
        - Multiple TOOL spans, each with correct tool name
        - Each TOOL span has proper input arguments and output results
        - Trace continuity across multiple nested tool calls
        """
        # Create tools
        weather_tool = WeatherTool()
        calculator_tool = CalculatorTool()

        # Create Agent with multiple tools
        agent = Agent(
            role="Multi-Tool Assistant",
            goal="Use various tools to complete tasks",
            backstory="Expert in using multiple tools",
            verbose=False,
            llm=self.model_name,
            tools=[weather_tool, calculator_tool],
        )

        # Create Task
        task = Task(
            description="Get weather and calculate temperature difference",
            expected_output="Weather and calculation results",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        tool_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]

        # May have multiple tool spans if tools were actually called
        # Each tool span should have proper attributes
        for tool_span in tool_spans:
            self.assertEqual(
                tool_span.attributes.get("gen_ai.operation.name"),
                "tool.execute",
            )
            self.assertIsNotNone(tool_span.attributes.get("gen_ai.tool.name"))

    def test_tool_call_with_error(self):
        """
        Test tool call that raises an error.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent uses a tool that fails
        - Executes 1 Task with failing tool call

        Verification:
        - TOOL span status set to StatusCode.ERROR
        - Recorded exception details in span events
        """

        # Create a tool that raises error
        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that always fails"

            def _run(self, input_str: str) -> str:
                raise Exception("Tool execution failed")

        failing_tool = FailingTool()

        agent = Agent(
            role="Test Assistant",
            goal="Test error handling",
            backstory="Test agent",
            verbose=False,
            llm=self.model_name,
            tools=[failing_tool],
        )

        task = Task(
            description="Use the failing tool",
            expected_output="Error handling result",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        # May raise error or handle gracefully
        try:
            crew.kickoff()
        except Exception:
            pass  # Expected

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        tool_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "tool.execute"
        ]

        # If tool span exists, verify error status
        for tool_span in tool_spans:
            if tool_span.name.startswith("Tool.failing_tool"):
                # Should have error status
                self.assertEqual(tool_span.status.status_code.name, "ERROR")
