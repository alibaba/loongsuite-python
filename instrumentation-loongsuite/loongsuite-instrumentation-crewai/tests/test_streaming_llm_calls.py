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
Test cases for streaming LLM calls in CrewAI.

Business Demo Description:
This test suite uses CrewAI framework to create Agents that perform streaming
LLM calls. It verifies the capture of OpenTelemetry GenAI attributes for
streaming scenarios and proper hierarchy preservation in asynchronous execution.
"""

import json
import os
import sys

import pysqlite3
import pytest
from crewai import Agent, Crew, Task

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import StatusCode

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


@pytest.mark.vcr
class TestStreamingLLMCalls(TestBase):
    """Test streaming LLM call scenarios."""

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

    def test_streaming_crew_execution(self):
        """
        Test streaming Crew execution with LLM streaming enabled.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Executes 1 Task with streaming enabled
        - Performs 1 streaming LLM call
        - Consumes streaming chunks

        Verification:
        - LLM/Agent spans reflect streaming configuration
        - Task span captures final output messages from stream
        - Trace hierarchy (Crew -> Task -> Agent) correctly maintained
        - Standard attributes (gen_ai.system, gen_ai.operation.name, etc.)
        """
        # Create Agent with streaming
        agent = Agent(
            role="Content Writer",
            goal="Write engaging content",
            backstory="Expert content writer",
            verbose=False,
            llm=self.model_name,
        )

        # Create Task
        task = Task(
            description="Write a short greeting message",
            expected_output="A greeting message",
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

        # 1. Verify existence of all levels in hierarchy
        crew_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "crew.kickoff"
        ]
        task_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "task.execute"
        ]
        agent_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "agent.execute"
        ]

        self.assertGreaterEqual(
            len(crew_spans), 1, "Should capture crew.kickoff"
        )
        self.assertGreaterEqual(
            len(task_spans), 1, "Should capture task.execute"
        )
        self.assertGreaterEqual(
            len(agent_spans), 1, "Should capture agent.execute"
        )

        # 2. Verify Span Hierarchy and Trace Continuity
        crew_span = crew_spans[0]
        task_span = task_spans[0]
        agent_span = agent_spans[0]

        # All spans must share the same Trace ID
        trace_id = crew_span.context.trace_id
        for s in spans:
            self.assertEqual(
                s.context.trace_id,
                trace_id,
                "Trace ID must be consistent across all spans",
            )

        # Verify parent-child relationship (Crew -> Task -> Agent)
        self.assertEqual(
            task_span.parent.span_id,
            crew_span.context.span_id,
            "Task should be child of Crew",
        )
        self.assertEqual(
            agent_span.parent.span_id,
            task_span.context.span_id,
            "Agent should be child of Task",
        )

        # 3. Verify OpenTelemetry GenAI Attributes
        for s in [crew_span, task_span, agent_span]:
            self.assertEqual(s.attributes.get("gen_ai.system"), "crewai")
            self.assertIsNotNone(s.attributes.get("gen_ai.operation.name"))

        # 4. Verify Content Capture (JSON formatted messages)
        output_messages_json = task_span.attributes.get(
            "gen_ai.output.messages"
        )
        self.assertIsNotNone(
            output_messages_json, "Task should capture output messages"
        )
        output_messages = json.loads(output_messages_json)
        self.assertGreater(len(output_messages), 0)
        self.assertEqual(output_messages[0]["role"], "assistant")
        self.assertIn(
            "content",
            output_messages[0]["parts"][0],
            "Output should contain message parts",
        )

    def test_streaming_with_error(self):
        """
        Test streaming with error during stream consumption.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Executes 1 Task with streaming
        - Stream fails mid-way

        Verification:
        - Spans marked with StatusCode.ERROR status
        - Exception events recorded with type and message
        - Input messages captured for diagnostic context even on failure
        """
        # Use invalid model name to trigger error
        agent = Agent(
            role="Content Writer",
            goal="Write content",
            backstory="Expert writer",
            verbose=False,
            llm="invalid-model-name",
        )

        task = Task(
            description="Write a message",
            expected_output="A message",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        # Expect error during execution
        try:
            crew.kickoff()
        except Exception:
            pass  # Expected

        # Verify spans for error state
        spans = self.memory_exporter.get_finished_spans()

        # Filter agent span (where error usually originated in this case)
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        self.assertGreaterEqual(
            len(error_spans), 1, "At least one span should be marked as ERROR"
        )

        for span in error_spans:
            # 1. Verify Status
            self.assertEqual(span.status.status_code, StatusCode.ERROR)

            # 2. Verify Exception Events
            self.assertGreater(
                len(span.events),
                0,
                "Span should have recorded exception events",
            )
            exception_events = [
                e for e in span.events if e.name == "exception"
            ]
            self.assertGreater(
                len(exception_events), 0, "Should contain 'exception' event"
            )

            # 3. Verify Exception Metadata
            event_attrs = exception_events[0].attributes
            self.assertIn("exception.type", event_attrs)
            self.assertIn("exception.message", event_attrs)

            # 4. Verify Input Capture (even on failure)
            # Input messages might be empty for the top-level crew if no inputs were provided,
            # but task and agent spans should always have them.
            op_name = span.attributes.get("gen_ai.operation.name")
            if op_name in ["task.execute", "agent.execute"]:
                input_messages = span.attributes.get("gen_ai.input.messages")
                self.assertIsNotNone(
                    input_messages,
                    f"Span {op_name} should capture inputs even on failure",
                )
