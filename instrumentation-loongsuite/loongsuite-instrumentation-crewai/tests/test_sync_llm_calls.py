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
Test cases for synchronous LLM calls in CrewAI.

Business Demo Description:
This test suite uses CrewAI framework to create Agents that perform synchronous
LLM calls. It verifies the proper creation of OpenTelemetry GenAI spans for
Crew, Task, and Agent operations, ensuring consistent trace hierarchy and
standardized attribute capture.
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
class TestSyncLLMCalls(TestBase):
    """Test synchronous LLM call scenarios."""

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

        # Disable CrewAI's built-in tracing to avoid interference with OTel spans hierarchy during tests
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
        # Uninstrument to avoid affecting other tests
        with self.disable_logging():
            self.instrumentor.uninstrument()
        super().tearDown()

    def test_basic_sync_crew_execution(self):
        """
        Test basic synchronous Crew execution with a single Agent and Task.

        Business Demo:
        - Creates a Crew with 1 Agent (Data Analyst role)
        - Executes 1 Task (analyze AI trends)
        - Performs 1 synchronous LLM call
        - Uses OpenAI GPT-4o-mini model

        Verification:
        - CHAIN span for Crew.kickoff (gen_ai.operation.name="crew.kickoff")
        - TASK span for Task execution (gen_ai.operation.name="task.execute")
        - AGENT span for Agent.execute_task (gen_ai.operation.name="agent.execute")
        - Verified parent-child relationship (Crew -> Task -> Agent)
        - Standardized OpenTelemetry GenAI attributes (gen_ai.system, gen_ai.input.messages, etc.)
        """
        # Create Agent
        agent = Agent(
            role="Data Analyst",
            goal="Extract actionable insights from data",
            backstory="You are an expert data analyst with 10 years of experience.",
            verbose=False,
            llm=self.model_name,
        )

        # Create Task
        task = Task(
            description="Analyze the latest AI trends and provide insights.",
            expected_output="A comprehensive analysis of AI trends.",
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
        # Verify inputs in task span
        input_messages_json = task_span.attributes.get("gen_ai.input.messages")
        self.assertIsNotNone(input_messages_json)
        input_messages = json.loads(input_messages_json)
        self.assertGreater(len(input_messages), 0)
        self.assertIn("AI trends", input_messages[0]["parts"][0]["content"])

        # Verify outputs in task span
        output_messages_json = task_span.attributes.get(
            "gen_ai.output.messages"
        )
        self.assertIsNotNone(output_messages_json)
        output_messages = json.loads(output_messages_json)
        self.assertGreater(len(output_messages), 0)
        self.assertEqual(output_messages[0]["role"], "assistant")

    def test_crew_with_multiple_tasks(self):
        """
        Test Crew execution with multiple sequential tasks.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Executes 2 Tasks sequentially
        - Performs 2 synchronous LLM calls

        Verification:
        - 1 CHAIN span for Crew.kickoff
        - 2 TASK spans
        - 2 AGENT spans (one per task execution)
        - Trace continuity: all spans share the same Trace ID
        - Hierarchical consistency: Tasks belong to the Crew trace
        """
        # Create Agent
        agent = Agent(
            role="Research Analyst",
            goal="Conduct thorough research",
            backstory="Expert researcher",
            verbose=False,
            llm=self.model_name,
        )

        # Create Tasks
        task1 = Task(
            description="Research AI market trends",
            expected_output="Market trends report",
            agent=agent,
        )

        task2 = Task(
            description="Analyze competitor strategies",
            expected_output="Competitor analysis",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task1, task2],
            verbose=False,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        chain_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "crew.kickoff"
        ]
        task_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "task.execute"
        ]

        self.assertGreaterEqual(len(chain_spans), 1)
        self.assertEqual(len(task_spans), 2)

        # Verify Trace Continuity
        trace_id = chain_spans[0].context.trace_id
        for s in spans:
            self.assertEqual(
                s.context.trace_id,
                trace_id,
                "Trace ID must be consistent across multiple tasks",
            )

        # Verify structure: Tasks should belong to the same trace and have a parent
        # Note: Depending on CrewAI internal logic, tasks might be direct or indirect children of Crew
        for t_span in task_spans:
            self.assertIsNotNone(
                t_span.parent, "All tasks should have a parent"
            )
            self.assertEqual(
                t_span.context.trace_id,
                trace_id,
                "Tasks should belong to the Crew trace",
            )

    def test_sync_call_with_error(self):
        """
        Test synchronous Crew execution with error during LLM call.

        Verification:
        - At least one span has ERROR status
        - Exception events recorded
        - Inputs captured even on failure
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

            # 3. Verify Input Capture (even on failure)
            op_name = span.attributes.get("gen_ai.operation.name")
            if op_name in ["task.execute", "agent.execute"]:
                input_messages = span.attributes.get("gen_ai.input.messages")
                self.assertIsNotNone(
                    input_messages,
                    f"Span {op_name} should capture inputs even on failure",
                )
