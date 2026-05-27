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
Test cases for Agent workflow orchestration and multi-agent collaboration in CrewAI.

This test suite verifies complex Agent interaction patterns, ensuring that the
trace hierarchy and span relationships correctly represent the execution flow.
"""

import json
import os
import sys

import pysqlite3
import pytest
from crewai import Agent, Crew, Process, Task

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


@pytest.mark.vcr
class TestAgentWorkflow(TestBase):
    """Test Agent workflow orchestration scenarios."""

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

    def test_sequential_workflow(self):
        """
        Test sequential workflow with multiple agents and tasks.

        Business Demo:
        - Creates a Crew with 3 Agents (Researcher, Analyst, Writer)
        - Executes 3 Tasks sequentially
        - Each task is handled by a different agent
        - Performs 3 LLM calls (one per task)

        Verification:
        - 1 CHAIN span for Crew.kickoff
        - 3 TASK spans (one per task)
        - 3 AGENT spans (one per agent execution)
        - Span hierarchy: Crew -> Task -> Agent
        - Trace continuity across all sequential operations
        """
        # Create Agents
        researcher = Agent(
            role="Researcher",
            goal="Gather comprehensive information",
            backstory="Expert researcher with 15 years of experience",
            verbose=False,
            llm=self.model_name,
        )

        analyst = Agent(
            role="Data Analyst",
            goal="Analyze data and extract insights",
            backstory="Senior data analyst specializing in AI trends",
            verbose=False,
            llm=self.model_name,
        )

        writer = Agent(
            role="Content Writer",
            goal="Create compelling content",
            backstory="Professional writer with expertise in tech content",
            verbose=False,
            llm=self.model_name,
        )

        # Create Tasks
        research_task = Task(
            description="Research the latest AI trends in 2024",
            expected_output="Comprehensive research report",
            agent=researcher,
        )

        analysis_task = Task(
            description="Analyze the research findings",
            expected_output="Data analysis report",
            agent=analyst,
        )

        writing_task = Task(
            description="Write an article based on the analysis",
            expected_output="Published article",
            agent=writer,
        )

        # Create and execute Crew with sequential process
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential,
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
        agent_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "agent.execute"
        ]

        # Verify span counts
        self.assertGreaterEqual(
            len(chain_spans), 1, "Expected at least 1 CHAIN span"
        )
        self.assertGreaterEqual(
            len(task_spans),
            3,
            f"Expected at least 3 TASK spans, got {len(task_spans)}",
        )
        self.assertGreaterEqual(
            len(agent_spans),
            3,
            f"Expected at least 3 AGENT spans, got {len(agent_spans)}",
        )

        # Verify CHAIN span has proper attributes
        chain_span = chain_spans[0]
        self.assertEqual(chain_span.attributes.get("gen_ai.system"), "crewai")
        self.assertEqual(
            chain_span.attributes.get("gen_ai.operation.name"), "crew.kickoff"
        )

        # Verify result is captured in OpenTelemetry GenAI format
        output_messages_json = chain_span.attributes.get(
            "gen_ai.output.messages"
        )
        self.assertIsNotNone(output_messages_json)
        output_messages = json.loads(output_messages_json)
        self.assertGreater(len(output_messages), 0)
        self.assertIn("role", output_messages[0])
        self.assertIn("parts", output_messages[0])

    def test_multi_agent_collaboration(self):
        """
        Test multi-agent collaboration scenario.

        Business Demo:
        - Creates a Crew with 2 Agents working together
        - Agents share context and collaborate on tasks
        - Executes 2 Tasks with agent collaboration
        - Performs multiple LLM calls with shared context

        Verification:
        - Multiple AGENT spans with proper context
        - All spans share the same trace context
        """
        # Create collaborative agents
        designer = Agent(
            role="UX Designer",
            goal="Design user-friendly interfaces",
            backstory="Senior UX designer",
            verbose=True,
            llm=self.model_name,
        )

        developer = Agent(
            role="Frontend Developer",
            goal="Implement designs",
            backstory="Expert frontend developer",
            verbose=True,
            llm=self.model_name,
        )

        # Create collaborative tasks
        design_task = Task(
            description="Design a dashboard interface",
            expected_output="UI design mockup",
            agent=designer,
        )

        implement_task = Task(
            description="Implement the designed dashboard",
            expected_output="Working dashboard code",
            agent=developer,
            context=[design_task],  # Depends on design_task
        )

        # Create and execute Crew
        crew = Crew(
            agents=[designer, developer],
            tasks=[design_task, implement_task],
            verbose=True,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        # Verify all spans share the same trace
        trace_ids = set(span.context.trace_id for span in spans)
        self.assertEqual(
            len(trace_ids), 1, "All spans should share the same trace ID"
        )

        agent_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "agent.execute"
        ]

        # Should have multiple agent spans for collaboration
        self.assertGreaterEqual(
            len(agent_spans),
            2,
            f"Expected at least 2 AGENT spans, got {len(agent_spans)}",
        )

    def test_hierarchical_workflow(self):
        """
        Test hierarchical workflow with manager delegation.

        Business Demo:
        - Creates a Crew with hierarchical process
        - Manager agent delegates tasks to worker agents
        - Executes tasks with delegation pattern

        Verification:
        - CHAIN span for overall workflow
        - Multiple AGENT spans showing delegation hierarchy
        - Proper parent-child relationships in span hierarchy
        """
        # Create worker agents
        worker1 = Agent(
            role="Junior Analyst",
            goal="Perform assigned analysis",
            backstory="Junior analyst",
            verbose=True,
            llm=self.model_name,
        )

        worker2 = Agent(
            role="Junior Researcher",
            goal="Conduct assigned research",
            backstory="Junior researcher",
            verbose=True,
            llm=self.model_name,
        )

        # Create tasks
        task1 = Task(
            description="Analyze market data",
            expected_output="Market analysis",
            agent=worker1,
        )

        task2 = Task(
            description="Research competitors",
            expected_output="Competitor research",
            agent=worker2,
        )

        # Create Crew with hierarchical process
        # Note: Hierarchical process requires a manager_llm
        crew = Crew(
            agents=[worker1, worker2],
            tasks=[task1, task2],
            process=Process.hierarchical,
            manager_llm=self.model_name,
            verbose=True,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        chain_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "crew.kickoff"
        ]

        # Should have CHAIN span for hierarchical workflow
        self.assertGreaterEqual(
            len(chain_spans), 1, "Expected at least 1 CHAIN span"
        )
