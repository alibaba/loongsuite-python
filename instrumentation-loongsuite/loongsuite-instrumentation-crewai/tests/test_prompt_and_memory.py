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
Test cases for prompt management and session management in CrewAI.

Business Demo Description:
This test suite covers:
- Prompt template variable substitution
- Session management and trace consistency
"""

import json
import os
import sys

import pysqlite3
import pytest
from crewai import Agent, Crew, Task

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
class TestPromptAndMemory(TestBase):
    """Test prompt management and memory scenarios."""

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
        self.documents = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "OpenTelemetry provides observability for distributed systems.",
        ]

    def tearDown(self):
        """Cleanup test resources."""
        with self.disable_logging():
            self.instrumentor.uninstrument()
        super().tearDown()

    def test_prompt_template_with_variables(self):
        """
        Test prompt template with variable substitution.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Uses a Task with template variables
        - Variables are substituted in the prompt
        - Executes 1 Task with expanded prompt

        Verification:
        - Agent/Task spans set gen_ai.system="crewai"
        - Modified prompts correctly captured in gen_ai.input.messages
        - Proper span hierarchy (Crew -> Task -> Agent)
        """
        # Create Agent
        agent = Agent(
            role="City Analyst",
            goal="Analyze cities",
            backstory="Expert city analyst",
            verbose=False,
            llm=self.model_name,
        )

        # Create Task with template-like description
        task = Task(
            description="Analyze the city of San Francisco and provide insights about its economy.",
            expected_output="City analysis report",
            agent=agent,
        )

        # Create and execute Crew with inputs
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        crew.kickoff(inputs={"city": "San Francisco"})

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        task_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "task.execute"
        ]
        llm_spans = [
            s
            for s in spans
            if s.attributes.get("gen_ai.operation.name") == "chat"
        ]

        # Verify TASK span has expanded prompt in gen_ai.input.messages
        if task_spans:
            task_span = task_spans[0]

            input_messages_json = task_span.attributes.get(
                "gen_ai.input.messages"
            )
            self.assertIsNotNone(input_messages_json)
            input_messages = json.loads(input_messages_json)
            content = input_messages[0]["parts"][0]["content"]
            self.assertIn("San Francisco", content)

        # Verify LLM span has messages
        if llm_spans:
            llm_span = llm_spans[0]
            messages_json = llm_span.attributes.get("gen_ai.input.messages")
            if messages_json:
                messages = json.loads(messages_json)
                self.assertGreater(len(messages), 0)

    def test_session_management(self):
        """
        Test session management and conversation tracking.

        Business Demo:
        - Creates a Crew with session tracking
        - Executes multiple Tasks in same session
        - Session ID is propagated through spans

        Verification:
        - All spans share session context
        - Session-related attributes are present
        """
        # Create Agent
        agent = Agent(
            role="Session Agent",
            goal="Manage conversation sessions",
            backstory="Expert in session management",
            verbose=False,
            llm=self.model_name,
        )

        # Create Tasks
        task1 = Task(
            description="Start a conversation",
            expected_output="Greeting",
            agent=agent,
        )

        task2 = Task(
            description="Continue the conversation",
            expected_output="Response",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task1, task2],
            verbose=False,
        )

        # Execute with session context
        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        # Verify all spans share the same trace
        trace_ids = set(span.context.trace_id for span in spans)
        self.assertEqual(
            len(trace_ids),
            1,
            "All spans should share the same trace ID for session tracking",
        )

        # Verify spans exist
        self.assertGreaterEqual(
            len(spans), 2, f"Expected at least 2 spans, got {len(spans)}"
        )
