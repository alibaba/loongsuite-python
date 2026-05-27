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
Test cases for error scenarios in CrewAI.

Business Demo Description:
This test suite covers various error scenarios including:
- API key authentication failures
- Network connectivity issues
- LLM service errors
- Task execution failures
All error scenarios should be properly traced with OpenTelemetry error status and exception events.
"""

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
class TestErrorScenarios(TestBase):
    """Test error handling scenarios."""

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

        # Disable CrewAI's built-in tracing to avoid interference
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
        self.model_name = "dashscope/qwen-turbo"

    def tearDown(self):
        """Cleanup test resources."""
        with self.disable_logging():
            self.instrumentor.uninstrument()
        super().tearDown()

    def test_api_key_missing(self):
        """
        Test execution with missing API key.

        Business Demo:
        - Creates a Crew with 1 Agent
        - API key is empty/missing
        - Executes 1 Task that fails due to authentication

        Verification:
        - Affected spans (e.g., Agent/Task) have ERROR status
        - Span records authentication exception in events
        - Input messages are still captured for context
        """
        # Temporarily remove API keys
        original_dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = ""

        try:
            agent = Agent(
                role="Test Agent",
                goal="Test authentication failure",
                backstory="Test agent",
                verbose=False,
                llm=self.model_name,
            )

            task = Task(
                description="Execute a task",
                expected_output="Task output",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
            )

            # Expect execution to fail
            try:
                crew.kickoff()
            except Exception:
                pass  # Expected

        finally:
            # Restore API keys
            if original_dashscope_key:
                os.environ["DASHSCOPE_API_KEY"] = original_dashscope_key
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        # Filter for error spans
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        self.assertGreater(
            len(error_spans), 0, "At least one span should be marked as ERROR"
        )

        # Verify the primary failed span (usually agent or task)
        found_exception = False
        for span in error_spans:
            for event in span.events:
                if event.name == "exception":
                    found_exception = True
                    break

            # Verify inputs are still there
            if span.attributes.get("gen_ai.operation.name") in [
                "task.execute",
                "agent.execute",
            ]:
                self.assertIsNotNone(
                    span.attributes.get("gen_ai.input.messages")
                )

        self.assertTrue(
            found_exception,
            "Should have recorded an exception event in at least one error span",
        )

    def test_network_error(self):
        """
        Test execution with network connectivity error.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Network connection fails during LLM call
        - Executes 1 Task that fails due to network error

        Verification:
        - Affected spans have ERROR status
        - Span records network exception in events
        """
        # Use invalid base URL to trigger network error
        original_dashscope_base = os.environ.get("DASHSCOPE_API_BASE")
        original_openai_base = os.environ.get("OPENAI_API_BASE")
        os.environ["DASHSCOPE_API_BASE"] = (
            "https://invalid-url-that-does-not-exist.com"
        )
        os.environ["OPENAI_API_BASE"] = (
            "https://invalid-url-that-does-not-exist.com"
        )

        try:
            agent = Agent(
                role="Test Agent",
                goal="Test network failure",
                backstory="Test agent",
                verbose=False,
                llm=self.model_name,
            )

            task = Task(
                description="Execute a task",
                expected_output="Task output",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
            )

            # Expect execution to fail
            try:
                crew.kickoff()
            except Exception:
                pass  # Expected
        finally:
            # Restore original base URLs
            if original_dashscope_base:
                os.environ["DASHSCOPE_API_BASE"] = original_dashscope_base
            if original_openai_base:
                os.environ["OPENAI_API_BASE"] = original_openai_base

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        self.assertGreater(
            len(error_spans),
            0,
            "Spans should be marked as ERROR on network failure",
        )

    def test_llm_service_error(self):
        """
        Test execution with LLM service error (e.g., rate limit, model unavailable).

        Business Demo:
        - Creates a Crew with 1 Agent
        - LLM service returns error (invalid model)
        - Executes 1 Task that fails due to service error

        Verification:
        - Spans have ERROR status
        - Exception events contain error details
        """
        # Use invalid model to trigger service error
        agent = Agent(
            role="Test Agent",
            goal="Test service error",
            backstory="Test agent",
            verbose=False,
            llm="invalid-model-name",
        )

        task = Task(
            description="Execute a task",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        # Expect execution to fail
        try:
            crew.kickoff()
        except Exception:
            pass  # Expected

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        self.assertGreater(
            len(error_spans), 0, "Expected error spans for invalid model"
        )

        # Verify specific details of the error inside the agent span
        agent_error_spans = [
            s
            for s in error_spans
            if s.attributes.get("gen_ai.operation.name") == "agent.execute"
        ]
        if agent_error_spans:
            span = agent_error_spans[0]
            self.assertEqual(span.status.status_code, StatusCode.ERROR)
            exception_events = [
                e for e in span.events if e.name == "exception"
            ]
            self.assertGreater(len(exception_events), 0)
