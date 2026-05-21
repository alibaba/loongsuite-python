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

# -*- coding: utf-8 -*-
"""
Integration tests for Google ADK Instrumentation using real SDK.

These tests use the actual Google ADK SDK to verify that instrumentation
works correctly with real API calls. They follow TDD principles and will
be used to validate the migration from direct trace/metrics implementation
to util-genai based implementation.
"""

import asyncio
import os

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import (
    InMemorySessionService,
)
from google.adk.tools import FunctionTool
from google.genai import types

from .conftest import (
    find_spans_by_operation,
)

# Test configuration
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "test_api_key")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "dashscope/qwen-plus"


def _metric_data_points(metrics_data, metric_name: str):
    """Return all data points for a named metric."""
    if not metrics_data:
        return []

    data_points = []
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == metric_name and hasattr(
                    metric.data, "data_points"
                ):
                    data_points.extend(metric.data.data_points)
    return data_points


# Simple tool functions for testing
# Use fixed return values to ensure VCR cassette matching
def get_current_time() -> str:
    """Get current time as a simple tool."""
    # Return fixed time for VCR cassette matching
    return "2024-01-01 12:00:00"


def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b


class TestGoogleAdkSDKIntegration:
    """Integration tests using real Google ADK SDK."""

    @pytest.fixture(scope="function")
    def session_service(self):
        """Create session service for tests."""
        return InMemorySessionService()

    @pytest.fixture(scope="function")
    def model(self):
        """Create LiteLlm model instance with fixed configuration for VCR matching."""
        return LiteLlm(
            model=DASHSCOPE_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.7,  # Fixed temperature for VCR matching
            max_tokens=100,  # Fixed max_tokens for VCR matching
        )

    @pytest.fixture(scope="function")
    def agent(self, model):
        """Create LlmAgent instance with tools.

        Uses fixed configuration to ensure VCR cassette matching:
        - Fixed agent name: "test_agent"
        - Fixed instruction and description
        - Fixed tool functions
        """
        time_tool = FunctionTool(func=get_current_time)
        calc_tool = FunctionTool(func=calculate_sum)

        agent = LlmAgent(
            name="test_agent",  # Fixed name for VCR matching
            model=model,
            instruction="You are a helpful assistant.",  # Fixed instruction
            description="Test agent for instrumentation",  # Fixed description
            tools=[time_tool, calc_tool],  # Fixed tools
        )
        return agent

    @pytest.fixture(scope="function")
    def runner(self, agent, session_service, instrument):
        """Create Runner instance after instrumentation is applied.

        Note: Runner must be created AFTER instrumentation to ensure
        the plugin is injected into Runner.__init__.

        Uses fixed app_name for VCR matching.
        """
        return Runner(
            app_name="test_app",  # Fixed app_name for VCR matching
            agent=agent,
            session_service=session_service,
        )

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_llm_call_creates_chat_span(
        self, instrument, span_exporter, runner, session_service
    ):
        """
        Test that LLM calls create chat spans.

        Expected spans:
        - invoke_agent {app_name} (Runner)
        - chat {model} (LLM)
        """
        # Create session
        session = await session_service.create_session(
            app_name="test_app",
            user_id="test_user",
            session_id="test_session_2",
        )

        # Create user message
        user_message = types.Content(
            role="user", parts=[types.Part(text="Hello, how are you?")]
        )

        # Clear spans before test
        span_exporter.clear()

        # Run conversation
        events = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=user_message,
        ):
            events.append(event)

        # Wait a bit for spans to be exported
        await asyncio.sleep(0.5)

        # Get finished spans
        spans = span_exporter.get_finished_spans()

        # Should have chat span
        chat_spans = find_spans_by_operation(spans, "chat")
        assert len(chat_spans) >= 1, "Should have at least one chat span"

        # Verify chat span attributes
        chat_span = chat_spans[0]
        assert chat_span.attributes.get("gen_ai.operation.name") == "chat"
        assert chat_span.attributes.get("gen_ai.span.kind") == "LLM"
        assert chat_span.attributes.get("gen_ai.provider.name") is not None
        assert chat_span.attributes.get("gen_ai.request.model") is not None
        assert chat_span.name.startswith("chat ")

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_agent_invocation_creates_agent_span(
        self, instrument, span_exporter, runner, session_service
    ):
        """
        Test that Agent invocation creates invoke_agent spans.

        Expected spans:
        - invoke_agent {app_name} (Runner)
        - invoke_agent {agent_name} (Agent)
        """
        # Create session with fixed IDs for VCR matching
        session = await session_service.create_session(
            app_name="test_app",  # Fixed app_name
            user_id="test_user",  # Fixed user_id
            session_id="test_session_4",  # Fixed session_id
        )

        # Create user message with fixed content for VCR matching
        user_message = types.Content(
            role="user",
            parts=[types.Part(text="Tell me a joke")],  # Fixed message content
        )

        # Clear spans before test
        span_exporter.clear()

        # Run conversation
        events = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=user_message,
        ):
            events.append(event)

        # Wait a bit for spans to be exported
        await asyncio.sleep(0.5)

        # Get finished spans
        spans = span_exporter.get_finished_spans()

        # Should have agent spans
        agent_spans = find_spans_by_operation(spans, "invoke_agent")
        assert len(agent_spans) >= 1, (
            "Should have at least one invoke_agent span"
        )

        # Verify agent span attributes
        agent_span = agent_spans[0]
        assert (
            agent_span.attributes.get("gen_ai.operation.name")
            == "invoke_agent"
        )
        assert agent_span.attributes.get("gen_ai.span.kind") == "AGENT"
        assert (
            agent_span.attributes.get("gen_ai.provider.name") == "google_adk"
        )

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_metrics_are_recorded(
        self, instrument, span_exporter, metric_reader, runner, session_service
    ):
        """
        Test that metrics are recorded for operations.

        Expected metrics:
        - gen_ai.client.operation.duration
        - gen_ai.client.token.usage (if tokens are available)
        """
        # Create session with fixed IDs for VCR matching
        session = await session_service.create_session(
            app_name="test_app",  # Fixed app_name
            user_id="test_user",  # Fixed user_id
            session_id="test_session_5",  # Fixed session_id
        )

        # Create user message with fixed content for VCR matching
        user_message = types.Content(
            role="user",
            parts=[types.Part(text="Hello")],  # Fixed message content
        )

        # Clear before test
        span_exporter.clear()

        # Run conversation
        events = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=user_message,
        ):
            events.append(event)

        # Wait a bit for metrics to be recorded
        await asyncio.sleep(0.5)

        # Get metrics
        metrics = metric_reader.get_metrics_data()

        # Should have operation duration metrics
        assert metrics is not None, "Should have metrics data"
        duration_points = _metric_data_points(
            metrics, "gen_ai.client.operation.duration"
        )
        assert duration_points, (
            "Should have gen_ai.client.operation.duration data points"
        )
        assert any(
            dict(point.attributes).get("gen_ai.operation.name") == "chat"
            for point in duration_points
        ), "Should record operation duration for chat"
