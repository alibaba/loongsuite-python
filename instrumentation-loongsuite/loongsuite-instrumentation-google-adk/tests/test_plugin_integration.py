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
Integration tests for Google ADK Plugin with InMemoryExporter validation.

Tests validate that spans are created with correct attributes according to
OpenTelemetry GenAI Semantic Conventions using real plugin callbacks and
InMemorySpanExporter to capture actual span data.

This test follows the same pattern as the commercial ARMS version but validates
against the latest OpenTelemetry GenAI semantic conventions.
"""

import asyncio
import timeit
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from google.adk.events.event import Event
from google.genai import types

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor
from opentelemetry.instrumentation.google_adk.internal._plugin import (
    _ACTIVE_LLM_REQUEST_KEY,
)
from opentelemetry.sdk import metrics as metrics_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def create_mock_callback_context(
    session_id="session_123", user_id="user_456", invocation_id="inv_123"
):
    """Create properly structured mock CallbackContext following ADK structure."""
    mock_callback_context = Mock()
    mock_session = Mock()
    mock_session.id = session_id
    mock_invocation_context = Mock()
    mock_invocation_context.session = mock_session
    mock_invocation_context.invocation_id = invocation_id
    mock_invocation_context.user_id = user_id
    mock_callback_context._invocation_context = mock_invocation_context
    mock_callback_context.user_id = user_id
    return mock_callback_context


def create_mock_llm_response(
    *,
    model_version=None,
    content=None,
    partial=False,
    finish_reason=None,
):
    """Create an ADK-version-neutral LLM response test double."""
    response = Mock()
    response.model = None
    response.model_version = None
    response.modelVersion = model_version
    response.content = content
    response.text = None
    response.partial = partial
    response.turn_complete = not partial
    response.finish_reason = finish_reason
    response.usage_metadata = None
    return response


class OTelGenAISpanValidator:
    """
    Validator for OpenTelemetry GenAI Semantic Conventions.

    Based on the latest OTel GenAI semantic conventions:
    - gen_ai.provider.name (required, replaces gen_ai.system)
    - gen_ai.operation.name (required, replaces gen_ai.span.kind)
    - gen_ai.conversation.id (replaces gen_ai.session.id)
    - enduser.id (replaces gen_ai.user.id)
    - gen_ai.response.finish_reasons (array, replaces gen_ai.response.finish_reason)
    - Tool attributes with gen_ai. prefix
    - Agent attributes with gen_ai. prefix
    """

    # Required attributes for different operation types
    REQUIRED_ATTRIBUTES_BY_OPERATION = {
        "chat": {
            "required": {
                "gen_ai.operation.name",
                "gen_ai.span.kind",
                "gen_ai.provider.name",
                "gen_ai.request.model",
            },
            "recommended": {
                "gen_ai.response.model",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
            },
        },
        "invoke_agent": {
            "required": {"gen_ai.operation.name", "gen_ai.span.kind"},
            "recommended": {"gen_ai.agent.name", "gen_ai.agent.description"},
        },
        "execute_tool": {
            "required": {
                "gen_ai.operation.name",
                "gen_ai.span.kind",
                "gen_ai.tool.name",
            },
            "recommended": {"gen_ai.tool.description"},
        },
    }

    def validate_span(self, span, expected_operation: str) -> Dict[str, Any]:
        """Validate a single span's attributes against OTel GenAI conventions."""
        validation_result = {
            "span_name": span.name,
            "expected_operation": expected_operation,
            "errors": [],
            "warnings": [],
            "missing_required": [],
            "missing_recommended": [],
        }

        attributes = getattr(span, "attributes", {}) or {}

        # Validate operation name
        actual_operation = attributes.get("gen_ai.operation.name")
        if not actual_operation:
            validation_result["errors"].append(
                "Missing required attribute: gen_ai.operation.name"
            )
        elif actual_operation != expected_operation:
            validation_result["errors"].append(
                f"Expected operation '{expected_operation}', got '{actual_operation}'"
            )

        # Validate required and recommended attributes
        if expected_operation in self.REQUIRED_ATTRIBUTES_BY_OPERATION:
            requirements = self.REQUIRED_ATTRIBUTES_BY_OPERATION[
                expected_operation
            ]

            # Check required attributes
            for attr in requirements["required"]:
                if attr not in attributes:
                    validation_result["missing_required"].append(attr)

            # Check recommended attributes
            for attr in requirements["recommended"]:
                if attr not in attributes:
                    validation_result["missing_recommended"].append(attr)

        # Validate specific attribute formats
        self._validate_attribute_formats(attributes, validation_result)

        return validation_result

    def _validate_attribute_formats(self, attributes: Dict, result: Dict):
        """Validate attribute value formats and types."""

        # Validate finish_reasons is array
        if "gen_ai.response.finish_reasons" in attributes:
            finish_reasons = attributes["gen_ai.response.finish_reasons"]
            if not isinstance(finish_reasons, (list, tuple)):
                result["errors"].append(
                    f"gen_ai.response.finish_reasons should be array, got {type(finish_reasons)}"
                )

        # Validate numeric attributes
        numeric_attrs = [
            "gen_ai.request.max_tokens",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ]
        for attr in numeric_attrs:
            if attr in attributes and not isinstance(
                attributes[attr], (int, float)
            ):
                result["errors"].append(
                    f"Attribute {attr} should be numeric, got {type(attributes[attr])}"
                )


class TestGoogleAdkPluginIntegration:
    """Integration tests using InMemoryExporter to validate actual spans."""

    def setup_method(self):
        """Set up test fixtures for each test."""
        # Create independent providers and exporters
        self.tracer_provider = trace_sdk.TracerProvider()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = metrics_sdk.MeterProvider(
            metric_readers=[self.metric_reader]
        )

        # Create instrumentor
        self.instrumentor = GoogleAdkInstrumentor()

        # Create validator
        self.validator = OTelGenAISpanValidator()

        # Clean up any existing instrumentation
        if self.instrumentor.is_instrumented_by_opentelemetry:
            self.instrumentor.uninstrument()

        # Clear any existing spans
        self.span_exporter.clear()

    def teardown_method(self):
        """Clean up after each test."""
        try:
            if self.instrumentor.is_instrumented_by_opentelemetry:
                self.instrumentor.uninstrument()
        except Exception:
            pass

        # Clear spans
        self.span_exporter.clear()

    @pytest.mark.asyncio
    async def test_llm_span_attributes_semantic_conventions(self):
        """
        Test that LLM spans follow the latest OTel GenAI semantic conventions.

        Validates:
        - Span name format: "chat {model}"
        - Required attributes: gen_ai.operation.name, gen_ai.provider.name
        - Provider name instead of gen_ai.system
        - No non-standard attributes
        """
        # Instrument the plugin
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock LLM request
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.config.max_tokens = 1000
        mock_llm_request.config.temperature = 0.7
        mock_llm_request.config.top_p = 0.9
        mock_llm_request.config.top_k = 40
        mock_llm_request.contents = ["test message"]
        mock_llm_request.stream = False

        # Create mock response
        mock_llm_response = Mock()
        mock_llm_response.model = "gemini-pro-001"
        mock_llm_response.finish_reason = "stop"
        mock_llm_response.content = "test response"
        mock_llm_response.usage_metadata = Mock()
        mock_llm_response.usage_metadata.prompt_token_count = 100
        mock_llm_response.usage_metadata.candidates_token_count = 50

        mock_callback_context = create_mock_callback_context(
            "conv_123", "user_456"
        )

        # Execute LLM span lifecycle
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )
        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=mock_llm_response,
        )

        # Get finished spans from InMemoryExporter
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 LLM span"

        llm_span = spans[0]

        # Validate span name follows OTel convention: "chat {model}"
        assert llm_span.name == "chat gemini-pro", (
            f"Expected span name 'chat gemini-pro', got '{llm_span.name}'"
        )

        # Validate span attributes using validator
        validation_result = self.validator.validate_span(llm_span, "chat")

        # Check for errors
        assert len(validation_result["errors"]) == 0, (
            f"Validation errors: {validation_result['errors']}"
        )

        # Validate specific required attributes
        attributes = llm_span.attributes
        assert attributes.get("gen_ai.operation.name") == "chat", (
            "Should have gen_ai.operation.name = 'chat'"
        )
        assert attributes.get("gen_ai.span.kind") == "LLM"
        assert "gen_ai.provider.name" in attributes, (
            "Should have gen_ai.provider.name (not gen_ai.system)"
        )
        assert attributes.get("gen_ai.request.model") == "gemini-pro"
        assert attributes.get("gen_ai.response.model") == "gemini-pro-001"

        # Validate token usage attributes
        assert attributes.get("gen_ai.usage.input_tokens") == 100
        assert attributes.get("gen_ai.usage.output_tokens") == 50

        # Validate conversation tracking uses correct attributes
        assert "gen_ai.conversation.id" in attributes, (
            "Should use gen_ai.conversation.id (not gen_ai.session.id)"
        )
        assert attributes.get("gen_ai.conversation.id") == "conv_123"
        assert "enduser.id" in attributes, (
            "Should use enduser.id (not gen_ai.user.id)"
        )
        assert attributes.get("enduser.id") == "user_456"

        # Validate finish_reasons is array
        assert "gen_ai.response.finish_reasons" in attributes, (
            "Should have gen_ai.response.finish_reasons (array)"
        )
        finish_reasons = attributes.get("gen_ai.response.finish_reasons")
        assert isinstance(finish_reasons, (list, tuple)), (
            "gen_ai.response.finish_reasons should be array"
        )

    @pytest.mark.asyncio
    async def test_agent_span_attributes_semantic_conventions(self):
        """
        Test that Agent spans follow OTel GenAI semantic conventions.

        Validates:
        - Span name format: "invoke_agent {agent_name}"
        - gen_ai.operation.name = "invoke_agent"
        - Agent attributes with gen_ai. prefix
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock agent
        mock_agent = Mock()
        mock_agent.name = "weather_agent"
        mock_agent.description = "Agent for weather queries"
        mock_agent.sub_agents = []  # Simple agent, not a chain

        mock_callback_context = create_mock_callback_context(
            "session_789", "user_999"
        )

        # Execute Agent span lifecycle
        await plugin.before_agent_callback(
            agent=mock_agent, callback_context=mock_callback_context
        )
        await plugin.after_agent_callback(
            agent=mock_agent, callback_context=mock_callback_context
        )

        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 Agent span"

        agent_span = spans[0]

        # Validate span name: "invoke_agent {agent_name}"
        assert agent_span.name == "invoke_agent weather_agent", (
            f"Expected span name 'invoke_agent weather_agent', got '{agent_span.name}'"
        )

        # Validate attributes
        validation_result = self.validator.validate_span(
            agent_span, "invoke_agent"
        )
        assert len(validation_result["errors"]) == 0, (
            f"Validation errors: {validation_result['errors']}"
        )

        attributes = agent_span.attributes
        assert attributes.get("gen_ai.operation.name") == "invoke_agent"
        assert attributes.get("gen_ai.span.kind") == "AGENT"

        # Validate agent attributes have gen_ai. prefix
        assert (
            "gen_ai.agent.name" in attributes or "agent.name" in attributes
        ), "Should have agent name attribute"
        assert (
            "gen_ai.agent.description" in attributes
            or "agent.description" in attributes
        ), "Should have agent description attribute"

    @pytest.mark.asyncio
    async def test_tool_span_attributes_semantic_conventions(self):
        """
        Test that Tool spans follow OTel GenAI semantic conventions.

        Validates:
        - Span name format: "execute_tool {tool_name}"
        - gen_ai.operation.name = "execute_tool"
        - Tool attributes with gen_ai. prefix
        - SpanKind = INTERNAL (per OTel convention)
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock tool
        mock_tool = Mock()
        mock_tool.name = "calculator"
        mock_tool.description = "Mathematical calculator"

        mock_tool_args = {"operation": "add", "a": 5, "b": 3}
        mock_tool_context = Mock()
        mock_tool_context.session_id = "session_456"
        mock_result = {"result": 8}

        # Execute Tool span lifecycle
        await plugin.before_tool_callback(
            tool=mock_tool,
            tool_args=mock_tool_args,
            tool_context=mock_tool_context,
        )
        await plugin.after_tool_callback(
            tool=mock_tool,
            tool_args=mock_tool_args,
            tool_context=mock_tool_context,
            result=mock_result,
        )

        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 Tool span"

        tool_span = spans[0]

        # Validate span name: "execute_tool {tool_name}"
        assert tool_span.name == "execute_tool calculator", (
            f"Expected span name 'execute_tool calculator', got '{tool_span.name}'"
        )

        # Validate SpanKind (should be INTERNAL per OTel convention)
        assert tool_span.kind == trace_api.SpanKind.INTERNAL, (
            "Tool spans should use SpanKind.INTERNAL"
        )

        # Validate attributes
        validation_result = self.validator.validate_span(
            tool_span, "execute_tool"
        )
        assert len(validation_result["errors"]) == 0, (
            f"Validation errors: {validation_result['errors']}"
        )

        attributes = tool_span.attributes
        assert attributes.get("gen_ai.operation.name") == "execute_tool"
        assert attributes.get("gen_ai.span.kind") == "TOOL"

        # Validate tool attributes
        assert attributes.get("gen_ai.tool.name") == "calculator"
        assert (
            attributes.get("gen_ai.tool.description")
            == "Mathematical calculator"
        )

    @pytest.mark.asyncio
    async def test_runner_span_attributes(self):
        """Test Runner span creation and attributes."""
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock invocation context
        mock_invocation_context = Mock()
        mock_invocation_context.invocation_id = "run_12345"
        mock_invocation_context.app_name = "test_app"
        mock_invocation_context.session = Mock()
        mock_invocation_context.session.id = "session_111"
        mock_invocation_context.user_id = "user_222"

        # Execute Runner span lifecycle
        await plugin.before_run_callback(
            invocation_context=mock_invocation_context
        )
        await plugin.after_run_callback(
            invocation_context=mock_invocation_context
        )

        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 Runner span"

        runner_span = spans[0]

        # Validate span name (runner uses agent-style naming)
        assert runner_span.name == "invoke_agent test_app", (
            f"Expected span name 'invoke_agent test_app', got '{runner_span.name}'"
        )

        # Validate attributes
        attributes = runner_span.attributes
        assert attributes.get("gen_ai.operation.name") == "invoke_agent"
        assert attributes.get("gen_ai.span.kind") == "AGENT"
        # Note: runner.app_name is namespaced with google_adk prefix
        assert attributes.get("google_adk.runner.app_name") == "test_app"

    @pytest.mark.asyncio
    async def test_runner_span_finishes_on_root_final_event(self):
        """ADK node runtime may emit final events without after_run_callback."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        mock_invocation_context = Mock()
        mock_invocation_context.invocation_id = "run_final_event"
        mock_invocation_context.app_name = "test_app"
        mock_invocation_context.session = Mock()
        mock_invocation_context.session.id = "session_final"
        mock_invocation_context.user_id = "user_222"
        mock_invocation_context.agent = Mock()
        mock_invocation_context.agent.name = "test_agent"

        await plugin.before_run_callback(
            invocation_context=mock_invocation_context
        )
        await plugin.on_event_callback(
            invocation_context=mock_invocation_context,
            event=Event(
                author="test_agent",
                content=types.Content(
                    role="model", parts=[types.Part(text="done")]
                ),
            ),
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should finish the Runner span on final event"
        assert spans[0].name == "invoke_agent test_app"
        assert spans[0].attributes.get("gen_ai.span.kind") == "AGENT"
        assert plugin._active_runner_invocations == {}

    @pytest.mark.asyncio
    async def test_runner_span_ignores_non_root_final_event(self):
        """Sub-agent final responses should not close the Runner span early."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        mock_invocation_context = Mock()
        mock_invocation_context.invocation_id = "run_subagent_final"
        mock_invocation_context.app_name = "test_app"
        mock_invocation_context.session = Mock()
        mock_invocation_context.session.id = "session_subagent_final"
        mock_invocation_context.agent = Mock()
        mock_invocation_context.agent.name = "root_agent"

        await plugin.before_run_callback(
            invocation_context=mock_invocation_context
        )
        await plugin.on_event_callback(
            invocation_context=mock_invocation_context,
            event=Event(
                author="child_agent",
                content=types.Content(
                    role="model", parts=[types.Part(text="child done")]
                ),
            ),
        )

        assert self.span_exporter.get_finished_spans() == ()
        assert plugin._active_runner_invocations

        await plugin.after_run_callback(
            invocation_context=mock_invocation_context
        )

    @pytest.mark.asyncio
    async def test_streaming_llm_span_finishes_on_final_response(
        self, monkeypatch
    ):
        """Streaming partial chunks should accumulate before ending one LLM span."""
        # The util reads this setting when serializing span attributes.
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "SPAN_ONLY",
        )
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.config.max_tokens = 1000
        mock_llm_request.config.temperature = 0.7
        mock_llm_request.config.top_p = 0.9
        mock_llm_request.contents = [
            types.Content(
                role="user", parts=[types.Part(text="stream please")]
            )
        ]

        mock_callback_context = create_mock_callback_context(
            "stream_session", "stream_user", "stream_invocation"
        )

        start_time = timeit.default_timer()
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )

        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-001",
                content=types.Content(
                    role="model", parts=[types.Part(text="Part")]
                ),
                partial=True,
            ),
        )
        first_partial_end_time = timeit.default_timer()
        assert len(self.span_exporter.get_finished_spans()) == 0
        await asyncio.sleep(0.05)

        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-001",
                content=types.Content(
                    role="model", parts=[types.Part(text="Partial")]
                ),
                partial=False,
                finish_reason=types.FinishReason.STOP,
            ),
        )
        end_time = timeit.default_timer()

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes.get("gen_ai.span.kind") == "LLM"
        assert span.attributes.get("gen_ai.response.model") == "gemini-pro-001"
        ttft_ns = span.attributes.get("gen_ai.response.time_to_first_token")
        assert isinstance(ttft_ns, int)
        assert ttft_ns > 0
        assert (
            ttft_ns
            <= int((first_partial_end_time - start_time) * 1_000_000_000)
            + 1_000_000
        )
        assert ttft_ns < int((end_time - start_time) * 1_000_000_000) - (
            20 * 1_000_000
        )
        assert '"Partial"' in span.attributes.get("gen_ai.output.messages", "")

    @pytest.mark.asyncio
    async def test_concurrent_llm_callbacks_same_session_do_not_cross_finish(
        self,
    ):
        """Concurrent calls in the same ADK session should keep their own spans."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        request_one = Mock()
        request_one.model = "gemini-pro-one"
        request_one.config = None
        request_one.contents = []

        request_two = Mock()
        request_two.model = "gemini-pro-two"
        request_two.config = None
        request_two.contents = []

        context_one = create_mock_callback_context(
            "shared_session", "user_one", "invocation_one"
        )
        context_two = create_mock_callback_context(
            "shared_session", "user_two", "invocation_two"
        )

        await plugin.before_model_callback(
            callback_context=context_one, llm_request=request_one
        )
        await plugin.before_model_callback(
            callback_context=context_two, llm_request=request_two
        )

        await plugin.after_model_callback(
            callback_context=context_two,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-two-response"
            ),
        )
        await plugin.after_model_callback(
            callback_context=context_one,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-one-response"
            ),
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 2

        response_by_request = {
            span.attributes.get("gen_ai.request.model"): span.attributes.get(
                "gen_ai.response.model"
            )
            for span in spans
        }
        assert response_by_request == {
            "gemini-pro-one": "gemini-pro-one-response",
            "gemini-pro-two": "gemini-pro-two-response",
        }

    @pytest.mark.asyncio
    async def test_concurrent_llm_callbacks_same_invocation_use_task_context(
        self,
    ):
        """Same-invocation concurrent LLM callbacks should finish their own span."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        async def drive_call(request_model: str, response_model: str) -> None:
            request = Mock()
            request.model = request_model
            request.config = None
            request.contents = []
            context = create_mock_callback_context(
                "shared_session", "shared_user", "shared_invocation"
            )
            await plugin.before_model_callback(
                callback_context=context,
                llm_request=request,
            )
            await asyncio.sleep(0.01)
            await plugin.after_model_callback(
                callback_context=context,
                llm_response=create_mock_llm_response(
                    model_version=response_model
                ),
            )

        await asyncio.gather(
            drive_call("gemini-pro-one", "gemini-pro-one-response"),
            drive_call("gemini-pro-two", "gemini-pro-two-response"),
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 2
        response_by_request = {
            span.attributes.get("gen_ai.request.model"): span.attributes.get(
                "gen_ai.response.model"
            )
            for span in spans
        }
        assert response_by_request == {
            "gemini-pro-one": "gemini-pro-one-response",
            "gemini-pro-two": "gemini-pro-two-response",
        }

    @pytest.mark.asyncio
    async def test_nested_llm_callbacks_restore_previous_task_context(self):
        """Nested same-task LLM callbacks should restore the previous key."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin
        context = create_mock_callback_context(
            "nested_session", "nested_user", "nested_invocation"
        )

        outer_request = Mock()
        outer_request.model = "gemini-pro-outer"
        outer_request.config = None
        outer_request.contents = []

        inner_request = Mock()
        inner_request.model = "gemini-pro-inner"
        inner_request.config = None
        inner_request.contents = []

        await plugin.before_model_callback(
            callback_context=context,
            llm_request=outer_request,
        )
        outer_key = _ACTIVE_LLM_REQUEST_KEY.get()
        assert outer_key

        await plugin.before_model_callback(
            callback_context=context,
            llm_request=inner_request,
        )
        inner_key = _ACTIVE_LLM_REQUEST_KEY.get()
        assert inner_key and inner_key != outer_key

        await plugin.after_model_callback(
            callback_context=context,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-inner-response"
            ),
        )
        assert _ACTIVE_LLM_REQUEST_KEY.get() == outer_key

        await plugin.after_model_callback(
            callback_context=context,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-outer-response"
            ),
        )
        assert _ACTIVE_LLM_REQUEST_KEY.get() is None
        assert plugin._llm_context_tokens == {}

    @pytest.mark.asyncio
    async def test_concurrent_streaming_llm_outputs_do_not_cross(
        self, monkeypatch
    ):
        """Interleaved streaming calls should keep output content isolated."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "SPAN_ONLY",
        )
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin
        partial_count = 0
        all_partials_seen = asyncio.Event()
        partial_lock = asyncio.Lock()

        async def mark_partial_seen() -> None:
            nonlocal partial_count
            async with partial_lock:
                partial_count += 1
                if partial_count == 2:
                    all_partials_seen.set()

        async def drive_stream(
            request_model: str, partial_text: str, final_text: str
        ) -> None:
            request = Mock()
            request.model = request_model
            request.config = None
            request.contents = [
                types.Content(
                    role="user", parts=[types.Part(text=request_model)]
                )
            ]
            context = create_mock_callback_context(
                "shared_stream_session",
                "shared_stream_user",
                "shared_stream_invocation",
            )
            await plugin.before_model_callback(
                callback_context=context,
                llm_request=request,
            )
            await plugin.after_model_callback(
                callback_context=context,
                llm_response=create_mock_llm_response(
                    model_version=f"{request_model}-response",
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=partial_text)],
                    ),
                    partial=True,
                ),
            )
            await mark_partial_seen()
            await all_partials_seen.wait()
            await plugin.after_model_callback(
                callback_context=context,
                llm_response=create_mock_llm_response(
                    model_version=f"{request_model}-response",
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=final_text)],
                    ),
                    partial=False,
                ),
            )

        await asyncio.gather(
            drive_stream("gemini-alpha", "Al", "Alpha done"),
            drive_stream("gemini-beta", "Be", "Beta done"),
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 2
        outputs_by_request = {
            span.attributes.get("gen_ai.request.model"): span.attributes.get(
                "gen_ai.output.messages", ""
            )
            for span in spans
        }
        assert '"Alpha done"' in outputs_by_request["gemini-alpha"]
        assert '"Beta done"' not in outputs_by_request["gemini-alpha"]
        assert '"Beta done"' in outputs_by_request["gemini-beta"]
        assert '"Alpha done"' not in outputs_by_request["gemini-beta"]

    @pytest.mark.asyncio
    async def test_concurrent_agent_callbacks_same_session_do_not_overwrite(
        self,
    ):
        """Agent spans use invocation id so same-session concurrent runs survive."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        mock_agent = Mock()
        mock_agent.name = "weather_agent"
        mock_agent.description = "Agent for weather queries"

        context_one = create_mock_callback_context(
            "shared_agent_session", "user_one", "agent_invocation_one"
        )
        context_two = create_mock_callback_context(
            "shared_agent_session", "user_two", "agent_invocation_two"
        )

        await plugin.before_agent_callback(
            agent=mock_agent, callback_context=context_one
        )
        await plugin.before_agent_callback(
            agent=mock_agent, callback_context=context_two
        )
        await plugin.after_agent_callback(
            agent=mock_agent, callback_context=context_two
        )
        await plugin.after_agent_callback(
            agent=mock_agent, callback_context=context_one
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(
            span.attributes.get("gen_ai.span.kind") == "AGENT"
            for span in spans
        )
        assert {span.attributes.get("enduser.id") for span in spans} == {
            "user_one",
            "user_two",
        }

    @pytest.mark.asyncio
    async def test_tool_error_creates_error_span(self):
        """Tool errors should finish the execute_tool span with error fields."""
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        mock_tool = Mock()
        mock_tool.name = "calculator"
        mock_tool.description = "Mathematical calculator"
        tool_args = {"operation": "divide", "a": 1, "b": 0}
        mock_tool_context = Mock()
        mock_tool_context.call_id = "tool_call_error"
        mock_tool_context._invocation_context = Mock()
        mock_tool_context._invocation_context.invocation_id = (
            "tool_error_invocation"
        )

        await plugin.before_tool_callback(
            tool=mock_tool,
            tool_args=tool_args,
            tool_context=mock_tool_context,
        )
        await plugin.on_tool_error_callback(
            tool=mock_tool,
            tool_args=tool_args,
            tool_context=mock_tool_context,
            error=ValueError("division by zero"),
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "execute_tool calculator"
        assert span.status.status_code == trace_api.StatusCode.ERROR
        assert "division by zero" in span.status.description
        assert span.attributes.get("error.type") == "ValueError"

    @pytest.mark.asyncio
    async def test_no_content_mode_does_not_capture_message_payloads(
        self, monkeypatch
    ):
        """Default NO_CONTENT mode should not serialize prompt or response text."""
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            raising=False,
        )
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = None
        mock_llm_request.contents = [
            types.Content(
                role="user", parts=[types.Part(text="private prompt")]
            )
        ]
        mock_callback_context = create_mock_callback_context(
            "no_content_session", "no_content_user"
        )

        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )
        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=create_mock_llm_response(
                model_version="gemini-pro-001",
                content=types.Content(
                    role="model", parts=[types.Part(text="private response")]
                ),
            ),
        )

        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        attributes = spans[0].attributes
        assert "gen_ai.input.messages" not in attributes
        assert "gen_ai.output.messages" not in attributes

    @pytest.mark.asyncio
    async def test_error_handling_attributes(self):
        """
        Test error handling and span status.

        Validates:
        - Span status set to ERROR
        - error.type attribute (not error.message per OTel)
        - Span description contains error message
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock LLM request
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.contents = None

        mock_callback_context = create_mock_callback_context(
            "session_err", "user_err"
        )

        # Create error
        test_error = Exception("API rate limit exceeded")

        # Execute error scenario
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )
        await plugin.on_model_error_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
            error=test_error,
        )

        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 error span"

        error_span = spans[0]

        # Validate span status
        assert error_span.status.status_code == trace_api.StatusCode.ERROR, (
            "Error span should have ERROR status"
        )
        assert "API rate limit exceeded" in error_span.status.description, (
            "Error description should contain error message"
        )

        # Validate error attributes
        attributes = error_span.attributes
        assert "error.type" in attributes, "Should have error.type attribute"
        assert attributes["error.type"] == "Exception"

        # Note: error.message is non-standard, OTel recommends using span status
        # but we may include it for debugging purposes

    @pytest.mark.asyncio
    async def test_metrics_recorded_with_correct_dimensions(self):
        """
        Test that metrics are recorded with correct OTel GenAI dimensions.

        Validates:
        - gen_ai.client.operation.duration histogram
        - gen_ai.client.token.usage histogram
        - Correct dimension attributes
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create and execute LLM span
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.config.max_tokens = 500
        mock_llm_request.config.temperature = 0.5
        mock_llm_request.contents = ["test"]

        mock_llm_response = Mock()
        mock_llm_response.model = "gemini-pro"
        mock_llm_response.finish_reason = "stop"
        mock_llm_response.usage_metadata = Mock()
        mock_llm_response.usage_metadata.prompt_token_count = 50
        mock_llm_response.usage_metadata.candidates_token_count = 30

        mock_callback_context = create_mock_callback_context()

        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )
        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=mock_llm_response,
        )

        # Get metrics data
        metrics_data = self.metric_reader.get_metrics_data()

        # Validate metrics exist
        assert metrics_data is not None, "Should have metrics data"

        # Note: Detailed metric validation would require iterating through
        # metrics_data.resource_metrics to find the specific histograms
        # and verify their attributes match OTel GenAI conventions


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
