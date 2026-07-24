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

# pylint: disable=too-many-lines

import os
import queue
import threading
import time
import unittest
from typing import Any, Mapping
from unittest.mock import MagicMock, patch

from opentelemetry import baggage as baggage_api
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.baggage import get_all as get_all_baggage
from opentelemetry.instrumentation._semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (  # pylint: disable=no-name-in-module
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai._multimodal_processing import (
    MultimodalProcessingMixin,
    _MultimodalAsyncTask,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GEN_AI_RERANK_DOCUMENTS_COUNT,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_SESSION_ID,
    GEN_AI_SKILL_DESCRIPTION,
    GEN_AI_SKILL_ID,
    GEN_AI_SKILL_NAME,
    GEN_AI_SKILL_VERSION,
    GEN_AI_SPAN_KIND,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GEN_AI_USER_ID,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
    RerankInvocation,
    RetrievalDocument,
    RetrievalInvocation,
)
from opentelemetry.util.genai.types import (
    Base64Blob,
    Blob,
    Error,
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    Uri,
)

_AGENT_NAME_BAGGAGE_KEY = "gen_ai.agent.name"


def patch_env_vars(
    stability_mode, content_capturing=None, emit_event=None, **extra_env_vars
):
    def decorator(test_case):
        env_vars = {
            OTEL_SEMCONV_STABILITY_OPT_IN: stability_mode,
        }
        if content_capturing is not None:
            env_vars[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
                content_capturing
            )
        if emit_event is not None:
            env_vars[OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT] = emit_event

        env_vars.update(extra_env_vars)

        @patch.dict(os.environ, env_vars)
        def wrapper(*args, **kwargs):
            # Reset state.
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            return test_case(*args, **kwargs)

        return wrapper

    return decorator


def _get_single_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _assert_span_time_order(span: ReadableSpan) -> None:
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.end_time >= span.start_time


def _get_span_attributes(span: ReadableSpan) -> Mapping[str, Any]:
    attrs = span.attributes
    assert attrs is not None
    return attrs


def _assert_span_attributes(
    span_attrs: Mapping[str, Any], expected_values: Mapping[str, Any]
) -> None:
    for key, value in expected_values.items():
        assert span_attrs.get(key) == value


class TestExtendedTelemetryHandler(unittest.TestCase):  # pylint: disable=too-many-public-methods
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )

        # Clear singleton if exists to avoid test interference
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")
        self.telemetry_handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

    def tearDown(self):
        # Clear singleton after test to avoid interference
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")
        # Clear spans, logs and reset the singleton telemetry handler so each test starts clean
        self.span_exporter.clear()
        self.log_exporter.clear()

    # ==================== Create Agent Tests ====================

    def test_create_agent_start_and_stop_creates_span(self):
        with self.telemetry_handler.create_agent() as invocation:
            invocation.provider = "openai"
            invocation.agent_name = "TestAgent"
            invocation.agent_id = "agent_123"
            invocation.agent_description = "A test agent"
            invocation.request_model = "gpt-4"
            invocation.server_address = "api.openai.com"
            invocation.server_port = 443
            invocation.attributes = {"custom_attr": "value"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "create_agent TestAgent")
        self.assertEqual(span.kind, trace.SpanKind.CLIENT)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "create_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GenAI.GEN_AI_AGENT_NAME: "TestAgent",
                GenAI.GEN_AI_AGENT_ID: "agent_123",
                GenAI.GEN_AI_AGENT_DESCRIPTION: "A test agent",
                GenAI.GEN_AI_REQUEST_MODEL: "gpt-4",
                ServerAttributes.SERVER_ADDRESS: "api.openai.com",
                ServerAttributes.SERVER_PORT: 443,
                "custom_attr": "value",
            },
        )

    def test_create_agent_without_name(self):
        with self.telemetry_handler.create_agent() as invocation:
            invocation.provider = "openai"
            invocation.agent_id = "agent_456"

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "create_agent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "create_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GenAI.GEN_AI_AGENT_ID: "agent_456",
            },
        )

    def test_create_agent_manual_start_and_stop(self):
        invocation = CreateAgentInvocation(
            provider="test-provider",
            agent_name="ManualAgent",
            attributes={"manual": True},
        )

        self.telemetry_handler.start_create_agent(invocation)
        assert invocation.span is not None
        invocation.agent_id = "manual_agent_789"
        self.telemetry_handler.stop_create_agent(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "create_agent ManualAgent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_AGENT_NAME: "ManualAgent",
                GenAI.GEN_AI_AGENT_ID: "manual_agent_789",
                "manual": True,
            },
        )

    def test_create_agent_error_handling(self):
        class CreateAgentError(RuntimeError):
            pass

        with self.assertRaises(CreateAgentError):
            with self.telemetry_handler.create_agent() as invocation:
                invocation.provider = "test-provider"
                invocation.agent_name = "ErrorAgent"
                raise CreateAgentError("Failed to create agent")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: CreateAgentError.__qualname__,
            },
        )

    # ==================== Embedding Tests ====================

    def test_embedding_start_and_stop_creates_span(self):
        with self.telemetry_handler.embedding() as invocation:
            invocation.request_model = "text-embedding-ada-002"
            invocation.provider = "openai"
            invocation.dimension_count = 1536
            invocation.encoding_formats = ["float"]
            invocation.input_tokens = 10
            invocation.server_address = "api.openai.com"
            invocation.server_port = 443
            invocation.attributes = {"custom": "value"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "embeddings text-embedding-ada-002")
        self.assertEqual(span.kind, trace.SpanKind.CLIENT)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "embeddings",
                GenAI.GEN_AI_REQUEST_MODEL: "text-embedding-ada-002",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GEN_AI_EMBEDDINGS_DIMENSION_COUNT: 1536,
                GenAI.GEN_AI_REQUEST_ENCODING_FORMATS: ("float",),
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 10,
                ServerAttributes.SERVER_ADDRESS: "api.openai.com",
                ServerAttributes.SERVER_PORT: 443,
                "custom": "value",
            },
        )

    def test_embedding_manual_start_and_stop(self):
        invocation = EmbeddingInvocation(
            request_model="text-embedding-v1",
            provider="test-provider",
            dimension_count=768,
        )

        self.telemetry_handler.start_embedding(invocation)
        assert invocation.span is not None
        invocation.input_tokens = 20
        self.telemetry_handler.stop_embedding(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "embeddings text-embedding-v1")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_EMBEDDINGS_DIMENSION_COUNT: 768,
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 20,
            },
        )

    def test_embedding_with_total_tokens(self):
        """Test that total_tokens is calculated when both input and output tokens are present."""
        with self.telemetry_handler.embedding() as invocation:
            invocation.request_model = "text-embedding-ada-002"
            invocation.provider = "openai"
            invocation.input_tokens = 15

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 15,
                GEN_AI_USAGE_TOTAL_TOKENS: 15,
            },
        )

    def test_embedding_error_handling(self):
        class EmbeddingError(RuntimeError):
            pass

        with self.assertRaises(EmbeddingError):
            with self.telemetry_handler.embedding() as invocation:
                invocation.request_model = "embedding-model"
                invocation.provider = "test"
                raise EmbeddingError("Embedding failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: EmbeddingError.__qualname__,
            },
        )

    # ==================== Execute Tool Tests ====================

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_execute_tool_start_and_stop_creates_span(self):
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "get_weather"
            invocation.tool_type = "function"
            invocation.tool_description = "Get weather info"
            invocation.tool_call_id = "call_123"
            invocation.tool_call_arguments = {"location": "Beijing"}
            invocation.tool_call_result = {"temp": 20, "conditions": "sunny"}
            invocation.attributes = {"custom": "tool_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "execute_tool get_weather")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "execute_tool",
                GenAI.GEN_AI_TOOL_NAME: "get_weather",
                GenAI.GEN_AI_TOOL_TYPE: "function",
                GenAI.GEN_AI_TOOL_DESCRIPTION: "Get weather info",
                GenAI.GEN_AI_TOOL_CALL_ID: "call_123",
                "custom": "tool_attr",
            },
        )
        # Check that arguments and result are present
        self.assertIn(GEN_AI_TOOL_CALL_ARGUMENTS, span_attrs)
        self.assertIn(GEN_AI_TOOL_CALL_RESULT, span_attrs)

    def test_execute_tool_without_sensitive_data(self):
        # Without experimental mode, sensitive data should not be recorded
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "secure_tool"
            invocation.tool_type = "function"
            invocation.tool_call_arguments = {"secret": "data"}
            invocation.tool_call_result = {"result": "value"}

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        # Arguments and result should not be present without opt-in
        self.assertNotIn(GEN_AI_TOOL_CALL_ARGUMENTS, span_attrs)
        self.assertNotIn(GEN_AI_TOOL_CALL_RESULT, span_attrs)

    def test_execute_tool_manual_start_and_stop(self):
        invocation = ExecuteToolInvocation(
            tool_name="manual_tool",
            tool_type="extension",
        )

        self.telemetry_handler.start_execute_tool(invocation)
        assert invocation.span is not None
        invocation.tool_description = "Manual tool execution"
        self.telemetry_handler.stop_execute_tool(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "execute_tool manual_tool")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_TOOL_NAME: "manual_tool",
                GenAI.GEN_AI_TOOL_TYPE: "extension",
                GenAI.GEN_AI_TOOL_DESCRIPTION: "Manual tool execution",
            },
        )

    def test_execute_tool_error_handling(self):
        class ToolExecutionError(RuntimeError):
            pass

        with self.assertRaises(ToolExecutionError):
            with self.telemetry_handler.execute_tool() as invocation:
                invocation.tool_name = "failing_tool"
                invocation.tool_type = "function"
                raise ToolExecutionError("Tool execution failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: ToolExecutionError.__qualname__,
            },
        )

    def test_execute_tool_writes_skill_attributes(self):
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "read_file"
            invocation.skill_name = "news"
            invocation.skill_id = "workspace:default:news"
            invocation.skill_description = "Latest news"
            invocation.skill_version = "1.0"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "execute_tool",
                GenAI.GEN_AI_TOOL_NAME: "read_file",
                GEN_AI_SKILL_NAME: "news",
                GEN_AI_SKILL_ID: "workspace:default:news",
                GEN_AI_SKILL_DESCRIPTION: "Latest news",
                GEN_AI_SKILL_VERSION: "1.0",
            },
        )

    def test_execute_tool_skips_empty_skill_attributes(self):
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "read_file"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        self.assertNotIn(GEN_AI_SKILL_NAME, span_attrs)
        self.assertNotIn(GEN_AI_SKILL_ID, span_attrs)
        self.assertNotIn(GEN_AI_SKILL_DESCRIPTION, span_attrs)
        self.assertNotIn(GEN_AI_SKILL_VERSION, span_attrs)

    def test_execute_tool_skips_blank_skill_attributes(self):
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "read_file"
            invocation.skill_name = ""
            invocation.skill_id = "   "
            invocation.skill_description = "\t"
            invocation.skill_version = "\n"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        self.assertNotIn(GEN_AI_SKILL_NAME, span_attrs)
        self.assertNotIn(GEN_AI_SKILL_ID, span_attrs)
        self.assertNotIn(GEN_AI_SKILL_DESCRIPTION, span_attrs)
        self.assertNotIn(GEN_AI_SKILL_VERSION, span_attrs)

    def test_execute_tool_skill_fields_override_custom_attributes(self):
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "read_file"
            invocation.skill_name = "news"
            invocation.skill_id = "workspace:default:news"
            invocation.skill_description = "Latest news"
            invocation.skill_version = "1.0"
            invocation.attributes = {
                GEN_AI_SKILL_NAME: "custom-name",
                GEN_AI_SKILL_ID: "custom-id",
                GEN_AI_SKILL_DESCRIPTION: "custom-description",
                GEN_AI_SKILL_VERSION: "custom-version",
            }

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_SKILL_NAME: "news",
                GEN_AI_SKILL_ID: "workspace:default:news",
                GEN_AI_SKILL_DESCRIPTION: "Latest news",
                GEN_AI_SKILL_VERSION: "1.0",
            },
        )

    # ==================== Invoke Agent Tests ====================

    def test_invoke_agent_start_and_stop_creates_span(self):
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "openai"
            invocation.agent_name = "CustomerAgent"
            invocation.agent_id = "agent_abc"
            invocation.agent_description = "Customer service agent"
            invocation.conversation_id = "conv_123"
            invocation.request_model = "gpt-4"
            invocation.temperature = 0.7
            invocation.max_tokens = 1000
            invocation.input_tokens = 50
            invocation.output_tokens = 200
            invocation.finish_reasons = ["stop"]
            invocation.response_id = "resp_456"
            invocation.attributes = {"custom": "agent_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent CustomerAgent")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GenAI.GEN_AI_AGENT_NAME: "CustomerAgent",
                GenAI.GEN_AI_AGENT_ID: "agent_abc",
                GenAI.GEN_AI_AGENT_DESCRIPTION: "Customer service agent",
                GenAI.GEN_AI_CONVERSATION_ID: "conv_123",
                GenAI.GEN_AI_REQUEST_MODEL: "gpt-4",
                GenAI.GEN_AI_REQUEST_TEMPERATURE: 0.7,
                GenAI.GEN_AI_REQUEST_MAX_TOKENS: 1000,
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 50,
                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS: 200,
                GEN_AI_USAGE_TOTAL_TOKENS: 250,
                GenAI.GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GenAI.GEN_AI_RESPONSE_ID: "resp_456",
                "custom": "agent_attr",
            },
        )

    def test_invoke_agent_without_name(self):
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_id = "agent_xyz"

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
                GenAI.GEN_AI_AGENT_ID: "agent_xyz",
            },
        )

    def test_invoke_agent_manual_start_and_stop(self):
        invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="ManualInvokeAgent",
        )

        self.telemetry_handler.start_invoke_agent(invocation)
        assert invocation.span is not None
        invocation.conversation_id = "manual_conv"
        invocation.input_tokens = 100
        self.telemetry_handler.stop_invoke_agent(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent ManualInvokeAgent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_CONVERSATION_ID: "manual_conv",
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 100,
            },
        )
        # Note: total_tokens is not set when only input_tokens is available

    def test_invoke_agent_propagates_agent_name_baggage(self):
        invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="BaggageAgent",
        )

        self.telemetry_handler.start_invoke_agent(invocation)
        try:
            current_baggage = get_all_baggage()
            self.assertEqual(
                current_baggage.get(_AGENT_NAME_BAGGAGE_KEY),
                "BaggageAgent",
            )
        finally:
            self.telemetry_handler.stop_invoke_agent(invocation)

        restored_baggage = get_all_baggage()
        self.assertNotIn(_AGENT_NAME_BAGGAGE_KEY, restored_baggage)

    def test_nested_invoke_agent_baggage_overrides_and_restores(self):
        parent_invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="ParentAgent",
        )
        child_invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="ChildAgent",
        )

        self.telemetry_handler.start_invoke_agent(parent_invocation)
        try:
            self.assertEqual(
                get_all_baggage().get(_AGENT_NAME_BAGGAGE_KEY),
                "ParentAgent",
            )

            self.telemetry_handler.start_invoke_agent(child_invocation)
            try:
                self.assertEqual(
                    get_all_baggage().get(_AGENT_NAME_BAGGAGE_KEY),
                    "ChildAgent",
                )
            finally:
                self.telemetry_handler.stop_invoke_agent(child_invocation)

            self.assertEqual(
                get_all_baggage().get(_AGENT_NAME_BAGGAGE_KEY),
                "ParentAgent",
            )
        finally:
            self.telemetry_handler.stop_invoke_agent(parent_invocation)

        restored_baggage = get_all_baggage()
        self.assertNotIn(_AGENT_NAME_BAGGAGE_KEY, restored_baggage)

    def test_agent_context_colors_llm_and_tool_spans(self):
        agent_invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="PlannerAgent",
        )

        self.telemetry_handler.start_invoke_agent(agent_invocation)
        try:
            llm_invocation = LLMInvocation(
                provider="openai",
                request_model="gpt-4o-mini",
            )
            self.telemetry_handler.start_llm(llm_invocation)
            self.telemetry_handler.stop_llm(llm_invocation)

            tool_invocation = ExecuteToolInvocation(tool_name="search")
            self.telemetry_handler.start_execute_tool(tool_invocation)
            self.telemetry_handler.stop_execute_tool(tool_invocation)
        finally:
            self.telemetry_handler.stop_invoke_agent(agent_invocation)

        spans = self.span_exporter.get_finished_spans()
        llm_span = next(
            span for span in spans if span.name == "chat gpt-4o-mini"
        )
        tool_span = next(
            span for span in spans if span.name == "execute_tool search"
        )
        self.assertEqual(
            llm_span.attributes.get(GenAI.GEN_AI_AGENT_NAME),
            "PlannerAgent",
        )
        self.assertEqual(
            tool_span.attributes.get(GenAI.GEN_AI_AGENT_NAME),
            "PlannerAgent",
        )

    def test_explicit_agent_parent_context_colors_llm_span(self):
        agent_invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="ExplicitParentAgent",
        )

        self.telemetry_handler.start_invoke_agent(agent_invocation)
        try:
            parent_context = context_api.get_current()
            llm_invocation = LLMInvocation(
                provider="openai",
                request_model="gpt-4o-mini",
            )
            self.telemetry_handler.start_llm(
                llm_invocation,
                context=parent_context,
            )
            self.telemetry_handler.stop_llm(llm_invocation)
        finally:
            self.telemetry_handler.stop_invoke_agent(agent_invocation)

        spans = self.span_exporter.get_finished_spans()
        llm_span = next(
            span for span in spans if span.name == "chat gpt-4o-mini"
        )
        self.assertEqual(
            llm_span.attributes.get(GenAI.GEN_AI_AGENT_NAME),
            "ExplicitParentAgent",
        )

    def test_invoke_agent_error_handling(self):
        class AgentInvocationError(RuntimeError):
            pass

        with self.assertRaises(AgentInvocationError):
            with self.telemetry_handler.invoke_agent() as invocation:
                invocation.provider = "test-provider"
                invocation.agent_name = "ErrorAgent"
                raise AgentInvocationError("Agent invocation failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: AgentInvocationError.__qualname__,
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_invoke_agent_with_messages(self):
        """Test that input/output messages are captured when content capturing is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "MessageAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Hello agent")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hello user")],
                    finish_reason="stop",
                )
            ]
            invocation.input_tokens = 10
            invocation.output_tokens = 20

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent MessageAgent")
        span_attrs = _get_span_attributes(span)

        # Verify messages are captured
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)
        self.assertIn(GenAI.GEN_AI_OUTPUT_MESSAGES, span_attrs)

        # Verify other attributes
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "test-provider",
                GenAI.GEN_AI_AGENT_NAME: "MessageAgent",
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 10,
                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS: 20,
                GEN_AI_USAGE_TOTAL_TOKENS: 30,
            },
        )

    def test_invoke_agent_without_content_capturing(self):
        """Test that messages are NOT captured when content capturing is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "NoContentAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Hello")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hi")],
                    finish_reason="stop",
                )
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify messages are NOT captured
        self.assertNotIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_MESSAGES, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_invoke_agent_with_tool_definitions(self):
        """Test that tool definitions are captured when content capturing is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "ToolAgent"
            invocation.tool_definitions = [
                FunctionToolDefinition(
                    name="get_weather",
                    description="Get weather information",
                    parameters={"location": "string"},
                )
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify tool definitions are captured
        self.assertIn("gen_ai.tool.definitions", span_attrs)

    def test_invoke_agent_with_tool_definitions_minimal_mode(self):
        """Test that only minimal tool info is captured when content capturing is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "ToolAgent"
            invocation.tool_definitions = [
                FunctionToolDefinition(
                    name="get_weather",
                    description="Get weather information",
                    parameters={"location": "string"},
                )
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify tool definitions are still captured (but with minimal info)
        self.assertIn("gen_ai.tool.definitions", span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_invoke_agent_with_system_instruction(self):
        """Test that system instruction is captured when content capturing is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "SystemAgent"
            invocation.system_instruction = [
                Text(content="You are a helpful assistant.")
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify system instruction is captured
        self.assertIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span_attrs)

    def test_invoke_agent_with_system_instruction_without_content_capturing(
        self,
    ):
        """Test that system instruction is NOT captured when content capturing is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "SystemAgent"
            invocation.system_instruction = [
                Text(content="You are a helpful assistant.")
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify system instruction is NOT captured
        self.assertNotIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="EVENT_ONLY",
        emit_event="true",
    )
    def test_invoke_agent_emits_event(self):
        """Test that invoke_agent emits events when emit_event is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "EventAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Hello agent")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hello user")],
                    finish_reason="stop",
                )
            ]
            invocation.input_tokens = 10
            invocation.output_tokens = 20

        # Check that event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record

        # Verify event name
        self.assertEqual(
            log_record.event_name,
            "gen_ai.client.agent.invoke.operation.details",
        )

        # Verify event attributes
        attrs = log_record.attributes
        self.assertIsNotNone(attrs)
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "invoke_agent")
        self.assertEqual(attrs[GenAI.GEN_AI_PROVIDER_NAME], "test-provider")
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_NAME], "EventAgent")
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, attrs)
        self.assertIn(GenAI.GEN_AI_OUTPUT_MESSAGES, attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_AND_EVENT",
        emit_event="true",
    )
    def test_invoke_agent_emits_event_and_span(self):
        """Test that invoke_agent emits both event and span when emit_event is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "CombinedAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Test query")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Test response")],
                    finish_reason="stop",
                )
            ]

        # Check span was created
        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        self.assertEqual(
            log_record.event_name,
            "gen_ai.client.agent.invoke.operation.details",
        )
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, log_record.attributes)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="EVENT_ONLY",
        emit_event="true",
    )
    def test_invoke_agent_emits_event_with_error(self):
        """Test that invoke_agent emits event with error when operation fails."""

        class AgentInvocationError(RuntimeError):
            pass

        with self.assertRaises(AgentInvocationError):
            with self.telemetry_handler.invoke_agent() as invocation:
                invocation.provider = "test-provider"
                invocation.agent_name = "ErrorAgent"
                invocation.input_messages = [
                    InputMessage(role="user", parts=[Text(content="Test")])
                ]
                raise AgentInvocationError("Agent failed")

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        attrs = log_record.attributes

        # Verify error attribute is present
        self.assertEqual(
            attrs[ErrorAttributes.ERROR_TYPE],
            AgentInvocationError.__qualname__,
        )
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "invoke_agent")

    def test_invoke_agent_does_not_emit_event_when_disabled(self):
        """Test that invoke_agent does not emit event when emit_event is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "NoEventAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Test")])
            ]

        # Check that no event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    # ==================== Retrieval Tests ====================

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_retrieval_start_and_stop_creates_span(self):
        with self.telemetry_handler.retrieval() as invocation:
            invocation.query = "Who is John's father?"
            invocation.server_address = "api.vectordb.com"
            invocation.server_port = 8080
            invocation.attributes = {"custom": "retrieval_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "retrieval")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "retrieval",
                GEN_AI_RETRIEVAL_QUERY_TEXT: "Who is John's father?",
                ServerAttributes.SERVER_ADDRESS: "api.vectordb.com",
                ServerAttributes.SERVER_PORT: 8080,
                "custom": "retrieval_attr",
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_retrieval_with_documents(self):
        documents = [
            RetrievalDocument(
                id="123",
                score=0.95,
                content="John's father is Mike",
                metadata={},
            ),
            RetrievalDocument(
                id="124",
                score=0.87,
                content="Mike is 45 years old",
                metadata={},
            ),
        ]
        with self.telemetry_handler.retrieval() as invocation:
            invocation.query = "Who is John's father?"
            invocation.documents = documents

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GEN_AI_RETRIEVAL_DOCUMENTS, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_retrieval_with_retrieval_documents(self):
        """Test retrieval with typed RetrievalDocument list."""
        documents = [
            RetrievalDocument(
                id="doc_123",
                score=0.95,
                content="John's father is Mike",
                metadata={"source": "kb1"},
            ),
            RetrievalDocument(
                id="doc_124",
                score=0.87,
                content="Mike is 45 years old",
                metadata={"source": "kb1"},
            ),
        ]
        with self.telemetry_handler.retrieval() as invocation:
            invocation.query = "Who is John's father?"
            invocation.documents = documents

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GEN_AI_RETRIEVAL_DOCUMENTS, span_attrs)
        docs_val = span_attrs[GEN_AI_RETRIEVAL_DOCUMENTS]
        self.assertIn("doc_123", docs_val)
        self.assertIn("doc_124", docs_val)
        self.assertIn("0.95", docs_val)
        self.assertIn("0.87", docs_val)

    @patch_env_vars(stability_mode="default")
    def test_retrieval_without_sensitive_data(self):
        # Without experimental mode, documents should not be recorded
        documents = [
            RetrievalDocument(id="123", score=0.9, content="sensitive data")
        ]
        with self.telemetry_handler.retrieval() as invocation:
            invocation.query = "test query"
            invocation.documents = documents

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertNotIn(GEN_AI_RETRIEVAL_DOCUMENTS, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="NO_CONTENT",
    )
    def test_retrieval_no_content_records_id_score_only(self):
        """When content capture is NO_CONTENT, query is omitted; documents record id and score only."""
        documents = [
            RetrievalDocument(
                id="doc_123",
                score=0.95,
                content="sensitive doc content",
                metadata={"secret": "data"},
            ),
        ]
        with self.telemetry_handler.retrieval() as invocation:
            invocation.query = "secret query"
            invocation.documents = documents

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertNotIn(
            GEN_AI_RETRIEVAL_QUERY_TEXT,
            span_attrs,
            "Query should NOT be captured when content capture is NO_CONTENT",
        )
        self.assertIn(
            GEN_AI_RETRIEVAL_DOCUMENTS,
            span_attrs,
            "Documents should be recorded with id and score only",
        )
        docs_val = span_attrs[GEN_AI_RETRIEVAL_DOCUMENTS]
        self.assertIn("doc_123", docs_val)
        self.assertIn("0.95", docs_val)
        self.assertNotIn(
            "sensitive doc content",
            docs_val,
            "Content should NOT be in documents when NO_CONTENT",
        )
        self.assertNotIn("secret", docs_val)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_retrieval_manual_start_and_stop(self):
        invocation = RetrievalInvocation()
        invocation.query = "manual query"

        self.telemetry_handler.start_retrieval(invocation)
        assert invocation.span is not None
        invocation.server_address = "localhost"
        self.telemetry_handler.stop_retrieval(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "retrieval")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_RETRIEVAL_QUERY_TEXT: "manual query",
                ServerAttributes.SERVER_ADDRESS: "localhost",
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_retrieval_span_name_with_data_source_id(self):
        """Span name should be 'retrieval {data_source_id}' per LoongSuite spec."""
        with self.telemetry_handler.retrieval() as invocation:
            invocation.data_source_id = "H7STPQYOND"
            invocation.query = "test query"
            invocation.provider = "chroma"
            invocation.request_model = "embedding-model"
            invocation.top_k = 5.0

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "retrieval H7STPQYOND")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "retrieval",
                GenAI.GEN_AI_DATA_SOURCE_ID: "H7STPQYOND",
                GenAI.GEN_AI_PROVIDER_NAME: "chroma",
                GenAI.GEN_AI_REQUEST_MODEL: "embedding-model",
                GenAI.GEN_AI_REQUEST_TOP_K: 5.0,
                GEN_AI_RETRIEVAL_QUERY_TEXT: "test query",
            },
        )

    def test_retrieval_error_handling(self):
        class RetrievalError(RuntimeError):
            pass

        with self.assertRaises(RetrievalError):
            with self.telemetry_handler.retrieval() as invocation:
                invocation.query = "error query"
                raise RetrievalError("Retrieval failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: RetrievalError.__qualname__,
            },
        )

    # ==================== Rerank Documents Tests ====================

    def test_rerank_start_and_stop_creates_span(self):
        with self.telemetry_handler.rerank() as invocation:
            invocation.provider = "cohere"
            invocation.request_model = "rerank-english-v2.0"
            invocation.top_k = 5
            invocation.documents_count = 10
            invocation.return_documents = False
            invocation.attributes = {"custom": "rerank_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "rerank_documents rerank-english-v2.0")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "rerank_documents",
                GenAI.GEN_AI_PROVIDER_NAME: "cohere",
                GenAI.GEN_AI_REQUEST_MODEL: "rerank-english-v2.0",
                GenAI.GEN_AI_REQUEST_TOP_K: 5,
                GEN_AI_RERANK_DOCUMENTS_COUNT: 10,
                "gen_ai.rerank.return_documents": False,
                "custom": "rerank_attr",
            },
        )

    def test_rerank_llm_reranker_attributes(self):
        with self.telemetry_handler.rerank() as invocation:
            invocation.provider = "openai"
            invocation.request_model = "gpt-4"
            invocation.temperature = 0.0
            invocation.max_tokens = 100
            invocation.scoring_prompt = "Rate relevance from 1-5"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_REQUEST_TEMPERATURE: 0.0,
                GenAI.GEN_AI_REQUEST_MAX_TOKENS: 100,
                "gen_ai.rerank.scoring_prompt": "Rate relevance from 1-5",
            },
        )

    def test_rerank_huggingface_attributes(self):
        with self.telemetry_handler.rerank() as invocation:
            invocation.provider = "huggingface"
            invocation.device = "cuda"
            invocation.batch_size = 32
            invocation.max_length = 512
            invocation.normalize = True

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                "gen_ai.rerank.device": "cuda",
                "gen_ai.rerank.batch_size": 32,
                "gen_ai.rerank.max_length": 512,
                "gen_ai.rerank.normalize": True,
            },
        )

    def test_rerank_manual_start_and_stop(self):
        invocation = RerankInvocation(
            provider="test-provider",
            request_model="rerank-model",
            top_k=3,
        )

        self.telemetry_handler.start_rerank(invocation)
        assert invocation.span is not None
        invocation.documents_count = 20
        self.telemetry_handler.stop_rerank(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "rerank_documents rerank-model")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_REQUEST_TOP_K: 3,
                GEN_AI_RERANK_DOCUMENTS_COUNT: 20,
            },
        )

    def test_rerank_error_handling(self):
        class RerankError(RuntimeError):
            pass

        with self.assertRaises(RerankError):
            with self.telemetry_handler.rerank() as invocation:
                invocation.provider = "test"
                raise RerankError("Rerank failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: RerankError.__qualname__,
            },
        )

    # ==================== Entry Tests ====================

    def test_entry_start_and_stop_creates_span(self):
        with self.telemetry_handler.entry() as invocation:
            invocation.session_id = "ddde34343-f93a-4477-33333-sdfsdaf"
            invocation.user_id = "u-lK8JddD"
            invocation.response_time_to_first_token = 1000000
            invocation.attributes = {"custom": "entry_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "enter_ai_application_system")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "enter",
                GEN_AI_SPAN_KIND: GenAiSpanKindValues.ENTRY.value,
                GEN_AI_SESSION_ID: "ddde34343-f93a-4477-33333-sdfsdaf",
                GEN_AI_USER_ID: "u-lK8JddD",
                "gen_ai.response.time_to_first_token": 1000000,
                "custom": "entry_attr",
            },
        )

    def test_entry_manual_start_and_stop(self):
        invocation = EntryInvocation(
            session_id="session_123",
            user_id="user_456",
        )

        self.telemetry_handler.start_entry(invocation)
        assert invocation.span is not None
        invocation.response_time_to_first_token = 500000
        self.telemetry_handler.stop_entry(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "enter_ai_application_system")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "enter",
                GEN_AI_SESSION_ID: "session_123",
                GEN_AI_USER_ID: "user_456",
                "gen_ai.response.time_to_first_token": 500000,
            },
        )

    def test_entry_error_handling(self):
        class EntryError(RuntimeError):
            pass

        with self.assertRaises(EntryError):
            with self.telemetry_handler.entry() as invocation:
                invocation.session_id = "session_err"
                raise EntryError("Entry failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: EntryError.__qualname__,
            },
        )

    def test_entry_propagates_baggage_for_child_spans(self):
        """session_id and user_id set at construction time are propagated
        to Baggage so that BaggageSpanProcessor copies them to child spans."""
        entry_inv = EntryInvocation(
            session_id="sess_bag_123",
            user_id="user_bag_456",
        )
        with self.telemetry_handler.entry(entry_inv):
            current_baggage = get_all_baggage()
            self.assertEqual(
                current_baggage.get("gen_ai.session.id"), "sess_bag_123"
            )
            self.assertEqual(
                current_baggage.get("gen_ai.user.id"), "user_bag_456"
            )

            with self.telemetry_handler.embedding() as emb_inv:
                emb_inv.request_model = "text-embedding-3-small"
                emb_inv.provider = "openai"

        restored_baggage = get_all_baggage()
        self.assertNotIn("gen_ai.session.id", restored_baggage)
        self.assertNotIn("gen_ai.user.id", restored_baggage)

    def test_entry_baggage_overwrites_existing(self):
        """If baggage already contains session_id/user_id, entry overwrites them."""
        ctx = baggage_api.set_baggage("gen_ai.session.id", "old_session")
        ctx = baggage_api.set_baggage("gen_ai.user.id", "old_user", ctx)
        token = context_api.attach(ctx)

        try:
            entry_inv = EntryInvocation(
                session_id="new_session",
                user_id="new_user",
            )
            with self.telemetry_handler.entry(entry_inv):
                current_baggage = baggage_api.get_all()
                self.assertEqual(
                    current_baggage.get("gen_ai.session.id"), "new_session"
                )
                self.assertEqual(
                    current_baggage.get("gen_ai.user.id"), "new_user"
                )
        finally:
            context_api.detach(token)

    def test_entry_baggage_only_session_id(self):
        """Only session_id is set, user_id should not appear in baggage."""
        entry_inv = EntryInvocation(session_id="sess_only")
        with self.telemetry_handler.entry(entry_inv):
            current_baggage = get_all_baggage()
            self.assertEqual(
                current_baggage.get("gen_ai.session.id"), "sess_only"
            )
            self.assertNotIn("gen_ai.user.id", current_baggage)

    def test_entry_no_baggage_when_values_not_set(self):
        """When neither session_id nor user_id is set, no baggage is propagated."""
        with self.telemetry_handler.entry():
            current_baggage = get_all_baggage()
            self.assertNotIn("gen_ai.session.id", current_baggage)
            self.assertNotIn("gen_ai.user.id", current_baggage)

    # ==================== ReAct Step Tests ====================

    def test_react_step_start_and_stop_creates_span(self):
        with self.telemetry_handler.react_step() as invocation:
            invocation.finish_reason = "stop"
            invocation.round = 1
            invocation.attributes = {"custom": "react_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "react step")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "react",
                GEN_AI_SPAN_KIND: GenAiSpanKindValues.STEP.value,
                GEN_AI_REACT_FINISH_REASON: "stop",
                GEN_AI_REACT_ROUND: 1,
                "custom": "react_attr",
            },
        )

    def test_react_step_manual_start_and_stop(self):
        invocation = ReactStepInvocation(
            finish_reason="error",
            round=1,
        )

        self.telemetry_handler.start_react_step(invocation)
        assert invocation.span is not None
        invocation.finish_reason = "stop"
        invocation.round = 2
        self.telemetry_handler.stop_react_step(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "react step")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "react",
                GEN_AI_REACT_FINISH_REASON: "stop",
                GEN_AI_REACT_ROUND: 2,
            },
        )

    def test_react_step_error_handling(self):
        class ReactStepError(RuntimeError):
            pass

        with self.assertRaises(ReactStepError):
            with self.telemetry_handler.react_step() as invocation:
                invocation.round = 1
                raise ReactStepError("ReAct step failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: ReactStepError.__qualname__,
            },
        )


class TestMultimodalProcessingMixin(  # pylint: disable=too-many-public-methods
    unittest.TestCase
):
    """Tests for MultimodalProcessingMixin.

    Uses orthogonal test design to maximize coverage with minimal test cases.
    """

    def setUp(self):
        # Reset class-level state before each test
        MultimodalProcessingMixin._async_queue = None
        MultimodalProcessingMixin._async_worker = None

    def tearDown(self):
        MultimodalProcessingMixin._async_queue = None
        MultimodalProcessingMixin._async_worker = None

    @staticmethod
    def _create_mock_handler(enabled=True):
        """Helper to create a MockHandler."""
        mixin = MultimodalProcessingMixin

        class MockHandler(mixin):
            def __init__(self):
                self._multimodal_enabled = enabled
                self._logger = MagicMock()

            def _get_uploader_and_pre_uploader(self):
                return MagicMock(), MagicMock()

            def _record_llm_metrics(self, *args, **kwargs):
                pass

        return MockHandler()

    @staticmethod
    def _create_invocation_with_multimodal(with_context=False):
        """Helper to create invocation with multimodal data."""
        invocation = LLMInvocation(request_model="gpt-4")
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[
                    Base64Blob(
                        mime_type="image/png", modality="image", content="data"
                    )
                ],
            )
        ]
        if with_context:
            invocation.context_token = context_api.attach(
                context_api.set_value("_test_key", "_test_value")
            )
            invocation.span = MagicMock()
        return invocation

    # ==================== Static Method Tests ====================

    def test_quick_has_multimodal_orthogonal_cases(self):
        """Test _quick_has_multimodal with all multimodal types and edge cases."""
        mixin = MultimodalProcessingMixin

        # No multimodal: Text only
        inv_text = LLMInvocation(request_model="gpt-4")
        inv_text.input_messages = [
            InputMessage(role="user", parts=[Text(content="Hello")])
        ]
        self.assertFalse(mixin._quick_has_multimodal(inv_text))

        # Edge cases: None, empty
        inv_none = LLMInvocation(request_model="gpt-4")
        inv_none.input_messages = None
        self.assertFalse(mixin._quick_has_multimodal(inv_none))

        inv_empty = LLMInvocation(request_model="gpt-4")
        inv_empty.input_messages = [InputMessage(role="user", parts=[])]
        self.assertFalse(mixin._quick_has_multimodal(inv_empty))

        # Has multimodal: Base64Blob in input
        inv_base64 = LLMInvocation(request_model="gpt-4")
        inv_base64.input_messages = [
            InputMessage(
                role="user",
                parts=[
                    Base64Blob(
                        mime_type="image/png", modality="image", content="x"
                    )
                ],
            )
        ]
        self.assertTrue(mixin._quick_has_multimodal(inv_base64))

        # Has multimodal: Blob in input
        inv_blob = LLMInvocation(request_model="gpt-4")
        inv_blob.input_messages = [
            InputMessage(
                role="user",
                parts=[
                    Blob(
                        mime_type="image/jpeg",
                        modality="image",
                        content=b"\xff",
                    )
                ],
            )
        ]
        self.assertTrue(mixin._quick_has_multimodal(inv_blob))

        # Has multimodal: Uri in output only
        inv_uri = LLMInvocation(request_model="gpt-4")
        inv_uri.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[
                    Uri(
                        mime_type="audio/mp3", modality="audio", uri="http://x"
                    )
                ],
                finish_reason="stop",
            )
        ]
        self.assertTrue(mixin._quick_has_multimodal(inv_uri))

    def test_compute_end_time_ns_all_branches(self):
        """Test _compute_end_time_ns with all time availability combinations."""
        mixin = MultimodalProcessingMixin

        # No monotonic times → current time
        inv1 = LLMInvocation(request_model="gpt-4")
        with patch(
            "opentelemetry.util.genai._multimodal_processing.time_ns",
            return_value=1000,
        ):
            self.assertEqual(mixin._compute_end_time_ns(inv1), 1000)

        # Has monotonic but no span._start_time → current time
        inv2 = LLMInvocation(request_model="gpt-4")
        inv2.monotonic_start_s = 100.0
        inv2.monotonic_end_s = 102.0
        mock_span = MagicMock(spec=[])  # No _start_time attribute
        inv2.span = mock_span
        with patch(
            "opentelemetry.util.genai._multimodal_processing.time_ns",
            return_value=2000,
        ):
            self.assertEqual(mixin._compute_end_time_ns(inv2), 2000)

        # All times available → computed time
        inv3 = LLMInvocation(request_model="gpt-4")
        inv3.monotonic_start_s = 100.0
        inv3.monotonic_end_s = 102.5
        mock_span3 = MagicMock()
        mock_span3._start_time = 5000000000000
        inv3.span = mock_span3
        self.assertEqual(mixin._compute_end_time_ns(inv3), 5002500000000)

    def test_extract_multimodal_metadata_orthogonal(self):
        """Test _extract_multimodal_metadata extracts only Uri parts."""
        mixin = MultimodalProcessingMixin

        # None/empty → empty lists
        self.assertEqual(
            mixin._extract_multimodal_metadata(None, None), ([], [])
        )

        # Text only → empty
        input_text = [InputMessage(role="user", parts=[Text(content="Hi")])]
        self.assertEqual(
            mixin._extract_multimodal_metadata(input_text, None), ([], [])
        )

        # Uri in input → extracted
        input_uri = [
            InputMessage(
                role="user",
                parts=[
                    Uri(
                        mime_type="image/png", modality="image", uri="http://x"
                    )
                ],
            )
        ]
        meta, _ = mixin._extract_multimodal_metadata(input_uri, None)
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]["type"], "uri")

        # Multiple Uris
        input_multi = [
            InputMessage(
                role="user",
                parts=[
                    Uri(
                        mime_type="image/png", modality="image", uri="http://1"
                    ),
                    Text(content="desc"),
                    Uri(
                        mime_type="image/jpeg",
                        modality="image",
                        uri="http://2",
                    ),
                ],
            )
        ]
        meta, _ = mixin._extract_multimodal_metadata(input_multi, None)
        self.assertEqual(len(meta), 2)

    # ==================== _init_multimodal Tests (via env vars) ====================

    @patch.dict(
        os.environ,
        {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "none"},
    )
    def test_init_multimodal_disabled_when_mode_none(self):
        """Test _init_multimodal with mode=none."""

        class Handler(MultimodalProcessingMixin):
            def _get_uploader_and_pre_uploader(self):
                return MagicMock(), MagicMock()

        handler = Handler()
        handler._init_multimodal()
        self.assertFalse(handler._multimodal_enabled)

    @patch_env_vars(
        "gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE="both",
    )
    def test_init_multimodal_enabled_or_disabled_by_uploader(self):
        """Test _init_multimodal enabled when uploader available, disabled when None."""

        class HandlerWithUploader(MultimodalProcessingMixin):
            def _get_uploader_and_pre_uploader(self):
                return MagicMock(), MagicMock()

        h1 = HandlerWithUploader()
        h1._init_multimodal()
        self.assertTrue(h1._multimodal_enabled)

        class HandlerWithoutUploader(MultimodalProcessingMixin):
            def _get_uploader_and_pre_uploader(self):
                return None, None

        h2 = HandlerWithoutUploader()
        h2._init_multimodal()
        self.assertFalse(h2._multimodal_enabled)

    # ==================== process_multimodal_stop/fail Tests ====================

    def test_process_multimodal_returns_false_on_precondition_failure(self):
        """Test process_multimodal_stop/fail returns False when preconditions not met."""
        handler = self._create_mock_handler(enabled=True)
        error = Error(message="err", type=RuntimeError)

        # context_token is None
        inv1 = self._create_invocation_with_multimodal()
        inv1.context_token = None
        inv1.span = MagicMock()
        self.assertFalse(
            handler.process_multimodal_stop(inv1, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
        )
        self.assertFalse(
            handler.process_multimodal_fail(inv1, error, method="fail_llm")  # pylint: disable=unexpected-keyword-arg
        )

        # span is None
        inv2 = self._create_invocation_with_multimodal()
        inv2.context_token = MagicMock()
        inv2.span = None
        self.assertFalse(
            handler.process_multimodal_stop(inv2, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
        )

        # No multimodal data
        inv3 = LLMInvocation(request_model="gpt-4")
        inv3.context_token = MagicMock()
        inv3.span = MagicMock()
        inv3.input_messages = [
            InputMessage(role="user", parts=[Text(content="Hi")])
        ]
        self.assertFalse(
            handler.process_multimodal_stop(inv3, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
        )

        # multimodal_enabled=False
        handler_disabled = self._create_mock_handler(enabled=False)
        inv4 = self._create_invocation_with_multimodal(with_context=True)
        self.assertFalse(
            handler_disabled.process_multimodal_stop(inv4, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
        )

    @patch_env_vars(
        "gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE="both",
    )
    def test_process_multimodal_fallback_on_queue_issues(self):
        """Test process_multimodal_stop/fail uses fallback when queue is None or full."""
        handler = self._create_mock_handler()
        inv = self._create_invocation_with_multimodal(with_context=True)
        error = Error(message="err", type=RuntimeError)

        with patch.object(MultimodalProcessingMixin, "_ensure_async_worker"):
            # Queue is None
            MultimodalProcessingMixin._async_queue = None
            with patch.object(handler, "_fallback_stop") as mock_end:
                self.assertTrue(
                    handler.process_multimodal_stop(inv, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
                )
                mock_end.assert_called_once()

            # Reset invocation context token (use real token for _safe_detach)
            inv.context_token = context_api.attach(
                context_api.set_value("_test_key", "_test_value")
            )
            with patch.object(handler, "_fallback_fail") as mock_fail:
                self.assertTrue(
                    handler.process_multimodal_fail(  # pylint: disable=unexpected-keyword-arg
                        inv, error, method="fail_llm"
                    )
                )
                mock_fail.assert_called_once()

            # Queue is full
            MultimodalProcessingMixin._async_queue = queue.Queue(maxsize=1)
            MultimodalProcessingMixin._async_queue.put("dummy")
            inv.context_token = context_api.attach(
                context_api.set_value("_test_key", "_test_value")
            )
            with patch.object(handler, "_fallback_stop") as mock_end2:
                self.assertTrue(
                    handler.process_multimodal_stop(inv, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
                )
                mock_end2.assert_called_once()

    @patch_env_vars(
        "gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE="both",
    )
    def test_process_multimodal_enqueues_task(self):
        """Test process_multimodal_stop/fail enqueues tasks correctly."""
        handler = self._create_mock_handler()
        error = Error(message="err", type=RuntimeError)

        with patch.object(MultimodalProcessingMixin, "_ensure_async_worker"):
            MultimodalProcessingMixin._async_queue = queue.Queue(maxsize=100)

            # stop
            inv1 = self._create_invocation_with_multimodal(with_context=True)
            self.assertTrue(
                handler.process_multimodal_stop(inv1, method="stop_llm")  # pylint: disable=unexpected-keyword-arg
            )
            task = MultimodalProcessingMixin._async_queue.get_nowait()
            self.assertEqual(task.method, "stop_llm")

            # fail
            inv2 = self._create_invocation_with_multimodal(with_context=True)
            self.assertTrue(
                handler.process_multimodal_fail(inv2, error, method="fail_llm")  # pylint: disable=unexpected-keyword-arg
            )
            task = MultimodalProcessingMixin._async_queue.get_nowait()
            self.assertEqual(task.method, "fail_llm")
            self.assertEqual(task.error, error)

    # ==================== Fallback / Async Methods Tests ====================

    def test_fallback_and_async_methods_handle_span_none(self):
        """Test fallback and async methods return early when span is None."""
        handler = self._create_mock_handler()
        inv = LLMInvocation(request_model="gpt-4")
        inv.span = None

        # Should not raise
        handler._fallback_stop(inv, "stop_llm")  # pylint: disable=no-member
        handler._fallback_fail(  # pylint: disable=no-member
            inv, Error(message="err", type=RuntimeError), "fail_llm"
        )
        handler._async_stop_llm(
            _MultimodalAsyncTask(
                invocation=inv, method="stop_llm", handler=handler
            )
        )
        handler._async_fail_llm(
            _MultimodalAsyncTask(
                invocation=inv,
                method="fail_llm",
                error=Error(message="err", type=RuntimeError),
                handler=handler,
            )
        )

        # error is None for async_fail_llm
        inv2 = LLMInvocation(request_model="gpt-4")
        inv2.span = MagicMock()
        handler._async_fail_llm(
            _MultimodalAsyncTask(
                invocation=inv2, method="fail_llm", error=None, handler=handler
            )
        )

    def test_fallback_methods_apply_attributes(self):
        """Test fallback methods apply correct attributes and end span."""
        handler = self._create_mock_handler()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000

        inv = LLMInvocation(request_model="gpt-4")
        inv.span = mock_span
        error = Error(message="err", type=ValueError)

        with patch(
            "opentelemetry.util.genai._multimodal_processing._apply_llm_finish_attributes"
        ) as m1, patch(
            "opentelemetry.util.genai._multimodal_processing._apply_error_attributes"
        ) as m2, patch(
            "opentelemetry.util.genai._multimodal_processing._maybe_emit_llm_event"
        ):  # fmt: skip
            handler._fallback_stop(inv, "stop_llm")  # pylint: disable=no-member
            m1.assert_called_with(mock_span, inv)
            mock_span.end.assert_called_once()

            mock_span.reset_mock()
            handler._fallback_fail(inv, error, "fail_llm")  # pylint: disable=no-member
            m2.assert_called_with(mock_span, error)
            mock_span.end.assert_called_once()

    def test_async_stop_and_fail_llm_process_correctly(self):
        """Test _async_stop_llm and _async_fail_llm process multimodal and end span."""
        handler = self._create_mock_handler()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000
        mock_span.get_span_context.return_value = MagicMock()

        inv = LLMInvocation(request_model="gpt-4")
        inv.span = mock_span
        inv.input_messages = [
            InputMessage(
                role="user",
                parts=[
                    Uri(
                        mime_type="image/png", modality="image", uri="http://x"
                    )
                ],
            )
        ]

        with patch(
            "opentelemetry.util.genai._multimodal_processing._apply_llm_finish_attributes"
        ) as m1, patch(
            "opentelemetry.util.genai._multimodal_processing._apply_error_attributes"
        ) as m2, patch(
            "opentelemetry.util.genai._multimodal_processing._maybe_emit_llm_event"
        ):  # fmt: skip
            handler._async_stop_llm(
                _MultimodalAsyncTask(
                    invocation=inv, method="stop_llm", handler=handler
                )
            )
            m1.assert_called_once()
            mock_span.end.assert_called_once()
            mock_span.set_attribute.assert_called()

            mock_span.reset_mock()
            error = Error(message="err", type=ValueError)
            handler._async_fail_llm(
                _MultimodalAsyncTask(
                    invocation=inv,
                    method="fail_llm",
                    error=error,
                    handler=handler,
                )
            )
            m2.assert_called_once()
            mock_span.end.assert_called_once()

    # ==================== Agent Async / Fallback / Dispatch Tests ====================

    @staticmethod
    def _create_agent_invocation_with_multimodal(with_context=False):
        """Helper to create InvokeAgentInvocation with multimodal data."""
        invocation = InvokeAgentInvocation(provider="test")
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[
                    Uri(
                        mime_type="image/png", modality="image", uri="http://x"
                    )
                ],
            )
        ]
        if with_context:
            invocation.context_token = context_api.attach(
                context_api.set_value("_test_key", "_test_value")
            )
            invocation.span = MagicMock()
        return invocation

    @staticmethod
    def _create_mock_handler_with_agent_metrics(enabled=True):
        """MockHandler that also has _record_extended_metrics."""
        mixin = MultimodalProcessingMixin

        class MockHandler(mixin):
            def __init__(self):
                self._multimodal_enabled = enabled
                self._logger = MagicMock()

            def _get_uploader_and_pre_uploader(self):
                return MagicMock(), MagicMock()

            def _record_llm_metrics(self, *args, **kwargs):
                pass

            def _record_extended_metrics(self, *args, **kwargs):
                pass

        return MockHandler()

    def test_dispatch_task_routes_agent_methods(self):
        """Test _dispatch_task dispatches stop_agent/fail_agent correctly."""
        handler = self._create_mock_handler_with_agent_metrics()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000
        mock_span.get_span_context.return_value = MagicMock()

        inv = self._create_agent_invocation_with_multimodal()
        inv.span = mock_span
        error = Error(message="err", type=RuntimeError)

        with patch(
            "opentelemetry.util.genai._multimodal_processing._apply_invoke_agent_finish_attributes"
        ) as m_attr, patch(
            "opentelemetry.util.genai._multimodal_processing._maybe_emit_invoke_agent_event"
        ):  # fmt: skip
            # stop_agent
            handler._dispatch_task(  # pylint: disable=no-member
                _MultimodalAsyncTask(
                    invocation=inv, method="stop_agent", handler=handler
                )
            )
            m_attr.assert_called_once()
            mock_span.end.assert_called_once()

            mock_span.reset_mock()
            m_attr.reset_mock()

            # fail_agent
            handler._dispatch_task(  # pylint: disable=no-member
                _MultimodalAsyncTask(
                    invocation=inv,
                    method="fail_agent",
                    error=error,
                    handler=handler,
                )
            )
            m_attr.assert_called_once()
            mock_span.end.assert_called_once()

    def test_async_stop_and_fail_agent_process_correctly(self):
        """Test _async_stop/fail_invoke_agent processes multimodal and end span."""
        handler = self._create_mock_handler_with_agent_metrics()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000
        mock_span.get_span_context.return_value = MagicMock()

        inv = self._create_agent_invocation_with_multimodal()
        inv.span = mock_span

        with patch(
            "opentelemetry.util.genai._multimodal_processing._apply_invoke_agent_finish_attributes"
        ) as m1, patch(
            "opentelemetry.util.genai._multimodal_processing._apply_error_attributes"
        ) as m2, patch(
            "opentelemetry.util.genai._multimodal_processing._maybe_emit_invoke_agent_event"
        ):  # fmt: skip
            handler._async_stop_invoke_agent(  # pylint: disable=no-member
                _MultimodalAsyncTask(
                    invocation=inv, method="stop_agent", handler=handler
                )
            )
            m1.assert_called_once()
            mock_span.end.assert_called_once()
            mock_span.set_attribute.assert_called()  # multimodal metadata

            mock_span.reset_mock()
            error = Error(message="err", type=ValueError)
            handler._async_fail_invoke_agent(  # pylint: disable=no-member
                _MultimodalAsyncTask(
                    invocation=inv,
                    method="fail_agent",
                    error=error,
                    handler=handler,
                )
            )
            m2.assert_called_with(mock_span, error)
            mock_span.end.assert_called_once()

    def test_agent_async_methods_handle_span_none(self):
        """Test agent async methods return early when span is None."""
        handler = self._create_mock_handler_with_agent_metrics()
        inv = InvokeAgentInvocation(provider="test")
        inv.span = None

        # Should not raise
        handler._async_stop_invoke_agent(  # pylint: disable=no-member
            _MultimodalAsyncTask(
                invocation=inv, method="stop_agent", handler=handler
            )
        )
        handler._async_fail_invoke_agent(  # pylint: disable=no-member
            _MultimodalAsyncTask(
                invocation=inv,
                method="fail_agent",
                error=Error(message="err", type=RuntimeError),
                handler=handler,
            )
        )

    def test_fallback_stop_agent_applies_attributes(self):
        """Test _fallback_stop with stop_agent method applies agent attributes."""
        handler = self._create_mock_handler_with_agent_metrics()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000

        inv = InvokeAgentInvocation(provider="test")
        inv.span = mock_span

        with patch(
            "opentelemetry.util.genai._multimodal_processing._apply_invoke_agent_finish_attributes"
        ) as m1, patch(
            "opentelemetry.util.genai._multimodal_processing._maybe_emit_invoke_agent_event"
        ):  # fmt: skip
            handler._fallback_stop(inv, "stop_agent")  # pylint: disable=no-member
            m1.assert_called_with(mock_span, inv)
            mock_span.end.assert_called_once()

    def test_fallback_fail_agent_applies_attributes(self):
        """Test _fallback_fail with fail_agent method applies agent attributes."""
        handler = self._create_mock_handler_with_agent_metrics()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000

        inv = InvokeAgentInvocation(provider="test")
        inv.span = mock_span
        error = Error(message="err", type=ValueError)

        with patch(
            "opentelemetry.util.genai._multimodal_processing._apply_invoke_agent_finish_attributes"
        ) as m1, patch(
            "opentelemetry.util.genai._multimodal_processing._apply_error_attributes"
        ) as m2, patch(
            "opentelemetry.util.genai._multimodal_processing._maybe_emit_invoke_agent_event"
        ):  # fmt: skip
            handler._fallback_fail(inv, error, "fail_agent")  # pylint: disable=no-member
            m1.assert_called_with(mock_span, inv)
            m2.assert_called_with(mock_span, error)
            mock_span.end.assert_called_once()

    # ==================== Worker & Lifecycle Tests ====================

    def test_ensure_worker_and_shutdown(self):
        """Test _ensure_async_worker creates resources and shutdown cleans them."""
        mixin = MultimodalProcessingMixin

        # Not started
        self.assertIsNone(mixin._async_worker)
        mixin.shutdown_multimodal_worker(timeout=0.1)  # Should not raise

        # Start
        mixin._ensure_async_worker()
        self.assertIsNotNone(mixin._async_queue)
        self.assertTrue(mixin._async_worker.is_alive())

        # Shutdown
        mixin.shutdown_multimodal_worker(timeout=2.0)
        self.assertIsNone(mixin._async_worker)
        self.assertIsNone(mixin._async_queue)

    def test_at_fork_reinit_resets_state(self):
        """Test _at_fork_reinit resets class-level state."""
        mixin = MultimodalProcessingMixin
        mixin._async_queue = queue.Queue()
        mixin._async_worker = threading.Thread(target=lambda: None)
        mixin._atexit_handler = object()

        mixin._at_fork_reinit()

        self.assertIsNone(mixin._async_queue)
        self.assertIsNone(mixin._async_worker)
        self.assertIsNone(mixin._atexit_handler)
        self.assertTrue(hasattr(mixin._async_lock, "acquire"))

    def test_async_worker_loop_processes_tasks(self):  # pylint: disable=no-self-use
        """Test _async_worker_loop processes stop/fail tasks and handles errors.

        Note: Method uses self.assertTrue but pylint doesn't detect it in nested code.
        """
        mixin = MultimodalProcessingMixin

        # Test 1: Processes stop task
        class Handler1(mixin):
            def __init__(self):
                self.called = False

            def _async_stop_llm(self, task):
                self.called = True

        handler1 = Handler1()
        mixin._async_queue = queue.Queue()
        inv1 = LLMInvocation(request_model="gpt-4")
        inv1.span = MagicMock()
        mixin._async_queue.put(
            _MultimodalAsyncTask(
                invocation=inv1, method="stop_llm", handler=handler1
            )
        )
        mixin._async_queue.put(None)

        worker_thread = threading.Thread(target=mixin._async_worker_loop)
        worker_thread.start()
        worker_thread.join(timeout=2.0)
        self.assertTrue(handler1.called)

        # Test 2: Skips task with None handler
        mixin._async_queue = queue.Queue()
        mixin._async_queue.put(
            _MultimodalAsyncTask(
                invocation=inv1, method="stop_llm", handler=None
            )
        )
        mixin._async_queue.put(None)
        worker_thread = threading.Thread(target=mixin._async_worker_loop)
        worker_thread.start()
        worker_thread.join(timeout=2.0)  # Should not raise

        # Test 3: Handles exception and ends span
        class Handler2(mixin):
            def _async_stop_llm(self, task):
                raise RuntimeError("error")

        mock_span = MagicMock()
        inv2 = LLMInvocation(request_model="gpt-4")
        inv2.span = mock_span
        inv2.monotonic_start_s = 100.0
        inv2.monotonic_end_s = 102.0

        mixin._async_queue = queue.Queue()
        mixin._async_queue.put(
            _MultimodalAsyncTask(
                invocation=inv2, method="stop_llm", handler=Handler2()
            )
        )
        mixin._async_queue.put(None)
        worker_thread = threading.Thread(target=mixin._async_worker_loop)
        worker_thread.start()
        worker_thread.join(timeout=2.0)
        mock_span.end.assert_called_once()

    def test_separate_and_upload(self):
        """Test _separate_and_upload calls uploader and handles exceptions."""

        class Handler(MultimodalProcessingMixin):
            pass

        handler = Handler()
        mock_span = MagicMock()
        mock_span._start_time = 1000000000
        mock_span.get_span_context.return_value = MagicMock()

        mock_uploader = MagicMock()
        mock_pre_uploader = MagicMock()
        mock_pre_uploader.pre_upload.return_value = [MagicMock(), MagicMock()]

        inv = LLMInvocation(request_model="gpt-4")

        handler._separate_and_upload(
            mock_span, inv, mock_uploader, mock_pre_uploader
        )
        mock_pre_uploader.pre_upload.assert_called_once()
        self.assertEqual(mock_uploader.upload.call_count, 2)

        # Exception handling
        mock_span2 = MagicMock()
        mock_span2.get_span_context.side_effect = RuntimeError("err")
        handler._separate_and_upload(
            mock_span2, inv, mock_uploader, mock_pre_uploader
        )  # Should not raise


class TestExtendedTelemetryHandlerShutdown(unittest.TestCase):
    """Tests for ExtendedTelemetryHandler shutdown behavior.

    Design: use the real worker loop and control task execution through
    mock task.handler._async_stop_llm.
    """

    def test_shutdown_waits_for_slow_task(self):
        """Test shutdown waits for slow task completion (poison-pill mode)."""
        # Reset state
        MultimodalProcessingMixin._async_queue = None
        MultimodalProcessingMixin._async_worker = None

        # Track task processing
        task_started = threading.Event()
        task_completed = threading.Event()

        try:
            # Ensure worker is started
            MultimodalProcessingMixin._ensure_async_worker()

            # Create a mock handler with slow processing
            mock_handler = MagicMock()

            def slow_stop(task):
                task_started.set()
                time.sleep(0.15)
                task_completed.set()

            mock_handler._dispatch_task = slow_stop

            mock_task = _MultimodalAsyncTask(
                invocation=MagicMock(), method="stop_llm", handler=mock_handler
            )
            MultimodalProcessingMixin._async_queue.put(mock_task)

            # Wait for the task to start
            self.assertTrue(
                task_started.wait(timeout=1.0), "Task did not start"
            )

            # Shutdown should wait for task completion
            # (the poison pill is queued after the task)
            MultimodalProcessingMixin.shutdown_multimodal_worker(timeout=5.0)

            # Verify the task has completed
            self.assertTrue(
                task_completed.is_set(), "Task should have completed"
            )
            # Idempotency: repeated shutdown should not fail
            MultimodalProcessingMixin.shutdown_multimodal_worker(timeout=1.0)
        finally:
            MultimodalProcessingMixin._async_queue = None
            MultimodalProcessingMixin._async_worker = None

    def test_shutdown_timeout_exits(self):
        """Test shutdown exits when timeout is reached."""
        # Reset state
        MultimodalProcessingMixin._async_queue = None
        MultimodalProcessingMixin._async_worker = None

        block_event = threading.Event()
        task_started = threading.Event()

        try:
            MultimodalProcessingMixin._ensure_async_worker()

            mock_handler = MagicMock()

            def blocking_stop(task):
                task_started.set()
                block_event.wait(timeout=5.0)

            mock_handler._dispatch_task = blocking_stop

            mock_task = _MultimodalAsyncTask(
                invocation=MagicMock(), method="stop_llm", handler=mock_handler
            )
            MultimodalProcessingMixin._async_queue.put(mock_task)

            # Wait for the task to start
            self.assertTrue(
                task_started.wait(timeout=1.0), "Task did not start"
            )

            # Shutdown timeout=0.3s, task blocks for 5s
            start = time.time()
            timeout = 0.3
            MultimodalProcessingMixin.shutdown_multimodal_worker(
                timeout=timeout
            )
            elapsed = time.time() - start

            # Verify it returns after timeout (cannot be shorter than timeout)
            self.assertLess(
                elapsed, timeout + 0.2, f"shutdown took {elapsed:.2f}s"
            )
            self.assertGreaterEqual(
                elapsed, timeout, f"shutdown too fast: {elapsed:.2f}s"
            )
        finally:
            block_event.set()
            time.sleep(0.1)
            MultimodalProcessingMixin._async_queue = None
            MultimodalProcessingMixin._async_worker = None


class TestExtendedHandlerAtexitShutdown(unittest.TestCase):
    def setUp(self):
        ExtendedTelemetryHandler._shutdown_called = False

    @patch.object(ExtendedTelemetryHandler, "_shutdown_uploader")
    @patch.object(ExtendedTelemetryHandler, "_shutdown_pre_uploader")
    @patch.object(ExtendedTelemetryHandler, "shutdown_multimodal_worker")
    def test_shutdown_sequence(
        self,
        mock_shutdown_worker: MagicMock,
        mock_shutdown_pre_uploader: MagicMock,
        mock_shutdown_uploader: MagicMock,
    ):
        calls = []

        mock_shutdown_worker.side_effect = lambda timeout: calls.append(
            ("handler", timeout)
        )
        mock_shutdown_pre_uploader.side_effect = lambda timeout: calls.append(
            ("pre_uploader", timeout)
        )
        mock_shutdown_uploader.side_effect = lambda timeout: calls.append(
            ("uploader", timeout)
        )

        ExtendedTelemetryHandler.shutdown(  # pylint: disable=no-member
            worker_timeout=1.0,
            pre_uploader_timeout=2.0,
            uploader_timeout=3.0,
        )

        self.assertEqual(
            calls,
            [("handler", 1.0), ("pre_uploader", 2.0), ("uploader", 3.0)],
        )

    @patch.object(ExtendedTelemetryHandler, "_shutdown_uploader")
    @patch.object(ExtendedTelemetryHandler, "_shutdown_pre_uploader")
    @patch.object(ExtendedTelemetryHandler, "shutdown_multimodal_worker")
    def test_shutdown_idempotent(  # pylint: disable=no-self-use
        self,
        mock_shutdown_worker: MagicMock,
        mock_shutdown_pre_uploader: MagicMock,
        mock_shutdown_uploader: MagicMock,
    ):
        ExtendedTelemetryHandler.shutdown()  # pylint: disable=no-member
        ExtendedTelemetryHandler.shutdown()  # pylint: disable=no-member

        mock_shutdown_worker.assert_called_once()
        mock_shutdown_pre_uploader.assert_called_once()
        mock_shutdown_uploader.assert_called_once()
