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

import os
import unittest
from typing import Any, Mapping
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.util.genai._configuration import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    is_experimental_mode,
)

# Backward compatibility for InMemoryLogExporter -> InMemoryLogRecordExporter rename
# Changed in opentelemetry-sdk@0.60b0
try:
    from opentelemetry.sdk._logs.export import (  # pylint: disable=no-name-in-module
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )
except ImportError:
    # Fallback to old name for compatibility with older SDK versions
    from opentelemetry.sdk._logs.export import (
        InMemoryLogExporter as InMemoryLogRecordExporter,
    )
    from opentelemetry.sdk._logs.export import (
        SimpleLogRecordProcessor,
    )

from opentelemetry.sdk._logs import LoggerProvider
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
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_memory import MemoryInvocation
from opentelemetry.util.genai.extended_semconv.gen_ai_memory_attributes import (
    GEN_AI_MEMORY_AGENT_ID,
    GEN_AI_MEMORY_APP_ID,
    GEN_AI_MEMORY_ID,
    GEN_AI_MEMORY_INPUT_MESSAGES,
    GEN_AI_MEMORY_LIMIT,
    GEN_AI_MEMORY_MEMORY_TYPE,
    GEN_AI_MEMORY_OPERATION,
    GEN_AI_MEMORY_OUTPUT_MESSAGES,
    GEN_AI_MEMORY_PAGE,
    GEN_AI_MEMORY_PAGE_SIZE,
    GEN_AI_MEMORY_RERANK,
    GEN_AI_MEMORY_RUN_ID,
    GEN_AI_MEMORY_THRESHOLD,
    GEN_AI_MEMORY_TOP_K,
    GEN_AI_MEMORY_USER_ID,
)


def patch_env_vars(stability_mode, content_capturing=None, emit_event=None):
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

        @patch.dict(os.environ, env_vars)
        def wrapper(*args, **kwargs):
            # Reset state.
            is_experimental_mode.cache_clear()
            is_experimental_mode()
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


class TestMemoryOperations(unittest.TestCase):  # pylint: disable=too-many-public-methods
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

        self.telemetry_handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

    def tearDown(self):
        # Clear spans, logs and reset the singleton telemetry handler so each test starts clean
        self.span_exporter.clear()
        self.log_exporter.clear()
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")

    # ==================== Memory Operation Tests ====================

    def test_memory_add_start_and_stop_creates_span(self):
        invocation = MemoryInvocation(operation="add")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"
            invocation.run_id = "run_789"
            invocation.input_messages = "Remember that user likes apples"
            invocation.server_address = "api.mem0.ai"
            invocation.server_port = 443
            invocation.attributes = {"custom": "value"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "memory_operation add")
        self.assertEqual(span.kind, trace.SpanKind.CLIENT)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "memory_operation",
                GEN_AI_MEMORY_OPERATION: "add",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_AGENT_ID: "agent_456",
                GEN_AI_MEMORY_RUN_ID: "run_789",
                ServerAttributes.SERVER_ADDRESS: "api.mem0.ai",
                ServerAttributes.SERVER_PORT: 443,
                "custom": "value",
            },
        )

    def test_memory_search_with_parameters(self):
        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"
            invocation.limit = 10
            invocation.threshold = 0.7
            invocation.rerank = True
            invocation.top_k = 5

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "memory_operation search")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "search",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_AGENT_ID: "agent_456",
                GEN_AI_MEMORY_LIMIT: 10,
                GEN_AI_MEMORY_THRESHOLD: 0.7,
                GEN_AI_MEMORY_RERANK: True,
                GEN_AI_MEMORY_TOP_K: 5,
            },
        )

    def test_memory_update_operation(self):
        invocation = MemoryInvocation(operation="update")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.memory_id = "mem_abc123"
            invocation.user_id = "user_123"
            invocation.input_messages = "Updated memory content"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "update",
                GEN_AI_MEMORY_ID: "mem_abc123",
                GEN_AI_MEMORY_USER_ID: "user_123",
            },
        )

    def test_memory_get_operation(self):
        invocation = MemoryInvocation(operation="get")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.memory_id = "mem_xyz789"
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "get",
                GEN_AI_MEMORY_ID: "mem_xyz789",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_AGENT_ID: "agent_456",
            },
        )

    def test_memory_get_all_with_pagination(self):
        invocation = MemoryInvocation(operation="get_all")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.page = 1
            invocation.page_size = 100

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "get_all",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_PAGE: 1,
                GEN_AI_MEMORY_PAGE_SIZE: 100,
            },
        )

    def test_memory_history_operation(self):
        invocation = MemoryInvocation(operation="history")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"
            invocation.run_id = "run_789"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "history",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_AGENT_ID: "agent_456",
                GEN_AI_MEMORY_RUN_ID: "run_789",
            },
        )

    def test_memory_with_app_id(self):
        """Test memory operation with app_id (for managed platforms)."""
        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.app_id = "app_001"
            invocation.user_id = "user_123"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "search",
                GEN_AI_MEMORY_APP_ID: "app_001",
                GEN_AI_MEMORY_USER_ID: "user_123",
            },
        )

    def test_memory_with_memory_type(self):
        invocation = MemoryInvocation(operation="add")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.memory_type = "procedural_memory"
            invocation.user_id = "user_123"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "add",
                GEN_AI_MEMORY_MEMORY_TYPE: "procedural_memory",
                GEN_AI_MEMORY_USER_ID: "user_123",
            },
        )

    def test_memory_manual_start_and_stop(self):
        invocation = MemoryInvocation(operation="search")
        invocation.user_id = "user_123"
        invocation.limit = 20

        self.telemetry_handler.start_memory(invocation)
        assert invocation.span is not None
        self.telemetry_handler.stop_memory(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "memory_operation search")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "search",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_LIMIT: 20,
            },
        )

    def test_memory_error_handling(self):
        class MemoryOperationError(RuntimeError):
            pass

        with self.assertRaises(MemoryOperationError):
            invocation = MemoryInvocation(operation="add")
            with self.telemetry_handler.memory(invocation) as invocation:
                invocation.user_id = "user_123"
                raise MemoryOperationError("Memory operation failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: MemoryOperationError.__qualname__,
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_memory_with_content_capturing(self):
        """Test that input/output messages are captured when content capturing is enabled."""
        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.input_messages = "What does the user like?"
            invocation.output_messages = "The user likes apples"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify messages are captured
        self.assertIn(GEN_AI_MEMORY_INPUT_MESSAGES, span_attrs)
        self.assertIn(GEN_AI_MEMORY_OUTPUT_MESSAGES, span_attrs)
        self.assertEqual(
            span_attrs[GEN_AI_MEMORY_INPUT_MESSAGES],
            "What does the user like?",
        )
        self.assertEqual(
            span_attrs[GEN_AI_MEMORY_OUTPUT_MESSAGES], "The user likes apples"
        )

    def test_memory_without_content_capturing(self):
        """Test that messages are NOT captured when content capturing is disabled."""
        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.input_messages = "What does the user like?"
            invocation.output_messages = "The user likes apples"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify messages are NOT captured
        self.assertNotIn(GEN_AI_MEMORY_INPUT_MESSAGES, span_attrs)
        self.assertNotIn(GEN_AI_MEMORY_OUTPUT_MESSAGES, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="EVENT_ONLY",
        emit_event="true",
    )
    def test_memory_emits_event(self):
        """Test that memory operation emits events when emit_event is enabled."""
        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"
            invocation.input_messages = "What does the user like?"
            invocation.output_messages = "The user likes apples"

        # Check that event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record

        # Verify event name
        self.assertEqual(
            log_record.event_name,
            "gen_ai.memory.operation.details",
        )

        # Verify event attributes
        attrs = log_record.attributes
        self.assertIsNotNone(attrs)
        self.assertEqual(
            attrs[GenAI.GEN_AI_OPERATION_NAME], "memory_operation"
        )
        self.assertEqual(attrs[GEN_AI_MEMORY_OPERATION], "search")
        self.assertEqual(attrs[GEN_AI_MEMORY_USER_ID], "user_123")
        self.assertEqual(attrs[GEN_AI_MEMORY_AGENT_ID], "agent_456")
        self.assertIn(GEN_AI_MEMORY_INPUT_MESSAGES, attrs)
        self.assertIn(GEN_AI_MEMORY_OUTPUT_MESSAGES, attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_AND_EVENT",
        emit_event="true",
    )
    def test_memory_emits_event_and_span(self):
        """Test that memory operation emits both event and span when emit_event is enabled."""
        invocation = MemoryInvocation(operation="add")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.input_messages = "User likes apples"

        # Check span was created
        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GEN_AI_MEMORY_INPUT_MESSAGES, span_attrs)

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        self.assertEqual(
            log_record.event_name,
            "gen_ai.memory.operation.details",
        )
        self.assertIn(GEN_AI_MEMORY_INPUT_MESSAGES, log_record.attributes)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="EVENT_ONLY",
        emit_event="true",
    )
    def test_memory_emits_event_with_error(self):
        """Test that memory operation emits event with error when operation fails."""

        class MemoryOperationError(RuntimeError):
            pass

        with self.assertRaises(MemoryOperationError):
            invocation = MemoryInvocation(operation="add")
            with self.telemetry_handler.memory(invocation) as invocation:
                invocation.user_id = "user_123"
                invocation.input_messages = "Test memory"
                raise MemoryOperationError("Memory operation failed")

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        attrs = log_record.attributes

        # Verify error attribute is present
        self.assertEqual(
            attrs[ErrorAttributes.ERROR_TYPE],
            MemoryOperationError.__qualname__,
        )
        self.assertEqual(
            attrs[GenAI.GEN_AI_OPERATION_NAME], "memory_operation"
        )

    def test_memory_does_not_emit_event_when_disabled(self):
        """Test that memory operation does not emit event when emit_event is disabled."""
        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.input_messages = "Test query"

        # Check that no event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    def test_memory_batch_update_operation(self):
        invocation = MemoryInvocation(operation="batch_update")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "batch_update",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_AGENT_ID: "agent_456",
            },
        )

    def test_memory_delete_operation(self):
        invocation = MemoryInvocation(operation="delete")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.memory_id = "mem_to_delete"
            invocation.user_id = "user_123"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "delete",
                GEN_AI_MEMORY_ID: "mem_to_delete",
                GEN_AI_MEMORY_USER_ID: "user_123",
            },
        )

    def test_memory_batch_delete_operation(self):
        invocation = MemoryInvocation(operation="batch_delete")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "batch_delete",
                GEN_AI_MEMORY_USER_ID: "user_123",
            },
        )

    def test_memory_delete_all_operation(self):
        invocation = MemoryInvocation(operation="delete_all")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.agent_id = "agent_456"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_MEMORY_OPERATION: "delete_all",
                GEN_AI_MEMORY_USER_ID: "user_123",
                GEN_AI_MEMORY_AGENT_ID: "agent_456",
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_memory_with_json_input_output(self):
        """Test that JSON input/output messages are properly serialized."""
        input_data = {"query": "What does user like?", "context": "shopping"}
        output_data = [
            {"memory_id": "mem1", "content": "User likes apples"},
            {"memory_id": "mem2", "content": "User likes oranges"},
        ]

        invocation = MemoryInvocation(operation="search")
        with self.telemetry_handler.memory(invocation) as invocation:
            invocation.user_id = "user_123"
            invocation.input_messages = input_data
            invocation.output_messages = output_data

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify messages are captured as JSON strings
        self.assertIn(GEN_AI_MEMORY_INPUT_MESSAGES, span_attrs)
        self.assertIn(GEN_AI_MEMORY_OUTPUT_MESSAGES, span_attrs)
        # Should be JSON strings
        self.assertIsInstance(span_attrs[GEN_AI_MEMORY_INPUT_MESSAGES], str)
        self.assertIsInstance(span_attrs[GEN_AI_MEMORY_OUTPUT_MESSAGES], str)
