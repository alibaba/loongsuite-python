# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from google.genai.types import (
    FunctionDeclarationDict,
    GenerateContentConfig,
    GoogleMaps,
    ToolDict,
)
from pydantic import BaseModel, Field

from opentelemetry import context as context_api
from opentelemetry.instrumentation.google_genai import (
    GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

from .base import TestCase

_is_mcp_imported = False
McpClientSession = McpTool = McpListToolsResult = None
try:
    from mcp import ClientSession as McpClientSession
    from mcp import ListToolsResult as McpListToolsResult
    from mcp import Tool as McpTool

    _is_mcp_imported = True
except ImportError:
    pass
# pylint: disable=too-many-public-methods

GEN_AI_TOOL_DEFINITIONS = getattr(
    gen_ai_attributes, "GEN_AI_TOOL_DEFINITIONS", "gen_ai.tool.definitions"
)


def _mock_callable_tool():
    """Description of some tool."""
    return "result"


def _mock_mcp_client_session() -> McpClientSession:
    mock_session = create_autospec(spec=McpClientSession, instance=True)

    mock_tool_obj = McpTool(
        name="mcp_tool",
        description="Tool from session",
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        },
    )
    mock_result = create_autospec(McpListToolsResult, instance=True)
    mock_result.tools = [mock_tool_obj]

    mock_session.list_tools = AsyncMock(return_value=mock_result)

    return mock_session


def _mock_mcp_tool():
    return McpTool(
        name="mcp_tool",
        description="A standalone mcp tool",
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        },
    )


def _mock_tool_dict() -> ToolDict:
    return ToolDict(
        function_declarations=[
            FunctionDeclarationDict(
                name="mock_tool",
                description="Description of mock tool.",
            ),
        ],
        google_maps=GoogleMaps(),
    )


class ExampleResponseSchema(BaseModel):
    name: str = Field(description="A Destination's Name")


class NonStreamingTestCase(TestCase):
    # The "setUp" function is defined by "unittest.TestCase" and thus
    # this name must be used. Uncertain why pylint doesn't seem to
    # recognize that this is a unit test class for which this is inherited.
    def setUp(self):  # pylint: disable=invalid-name
        super().setUp()
        if self.__class__ == NonStreamingTestCase:
            raise unittest.SkipTest("Skipping testcase base.")
        self.tools = [
            _mock_callable_tool,
            _mock_tool_dict(),
        ]
        if _is_mcp_imported:
            self.tools.append(_mock_mcp_client_session())
            self.tools.append(_mock_mcp_tool())
        self.base_tools_definition = (
            {
                "name": "_mock_callable_tool",
                "description": "Description of some tool.",
                "parameters": None,
                "type": "function",
            },
            {
                "name": "mock_tool",
                "description": "Description of mock tool.",
                "parameters": None,
                "type": "function",
            },
            {
                "name": "google_maps",
                "type": "google_maps",
            },
        )
        if _is_mcp_imported:
            self.mcp_tools_no_content = (
                (
                    {
                        "name": "mcp_tool",
                        "description": "A standalone mcp tool",
                        "parameters": None,
                        "type": "function",
                    },
                ),
                (
                    {
                        "name": "mcp_tool",
                        "description": "Tool from session",
                        "parameters": None,
                        "type": "function",
                    },
                    {
                        "name": "mcp_tool",
                        "description": "A standalone mcp tool",
                        "parameters": None,
                        "type": "function",
                    },
                ),
            )
            self.mcp_tools_with_content = (
                (
                    {
                        "name": "mcp_tool",
                        "description": "A standalone mcp tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                        },
                        "type": "function",
                    },
                ),
                (
                    {
                        "name": "mcp_tool",
                        "description": "Tool from session",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                        },
                        "type": "function",
                    },
                    {
                        "name": "mcp_tool",
                        "description": "A standalone mcp tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                        },
                        "type": "function",
                    },
                ),
            )

    def generate_content(self, *args, **kwargs):
        raise NotImplementedError("Must implement 'generate_content'.")

    @property
    def expected_function_name(self):
        raise NotImplementedError("Must implement 'expected_function_name'.")

    def _generate_and_get_span(self, config):
        self.generate_content(
            model="gemini-2.0-flash",
            contents="Some input prompt",
            config=config,
        )
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        return self.otel.get_span_named("generate_content gemini-2.0-flash")

    def test_instrumentation_does_not_break_core_functionality(self):
        self.configure_valid_response(text="Yep, it works!")
        response = self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.assertEqual(response.text, "Yep, it works!")

    def test_generates_span(self):
        self.configure_valid_response(text="Yep, it works!")
        response = self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.assertEqual(response.text, "Yep, it works!")
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")

    def test_model_reflected_into_span_name(self):
        self.configure_valid_response(text="Yep, it works!")
        response = self.generate_content(
            model="gemini-1.5-flash", contents="Does this work?"
        )
        self.assertEqual(response.text, "Yep, it works!")
        self.otel.assert_has_span_named("generate_content gemini-1.5-flash")

    def test_generated_span_has_minimal_genai_attributes(self):
        self.configure_valid_response(text="Yep, it works!")
        self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.provider.name"], "gemini")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "generate_content"
        )
        self.assertEqual(
            span.attributes["server.address"],
            "generativelanguage.googleapis.com",
        )

    def test_generated_span_has_extra_genai_attributes(self):
        self.configure_valid_response(text="Yep, it works!")
        tok = context_api.attach(
            context_api.set_value(
                GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
                {"extra_attribute_key": "extra_attribute_value"},
            )
        )
        try:
            self.generate_content(
                model="gemini-2.0-flash", contents="Does this work?"
            )
            self.otel.assert_has_span_named(
                "generate_content gemini-2.0-flash"
            )
            span = self.otel.get_span_named(
                "generate_content gemini-2.0-flash"
            )
            self.assertEqual(
                span.attributes["extra_attribute_key"], "extra_attribute_value"
            )
        finally:
            context_api.detach(tok)

    def test_span_and_event_still_written_when_response_is_exception(self):
        self.configure_exception(ValueError("Uh oh!"))
        with pytest.raises(ValueError):
            self.generate_content(
                model="gemini-2.0-flash", contents="Does this work?"
            )
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        assert (
            span.attributes["error.type"]
            == event.attributes["error.type"]
            == "ValueError"
        )

    def test_generated_span_has_vertex_ai_system_when_configured(self):
        self.set_use_vertex(True)
        self.configure_valid_response(text="Yep, it works!")
        self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.provider.name"], "vertex_ai")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "generate_content"
        )

    def test_generated_span_counts_tokens(self):
        self.configure_valid_response(
            input_tokens=123,
            output_tokens=456,
            cached_tokens=50,
            thinking_tokens=17,
        )
        self.generate_content(model="gemini-2.0-flash", contents="Some input")
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 123)
        self.assertEqual(
            span.attributes["gen_ai.usage.output_tokens"], 456 + 17
        )
        self.assertEqual(
            span.attributes["gen_ai.usage.cache_read.input_tokens"], 50
        )
        self.assertEqual(
            span.attributes["gen_ai.usage.reasoning.output_tokens"], 17
        )

    def test_generated_span_records_response_model(self):
        self.configure_valid_response(model_version="gemini-2.0-flash-001")
        self.generate_content(model="gemini-2.0-flash", contents="Some input")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(
            span.attributes["gen_ai.response.model"], "gemini-2.0-flash-001"
        )
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        self.assertEqual(
            event.attributes["gen_ai.response.model"], "gemini-2.0-flash-001"
        )

    @patch.dict(
        "os.environ",
        {
            "OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_INCLUDES": "gcp.gen_ai.operation.config.response_schema",
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
        },
    )
    def test_log_event_no_content_capture(self):
        self.configure_valid_response(
            text="Some response content",
            cached_tokens=50,
            thinking_tokens=17,
        )
        self.generate_content(
            model="gemini-2.0-flash",
            contents="Some input",
            config=GenerateContentConfig(
                system_instruction="System instruction",
                response_schema=ExampleResponseSchema,
                tools=self.tools,
            ),
        )
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        self.assertEqual(
            event.attributes["gen_ai.usage.cache_read.input_tokens"],
            50,
        )
        self.assertNotIn(
            gen_ai_attributes.GEN_AI_RESPONSE_ID, event.attributes
        )
        self.assertEqual(
            event.attributes["gen_ai.usage.reasoning.output_tokens"],
            17,
        )
        self.assertEqual(
            event.attributes["gen_ai.usage.output_tokens"],
            17,
        )
        assert (
            event.attributes["gcp.gen_ai.operation.config.response_schema"]
            == "<class 'tests.generate_content.nonstreaming_base.ExampleResponseSchema'>"
        )

        self.assertNotIn(
            gen_ai_attributes.GEN_AI_INPUT_MESSAGES,
            event.attributes,
        )
        self.assertNotIn(
            gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES,
            event.attributes,
        )
        self.assertNotIn(
            gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS,
            event.attributes,
        )
        if _is_mcp_imported:
            self.assertIn(
                event.attributes[GEN_AI_TOOL_DEFINITIONS],
                [
                    self.base_tools_definition + mcp_var
                    for mcp_var in self.mcp_tools_no_content
                ],
            )
        else:
            self.assertEqual(
                event.attributes[GEN_AI_TOOL_DEFINITIONS],
                self.base_tools_definition,
            )

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
        },
    )
    def test_log_event_with_content_capture(self):
        self.configure_valid_response(
            text="Some response content",
            cached_tokens=50,
            thinking_tokens=17,
            response_id="qwerty17",
        )
        self.generate_content(
            model="gemini-2.0-flash",
            contents="Some input",
            config=GenerateContentConfig(
                system_instruction="System instruction",
                response_schema=ExampleResponseSchema,
                tools=self.tools,
            ),
        )
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        self.assertEqual(
            event.attributes[gen_ai_attributes.GEN_AI_RESPONSE_ID], "qwerty17"
        )
        self.assertEqual(
            event.attributes["gen_ai.usage.cache_read.input_tokens"],
            50,
        )
        self.assertEqual(
            event.attributes["gen_ai.usage.reasoning.output_tokens"],
            17,
        )
        self.assertEqual(
            event.attributes["gen_ai.usage.output_tokens"],
            17,
        )
        self.assertEqual(
            event.attributes[gen_ai_attributes.GEN_AI_INPUT_MESSAGES],
            (
                {
                    "role": "user",
                    "parts": ({"content": "Some input", "type": "text"},),
                },
            ),
        )
        self.assertEqual(
            event.attributes[gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES],
            (
                {
                    "role": "assistant",
                    "parts": (
                        {"content": "Some response content", "type": "text"},
                    ),
                    "finish_reason": "",
                },
            ),
        )
        self.assertEqual(
            event.attributes[gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS],
            ({"content": "System instruction", "type": "text"},),
        )
        if _is_mcp_imported:
            self.assertIn(
                event.attributes[GEN_AI_TOOL_DEFINITIONS],
                [
                    self.base_tools_definition + mcp_var
                    for mcp_var in self.mcp_tools_with_content
                ],
            )
        else:
            self.assertEqual(
                event.attributes[GEN_AI_TOOL_DEFINITIONS],
                self.base_tools_definition,
            )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT"},
    )
    def test_span_attributes_no_content_capture(self):
        self.configure_valid_response(
            text="Some response content",
            cached_tokens=50,
            thinking_tokens=19,
            response_id="qwerty17",
        )
        self.generate_content(
            model="gemini-2.0-flash",
            contents="Some input",
            config=GenerateContentConfig(
                system_instruction="System instruction",
                response_schema=ExampleResponseSchema,
                tools=self.tools,
            ),
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.provider.name"], "gemini")
        self.assertEqual(
            span.attributes["gen_ai.usage.cache_read.input_tokens"],
            50,
        )
        self.assertEqual(
            span.attributes[gen_ai_attributes.GEN_AI_RESPONSE_ID], "qwerty17"
        )
        self.assertEqual(
            span.attributes["gen_ai.usage.reasoning.output_tokens"],
            19,
        )
        self.assertEqual(
            span.attributes["gen_ai.usage.output_tokens"],
            19,
        )
        for attribute in (
            gen_ai_attributes.GEN_AI_INPUT_MESSAGES,
            gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES,
            gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS,
        ):
            self.assertNotIn(attribute, span.attributes)
        if _is_mcp_imported:
            self.assertIn(
                span.attributes[GEN_AI_TOOL_DEFINITIONS],
                [
                    '[{"name":"_mock_callable_tool","description":"Description of some tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description of mock tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"},{"name":"mcp_tool","description":"Tool from session","parameters":null,"type":"function"},{"name":"mcp_tool","description":"A standalone mcp tool","parameters":null,"type":"function"}]',
                    '[{"name":"_mock_callable_tool","description":"Description of some tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description of mock tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"},{"name":"mcp_tool","description":"A standalone mcp tool","parameters":null,"type":"function"}]',
                ],
            )
        else:
            self.assertEqual(
                span.attributes[GEN_AI_TOOL_DEFINITIONS],
                '[{"name":"_mock_callable_tool","description":"Description of some tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description of mock tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"}]',
            )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_span_attributes_with_content_capture(self):
        self.configure_valid_response(
            text="Some response content",
            cached_tokens=50,
            thinking_tokens=19,
        )
        self.generate_content(
            model="gemini-2.0-flash",
            contents="Some input",
            config=GenerateContentConfig(
                system_instruction="System instruction",
                response_schema=ExampleResponseSchema,
                tools=self.tools,
            ),
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.provider.name"], "gemini")
        self.assertEqual(
            span.attributes["gen_ai.usage.cache_read.input_tokens"],
            50,
        )
        self.assertEqual(
            span.attributes["gen_ai.usage.reasoning.output_tokens"],
            19,
        )
        self.assertEqual(
            span.attributes["gen_ai.usage.output_tokens"],
            19,
        )
        self.assertEqual(
            span.attributes[gen_ai_attributes.GEN_AI_INPUT_MESSAGES],
            '[{"role":"user","parts":[{"content":"Some input","type":"text"}]}]',
        )
        self.assertEqual(
            span.attributes[gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES],
            '[{"role":"assistant","parts":[{"content":"Some response content","type":"text"}],"finish_reason":""}]',
        )
        self.assertEqual(
            span.attributes[gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS],
            '[{"content":"System instruction","type":"text"}]',
        )
        if _is_mcp_imported:
            self.assertIn(
                span.attributes[GEN_AI_TOOL_DEFINITIONS],
                [
                    '[{"name":"_mock_callable_tool","description":"Description of some tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description of mock tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"},{"name":"mcp_tool","description":"Tool from session","parameters":{"type":"object","properties":{"id":{"type":"integer"}}},"type":"function"},{"name":"mcp_tool","description":"A standalone mcp tool","parameters":{"type":"object","properties":{"id":{"type":"integer"}}},"type":"function"}]',
                    '[{"name":"_mock_callable_tool","description":"Description of some tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description of mock tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"},{"name":"mcp_tool","description":"A standalone mcp tool","parameters":{"type":"object","properties":{"id":{"type":"integer"}}},"type":"function"}]',
                ],
            )
        else:
            self.assertEqual(
                span.attributes[GEN_AI_TOOL_DEFINITIONS],
                '[{"name":"_mock_callable_tool","description":"Description of some tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description of mock tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"}]',
            )

    def test_log_has_extra_genai_attributes(self):
        self.configure_valid_response(text="Yep, it works!")
        tok = context_api.attach(
            context_api.set_value(
                GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
                {"extra_attribute_key": "extra_attribute_value"},
            )
        )
        try:
            self.generate_content(
                model="gemini-2.0-flash",
                contents="Does this work?",
            )
            self.otel.assert_has_event_named(
                "gen_ai.client.inference.operation.details"
            )
            event = self.otel.get_event_named(
                "gen_ai.client.inference.operation.details"
            )
            assert (
                event.attributes["extra_attribute_key"]
                == "extra_attribute_value"
            )
        finally:
            context_api.detach(tok)

    def test_records_metrics_data(self):
        self.configure_valid_response()
        self.generate_content(model="gemini-2.0-flash", contents="Some input")
        self.otel.assert_has_metrics_data_named("gen_ai.client.token.usage")
        self.otel.assert_has_metrics_data_named(
            "gen_ai.client.operation.duration"
        )
