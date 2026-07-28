# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
import unittest.mock

from opentelemetry.instrumentation.google_genai._compat import GenericPart
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    _interactions_input_to_messages,
    _interactions_response_to_messages,
)
from opentelemetry.util.genai.types import (
    Text,
    ToolCallResponse,
    Uri,
)
from opentelemetry.util.genai.types import (
    ToolCall as ToolCallRequest,
)


@unittest.skipIf(
    not _HAS_INTERACTIONS,
    "Interactions are not supported in this version of google-genai",
)
class TestInteractionsParser(unittest.TestCase):
    def test_input_to_messages_none(self) -> None:
        self.assertEqual(_interactions_input_to_messages(None), [])

    def test_input_to_messages_str(self) -> None:
        messages = _interactions_input_to_messages("Hello world")
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello world")

    def test_input_to_messages_list_of_strings(self) -> None:
        messages = _interactions_input_to_messages(["Hello", "world"])
        self.assertEqual(len(messages[0].parts), 2)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello")
        self.assertIsInstance(messages[0].parts[1], Text)
        self.assertEqual(messages[0].parts[1].content, "world")

    def test_input_to_messages_text_step(self) -> None:
        steps = [{"type": "text", "text": "Hello text step"}]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello text step")

    def test_input_to_messages_document_step(self) -> None:
        steps = [
            {
                "type": "document",
                "mime_type": "application/pdf",
                "uri": "https://example.com/doc.pdf",
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Uri)
        self.assertEqual(messages[0].parts[0].mime_type, "application/pdf")
        self.assertEqual(messages[0].parts[0].modality, "document")
        self.assertEqual(
            messages[0].parts[0].uri, "https://example.com/doc.pdf"
        )

    def test_input_to_messages_tool_call_step(self) -> None:
        steps = [
            {
                "type": "function_call",
                "id": "call-123",
                "name": "calc",
                "arguments": {"x": 5},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ToolCallRequest)
        self.assertEqual(messages[0].parts[0].id, "call-123")
        self.assertEqual(messages[0].parts[0].name, "calc")
        self.assertEqual(messages[0].parts[0].arguments, {"x": 5})

    def test_input_to_messages_tool_result_step(self) -> None:
        steps = [
            {
                "type": "function_result",
                "call_id": "call-123",
                "result": {"val": 10},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ToolCallResponse)
        self.assertEqual(messages[0].parts[0].id, "call-123")
        self.assertEqual(messages[0].parts[0].response, {"val": 10})

    def test_input_to_messages_generic_fallback(self) -> None:
        steps = [{"type": "some_unsupported_type"}]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], GenericPart)
        self.assertEqual(messages[0].parts[0].value, "dict")

    def test_input_to_messages_single_non_sequence_step(self) -> None:
        step = {"type": "text", "text": "Hello single step"}
        messages = _interactions_input_to_messages(step)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Hello single step")

    def test_input_to_messages_none_type_fall_through(self) -> None:
        step = {"other_field": "no type specified"}
        messages = _interactions_input_to_messages(step)
        self.assertEqual(len(messages[0].parts), 0)

    def test_response_to_messages(self) -> None:
        mock_interaction = unittest.mock.MagicMock()
        mock_interaction.output_text = "Model response text"

        messages = _interactions_response_to_messages(mock_interaction)

        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[0].finish_reason, "stop")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], Text)
        self.assertEqual(messages[0].parts[0].content, "Model response text")
