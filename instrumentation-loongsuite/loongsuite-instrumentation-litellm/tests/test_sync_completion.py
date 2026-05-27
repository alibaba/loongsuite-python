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
Test cases for synchronous LiteLLM completion calls.

This module tests basic synchronous text generation functionality using LiteLLM's
completion API with various models and configurations.
"""

import json
import os
from unittest.mock import patch

import litellm
import pytest

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.types import ContentCapturingMode


@pytest.mark.vcr
class TestSyncCompletion(TestBase):
    """
    Test synchronous completion calls with LiteLLM.
    """

    def setUp(self):
        super().setUp()
        # Set up environment variables for testing
        os.environ["OPENAI_API_KEY"] = os.environ.get(
            "OPENAI_API_KEY", "sk-..."
        )
        os.environ["DASHSCOPE_API_KEY"] = os.environ.get(
            "DASHSCOPE_API_KEY", "sk-..."
        )
        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ["DASHSCOPE_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Simpler approach: Mock the utility functions that check for experimental mode

        self.patch_experimental = patch(
            "opentelemetry.util.genai.span_utils.is_experimental_mode",
            return_value=True,
        )
        self.patch_content_mode = patch(
            "opentelemetry.util.genai.span_utils.get_content_capturing_mode",
            return_value=ContentCapturingMode.SPAN_ONLY,
        )

        self.patch_experimental.start()
        self.patch_content_mode.start()

        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
        )
        # Use model aliases to map to openai provider while keeping model name
        litellm.model_alias_map = {
            "qwen-turbo": "openai/qwen-turbo",
            "qwen-plus": "openai/qwen-plus",
        }

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()

        # Stop patches
        self.patch_experimental.stop()
        self.patch_content_mode.stop()

    def test_basic_sync_completion(self):
        """
        Test basic synchronous text generation.

        The test verifies:
        - A span is created with gen_ai.span.kind = "LLM"
        - Required span attributes are present (model, provider, tokens)
        - Input and output messages are captured (Experimental mode)
        - Token usage and finish reasons are recorded
        """

        # Business demo: Simple chat completion
        response = litellm.completion(
            model="qwen-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                }
            ],
            temperature=0.7,
            max_tokens=50,
        )

        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "choices"))
        self.assertGreater(len(response.choices), 0)

        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(
            len(spans), 1, "Expected exactly one span for completion call"
        )

        span = spans[0]

        # Verify span kind and operation name
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertEqual(span.attributes.get("gen_ai.operation.name"), "chat")

        # Verify model and provider
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"), "qwen-turbo"
        )
        self.assertEqual(
            span.attributes.get("gen_ai.provider.name"), "dashscope"
        )

        # Verify token usage
        self.assertGreater(span.attributes.get("gen_ai.usage.input_tokens"), 0)
        self.assertGreater(
            span.attributes.get("gen_ai.usage.output_tokens"), 0
        )

        # Verify Content Capture (Experimental Mode)
        self.assertIn("gen_ai.input.messages", span.attributes)
        input_messages = json.loads(
            span.attributes.get("gen_ai.input.messages")
        )
        self.assertEqual(input_messages[0]["role"], "user")
        self.assertIn("capital of France", str(input_messages[0]["parts"]))

        self.assertIn("gen_ai.output.messages", span.attributes)
        output_messages = json.loads(
            span.attributes.get("gen_ai.output.messages")
        )
        self.assertEqual(output_messages[0]["role"], "assistant")
        self.assertGreater(len(output_messages[0]["parts"]), 0)

        # Verify recommended attributes
        self.assertEqual(
            span.attributes.get("gen_ai.request.temperature"), 0.7
        )
        self.assertEqual(span.attributes.get("gen_ai.request.max_tokens"), 50)
        self.assertIn("gen_ai.response.id", span.attributes)
        self.assertIn("gen_ai.response.model", span.attributes)
        self.assertIn("gen_ai.response.finish_reasons", span.attributes)

    def test_sync_completion_with_multiple_messages(self):
        """
        Test synchronous completion with multi-turn conversation.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise answers.",
            },
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What is 3+3?"},
        ]

        # Business demo: Multi-turn conversation
        response = litellm.completion(
            model="qwen-turbo",
            messages=messages,
            temperature=0.1,
        )
        # Verify response
        self.assertIsNotNone(response)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        # Verify all messages captured in sequence
        self.assertIn("gen_ai.input.messages", span.attributes)
        input_messages = json.loads(
            span.attributes.get("gen_ai.input.messages")
        )
        self.assertEqual(len(input_messages), 4)
        self.assertEqual(input_messages[0]["role"], "system")
        self.assertEqual(input_messages[1]["role"], "user")
        self.assertEqual(input_messages[2]["role"], "assistant")
        self.assertEqual(input_messages[3]["role"], "user")

        output_messages = json.loads(
            span.attributes.get("gen_ai.output.messages")
        )
        self.assertGreater(len(output_messages), 0)
        self.assertEqual(output_messages[0]["role"], "assistant")

    def test_sync_completion_with_parameters(self):
        """
        Test capturing of various LLM parameters.
        """

        response = litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "Tell me a short joke."}],
            temperature=0.9,
            max_tokens=100,
            top_p=0.95,
            seed=42,
            stop=["END"],
        )
        # Verify response
        self.assertIsNotNone(response)

        span = self.get_finished_spans()[0]

        # Verify advanced parameters
        self.assertEqual(
            span.attributes.get("gen_ai.request.temperature"), 0.9
        )
        self.assertEqual(span.attributes.get("gen_ai.request.max_tokens"), 100)
        self.assertEqual(span.attributes.get("gen_ai.request.top_p"), 0.95)
        self.assertEqual(span.attributes.get("gen_ai.request.seed"), 42)

        # Verify stop sequences (stored as list/tuple in attributes)
        stop_seq = span.attributes.get("gen_ai.request.stop_sequences")
        self.assertIn("END", stop_seq)
