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

"""Tests for QwenAgentInstrumentor."""

from unittest.mock import MagicMock

from qwen_agent.llm.schema import ContentItem, FunctionCall, Message

from opentelemetry.instrumentation.qwen_agent import QwenAgentInstrumentor
from opentelemetry.instrumentation.qwen_agent.utils import (
    _convert_qwen_messages_to_input_messages,
    _convert_qwen_messages_to_output_messages,
    _create_llm_invocation,
    _get_provider_name,
)
from opentelemetry.util.genai.types import ToolCall


class TestQwenAgentInstrumentor:
    """Test the instrumentor lifecycle."""

    def test_instrument_and_uninstrument(
        self, tracer_provider, logger_provider, meter_provider
    ):
        """Test that instrument/uninstrument works without errors."""
        instrumentor = QwenAgentInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        assert instrumentor._handler is not None
        instrumentor.uninstrument()
        assert instrumentor._handler is None

    def test_instrumentation_dependencies(self):
        """Test that dependencies are correctly specified."""
        instrumentor = QwenAgentInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert ("qwen-agent >= 0.0.20",) == deps


class TestProviderName:
    """Test provider name detection."""

    def test_dashscope_model_type(self):
        llm = MagicMock()
        llm.model_type = "qwen_dashscope"
        assert _get_provider_name(llm) == "dashscope"

    def test_oai_model_type(self):
        llm = MagicMock()
        llm.model_type = "oai"
        assert _get_provider_name(llm) == "openai"

    def test_unknown_model_type(self):
        llm = MagicMock()
        llm.model_type = "unknown_custom"
        type(llm).__name__ = "CustomModel"
        assert _get_provider_name(llm) == "dashscope"

    def test_class_name_fallback_dashscope(self):
        llm = MagicMock()
        llm.model_type = "custom"
        type(llm).__name__ = "QwenDashScopeChat"
        assert _get_provider_name(llm) == "dashscope"


class TestMessageConversion:
    """Test qwen-agent message to GenAI type conversion."""

    def test_convert_simple_user_message(self):
        """Test converting a simple user text message."""
        messages = [Message(role="user", content="Hello")]
        result = _convert_qwen_messages_to_input_messages(messages)
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert result[0].parts[0].content == "Hello"

    def test_convert_function_call_message(self):
        """Test converting a message with function_call."""
        msg = Message(
            role="assistant",
            content="",
            function_call=FunctionCall(
                name="get_weather",
                arguments='{"city": "Beijing"}',
            ),
        )
        result = _convert_qwen_messages_to_output_messages([msg])
        assert len(result) == 1
        assert result[0].finish_reason == "tool_calls"
        # Should have a ToolCall part
        tool_calls = [p for p in result[0].parts if isinstance(p, ToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

    def test_convert_function_response_message(self):
        """Test converting a function role message (tool result)."""
        msg = Message(
            role="function", name="get_weather", content="Sunny, 25°C"
        )
        result = _convert_qwen_messages_to_input_messages([msg])
        assert len(result) == 1
        assert result[0].role == "function"

    def test_convert_empty_messages(self):
        """Test converting empty message list."""
        result = _convert_qwen_messages_to_input_messages([])
        assert result == []

    def test_convert_multimodal_content(self):
        """Test converting message with ContentItem list."""
        msg = Message(
            role="user",
            content=[ContentItem(text="Describe this image")],
        )
        result = _convert_qwen_messages_to_input_messages([msg])
        assert len(result) == 1
        assert result[0].parts[0].content == "Describe this image"


class TestLLMInvocation:
    """Test LLM invocation creation."""

    def test_create_basic_invocation(self):
        """Test creating a basic LLM invocation."""
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"

        messages = [Message(role="user", content="Hi")]
        invocation = _create_llm_invocation(llm, messages)

        assert invocation.request_model == "qwen-max"
        assert invocation.provider == "dashscope"
        assert len(invocation.input_messages) == 1

    def test_create_invocation_with_functions(self):
        """Test creating invocation with tool definitions."""
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"

        messages = [Message(role="user", content="What's the weather?")]
        functions = [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        invocation = _create_llm_invocation(llm, messages, functions=functions)

        # P1 fix: tool_definitions are now FunctionToolDefinition objects on invocation.tool_definitions
        assert len(invocation.tool_definitions) == 1
        tool_def = invocation.tool_definitions[0]
        assert tool_def.name == "get_weather"
        assert tool_def.description == "Get weather info"
