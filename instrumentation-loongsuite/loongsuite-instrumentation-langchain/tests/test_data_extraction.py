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

"""Unit tests for LangChain data extraction helper functions."""

from opentelemetry.instrumentation.langchain.internal._tracer import (
    _extract_deepagents_skills_metadata,
    _extract_tool_call_arguments,
)
from opentelemetry.instrumentation.langchain.internal._utils import (
    _convert_lc_message_to_input,
    _extract_finish_reasons,
    _extract_llm_input_messages,
    _extract_llm_output_messages,
    _extract_model_name,
    _extract_provider,
    _extract_response_model,
    _extract_token_usage,
    _extract_tool_definitions,
    _parse_token_usage_dict,
    _safe_json,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    Text,
    ToolCall,
    ToolCallResponse,
)


class _FakeRun:
    """Minimal stub mimicking a LangChain Run object."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "test")
        self.inputs = kwargs.get("inputs", {})
        self.outputs = kwargs.get("outputs", {})
        self.extra = kwargs.get("extra", {})
        self.serialized = kwargs.get("serialized", {})
        self.error = kwargs.get("error", None)


class TestExtractModelName:
    def test_from_invocation_params_model_name(self):
        run = _FakeRun(extra={"invocation_params": {"model_name": "gpt-4"}})
        assert _extract_model_name(run) == "gpt-4"

    def test_from_invocation_params_model(self):
        run = _FakeRun(extra={"invocation_params": {"model": "claude-3"}})
        assert _extract_model_name(run) == "claude-3"

    def test_none_when_missing(self):
        run = _FakeRun()
        assert _extract_model_name(run) is None


class TestExtractProvider:
    def test_from_serialized_id(self):
        run = _FakeRun(
            serialized={"id": ["langchain", "llms", "openai", "ChatOpenAI"]}
        )
        assert _extract_provider(run) == "openai"

    def test_default_langchain(self):
        run = _FakeRun()
        assert _extract_provider(run) == "langchain"


class TestExtractToolCallArguments:
    def test_input_key_takes_precedence(self):
        assert _extract_tool_call_arguments({"input": "hello"}) == "hello"

    def test_query_key_is_used_when_input_missing(self):
        assert _extract_tool_call_arguments({"query": "search text"}) == (
            "search text"
        )

    def test_structured_inputs_are_preserved(self):
        args = {"file_path": "/skills/probe/SKILL.md", "limit": 1000}
        assert _extract_tool_call_arguments(args) == args

    def test_none_becomes_empty_string(self):
        assert _extract_tool_call_arguments(None) == ""

    def test_non_dict_input_is_returned(self):
        assert _extract_tool_call_arguments(["raw"]) == ["raw"]


class TestExtractDeepAgentsSkillsMetadata:
    def test_top_level_metadata(self):
        assert _extract_deepagents_skills_metadata(
            {"skills_metadata": [{"name": "probe"}]}
        ) == [{"name": "probe"}]

    def test_nested_output_metadata(self):
        assert _extract_deepagents_skills_metadata(
            {"output": {"skills_metadata": [{"name": "nested"}]}}
        ) == [{"name": "nested"}]

    def test_ignores_non_dict_items(self):
        assert _extract_deepagents_skills_metadata(
            {"skills_metadata": [{"name": "probe"}, "bad"]}
        ) == [{"name": "probe"}]


class TestConvertMessage:
    def test_human_message(self):
        msg = {
            "id": ["langchain", "schema", "HumanMessage"],
            "kwargs": {"content": "Hello"},
        }
        result = _convert_lc_message_to_input(msg)
        assert result is not None
        assert result.role == "user"
        assert len(result.parts) == 1
        assert isinstance(result.parts[0], Text)
        assert result.parts[0].content == "Hello"

    def test_ai_message(self):
        msg = {
            "id": ["langchain", "schema", "AIMessage"],
            "kwargs": {"content": "Hi there"},
        }
        result = _convert_lc_message_to_input(msg)
        assert result is not None
        assert result.role == "assistant"

    def test_message_with_tool_calls(self):
        msg = {
            "id": ["langchain", "schema", "AIMessage"],
            "kwargs": {
                "content": "",
                "tool_calls": [
                    {"name": "search", "args": {"q": "test"}, "id": "tc1"}
                ],
            },
        }
        result = _convert_lc_message_to_input(msg)
        assert result is not None
        assert any(isinstance(p, ToolCall) for p in result.parts)

    def test_tool_message(self):
        """ToolMessage (role=tool) should be converted to ToolCallResponse, not Text."""
        msg = {
            "id": ["langchain", "schema", "ToolMessage"],
            "kwargs": {
                "content": "search result: 42",
                "tool_call_id": "call_abc123",
            },
        }
        result = _convert_lc_message_to_input(msg)
        assert result is not None
        assert result.role == "tool"
        assert len(result.parts) == 1
        assert isinstance(result.parts[0], ToolCallResponse)
        assert result.parts[0].response == "search result: 42"
        assert result.parts[0].id == "call_abc123"

    def test_tool_message_empty_content(self):
        """ToolMessage with empty content should still produce ToolCallResponse."""
        msg = {
            "id": ["langchain", "schema", "ToolMessage"],
            "kwargs": {
                "content": "",
                "tool_call_id": "call_xyz",
            },
        }
        result = _convert_lc_message_to_input(msg)
        assert result is not None
        assert result.role == "tool"
        assert len(result.parts) == 1
        assert isinstance(result.parts[0], ToolCallResponse)
        assert result.parts[0].response == ""
        assert result.parts[0].id == "call_xyz"

    def test_none_for_non_dict(self):
        assert _convert_lc_message_to_input("not a dict") is None


class TestExtractToolDefinitions:
    def test_from_invocation_params_openai_format(self):
        """Tools in OpenAI format: {type: function, function: {...}}."""
        run = _FakeRun(
            extra={
                "invocation_params": {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {"type": "object"},
                            },
                        },
                    ]
                }
            }
        )
        result = _extract_tool_definitions(run)
        assert len(result) == 1
        assert isinstance(result[0], FunctionToolDefinition)
        assert result[0].name == "get_weather"
        assert result[0].description == "Get weather"

    def test_from_invocation_params_flat_format(self):
        """Tools in flat format: {name, description, parameters}."""
        run = _FakeRun(
            extra={
                "invocation_params": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search tool",
                            "parameters": {},
                        },
                    ]
                }
            }
        )
        result = _extract_tool_definitions(run)
        assert len(result) == 1
        assert result[0].name == "search"

    def test_from_inputs(self):
        """Tools in run.inputs when not in invocation_params."""
        run = _FakeRun(
            inputs={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "description": "Do math",
                            "parameters": {},
                        },
                    },
                ]
            }
        )
        result = _extract_tool_definitions(run)
        assert len(result) == 1
        assert result[0].name == "calculator"

    def test_empty_when_no_tools(self):
        run = _FakeRun(extra={}, inputs={})
        assert _extract_tool_definitions(run) == []


class TestExtractLLMInputMessages:
    def test_from_messages_field(self):
        run = _FakeRun(
            inputs={
                "messages": [
                    [
                        {
                            "id": ["langchain", "schema", "HumanMessage"],
                            "kwargs": {"content": "Hi"},
                        }
                    ]
                ]
            }
        )
        messages = _extract_llm_input_messages(run)
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_from_prompts_field(self):
        run = _FakeRun(inputs={"prompts": ["Tell me a joke"]})
        messages = _extract_llm_input_messages(run)
        assert len(messages) == 1
        assert messages[0].parts[0].content == "Tell me a joke"

    def test_empty_inputs(self):
        run = _FakeRun(inputs={})
        messages = _extract_llm_input_messages(run)
        assert messages == []

    def test_messages_with_tool_message(self):
        """Messages containing ToolMessage should convert to ToolCallResponse."""
        run = _FakeRun(
            inputs={
                "messages": [
                    [
                        {
                            "id": ["langchain", "schema", "HumanMessage"],
                            "kwargs": {"content": "search for x"},
                        },
                        {
                            "id": ["langchain", "schema", "ToolMessage"],
                            "kwargs": {
                                "content": "found: 42",
                                "tool_call_id": "call_123",
                            },
                        },
                    ]
                ]
            }
        )
        messages = _extract_llm_input_messages(run)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "tool"
        assert isinstance(messages[1].parts[0], ToolCallResponse)
        assert messages[1].parts[0].response == "found: 42"
        assert messages[1].parts[0].id == "call_123"


class TestExtractLLMOutputMessages:
    def test_basic_generation(self):
        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Hello world",
                            "generation_info": {"finish_reason": "stop"},
                        }
                    ]
                ]
            }
        )
        messages = _extract_llm_output_messages(run)
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].finish_reason == "stop"

    def test_empty_outputs(self):
        run = _FakeRun(outputs={})
        assert _extract_llm_output_messages(run) == []


class TestParseTokenUsageDict:
    """Unit tests for _parse_token_usage_dict."""

    def test_prompt_completion_tokens(self):
        """Standard OpenAI format: prompt_tokens, completion_tokens."""
        inp, out = _parse_token_usage_dict(
            {"prompt_tokens": 10, "completion_tokens": 20}
        )
        assert inp == 10
        assert out == 20

    def test_input_output_tokens(self):
        """Anthropic/Claude format: input_tokens, output_tokens."""
        inp, out = _parse_token_usage_dict(
            {"input_tokens": 5, "output_tokens": 15}
        )
        assert inp == 5
        assert out == 15

    def test_azure_format_pascal_case(self):
        """Azure format: PromptTokens, CompletionTokens (PascalCase)."""
        inp, out = _parse_token_usage_dict(
            {"PromptTokens": 100, "CompletionTokens": 50}
        )
        assert inp == 100
        assert out == 50

    def test_partial_input_only(self):
        """Only input tokens present, output is None."""
        inp, out = _parse_token_usage_dict({"prompt_tokens": 42})
        assert inp == 42
        assert out is None

    def test_partial_output_only(self):
        """Only output tokens present, input is None."""
        inp, out = _parse_token_usage_dict({"completion_tokens": 8})
        assert inp is None
        assert out == 8

    def test_non_dict_returns_none(self):
        """Non-dict input returns (None, None)."""
        assert _parse_token_usage_dict(None) == (None, None)
        assert _parse_token_usage_dict("not a dict") == (None, None)
        assert _parse_token_usage_dict([]) == (None, None)

    def test_empty_dict(self):
        """Empty dict returns (None, None)."""
        assert _parse_token_usage_dict({}) == (None, None)


class TestExtractTokenUsage:
    def test_from_llm_output(self):
        run = _FakeRun(
            outputs={
                "llm_output": {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                    }
                }
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 10
        assert out == 20

    def test_alternative_keys(self):
        run = _FakeRun(
            outputs={
                "llm_output": {
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 15,
                    }
                }
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 5
        assert out == 15

    def test_from_generation_info_token_usage(self):
        """Providers that don't populate llm_output may put token_usage in generation_info."""
        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Hello",
                            "generation_info": {
                                "finish_reason": "stop",
                                "token_usage": {
                                    "input_tokens": 39,
                                    "output_tokens": 8,
                                },
                            },
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 39
        assert out == 8

    def test_from_generation_info_usage(self):
        """generation_info may use 'usage' key with prompt_tokens/completion_tokens."""
        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Hi",
                            "generation_info": {
                                "usage": {
                                    "prompt_tokens": 12,
                                    "completion_tokens": 6,
                                }
                            },
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 12
        assert out == 6

    def test_from_message_response_metadata_dict(self):
        """Token usage may be in message.kwargs.response_metadata (serialized format)."""
        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Response",
                            "message": {
                                "kwargs": {
                                    "content": "Response",
                                    "response_metadata": {
                                        "token_usage": {
                                            "prompt_tokens": 100,
                                            "completion_tokens": 25,
                                        }
                                    },
                                }
                            },
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 100
        assert out == 25

    def test_from_message_response_metadata_object(self):
        """Token usage may be in message.response_metadata (object format, not serialized)."""

        class _FakeMessage:
            response_metadata = {
                "token_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                }
            }

        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Response",
                            "message": _FakeMessage(),
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 50
        assert out == 10

    def test_llm_output_takes_precedence(self):
        """When both llm_output and generation_info have token_usage, prefer llm_output."""
        run = _FakeRun(
            outputs={
                "llm_output": {
                    "token_usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                    }
                },
                "generations": [
                    [
                        {
                            "generation_info": {
                                "token_usage": {
                                    "input_tokens": 99,
                                    "output_tokens": 99,
                                }
                            }
                        }
                    ]
                ],
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 1
        assert out == 2

    def test_from_message_usage_metadata_dict(self):
        """Token usage may be in message.kwargs.usage_metadata (serialized format)."""
        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Response",
                            "message": {
                                "kwargs": {
                                    "content": "Response",
                                    "usage_metadata": {
                                        "input_tokens": 30,
                                        "output_tokens": 12,
                                    },
                                }
                            },
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 30
        assert out == 12

    def test_from_message_usage_metadata_object(self):
        """Token usage may be in message.usage_metadata (object format)."""

        class _FakeMessage:
            usage_metadata = {
                "prompt_tokens": 80,
                "completion_tokens": 20,
            }

        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Response",
                            "message": _FakeMessage(),
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 80
        assert out == 20

    def test_llm_output_azure_format(self):
        """llm_output may use Azure-style PromptTokens/CompletionTokens."""
        run = _FakeRun(
            outputs={
                "llm_output": {
                    "token_usage": {
                        "PromptTokens": 200,
                        "CompletionTokens": 75,
                    }
                }
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 200
        assert out == 75

    def test_response_metadata_usage_key(self):
        """response_metadata may use 'usage' key instead of 'token_usage'."""
        run = _FakeRun(
            outputs={
                "generations": [
                    [
                        {
                            "text": "Hi",
                            "message": {
                                "kwargs": {
                                    "response_metadata": {
                                        "usage": {
                                            "prompt_tokens": 15,
                                            "completion_tokens": 7,
                                        }
                                    },
                                }
                            },
                        }
                    ]
                ]
            }
        )
        inp, out = _extract_token_usage(run)
        assert inp == 15
        assert out == 7

    def test_no_token_usage(self):
        run = _FakeRun(outputs={})
        inp, out = _extract_token_usage(run)
        assert inp is None
        assert out is None


class TestExtractFinishReasons:
    def test_basic(self):
        run = _FakeRun(
            outputs={
                "generations": [
                    [{"generation_info": {"finish_reason": "stop"}}]
                ]
            }
        )
        reasons = _extract_finish_reasons(run)
        assert reasons == ["stop"]

    def test_none_when_empty(self):
        run = _FakeRun(outputs={})
        assert _extract_finish_reasons(run) is None


class TestExtractResponseModel:
    def test_from_llm_output(self):
        run = _FakeRun(outputs={"llm_output": {"model_name": "gpt-4-turbo"}})
        assert _extract_response_model(run) == "gpt-4-turbo"

    def test_none_when_missing(self):
        run = _FakeRun(outputs={})
        assert _extract_response_model(run) is None


class TestSafeJson:
    def test_basic_dict(self):
        assert '"a": 1' in _safe_json({"a": 1})

    def test_truncation(self):
        result = _safe_json({"x": "a" * 10000}, max_len=100)
        assert result.endswith("...[truncated]")
        assert len(result) <= 114  # max_len(100) + len("...[truncated]")(14)

    def test_non_serializable(self):
        result = _safe_json(object())
        assert isinstance(result, str)
