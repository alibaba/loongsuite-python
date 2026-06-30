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

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.instrumentation.autogen.semantic_conventions import (
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_AGENT_NAME,
)
from opentelemetry.instrumentation.autogen.utils import (
    apply_create_result,
    make_agent_invocation,
    make_llm_invocation,
    to_input_messages,
    tool_definitions,
)
from opentelemetry.util.genai.types import (
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)


@dataclass
class SystemMessage:
    content: str


@dataclass
class UserMessage:
    content: str


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class AssistantMessage:
    content: list
    thought: str | None = None


@dataclass
class FunctionExecutionResult:
    call_id: str
    content: str


@dataclass
class FunctionExecutionResultMessage:
    content: list[FunctionExecutionResult]


@dataclass
class FunctionCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class CreateResult:
    content: str
    finish_reason: str
    usage: Usage


class ModelClient:
    _create_args = {"model": "qwen-plus"}
    _resolved_model = "qwen-plus-2025-04-28"


class RawConfigModelClient:
    _raw_config = {"model": "qwen-fake"}


class DashScopeOpenAIClient(ModelClient):
    class _client:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"


class Tool:
    def schema(self):
        return {
            "name": "lookup",
            "description": "Lookup data",
            "parameters": {"type": "object"},
        }


def test_to_input_messages_converts_autogen_agentchat_messages():
    messages = to_input_messages(
        [
            SystemMessage("system"),
            UserMessage("hello"),
            AssistantMessage(
                [FunctionCall("call-1", "lookup", {"q": "x"})],
                thought="thinking",
            ),
            FunctionExecutionResultMessage(
                [FunctionExecutionResult("call-1", "ok")]
            ),
        ]
    )

    assert messages[0].role == "system"
    assert messages[0].parts == [Text(content="system")]
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert isinstance(messages[2].parts[0], ToolCall)
    assert messages[2].parts[0].name == "lookup"
    assert messages[2].parts[1] == Reasoning(content="thinking")
    assert messages[3].role == "tool"
    assert messages[3].parts == [ToolCallResponse(response="ok", id="call-1")]


def test_to_input_messages_converts_agentchat_text_messages():
    messages = to_input_messages(
        [
            TextMessage("task", "user"),
            TextMessage("answer", "assistant"),
        ]
    )

    assert messages[0].role == "user"
    assert messages[0].parts == [Text(content="task")]
    assert messages[1].role == "assistant"
    assert messages[1].parts == [Text(content="answer")]


def test_make_llm_invocation_uses_model_client_metadata_and_tools():
    invocation = make_llm_invocation(
        ModelClient(),
        [UserMessage("hello")],
        [Tool()],
        agent_name="assistant",
        output_type="json",
    )

    assert invocation.request_model == "qwen-plus"
    assert invocation.response_model_name == "qwen-plus-2025-04-28"
    assert invocation.provider == AUTOGEN_PROVIDER_NAME
    assert invocation.attributes[GEN_AI_AGENT_NAME] == "assistant"
    assert invocation.input_messages[0].parts == [Text(content="hello")]
    assert invocation.tool_definitions == tool_definitions([Tool()])
    assert invocation.output_type == "json"


def test_make_llm_invocation_uses_raw_config_model_fallback():
    invocation = make_llm_invocation(RawConfigModelClient(), [], [])

    assert invocation.request_model == "qwen-fake"
    assert invocation.response_model_name == "qwen-fake"


def test_make_llm_invocation_detects_dashscope_openai_compatible_endpoint():
    invocation = make_llm_invocation(
        DashScopeOpenAIClient(),
        [UserMessage("hello")],
        [],
    )

    assert invocation.provider == "dashscope"


def test_apply_create_result_populates_output_and_usage():
    invocation = make_llm_invocation(ModelClient(), [], [])

    apply_create_result(
        invocation,
        CreateResult(
            "done", "stop", Usage(prompt_tokens=3, completion_tokens=5)
        ),
    )

    assert invocation.finish_reasons == ["stop"]
    assert invocation.input_tokens == 3
    assert invocation.output_tokens == 5
    assert invocation.output_messages[0].parts == [Text(content="done")]
    assert invocation.output_messages[0].finish_reason == "stop"


def test_apply_create_result_normalizes_tool_call_finish_reason():
    invocation = make_llm_invocation(ModelClient(), [], [])

    apply_create_result(
        invocation,
        CreateResult(
            [FunctionCall("call-1", "lookup", {"q": "x"})],
            "function_calls",
            Usage(prompt_tokens=3, completion_tokens=5),
        ),
    )

    assert invocation.finish_reasons == ["tool_calls"]
    assert isinstance(invocation.output_messages[0].parts[0], ToolCall)
    assert invocation.output_messages[0].finish_reason == "tool_calls"


def test_make_agent_invocation_uses_assistant_metadata():
    class AssistantAgent:
        _name = "assistant"
        _description = "answers"
        _model_client = ModelClient()
        _system_messages = [SystemMessage("system")]
        _tools = [Tool()]

    invocation = make_agent_invocation(AssistantAgent())

    assert invocation.provider == AUTOGEN_PROVIDER_NAME
    assert invocation.agent_name == "assistant"
    assert invocation.agent_description == "answers"
    assert invocation.request_model == "qwen-plus"
    assert invocation.response_model_name == "qwen-plus-2025-04-28"
    assert invocation.input_messages[0].parts == [Text(content="system")]
    assert invocation.tool_definitions == tool_definitions([Tool()])
