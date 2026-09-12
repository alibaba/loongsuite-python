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

"""Tests for opentelemetry.instrumentation.minisweagent.internal.conversation.

Covers the pure-function helpers that convert mini-swe-agent trajectory
dicts into OpenTelemetry GenAI semantic-convention message types.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

# =====================================================================
# Helpers under test — imported after conftest injects stub modules.
# =====================================================================


def _conv():
    """Lazy import so stub modules are in place."""
    from opentelemetry.instrumentation.minisweagent.internal import (
        conversation,
    )

    return conversation


# =====================================================================
# split_system_messages
# =====================================================================


class TestSplitSystemMessages:
    """Tests for ``split_system_messages``."""

    def test_empty_list(self):
        sys_parts, rest = _conv().split_system_messages([])
        assert sys_parts == []
        assert rest == []

    def test_only_system_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
        ]
        sys_parts, rest = _conv().split_system_messages(msgs)
        assert len(sys_parts) == 2
        assert all(isinstance(p, Text) for p in sys_parts)
        assert sys_parts[0].content == "You are helpful."
        assert sys_parts[1].content == "Be concise."
        assert rest == []

    def test_no_system_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        sys_parts, rest = _conv().split_system_messages(msgs)
        assert sys_parts == []
        assert rest == msgs

    def test_mixed_messages(self):
        msgs = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Usr"},
            {"role": "assistant", "content": "Asst"},
        ]
        sys_parts, rest = _conv().split_system_messages(msgs)
        assert len(sys_parts) == 1
        assert sys_parts[0].content == "Sys"
        assert len(rest) == 2
        assert rest[0]["role"] == "user"
        assert rest[1]["role"] == "assistant"

    def test_non_dict_entries_are_skipped(self):
        msgs: list[Any] = [
            "not a dict",
            42,
            {"role": "user", "content": "ok"},
        ]
        sys_parts, rest = _conv().split_system_messages(msgs)
        assert sys_parts == []
        assert len(rest) == 1

    def test_system_message_without_content(self):
        msgs = [{"role": "system"}]
        sys_parts, rest = _conv().split_system_messages(msgs)
        assert len(sys_parts) == 1
        assert sys_parts[0].content == ""
        assert rest == []


# =====================================================================
# _text_parts
# =====================================================================


class TestTextParts:
    """Tests for ``_text_parts``."""

    def test_none_content(self):
        assert _conv()._text_parts(None) == []

    def test_empty_string(self):
        assert _conv()._text_parts("") == []

    def test_whitespace_only(self):
        assert _conv()._text_parts("   ") == []

    def test_nonempty_content(self):
        parts = _conv()._text_parts("Hello world")
        assert len(parts) == 1
        assert isinstance(parts[0], Text)
        assert parts[0].content == "Hello world"

    def test_numeric_content_converted_to_str(self):
        # The function calls str(content), so non-string is tolerated.
        parts = _conv()._text_parts(123)  # type: ignore[arg-type]
        assert len(parts) == 1
        assert parts[0].content == "123"


# =====================================================================
# _normalized_tool_calls
# =====================================================================


class TestNormalizedToolCalls:
    """Tests for ``_normalized_tool_calls``."""

    def test_no_tool_calls_no_actions(self):
        msg: dict[str, Any] = {"role": "assistant", "content": "hi"}
        assert _conv()._normalized_tool_calls(msg) == []

    def test_dict_tool_calls_with_function_dict(self):
        msg: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ]
        }
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 1
        tc = result[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "tc_1"
        assert tc.name == "bash"
        assert tc.arguments == {"command": "ls"}

    def test_object_style_tool_calls(self):
        """When tool_call items have attribute-style .function / .id access."""

        class FnObj:
            name = "bash"
            arguments = '{"command": "pwd"}'

        class TCObj:
            id = "tc_2"
            function = FnObj()

        msg: dict[str, Any] = {"tool_calls": [TCObj()]}
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 1
        assert result[0].id == "tc_2"
        assert result[0].name == "bash"
        assert result[0].arguments == {"command": "pwd"}

    def test_invalid_json_arguments_become_raw(self):
        msg: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "tc_3",
                    "function": {
                        "name": "bash",
                        "arguments": "not valid json {{{",
                    },
                }
            ]
        }
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 1
        assert result[0].arguments == {"raw": "not valid json {{{"}

    def test_non_string_arguments(self):
        """When arguments is already a dict (not a JSON string)."""
        msg: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "tc_4",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "echo hi"},
                    },
                }
            ]
        }
        result = _conv()._normalized_tool_calls(msg)
        assert result[0].arguments == {"command": "echo hi"}

    def test_none_arguments_defaults_to_empty_dict(self):
        msg: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "tc_5",
                    "function": {
                        "name": "bash",
                        "arguments": None,
                    },
                }
            ]
        }
        result = _conv()._normalized_tool_calls(msg)
        assert result[0].arguments == {}

    def test_fallback_to_extra_actions(self):
        """When ``tool_calls`` is missing/empty, fall back to ``extra.actions``."""
        msg: dict[str, Any] = {
            "extra": {
                "actions": [
                    {"command": "ls -la", "tool_call_id": "act_1"},
                    {"command": "pwd"},
                ]
            }
        }
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 2
        assert result[0].name == "bash"
        assert result[0].arguments == {"command": "ls -la"}
        assert result[0].id == "act_1"
        assert result[1].id is None
        assert result[1].arguments == {"command": "pwd"}

    def test_actions_ignored_when_tool_calls_present(self):
        """``extra.actions`` are ignored if ``tool_calls`` is non-empty."""
        msg: dict[str, Any] = {
            "tool_calls": [
                {"id": "tc_x", "function": {"name": "bash", "arguments": "{}"}}
            ],
            "extra": {
                "actions": [{"command": "should be ignored"}],
            },
        }
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 1
        assert result[0].id == "tc_x"

    def test_action_without_command_is_skipped(self):
        msg: dict[str, Any] = {
            "extra": {
                "actions": [
                    {"not_command": "nope"},
                    {"command": "valid"},
                ]
            }
        }
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 1
        assert result[0].arguments == {"command": "valid"}

    def test_tool_call_without_function_uses_defaults(self):
        """A tool_call dict missing 'function' should still produce a ToolCall with defaults."""
        msg: dict[str, Any] = {"tool_calls": [{"id": "tc_nofn"}]}
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 1
        assert result[0].name == "bash"
        assert result[0].arguments == {}

    def test_multiple_tool_calls(self):
        msg: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": "t1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"a"}',
                    },
                },
                {
                    "id": "t2",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"b"}',
                    },
                },
            ]
        }
        result = _conv()._normalized_tool_calls(msg)
        assert len(result) == 2
        assert result[0].arguments == {"command": "a"}
        assert result[1].arguments == {"command": "b"}


# =====================================================================
# _message_to_semconv_messages
# =====================================================================


class TestMessageToSemconvMessages:
    """Tests for ``_message_to_semconv_messages``."""

    def test_user_message(self):
        msg = {"role": "user", "content": "Hello"}
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        assert isinstance(result[0], InputMessage)
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], Text)
        assert result[0].parts[0].content == "Hello"

    def test_user_message_empty_content(self):
        msg = {"role": "user", "content": ""}
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        assert result[0].parts == []

    def test_tool_message(self):
        msg = {"role": "tool", "tool_call_id": "tc_1", "content": "output"}
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        assert isinstance(result[0], InputMessage)
        assert result[0].role == "tool"
        assert len(result[0].parts) == 1
        part = result[0].parts[0]
        assert isinstance(part, ToolCallResponse)
        assert part.id == "tc_1"
        assert part.response == "output"

    def test_tool_message_without_id(self):
        msg = {"role": "tool", "content": "data"}
        result = _conv()._message_to_semconv_messages(msg)
        part = result[0].parts[0]
        assert isinstance(part, ToolCallResponse)
        assert part.id is None

    def test_tool_message_non_string_id(self):
        msg = {"role": "tool", "tool_call_id": 42, "content": "data"}
        result = _conv()._message_to_semconv_messages(msg)
        part = result[0].parts[0]
        assert part.id is None  # non-string ids become None

    def test_assistant_message_text_only(self):
        msg = {"role": "assistant", "content": "I'll help"}
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        out = result[0]
        assert isinstance(out, OutputMessage)
        assert out.role == "assistant"
        assert out.finish_reason == "stop"
        assert len(out.parts) == 1
        assert isinstance(out.parts[0], Text)

    def test_assistant_message_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Running command",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"ls"}',
                    },
                }
            ],
        }
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        out = result[0]
        assert isinstance(out, OutputMessage)
        assert out.finish_reason == "tool_calls"
        # Should have text part + tool call part
        assert len(out.parts) == 2
        assert isinstance(out.parts[0], Text)
        assert isinstance(out.parts[1], ToolCall)

    def test_assistant_message_with_extra_actions(self):
        msg = {
            "role": "assistant",
            "content": "",
            "extra": {"actions": [{"command": "pwd"}]},
        }
        result = _conv()._message_to_semconv_messages(msg)
        out = result[0]
        assert isinstance(out, OutputMessage)
        assert out.finish_reason == "tool_calls"

    def test_assistant_message_empty_content_no_tools(self):
        """Empty assistant message with no tools should get a placeholder Text part."""
        msg = {"role": "assistant", "content": ""}
        result = _conv()._message_to_semconv_messages(msg)
        out = result[0]
        assert isinstance(out, OutputMessage)
        assert len(out.parts) == 1
        assert isinstance(out.parts[0], Text)
        assert out.parts[0].content == ""

    def test_exit_message(self):
        msg = {"role": "exit", "content": "Goodbye"}
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        inp = result[0]
        assert isinstance(inp, InputMessage)
        assert inp.role == "user"
        assert len(inp.parts) == 1
        assert "EXIT: Goodbye" in inp.parts[0].content

    def test_exit_message_no_content(self):
        msg = {"role": "exit"}
        result = _conv()._message_to_semconv_messages(msg)
        inp = result[0]
        assert "EXIT:" in inp.parts[0].content

    def test_unknown_role(self):
        msg = {"role": "custom_role", "content": "custom data"}
        result = _conv()._message_to_semconv_messages(msg)
        assert len(result) == 1
        inp = result[0]
        assert isinstance(inp, InputMessage)
        assert inp.role == "custom_role"
        assert inp.parts[0].content == "custom data"

    def test_none_role(self):
        msg = {"content": "orphan content"}
        result = _conv()._message_to_semconv_messages(msg)
        inp = result[0]
        assert inp.role == "unknown"

    def test_assistant_none_content_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_n", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }
        result = _conv()._message_to_semconv_messages(msg)
        out = result[0]
        assert isinstance(out, OutputMessage)
        # content is None => no text part, only tool call
        assert any(isinstance(p, ToolCall) for p in out.parts)
        assert out.finish_reason == "tool_calls"


# =====================================================================
# bash_tool_definition
# =====================================================================


class TestBashToolDefinition:
    """Tests for ``bash_tool_definition``."""

    def test_returns_function_tool_definition(self):
        result = _conv().bash_tool_definition()
        assert isinstance(result, FunctionToolDefinition)
        assert result.name == "bash"
        assert result.description == "Execute a bash command"
        assert "properties" in result.parameters
        assert result.type == "function"


# =====================================================================
# build_invoke_payload_from_messages
# =====================================================================


class TestBuildInvokePayloadFromMessages:
    """Tests for ``build_invoke_payload_from_messages``."""

    def test_empty_messages(self):
        payload = _conv().build_invoke_payload_from_messages([])
        assert payload["system_instruction"] == []
        assert payload["input_messages"] == []
        assert payload["output_messages"] == []
        assert len(payload["tool_definitions"]) == 1  # bash tool

    def test_full_trajectory(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix the bug"},
            {
                "role": "assistant",
                "content": "Running ls",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "file.py"},
            {"role": "assistant", "content": "Fixed."},
        ]
        payload = _conv().build_invoke_payload_from_messages(messages)

        # system instruction extracted
        assert len(payload["system_instruction"]) == 1
        assert payload["system_instruction"][0].content == "You are helpful."

        # input: user + tool
        assert len(payload["input_messages"]) == 2
        assert payload["input_messages"][0].role == "user"
        assert payload["input_messages"][1].role == "tool"

        # output: two assistant messages
        assert len(payload["output_messages"]) == 2
        assert all(
            isinstance(m, OutputMessage) for m in payload["output_messages"]
        )

        # tool definitions
        assert len(payload["tool_definitions"]) == 1
        assert isinstance(
            payload["tool_definitions"][0], FunctionToolDefinition
        )

    def test_system_only(self):
        messages = [
            {"role": "system", "content": "Sys1"},
            {"role": "system", "content": "Sys2"},
        ]
        payload = _conv().build_invoke_payload_from_messages(messages)
        assert len(payload["system_instruction"]) == 2
        assert payload["input_messages"] == []
        assert payload["output_messages"] == []

    def test_exit_message_in_trajectory(self):
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "exit", "content": "user cancelled"},
        ]
        payload = _conv().build_invoke_payload_from_messages(messages)
        # Both become InputMessage (exit -> user role)
        assert len(payload["input_messages"]) == 2
        assert payload["output_messages"] == []


# =====================================================================
# build_invoke_agent_payload
# =====================================================================


class TestBuildInvokeAgentPayload:
    """Tests for ``build_invoke_agent_payload``."""

    def test_agent_with_messages(self, stub_agent):
        payload = _conv().build_invoke_agent_payload(stub_agent)
        assert len(payload["system_instruction"]) == 1
        assert len(payload["input_messages"]) >= 1
        assert len(payload["output_messages"]) >= 1

    def test_agent_without_messages(self):
        class BareAgent:
            messages = None

        payload = _conv().build_invoke_agent_payload(BareAgent())
        assert payload["system_instruction"] == []
        assert payload["input_messages"] == []
        assert payload["output_messages"] == []

    def test_agent_with_non_dict_messages(self):
        class MixedAgent:
            messages = ["not_a_dict", 42, {"role": "user", "content": "ok"}]

        payload = _conv().build_invoke_agent_payload(MixedAgent())
        # Only the dict message should be processed
        assert len(payload["input_messages"]) == 1


# =====================================================================
# apply_payload_to_entry_invocation
# =====================================================================


class TestApplyPayloadToEntryInvocation:
    """Tests for ``apply_payload_to_entry_invocation``."""

    def test_sets_all_fields(self):
        class FakeEntry:
            input_messages = None
            output_messages = None
            system_instruction = None
            tool_definitions = None

        entry = FakeEntry()
        payload = {
            "input_messages": [
                InputMessage(role="user", parts=[Text(content="hi")])
            ],
            "output_messages": [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="hello")],
                    finish_reason="stop",
                )
            ],
            "system_instruction": [Text(content="sys")],
            "tool_definitions": [
                FunctionToolDefinition(
                    name="bash", description="desc", parameters={}
                )
            ],
        }
        _conv().apply_payload_to_entry_invocation(entry, payload)
        assert entry.input_messages == payload["input_messages"]
        assert entry.output_messages == payload["output_messages"]
        assert entry.system_instruction == payload["system_instruction"]
        assert entry.tool_definitions == payload["tool_definitions"]
