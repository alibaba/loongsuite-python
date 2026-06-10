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

# -*- coding: utf-8 -*-
"""
Tests for utility functions in opentelemetry.instrumentation.agentscope.utils
"""

from types import SimpleNamespace

from agentscope.message import Msg, ToolResultBlock
from agentscope.tracing._converter import (
    _convert_block_to_part as _convert_block_to_part_framework,
)

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.instrumentation.agentscope import utils as utils_module
from opentelemetry.instrumentation.agentscope.utils import (
    _convert_block_to_part as _convert_block_to_part_local,
)
from opentelemetry.instrumentation.agentscope.utils import (
    apply_entry_baggage_identity,
    convert_agentscope_messages_to_genai_format,
    create_agent_invocation,
    create_embedding_invocation,
    create_llm_invocation,
    entry_baggage_identity_attributes,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)
from opentelemetry.util.genai.extended_types import ReactStepInvocation
from opentelemetry.util.genai.types import ToolCallResponse


class TestUtils:
    def test_convert_msg_with_tool_result(self):
        """Test conversion of AgentScope Msg with ToolResultBlock (End-to-End)"""
        # Construct a Msg object simulating a tool execution result
        tool_result_block = ToolResultBlock(
            type="tool_result",
            id="call_test_123",
            name="test_tool",
            output="Tool execution success",
        )

        # AgentScope Msg enforces role to be 'user', 'assistant', or 'system'
        msg = Msg(name="tool", role="user", content=[tool_result_block])

        converted_messages = convert_agentscope_messages_to_genai_format([msg])

        assert len(converted_messages) == 1
        assert converted_messages[0].role == "user"
        assert len(converted_messages[0].parts) == 1

        part = converted_messages[0].parts[0]
        assert isinstance(part, ToolCallResponse)
        assert part.id == "call_test_123"
        assert part.response == "Tool execution success"

    def test_convert_with_local_converter(self):
        """Test conversion using instrumentation's local converter (produces 'result')"""
        block = {
            "type": "tool_result",
            "id": "id_local",
            "name": "tool_local",
            "output": "local output",
        }
        # This uses the copy in utils.py which uses 'result' key
        part = _convert_block_to_part_local(block)

        # Verify internal behavior of local converter
        assert "result" in part
        assert part["result"] == "local output"

        # Verify conversion handles it
        msg_dict = {"role": "tool", "parts": [part]}
        converted = convert_agentscope_messages_to_genai_format(msg_dict)

        assert len(converted) == 1
        part_obj = converted[0].parts[0]
        assert isinstance(part_obj, ToolCallResponse)
        assert part_obj.response == "local output"

    def test_convert_with_framework_converter(self):
        """Test conversion using agentscope framework converter (produces 'response')"""
        block = {
            "type": "tool_result",
            "id": "id_framework",
            "name": "tool_framework",
            "output": "framework output",
        }
        # This uses the official agentscope converter which uses 'response' key
        part = _convert_block_to_part_framework(block)

        # Verify internal behavior of framework converter
        assert "response" in part
        assert part["response"] == "framework output"

        # Verify conversion handles it
        msg_dict = {"role": "tool", "parts": [part]}
        converted = convert_agentscope_messages_to_genai_format(msg_dict)

        assert len(converted) == 1
        part_obj = converted[0].parts[0]
        assert isinstance(part_obj, ToolCallResponse)
        assert part_obj.response == "framework output"

    def test_create_agent_invocation_prefers_entry_baggage_identity(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            utils_module._config, "run_id", "agentscope-run-id"
        )
        ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
        ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user", ctx)
        token = otel_context.attach(ctx)
        try:
            invocation = create_agent_invocation(
                SimpleNamespace(
                    model=None,
                    name="TestAgent",
                    id="agent-id",
                    sys_prompt=None,
                ),
                tuple(),
                {},
            )
        finally:
            otel_context.detach(token)

        assert invocation.conversation_id == "entry-session"
        assert invocation.attributes[GEN_AI_SESSION_ID] == "entry-session"
        assert invocation.attributes[GEN_AI_USER_ID] == "entry-user"

    def test_entry_baggage_identity_attributes(self):
        ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
        ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user", ctx)
        token = otel_context.attach(ctx)
        try:
            attributes = entry_baggage_identity_attributes()
        finally:
            otel_context.detach(token)

        assert attributes == {
            GEN_AI_SESSION_ID: "entry-session",
            GEN_AI_USER_ID: "entry-user",
        }

    def test_create_agent_invocation_falls_back_to_agentscope_run_id(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            utils_module._config, "run_id", "agentscope-run-id"
        )

        invocation = create_agent_invocation(
            SimpleNamespace(
                model=None,
                name="TestAgent",
                id="agent-id",
                sys_prompt=None,
            ),
            tuple(),
            {},
        )

        assert invocation.conversation_id == "agentscope-run-id"
        assert GEN_AI_SESSION_ID not in invocation.attributes
        assert GEN_AI_USER_ID not in invocation.attributes

    def test_model_invocations_copy_entry_baggage_identity(self):
        ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
        ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user", ctx)
        token = otel_context.attach(ctx)
        try:
            llm_invocation = create_llm_invocation(
                SimpleNamespace(model_name="qwen-max"),
                tuple(),
                {},
            )
            embedding_invocation = create_embedding_invocation(
                SimpleNamespace(model_name="text-embedding-v4"),
                tuple(),
                {},
            )
        finally:
            otel_context.detach(token)

        assert llm_invocation.conversation_id == "entry-session"
        assert llm_invocation.attributes[GEN_AI_SESSION_ID] == "entry-session"
        assert llm_invocation.attributes[GEN_AI_USER_ID] == "entry-user"
        assert (
            embedding_invocation.attributes[GEN_AI_SESSION_ID]
            == "entry-session"
        )
        assert embedding_invocation.attributes[GEN_AI_USER_ID] == "entry-user"

    def test_react_step_invocation_copies_entry_baggage_identity(self):
        ctx = baggage.set_baggage(GEN_AI_SESSION_ID, "entry-session")
        ctx = baggage.set_baggage(GEN_AI_USER_ID, "entry-user", ctx)
        token = otel_context.attach(ctx)
        try:
            invocation = ReactStepInvocation(round=1)
            apply_entry_baggage_identity(invocation)
        finally:
            otel_context.detach(token)

        assert invocation.attributes[GEN_AI_SESSION_ID] == "entry-session"
        assert invocation.attributes[GEN_AI_USER_ID] == "entry-user"
