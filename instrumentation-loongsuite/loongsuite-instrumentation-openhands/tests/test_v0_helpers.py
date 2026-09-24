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

"""Tests for V0 wrapper helper functions (non-wrapper code in v0_wrappers.py).

Covers: _set_io, _extract_model_from_config, _state_to_input_messages,
_final_state_to_output, _entry_input_messages_from_initial, _entry_io_from_state,
_action_event_to_parts, _observation_event_to_parts,
_history_to_input_messages_schema, _history_to_output_messages_schema,
_agent_to_system_instructions, _action_type_value, _is_real_tool_call,
_extract_tool_name, _extract_tool_call_id, _runtime_sid,
_tool_call_arguments, _coerce_tool_arguments, _observation_to_result,
_annotate_observation, _first_preview_field, _close_open_step,
_capture_agent_io_attributes.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def tracer():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider._exporter = exporter
    return provider.get_tracer(__name__), exporter


@pytest.fixture(autouse=True)
def _reset():
    yield
    trace_api._TRACER_PROVIDER = None


# ---------------------------------------------------------------------------
# _set_io
# ---------------------------------------------------------------------------


def test_set_io_all_fields(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _set_io,
    )

    tr, exporter = tracer
    with tr.start_as_current_span("test") as span:
        _set_io(
            span,
            input_value="in",
            output_value="out",
            input_messages="im",
            output_messages="om",
        )

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["input.value"] == "in"
    assert attrs["input.mime_type"] == "application/json"
    assert attrs["output.value"] == "out"
    assert attrs["output.mime_type"] == "application/json"
    assert attrs["gen_ai.input.messages"] == "im"
    assert attrs["gen_ai.output.messages"] == "om"


def test_set_io_empty_fields(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _set_io,
    )

    tr, exporter = tracer
    with tr.start_as_current_span("test") as span:
        _set_io(span)

    attrs = exporter.get_finished_spans()[0].attributes
    assert "input.value" not in attrs
    assert "output.value" not in attrs
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs


# ---------------------------------------------------------------------------
# _extract_model_from_config
# ---------------------------------------------------------------------------


def test_extract_model_from_config_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_model_from_config,
    )

    assert _extract_model_from_config(None) == ""


def test_extract_model_from_config_llms_dict():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_model_from_config,
    )

    class LLM:
        model = "gpt-4"

    class Config:
        llms = {"default": LLM()}

    assert _extract_model_from_config(Config()) == "gpt-4"


def test_extract_model_from_config_llm_attr():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_model_from_config,
    )

    class LLM:
        model = "claude-3"

    class Config:
        llms = None
        llm = LLM()

    assert _extract_model_from_config(Config()) == "claude-3"


def test_extract_model_from_config_empty():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_model_from_config,
    )

    class Config:
        llms = {}
        llm = None

    assert _extract_model_from_config(Config()) == ""


# ---------------------------------------------------------------------------
# _state_to_input_messages
# ---------------------------------------------------------------------------


def test_state_to_input_messages_empty():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _state_to_input_messages,
    )

    class State:
        history = []

    # Empty history produces "[]" from to_json_str
    result = _state_to_input_messages(State())
    assert result == "" or json.loads(result) == []


def test_state_to_input_messages_no_list():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _state_to_input_messages,
    )

    class State:
        history = "not a list"

    assert _state_to_input_messages(State()) == ""


def test_state_to_input_messages_message_action():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _state_to_input_messages,
    )

    class MessageAction:
        content = "hi"
        source = "user"

    class State:
        history = [MessageAction()]

    result = _state_to_input_messages(State())
    parsed = json.loads(result)
    assert parsed[0]["role"] == "user"
    assert parsed[0]["content"] == "hi"


def test_state_to_input_messages_system_message():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _state_to_input_messages,
    )

    class SystemMessageAction:
        content = "system prompt"
        source = "system"

    class State:
        history = [SystemMessageAction()]

    result = _state_to_input_messages(State())
    parsed = json.loads(result)
    assert parsed[0]["role"] == "assistant"  # source != "user" -> assistant


def test_state_to_input_messages_action_type():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _state_to_input_messages,
    )

    class CmdRunAction:
        thought = "running cmd"
        command = None
        code = None

    class State:
        history = [CmdRunAction()]

    result = _state_to_input_messages(State())
    parsed = json.loads(result)
    assert parsed[0]["role"] == "assistant"
    assert "running cmd" in parsed[0]["content"]


def test_state_to_input_messages_observation():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _state_to_input_messages,
    )

    class CmdOutputObservation:
        content = "output here"

    class State:
        history = [CmdOutputObservation()]

    result = _state_to_input_messages(State())
    parsed = json.loads(result)
    assert parsed[0]["role"] == "tool"
    assert "output here" in parsed[0]["content"]


# ---------------------------------------------------------------------------
# _final_state_to_output
# ---------------------------------------------------------------------------


def test_final_state_to_output_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _final_state_to_output,
    )

    assert _final_state_to_output(None) == ""


def test_final_state_to_output_basic():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _final_state_to_output,
    )

    class AgentState:
        value = "finished"

    class AgentFinishAction:
        final_thought = "I'm done"
        thought = "done thinking"
        outputs = {"result": "success"}

    class State:
        agent_state = AgentState()
        last_error = None
        iteration = 5
        history = [AgentFinishAction()]

    result = _final_state_to_output(State())
    parsed = json.loads(result)
    assert parsed["agent_state"] == "finished"
    assert parsed["iteration"] == "5"
    assert "I'm done" in parsed["final_thought"]


def test_final_state_to_output_with_error():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _final_state_to_output,
    )

    class AgentState:
        value = "error"

    class State:
        agent_state = AgentState()
        last_error = "something went wrong"
        iteration = None
        history = []

    result = _final_state_to_output(State())
    parsed = json.loads(result)
    assert "something went wrong" in parsed["last_error"]


# ---------------------------------------------------------------------------
# _entry_input_messages_from_initial / _entry_io_from_state
# ---------------------------------------------------------------------------


def test_entry_input_messages_from_initial():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _entry_input_messages_from_initial,
    )

    class Msg:
        content = "do something"

    result = _entry_input_messages_from_initial(Msg())
    parsed = json.loads(result)
    assert parsed[0]["role"] == "user"
    assert "do something" in str(parsed[0]["parts"])


def test_entry_input_messages_from_initial_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _entry_input_messages_from_initial,
    )

    assert _entry_input_messages_from_initial(None) == ""


def test_entry_io_from_state():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _entry_io_from_state,
    )

    class MessageAction:
        content = "user input"
        source = "user"

    class AgentFinishAction:
        final_thought = "done"
        outputs = {}

    class State:
        history = [MessageAction(), AgentFinishAction()]

    input_msgs, output_msgs = _entry_io_from_state(State())
    assert input_msgs  # Should be non-empty
    assert output_msgs  # Should be non-empty
    assert "user input" in input_msgs


def test_entry_io_from_state_no_history():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _entry_io_from_state,
    )

    class State:
        history = []
        agent_state = None
        last_error = None
        iteration = None

    input_msgs, output_msgs = _entry_io_from_state(State())
    assert input_msgs == ""
    # output_msgs may be empty or a fallback structure
    # _final_state_to_output with no meaningful attrs returns ""
    # then output_messages stays ""


# ---------------------------------------------------------------------------
# _action_event_to_parts
# ---------------------------------------------------------------------------


def test_action_event_to_parts_with_thought():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_event_to_parts,
    )

    class Ev:
        thought = "thinking"
        tool_call_metadata = None
        action = None

    parts = _action_event_to_parts(Ev())
    assert parts[0]["type"] == "text"
    assert parts[0]["content"] == "thinking"


def test_action_event_to_parts_with_tool_call_metadata():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_event_to_parts,
    )

    class Fn:
        name = "bash"
        arguments = '{"cmd": "ls"}'

    class TC:
        id = "tc1"
        function = Fn()

    class Msg:
        tool_calls = [TC()]

    class Choice:
        message = Msg()

    class ModelResp:
        choices = [Choice()]

    class TCM:
        function_name = "bash"
        tool_call_id = "tc1"
        model_response = ModelResp()

    class Ev:
        thought = ""
        tool_call_metadata = TCM()
        action = None

    parts = _action_event_to_parts(Ev())
    assert any(p["type"] == "tool_call" for p in parts)
    tool_part = [p for p in parts if p["type"] == "tool_call"][0]
    assert tool_part["name"] == "bash"
    assert tool_part["arguments"] == {"cmd": "ls"}


def test_action_event_to_parts_fallback():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_event_to_parts,
    )

    class Ev:
        thought = ""
        tool_call_metadata = None
        action = "run"

    parts = _action_event_to_parts(Ev())
    assert parts[0]["type"] == "tool_call"
    assert parts[0]["name"] == "run"


def test_action_event_to_parts_with_command_fallback():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_event_to_parts,
    )

    class TCM:
        function_name = "execute_bash"
        tool_call_id = "tc1"
        model_response = None

    class Ev:
        thought = ""
        tool_call_metadata = TCM()
        action = None
        command = "ls -la"
        code = None
        path = None
        url = None
        content = None
        task_list = None
        old_str = None
        new_str = None
        file_text = None

    parts = _action_event_to_parts(Ev())
    tool_part = [p for p in parts if p["type"] == "tool_call"][0]
    assert tool_part["arguments"]["command"] == "ls -la"


# ---------------------------------------------------------------------------
# _observation_event_to_parts
# ---------------------------------------------------------------------------


def test_observation_event_to_parts():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _observation_event_to_parts,
    )

    class TCM:
        tool_call_id = "tc1"

    class Obs:
        tool_call_metadata = TCM()
        content = "output"
        exit_code = 0
        error = None
        stdout = None
        stderr = None
        url = None

    parts = _observation_event_to_parts(Obs())
    assert parts[0]["type"] == "tool_call_response"
    assert parts[0]["id"] == "tc1"
    assert parts[0]["result"]["content"] == "output"


def test_observation_event_to_parts_no_tcm():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _observation_event_to_parts,
    )

    class Obs:
        tool_call_metadata = None
        content = "data"
        exit_code = None
        error = None
        stdout = None
        stderr = None
        url = None

    parts = _observation_event_to_parts(Obs())
    assert parts[0]["id"] == ""
    assert parts[0]["result"]["content"] == "data"


# ---------------------------------------------------------------------------
# _history_to_input_messages_schema
# ---------------------------------------------------------------------------


def test_history_to_input_messages_schema_empty():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_input_messages_schema,
    )

    assert _history_to_input_messages_schema([]) == []
    assert _history_to_input_messages_schema(None) == []


def test_history_to_input_messages_schema_message_action():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_input_messages_schema,
    )

    class MessageAction:
        content = "hello"
        source = "user"

    result = _history_to_input_messages_schema([MessageAction()])
    assert result[0]["role"] == "user"
    assert result[0]["parts"][0]["type"] == "text"
    assert result[0]["parts"][0]["content"] == "hello"


def test_history_to_input_messages_schema_system_message_skipped():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_input_messages_schema,
    )

    class SystemMessageAction:
        content = "system prompt"

    result = _history_to_input_messages_schema([SystemMessageAction()])
    assert result == []  # SystemMessageAction is skipped


def test_history_to_input_messages_schema_folds_same_role():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_input_messages_schema,
    )

    class MessageAction:
        def __init__(self, content):
            self.content = content
            self.source = "user"

    result = _history_to_input_messages_schema(
        [MessageAction("hi"), MessageAction("there")]
    )
    assert len(result) == 1  # Folded into one message
    assert len(result[0]["parts"]) == 2


def test_history_to_input_messages_schema_observation():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_input_messages_schema,
    )

    class CmdOutputObservation:
        content = "output"
        tool_call_metadata = None
        exit_code = None
        error = None
        stdout = None
        stderr = None
        url = None

    result = _history_to_input_messages_schema([CmdOutputObservation()])
    assert result[0]["role"] == "tool"


def test_history_to_input_messages_schema_unknown_event():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_input_messages_schema,
    )

    class UnknownEvent:
        pass

    result = _history_to_input_messages_schema([UnknownEvent()])
    assert result[0]["role"] == "system"


# ---------------------------------------------------------------------------
# _history_to_output_messages_schema
# ---------------------------------------------------------------------------


def test_history_to_output_messages_schema_empty():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_output_messages_schema,
    )

    assert _history_to_output_messages_schema([]) == []


def test_history_to_output_messages_schema_finish_action():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_output_messages_schema,
    )

    class AgentFinishAction:
        final_thought = "Done."
        outputs = {"result": 42}
        thought = None
        tool_call_metadata = None
        action = None

    result = _history_to_output_messages_schema([AgentFinishAction()])
    assert result[0]["role"] == "assistant"
    assert result[0]["finish_reason"] == "stop"
    assert any("Done." in p.get("content", "") for p in result[0]["parts"])


def test_history_to_output_messages_schema_stops_at_user():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_output_messages_schema,
    )

    class MessageAction:
        content = "user msg"
        source = "user"
        thought = None
        tool_call_metadata = None
        action = None

    class AgentFinishAction:
        final_thought = "done"
        outputs = {}
        thought = None
        tool_call_metadata = None
        action = None

    result = _history_to_output_messages_schema(
        [MessageAction(), AgentFinishAction()]
    )
    assert len(result) == 1
    # Only the finish action should be in the output
    assert any("done" in p.get("content", "") for p in result[0]["parts"])


def test_history_to_output_messages_schema_stops_at_observation():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _history_to_output_messages_schema,
    )

    class CmdOutputObservation:
        content = "obs"

    class AgentFinishAction:
        final_thought = "finished"
        outputs = {}
        thought = None
        tool_call_metadata = None
        action = None

    result = _history_to_output_messages_schema(
        [CmdOutputObservation(), AgentFinishAction()]
    )
    assert len(result) == 1
    assert any("finished" in p.get("content", "") for p in result[0]["parts"])


# ---------------------------------------------------------------------------
# _agent_to_system_instructions
# ---------------------------------------------------------------------------


def test_agent_to_system_instructions_via_method():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _agent_to_system_instructions,
    )

    class SysMsg:
        content = "You are helpful."

    class Agent:
        def get_system_message(self):
            return SysMsg()

    result = _agent_to_system_instructions(Agent(), None)
    assert result[0]["type"] == "text"
    assert result[0]["content"] == "You are helpful."


def test_agent_to_system_instructions_via_history():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _agent_to_system_instructions,
    )

    class SystemMessageAction:
        content = "System prompt here."

    class State:
        history = [SystemMessageAction()]

    result = _agent_to_system_instructions(None, State())
    assert result[0]["content"] == "System prompt here."


def test_agent_to_system_instructions_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _agent_to_system_instructions,
    )

    class State:
        history = []

    result = _agent_to_system_instructions(None, State())
    assert result == []


# ---------------------------------------------------------------------------
# _action_type_value
# ---------------------------------------------------------------------------


def test_action_type_value_basic():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_type_value,
    )

    class Action:
        action = "run"

    assert _action_type_value(Action()) == "run"


def test_action_type_value_enum_like():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_type_value,
    )

    class ActionType:
        value = "message"

    class Action:
        action = ActionType()

    assert _action_type_value(Action()) == "message"


def test_action_type_value_enum_str():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_type_value,
    )

    class Action:
        action = "ActionType.RUN"

    assert _action_type_value(Action()) == "run"


def test_action_type_value_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _action_type_value,
    )

    class Action:
        action = None

    assert _action_type_value(Action()) == ""


# ---------------------------------------------------------------------------
# _is_real_tool_call
# ---------------------------------------------------------------------------


def test_is_real_tool_call_internal():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _is_real_tool_call,
    )

    class Action:
        action = "message"
        tool_call_metadata = None

    assert _is_real_tool_call(Action()) is False


def test_is_real_tool_call_with_metadata():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _is_real_tool_call,
    )

    class Action:
        action = "run"
        tool_call_metadata = object()

    assert _is_real_tool_call(Action()) is True


def test_is_real_tool_call_in_whitelist():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _is_real_tool_call,
    )

    class Action:
        action = "recall"
        tool_call_metadata = None

    assert _is_real_tool_call(Action()) is True


def test_is_real_tool_call_unknown():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _is_real_tool_call,
    )

    class Action:
        action = "unknown_type"
        tool_call_metadata = None

    assert _is_real_tool_call(Action()) is False


def test_is_real_tool_call_no_action():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _is_real_tool_call,
    )

    class Action:
        action = None
        tool_call_metadata = None

    assert _is_real_tool_call(Action()) is False


def test_is_real_tool_call_internal_with_metadata():
    """Internal actions should be dropped even with tool_call_metadata."""
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _is_real_tool_call,
    )

    class Action:
        action = "message"
        tool_call_metadata = object()

    assert _is_real_tool_call(Action()) is False


# ---------------------------------------------------------------------------
# _extract_tool_name / _extract_tool_call_id
# ---------------------------------------------------------------------------


def test_extract_tool_name_with_metadata():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_tool_name,
    )

    class TCM:
        function_name = "execute_bash"

    class Action:
        action = "run"
        tool_call_metadata = TCM()

    name, action_type = _extract_tool_name(Action())
    assert name == "execute_bash"
    assert action_type == "run"


def test_extract_tool_name_fallback():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_tool_name,
    )

    class Action:
        action = "run"
        tool_call_metadata = None

    name, action_type = _extract_tool_name(Action())
    assert name == "bash"
    assert action_type == "run"


def test_extract_tool_name_unknown_type():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_tool_name,
    )

    class Action:
        action = "custom_action"
        tool_call_metadata = None

    name, action_type = _extract_tool_name(Action())
    assert name == "custom_action"


def test_extract_tool_call_id():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_tool_call_id,
    )

    class TCM:
        tool_call_id = "call_123"

    class Action:
        tool_call_metadata = TCM()

    assert _extract_tool_call_id(Action()) == "call_123"


def test_extract_tool_call_id_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _extract_tool_call_id,
    )

    class Action:
        tool_call_metadata = None

    assert _extract_tool_call_id(Action()) == ""


# ---------------------------------------------------------------------------
# _runtime_sid
# ---------------------------------------------------------------------------


def test_runtime_sid_direct():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _runtime_sid,
    )

    class Runtime:
        sid = "my-sid"

    assert _runtime_sid(Runtime()) == "my-sid"


def test_runtime_sid_from_event_stream():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _runtime_sid,
    )

    class ES:
        sid = "es-sid"

    class Runtime:
        sid = None
        event_stream = ES()

    assert _runtime_sid(Runtime()) == "es-sid"


def test_runtime_sid_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _runtime_sid,
    )

    class Runtime:
        sid = None
        event_stream = None

    assert _runtime_sid(Runtime()) == ""


# ---------------------------------------------------------------------------
# _coerce_tool_arguments
# ---------------------------------------------------------------------------


def test_coerce_tool_arguments_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _coerce_tool_arguments,
    )

    assert _coerce_tool_arguments(None) == {}
    assert _coerce_tool_arguments("") == {}
    assert _coerce_tool_arguments([]) == {}
    assert _coerce_tool_arguments({}) == {}


def test_coerce_tool_arguments_dict():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _coerce_tool_arguments,
    )

    assert _coerce_tool_arguments({"a": 1}) == {"a": 1}


def test_coerce_tool_arguments_json_str():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _coerce_tool_arguments,
    )

    assert _coerce_tool_arguments('{"a": 1}') == {"a": 1}


def test_coerce_tool_arguments_non_dict_json():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _coerce_tool_arguments,
    )

    assert _coerce_tool_arguments("[1, 2]") == {"value": [1, 2]}


def test_coerce_tool_arguments_invalid_json():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _coerce_tool_arguments,
    )

    assert _coerce_tool_arguments("not json") == {"raw": "not json"}


def test_coerce_tool_arguments_other_type():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _coerce_tool_arguments,
    )

    assert _coerce_tool_arguments(42) == {"value": 42}


# ---------------------------------------------------------------------------
# _tool_call_arguments
# ---------------------------------------------------------------------------


def test_tool_call_arguments_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _tool_call_arguments,
    )

    assert _tool_call_arguments(None) == {}


def test_tool_call_arguments_from_tcm_direct():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _tool_call_arguments,
    )

    class TCM:
        arguments = {"cmd": "ls"}
        model_response = None

    class Action:
        tool_call_metadata = TCM()

    assert _tool_call_arguments(Action()) == {"cmd": "ls"}


def test_tool_call_arguments_from_model_response():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _tool_call_arguments,
    )

    class Fn:
        arguments = '{"path": "/tmp"}'

    class TC:
        id = "tc1"
        function = Fn()

    class Msg:
        tool_calls = [TC()]

    class Choice:
        message = Msg()

    class ModelResp:
        choices = [Choice()]

    class TCM:
        arguments = None
        tool_call_id = "tc1"
        model_response = ModelResp()

    class Action:
        tool_call_metadata = TCM()

    assert _tool_call_arguments(Action()) == {"path": "/tmp"}


def test_tool_call_arguments_fallback_to_fields():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _tool_call_arguments,
    )

    class Action:
        tool_call_metadata = None
        command = "echo hello"
        code = None
        path = None
        url = None
        content = None
        task_list = None
        name = None
        arguments = None
        thought = "thinking"
        is_input = None
        blocking = None
        keep_prompt = None
        translated_ipython_code = None
        browser_actions = None
        agent_state = None
        outputs = None
        final_thought = None
        old_str = None
        new_str = None
        view_range = None
        file_text = None
        insert_line = None
        start_line = None
        end_line = None

    args = _tool_call_arguments(Action())
    assert args["command"] == "echo hello"
    assert args["thought"] == "thinking"


# ---------------------------------------------------------------------------
# _observation_to_result
# ---------------------------------------------------------------------------


def test_observation_to_result_none():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _observation_to_result,
    )

    assert _observation_to_result(None) == {}


def test_observation_to_result_basic():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _observation_to_result,
    )

    class Obs:
        content = "output text"
        exit_code = 0
        error = None
        interpreter_details = None
        command = None
        stdout = None
        stderr = None
        url = None
        screenshot = None
        outputs = None

    result = _observation_to_result(Obs())
    assert result["content"] == "output text"
    assert result["exit_code"] == 0


# ---------------------------------------------------------------------------
# _annotate_observation
# ---------------------------------------------------------------------------


def test_annotate_observation_none(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _annotate_observation,
    )

    tr, exporter = tracer
    with tr.start_as_current_span("test") as span:
        _annotate_observation(span, None)

    attrs = exporter.get_finished_spans()[0].attributes
    assert "openhands.observation.type" not in attrs


def test_annotate_observation_with_error(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _annotate_observation,
    )

    class Obs:
        observation = "run"
        exit_code = None
        error = "something failed"
        content = None
        interpreter_details = None
        command = None
        stdout = None
        stderr = None
        url = None
        screenshot = None
        outputs = None

    tr, exporter = tracer
    with tr.start_as_current_span("test") as span:
        _annotate_observation(span, Obs())

    s = exporter.get_finished_spans()[0]
    assert s.attributes["openhands.observation.error"] == "something failed"
    assert s.status.status_code.name == "ERROR"


def test_annotate_observation_nonzero_exit(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _annotate_observation,
    )

    class Obs:
        observation = "run"
        exit_code = 1
        error = None
        content = "fail"
        interpreter_details = None
        command = None
        stdout = None
        stderr = None
        url = None
        screenshot = None
        outputs = None

    tr, exporter = tracer
    with tr.start_as_current_span("test") as span:
        _annotate_observation(span, Obs())

    s = exporter.get_finished_spans()[0]
    assert s.attributes["openhands.action.exit_code"] == 1
    assert s.status.status_code.name == "ERROR"


def test_annotate_observation_invalid_exit_code(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _annotate_observation,
    )

    class Obs:
        observation = "run"
        exit_code = "not_a_number"
        error = None
        content = None
        interpreter_details = None
        command = None
        stdout = None
        stderr = None
        url = None
        screenshot = None
        outputs = None

    tr, exporter = tracer
    with tr.start_as_current_span("test") as span:
        _annotate_observation(span, Obs())

    # Should not raise, exit_code parsing just skipped
    s = exporter.get_finished_spans()[0]
    assert "openhands.action.exit_code" not in s.attributes


# ---------------------------------------------------------------------------
# _first_preview_field
# ---------------------------------------------------------------------------


def test_first_preview_field():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _first_preview_field,
    )

    class Action:
        command = "ls"
        code = None
        path = None
        url = None
        content = None

    field, value = _first_preview_field(Action())
    assert field == "command"
    assert value == "ls"


def test_first_preview_field_empty():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _first_preview_field,
    )

    class Action:
        command = None
        code = None
        path = None
        url = None
        content = None

    field, value = _first_preview_field(Action())
    assert field == ""
    assert value == ""


# ---------------------------------------------------------------------------
# _close_open_step
# ---------------------------------------------------------------------------


def test_close_open_step_no_span():
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _close_open_step,
    )

    class Ctrl:
        _otel_oh_step_span = None
        id = "sid"

    _close_open_step(Ctrl())  # Should not raise


def test_close_open_step_with_span(tracer):
    from opentelemetry.instrumentation.openhands.internal.session_context import (
        clear_all,
        get_context,
        store_context,
    )
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _AGENT_CTX_ATTR,
        _STEP_SPAN_ATTR,
        _close_open_step,
    )

    clear_all()
    tr, exporter = tracer
    span = tr.start_span("step")
    agent_ctx = mock.MagicMock()

    class Ctrl:
        id = "sid"

    ctrl = Ctrl()
    setattr(ctrl, _STEP_SPAN_ATTR, span)
    setattr(ctrl, _AGENT_CTX_ATTR, agent_ctx)

    store_context("sid", mock.MagicMock())
    _close_open_step(ctrl)

    assert getattr(ctrl, _STEP_SPAN_ATTR) is None
    # Should have restored to agent context
    ctx = get_context("sid")
    assert ctx is agent_ctx
    clear_all()


# ---------------------------------------------------------------------------
# _capture_agent_io_attributes
# ---------------------------------------------------------------------------


def test_capture_agent_io_attributes(tracer):
    from opentelemetry.instrumentation.openhands.internal.v0_wrappers import (
        _capture_agent_io_attributes,
    )

    class SystemMessageAction:
        content = "You are helpful."

    class MessageAction:
        content = "do something"
        source = "user"

    class AgentFinishAction:
        final_thought = "done"
        outputs = {}
        thought = None
        tool_call_metadata = None
        action = None

    class State:
        history = [SystemMessageAction(), MessageAction(), AgentFinishAction()]

    tr, exporter = tracer
    with tr.start_as_current_span("agent") as span:
        _capture_agent_io_attributes(span, None, None, State())

    attrs = exporter.get_finished_spans()[0].attributes
    assert "gen_ai.system_instructions" in attrs
    assert "gen_ai.input.messages" in attrs
    assert "gen_ai.output.messages" in attrs
