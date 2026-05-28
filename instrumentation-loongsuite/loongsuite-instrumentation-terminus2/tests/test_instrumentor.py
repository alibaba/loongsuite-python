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

"""Tests for the Terminus2Instrumentor: lifecycle, span creation, and
parent-child relationships for all span types (ENTRY, AGENT, TOOL, STEP,
TASK, CHAIN).
"""

from __future__ import annotations

import json

import pytest

from opentelemetry.instrumentation.terminus2 import (
    _AGENT_NAME,
    _FRAMEWORK,
    _GEN_AI_FRAMEWORK,
    _GEN_AI_INPUT_MESSAGES,
    _GEN_AI_OPERATION_NAME,
    _GEN_AI_OUTPUT_MESSAGES,
    _GEN_AI_PROVIDER_NAME,
    _GEN_AI_REACT_FINISH_REASON,
    _GEN_AI_REACT_ROUND,
    _GEN_AI_REQUEST_MODEL,
    _GEN_AI_SPAN_KIND,
    _GEN_AI_SYSTEM_INSTRUCTIONS,
    _GEN_AI_TOOL_CALL_ARGUMENTS,
    _GEN_AI_TOOL_CALL_RESULT,
    _GEN_AI_TOOL_DEFINITIONS,
    _GEN_AI_TOOL_DESCRIPTION,
    _GEN_AI_TOOL_NAME,
    _GEN_AI_TOOL_TYPE,
    _OP_ENTER,
    _OP_EXECUTE_TOOL,
    _OP_INVOKE_AGENT,
    _OP_REACT,
    _OP_RUN_TASK,
    _OP_TASK,
    _SPAN_KIND_AGENT,
    _SPAN_KIND_CHAIN,
    _SPAN_KIND_ENTRY,
    _SPAN_KIND_STEP,
    _SPAN_KIND_TASK,
    _SPAN_KIND_TOOL,
    _TERMINAL_TOOL_DESCRIPTION,
    _TERMINAL_TOOL_NAME,
    _TOOL_TYPE_EXTENSION,
    Terminus2Instrumentor,
    _end_current_step,
    _react_round_counter,
    _try_unwrap,
    _try_wrap,
)
from opentelemetry.trace import StatusCode

from .conftest import (
    AgentResult,
    Chat,
    Command,
    ParseResult,
    Terminus2,
    TerminusJSONPlainParser,
    TerminusXMLPlainParser,
)

# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    """Instrument / uninstrument correctness."""

    def test_instrumentation_dependencies(self):
        instrumentor = Terminus2Instrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert deps == ("terminal-bench >= 0.1.0",)

    def test_instrument_and_uninstrument(self, tracer_provider, span_exporter):
        """Instrument should wrap targets; uninstrument should restore them."""
        instrumentor = Terminus2Instrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        # After instrumenting, calling perform_task should produce a span
        agent = Terminus2()
        agent.perform_task("test instruction")
        spans = span_exporter.get_finished_spans()
        assert any(s.name == "enter_ai_application_system" for s in spans)

        instrumentor.uninstrument()
        span_exporter.clear()

        # After uninstrumenting, no new spans should be produced
        agent.perform_task("test instruction again")
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_double_instrument_is_idempotent(
        self, tracer_provider, span_exporter
    ):
        """Instrumenting twice should not double-wrap."""
        instrumentor = Terminus2Instrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        # Second instrument call -- _try_wrap should detect the marker
        instrumentor2 = Terminus2Instrumentor()
        instrumentor2.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        agent = Terminus2()
        agent.perform_task("test instruction")
        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        # Only one ENTRY span, not two
        assert len(entry_spans) == 1

        instrumentor.uninstrument()


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY span: Terminus2.perform_task
# ═══════════════════════════════════════════════════════════════════════════


class TestEntrySpan:
    """Tests for the ENTRY span produced by perform_task."""

    def test_basic_entry_span(self, instrument, span_exporter):
        agent = Terminus2()
        agent._model_name = "gpt-4o"
        agent._perform_task_override = AgentResult(
            failure_mode=None, timestamped_markers=["m1"]
        )

        agent.perform_task("Install vim")

        spans = span_exporter.get_finished_spans()
        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        assert len(entry) == 1
        s = entry[0]

        assert s.attributes[_GEN_AI_SPAN_KIND] == _SPAN_KIND_ENTRY
        assert s.attributes[_GEN_AI_OPERATION_NAME] == _OP_ENTER
        assert s.attributes[_GEN_AI_FRAMEWORK] == _FRAMEWORK
        assert s.attributes[_GEN_AI_REQUEST_MODEL] == "gpt-4o"
        assert s.attributes[_GEN_AI_PROVIDER_NAME] == "openai"

        # Input message
        input_msgs = json.loads(s.attributes[_GEN_AI_INPUT_MESSAGES])
        assert input_msgs[0]["role"] == "user"
        assert input_msgs[0]["parts"][0]["content"] == "Install vim"

        # Output message
        output_msgs = json.loads(s.attributes[_GEN_AI_OUTPUT_MESSAGES])
        assert output_msgs[0]["role"] == "assistant"
        output_content = json.loads(output_msgs[0]["parts"][0]["content"])
        assert output_content["failure_mode"] == "none"
        assert output_content["marker_count"] == 1

        assert s.attributes["terminus2.failure_mode"] == "none"
        assert s.status.status_code == StatusCode.OK

    def test_entry_span_error(self, instrument, span_exporter):
        """Error in perform_task should record exception on ENTRY span."""
        agent = Terminus2()
        agent._perform_task_error = RuntimeError("task failed")

        with pytest.raises(RuntimeError, match="task failed"):
            agent.perform_task("do something")

        spans = span_exporter.get_finished_spans()
        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        assert len(entry) == 1
        s = entry[0]
        assert s.status.status_code == StatusCode.ERROR
        assert len(s.events) > 0  # exception event recorded

    def test_entry_span_with_failure_mode_enum(
        self, instrument, span_exporter
    ):
        """failure_mode with a .value attribute (e.g. enum) should be serialized."""
        from enum import Enum

        class FM(Enum):
            TIMEOUT = "timeout"

        agent = Terminus2()
        agent._perform_task_override = AgentResult(
            failure_mode=FM.TIMEOUT, timestamped_markers=[]
        )

        agent.perform_task("test")

        spans = span_exporter.get_finished_spans()
        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        assert entry[0].attributes["terminus2.failure_mode"] == "timeout"

    def test_entry_span_empty_instruction(self, instrument, span_exporter):
        """When instruction is empty, gen_ai.input.messages should not be set."""
        agent = Terminus2()
        agent.perform_task("")

        spans = span_exporter.get_finished_spans()
        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        assert _GEN_AI_INPUT_MESSAGES not in entry[0].attributes

    def test_entry_span_instruction_via_kwargs(
        self, instrument, span_exporter
    ):
        """Instruction passed as keyword argument should be captured."""
        agent = Terminus2()
        agent.perform_task(instruction="hello via kwarg")

        spans = span_exporter.get_finished_spans()
        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        input_msgs = json.loads(entry[0].attributes[_GEN_AI_INPUT_MESSAGES])
        assert input_msgs[0]["parts"][0]["content"] == "hello via kwarg"


# ═══════════════════════════════════════════════════════════════════════════
# AGENT span: Terminus2._run_agent_loop
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentSpan:
    """Tests for the AGENT span produced by _run_agent_loop."""

    def test_basic_agent_span(self, instrument, span_exporter):
        agent = Terminus2()
        agent._model_name = "claude-3-5-sonnet"
        agent._parser_name = "xml"
        agent._prompt_template = "You are a terminal agent."
        agent._pending_completion = True

        chat = Chat(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "I will help"},
            ]
        )

        agent._run_agent_loop(
            "initial prompt",
            None,
            chat,
            None,
            "original instruction",
        )

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        assert len(agent_spans) == 1
        s = agent_spans[0]

        assert s.attributes[_GEN_AI_SPAN_KIND] == _SPAN_KIND_AGENT
        assert s.attributes[_GEN_AI_OPERATION_NAME] == _OP_INVOKE_AGENT
        assert s.attributes[_GEN_AI_FRAMEWORK] == _FRAMEWORK
        assert s.attributes["gen_ai.agent.name"] == _AGENT_NAME
        assert s.attributes[_GEN_AI_REQUEST_MODEL] == "claude-3-5-sonnet"
        assert s.attributes[_GEN_AI_PROVIDER_NAME] == "anthropic"
        assert s.attributes["terminus2.parser"] == "xml"
        assert (
            s.attributes[_GEN_AI_SYSTEM_INSTRUCTIONS]
            == "You are a terminal agent."
        )
        assert _GEN_AI_TOOL_DEFINITIONS in s.attributes

        # Input messages
        input_msgs = json.loads(s.attributes[_GEN_AI_INPUT_MESSAGES])
        assert input_msgs[0]["parts"][0]["content"] == "original instruction"

        # Output messages
        output_msgs = json.loads(s.attributes[_GEN_AI_OUTPUT_MESSAGES])
        output_content = json.loads(output_msgs[0]["parts"][0]["content"])
        assert output_content["pending_completion"] is True
        assert output_content["final_assistant_message"] == "I will help"
        assert s.attributes["terminus2.pending_completion"] is True

        assert s.status.status_code == StatusCode.OK

    def test_agent_span_rounds_counter(self, instrument, span_exporter):
        """The react rounds counter should be recorded on the AGENT span."""
        agent = Terminus2()

        # Simulate 3 rounds by manually setting the counter
        _react_round_counter.set(3)
        agent._run_agent_loop("p", None, Chat(), None, "inst")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        # Counter is reset to 0 by the wrapper at the start of
        # _run_agent_loop, so it should be 0 (no _handle_llm_interaction
        # calls happened inside the stub).
        assert agent_spans[0].attributes["terminus2.react.rounds"] == 0

    def test_agent_span_error(self, instrument, span_exporter):
        agent = Terminus2()
        agent._run_agent_loop_error = ValueError("loop exploded")

        with pytest.raises(ValueError, match="loop exploded"):
            agent._run_agent_loop("p", None, Chat(), None, "inst")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        assert len(agent_spans) == 1
        assert agent_spans[0].status.status_code == StatusCode.ERROR

    def test_agent_span_no_original_instruction(
        self, instrument, span_exporter
    ):
        """When original_instruction is empty, input messages not set."""
        agent = Terminus2()
        agent._run_agent_loop("prompt", None, Chat(), None, "")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        assert _GEN_AI_INPUT_MESSAGES not in agent_spans[0].attributes

    def test_agent_span_no_system_instructions(
        self, instrument, span_exporter
    ):
        """When _prompt_template is empty, system_instructions not set."""
        agent = Terminus2()
        agent._prompt_template = ""
        agent._run_agent_loop("prompt", None, Chat(), None, "inst")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        assert _GEN_AI_SYSTEM_INSTRUCTIONS not in agent_spans[0].attributes

    def test_agent_span_empty_chat_messages(self, instrument, span_exporter):
        """When chat has no messages, final_assistant_message is empty."""
        agent = Terminus2()
        agent._run_agent_loop("prompt", None, Chat(messages=[]), None, "inst")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        output_msgs = json.loads(
            agent_spans[0].attributes[_GEN_AI_OUTPUT_MESSAGES]
        )
        output_content = json.loads(output_msgs[0]["parts"][0]["content"])
        assert output_content["final_assistant_message"] == ""

    def test_agent_span_description_attribute(self, instrument, span_exporter):
        """Agent description should be set."""
        agent = Terminus2()
        agent._run_agent_loop("p", None, Chat(), None, "inst")

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        assert "gen_ai.agent.description" in agent_spans[0].attributes


# ═══════════════════════════════════════════════════════════════════════════
# TOOL span: Terminus2._execute_commands
# ═══════════════════════════════════════════════════════════════════════════


class TestToolSpan:
    """Tests for the TOOL span produced by _execute_commands."""

    def test_basic_tool_span(self, instrument, span_exporter):
        agent = Terminus2()
        agent._execute_commands_override = (False, "total 42\ndrwxr-xr-x")

        cmds = [
            Command(keystrokes="ls -la", duration_sec=3.0),
            Command(keystrokes="cat file.txt", duration_sec=5.0),
        ]

        result = agent._execute_commands(cmds)
        assert result == (False, "total 42\ndrwxr-xr-x")

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.name == f"execute_tool {_TERMINAL_TOOL_NAME}"
        ]
        assert len(tool_spans) == 1
        s = tool_spans[0]

        assert s.attributes[_GEN_AI_SPAN_KIND] == _SPAN_KIND_TOOL
        assert s.attributes[_GEN_AI_OPERATION_NAME] == _OP_EXECUTE_TOOL
        assert s.attributes[_GEN_AI_FRAMEWORK] == _FRAMEWORK
        assert s.attributes[_GEN_AI_TOOL_NAME] == _TERMINAL_TOOL_NAME
        assert (
            s.attributes[_GEN_AI_TOOL_DESCRIPTION]
            == _TERMINAL_TOOL_DESCRIPTION
        )
        assert s.attributes[_GEN_AI_TOOL_TYPE] == _TOOL_TYPE_EXTENSION
        assert s.attributes["terminus2.commands.count"] == 2

        # Arguments should be serialized command list
        args_parsed = json.loads(s.attributes[_GEN_AI_TOOL_CALL_ARGUMENTS])
        assert len(args_parsed) == 2
        assert args_parsed[0]["keystrokes"] == "ls -la"
        assert args_parsed[0]["duration_sec"] == 3.0

        # Result should be terminal output
        assert s.attributes[_GEN_AI_TOOL_CALL_RESULT] == "total 42\ndrwxr-xr-x"
        assert s.attributes["terminus2.terminal.timeout"] is False
        assert s.status.status_code == StatusCode.OK

    def test_tool_span_timeout(self, instrument, span_exporter):
        agent = Terminus2()
        agent._execute_commands_override = (True, "partial output")

        agent._execute_commands([Command(keystrokes="sleep 999")])

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.name == f"execute_tool {_TERMINAL_TOOL_NAME}"
        ]
        assert tool_spans[0].attributes["terminus2.terminal.timeout"] is True

    def test_tool_span_none_output(self, instrument, span_exporter):
        """When terminal output is None, tool.call.result not set."""
        agent = Terminus2()
        agent._execute_commands_override = (False, None)

        agent._execute_commands([])

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.name == f"execute_tool {_TERMINAL_TOOL_NAME}"
        ]
        assert _GEN_AI_TOOL_CALL_RESULT not in tool_spans[0].attributes

    def test_tool_span_error(self, instrument, span_exporter):
        agent = Terminus2()
        agent._execute_commands_error = OSError("tmux died")

        with pytest.raises(OSError, match="tmux died"):
            agent._execute_commands([Command(keystrokes="x")])

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.name == f"execute_tool {_TERMINAL_TOOL_NAME}"
        ]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR

    def test_tool_span_empty_commands(self, instrument, span_exporter):
        """Empty command list should produce a TOOL span with count=0."""
        agent = Terminus2()
        agent._execute_commands([])

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.name == f"execute_tool {_TERMINAL_TOOL_NAME}"
        ]
        assert tool_spans[0].attributes["terminus2.commands.count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# STEP span: Terminus2._handle_llm_interaction
# ═══════════════════════════════════════════════════════════════════════════


class TestStepSpan:
    """Tests for the STEP span produced by _handle_llm_interaction."""

    def _reset_step_state(self):
        _end_current_step()
        _react_round_counter.set(0)

    def test_basic_step_span(self, instrument, span_exporter):
        self._reset_step_state()
        agent = Terminus2()
        agent._handle_llm_override = (
            [Command(keystrokes="echo hi")],
            False,
            "",
        )

        result = agent._handle_llm_interaction()
        commands, is_complete, feedback = result
        assert not is_complete

        # STEP span stays open -- force close it for export
        _end_current_step(finish_reason="test_end")

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) == 1
        s = step_spans[0]

        assert s.attributes[_GEN_AI_SPAN_KIND] == _SPAN_KIND_STEP
        assert s.attributes[_GEN_AI_OPERATION_NAME] == _OP_REACT
        assert s.attributes[_GEN_AI_FRAMEWORK] == _FRAMEWORK
        assert s.attributes[_GEN_AI_REACT_ROUND] == 1

    def test_step_span_complete(self, instrument, span_exporter):
        """When is_task_complete=True, finish_reason should be 'complete'."""
        self._reset_step_state()
        agent = Terminus2()
        agent._handle_llm_override = ([], True, "")

        agent._handle_llm_interaction()

        # Close step to export
        _end_current_step()

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert (
            step_spans[0].attributes[_GEN_AI_REACT_FINISH_REASON] == "complete"
        )

    def test_step_span_parse_error(self, instrument, span_exporter):
        """When feedback contains ERROR:, finish_reason = 'parse_error'."""
        self._reset_step_state()
        agent = Terminus2()
        agent._handle_llm_override = ([], False, "ERROR: invalid JSON")

        agent._handle_llm_interaction()

        _end_current_step()

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert (
            step_spans[0].attributes[_GEN_AI_REACT_FINISH_REASON]
            == "parse_error"
        )

    def test_step_span_next_round_closes_previous(
        self, instrument, span_exporter
    ):
        """Calling _handle_llm_interaction twice closes the first STEP."""
        self._reset_step_state()
        agent = Terminus2()
        agent._handle_llm_override = ([], False, "")

        agent._handle_llm_interaction()  # round 1
        agent._handle_llm_interaction()  # round 2 -- closes round 1

        # Close round 2
        _end_current_step(finish_reason="loop_end")

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) == 2
        # First span should have finish_reason="next_round"
        assert (
            step_spans[0].attributes.get(_GEN_AI_REACT_FINISH_REASON)
            == "next_round"
        )
        # Round numbers should increment
        assert step_spans[0].attributes[_GEN_AI_REACT_ROUND] == 1
        assert step_spans[1].attributes[_GEN_AI_REACT_ROUND] == 2

    def test_step_span_error(self, instrument, span_exporter):
        self._reset_step_state()
        agent = Terminus2()
        agent._handle_llm_error = ConnectionError("LLM timeout")

        with pytest.raises(ConnectionError, match="LLM timeout"):
            agent._handle_llm_interaction()

        # On error the span records exception + error status but stays open;
        # close it to export.
        _end_current_step()

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) == 1
        assert (
            step_spans[0].attributes.get(_GEN_AI_REACT_FINISH_REASON)
            == "error"
        )
        assert step_spans[0].status.status_code == StatusCode.ERROR

    def test_step_span_no_finish_reason_on_continue(
        self, instrument, span_exporter
    ):
        """When not complete and no error, no finish_reason is set by the
        wrapper itself -- only set when closed externally."""
        self._reset_step_state()
        agent = Terminus2()
        agent._handle_llm_override = ([], False, "thinking...")

        agent._handle_llm_interaction()

        # Close without a reason
        _end_current_step()

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        # No finish_reason attribute was set by the wrapper (feedback does
        # not contain "ERROR:"), and _end_current_step was called with None.
        assert _GEN_AI_REACT_FINISH_REASON not in step_spans[0].attributes


# ═══════════════════════════════════════════════════════════════════════════
# TASK span: parser.parse_response
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskSpan:
    """Tests for the TASK span produced by parse_response."""

    def test_json_parser_task_span(self, instrument, span_exporter):
        parser = TerminusJSONPlainParser()
        cmds = [Command(keystrokes="ls")]
        parser._parse_response_override = ParseResult(
            commands=cmds,
            is_task_complete=True,
            error=None,
            warning="minor issue",
        )

        parser.parse_response("some LLM response")

        spans = span_exporter.get_finished_spans()
        task_spans = [s for s in spans if s.name == "run_task parse_response"]
        assert len(task_spans) == 1
        s = task_spans[0]

        assert s.attributes[_GEN_AI_SPAN_KIND] == _SPAN_KIND_TASK
        assert s.attributes[_GEN_AI_OPERATION_NAME] == _OP_RUN_TASK
        assert s.attributes[_GEN_AI_FRAMEWORK] == _FRAMEWORK
        assert s.attributes["terminus2.parser"] == "json"
        assert s.attributes["terminus2.task_complete"] is True
        assert s.attributes["terminus2.commands.count"] == 1
        assert s.attributes["terminus2.parse.warning"] == "minor issue"
        assert "terminus2.parse.error" not in s.attributes

        # Input messages (the LLM response being parsed)
        input_msgs = json.loads(s.attributes[_GEN_AI_INPUT_MESSAGES])
        assert input_msgs[0]["role"] == "assistant"
        assert input_msgs[0]["parts"][0]["content"] == "some LLM response"

        # Output messages (structured parse result)
        output_msgs = json.loads(s.attributes[_GEN_AI_OUTPUT_MESSAGES])
        output_content = json.loads(output_msgs[0]["parts"][0]["content"])
        assert output_content["is_task_complete"] is True
        assert len(output_content["commands"]) == 1
        assert output_content["warning"] == "minor issue"

        assert s.status.status_code == StatusCode.OK

    def test_xml_parser_task_span(self, instrument, span_exporter):
        parser = TerminusXMLPlainParser()
        parser._parse_response_override = ParseResult(
            commands=[],
            is_task_complete=False,
            error="parse failure",
            warning=None,
        )

        parser.parse_response("<xml>response</xml>")

        spans = span_exporter.get_finished_spans()
        task_spans = [s for s in spans if s.name == "run_task parse_response"]
        assert len(task_spans) == 1
        s = task_spans[0]

        assert s.attributes["terminus2.parser"] == "xml"
        assert s.attributes["terminus2.task_complete"] is False
        assert s.attributes["terminus2.commands.count"] == 0
        assert s.attributes["terminus2.parse.error"] == "parse failure"
        assert "terminus2.parse.warning" not in s.attributes

    def test_task_span_error(self, instrument, span_exporter):
        parser = TerminusJSONPlainParser()
        parser._parse_response_error = ValueError("bad json")

        with pytest.raises(ValueError, match="bad json"):
            parser.parse_response("not json")

        spans = span_exporter.get_finished_spans()
        task_spans = [s for s in spans if s.name == "run_task parse_response"]
        assert len(task_spans) == 1
        assert task_spans[0].status.status_code == StatusCode.ERROR

    def test_task_span_none_response(self, instrument, span_exporter):
        """Passing None as response should not crash the wrapper."""
        parser = TerminusJSONPlainParser()
        # Default stub returns ParseResult() with empty values
        parser.parse_response(None)

        spans = span_exporter.get_finished_spans()
        task_spans = [s for s in spans if s.name == "run_task parse_response"]
        assert len(task_spans) == 1
        # None is not None => input messages should not be set (but the
        # code checks ``if response_text is not None``). None IS None, so
        # input messages should NOT be set.
        assert _GEN_AI_INPUT_MESSAGES not in task_spans[0].attributes

    def test_task_span_with_error_and_warning(self, instrument, span_exporter):
        """Both error and warning should be recorded when present."""
        parser = TerminusJSONPlainParser()
        parser._parse_response_override = ParseResult(
            commands=[],
            is_task_complete=False,
            error="something broke",
            warning="also this warning",
        )

        parser.parse_response("test")

        spans = span_exporter.get_finished_spans()
        task_spans = [s for s in spans if s.name == "run_task parse_response"]
        s = task_spans[0]
        assert s.attributes["terminus2.parse.error"] == "something broke"
        assert s.attributes["terminus2.parse.warning"] == "also this warning"


# ═══════════════════════════════════════════════════════════════════════════
# CHAIN span: Terminus2._summarize
# ═══════════════════════════════════════════════════════════════════════════


class TestChainSpan:
    """Tests for the CHAIN span produced by _summarize."""

    def test_basic_chain_span(self, instrument, span_exporter):
        agent = Terminus2()
        agent._summarize()

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name == "chain summarize"]
        assert len(chain_spans) == 1
        s = chain_spans[0]

        assert s.attributes[_GEN_AI_SPAN_KIND] == _SPAN_KIND_CHAIN
        assert s.attributes[_GEN_AI_OPERATION_NAME] == _OP_TASK
        assert s.attributes[_GEN_AI_FRAMEWORK] == _FRAMEWORK
        assert s.status.status_code == StatusCode.OK

    def test_chain_span_error(self, instrument, span_exporter):
        agent = Terminus2()
        agent._summarize_error = MemoryError("context overflow")

        with pytest.raises(MemoryError, match="context overflow"):
            agent._summarize()

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name == "chain summarize"]
        assert len(chain_spans) == 1
        assert chain_spans[0].status.status_code == StatusCode.ERROR


# ═══════════════════════════════════════════════════════════════════════════
# Parent-child relationships
# ═══════════════════════════════════════════════════════════════════════════


class TestParentChildRelationships:
    """Verify span hierarchy: ENTRY > AGENT > STEP > TOOL."""

    def test_entry_agent_hierarchy(self, tracer_provider, span_exporter):
        """AGENT span should be a child of ENTRY span."""
        _end_current_step()
        _react_round_counter.set(0)

        instrumentor = Terminus2Instrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        agent = Terminus2()
        agent._model_name = "gpt-4o"

        # perform_task calls _run_agent_loop internally
        def perform_with_loop(self, instruction):
            self._run_agent_loop("prompt", None, Chat(), None, instruction)
            return AgentResult()

        # We cannot patch __wrapped__, so we make the stub delegate.
        # Override perform_task to call _run_agent_loop. But since the
        # wrapper has already captured the original perform_task, and the
        # original perform_task checks _perform_task_override... we need a
        # different approach. Instead, we set _perform_task_override to
        # None and patch the original method.
        #
        # Actually, we need the original perform_task (the one the wrapper
        # captured) to call _run_agent_loop. The simplest way: before
        # instrumenting, change the class method. But instrument fixture
        # already ran. So let's use a fresh instrumentor.
        instrumentor.uninstrument()

        # Temporarily replace the stub's perform_task to call _run_agent_loop
        original_perform = Terminus2.perform_task

        def perform_with_agent_loop(self, instruction=""):
            self._run_agent_loop("prompt", None, Chat(), None, instruction)
            return AgentResult()

        Terminus2.perform_task = perform_with_agent_loop

        instrumentor2 = Terminus2Instrumentor()
        instrumentor2.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent.perform_task("test hierarchy")
        finally:
            instrumentor2.uninstrument()
            Terminus2.perform_task = original_perform

        spans = span_exporter.get_finished_spans()
        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        agent_s = [s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"]

        assert len(entry) == 1
        assert len(agent_s) == 1

        # AGENT's parent should be ENTRY
        assert agent_s[0].parent is not None
        assert agent_s[0].parent.span_id == entry[0].context.span_id

    def test_agent_step_tool_hierarchy(self, tracer_provider, span_exporter):
        """STEP and TOOL should be children of AGENT; TOOL child of STEP."""
        _end_current_step()
        _react_round_counter.set(0)

        # Replace stubs to chain calls: _run_agent_loop calls
        # _handle_llm_interaction then _execute_commands
        original_loop = Terminus2._run_agent_loop
        original_llm = Terminus2._handle_llm_interaction
        original_exec = Terminus2._execute_commands

        cmds = [Command(keystrokes="pwd")]

        def loop_body(
            self,
            initial_prompt,
            session=None,
            chat=None,
            logging_dir=None,
            original_instruction="",
        ):
            self._handle_llm_interaction()
            self._execute_commands(cmds)
            return None

        def llm_body(self, *a, **kw):
            return (cmds, True, "")

        def exec_body(self, c):
            return (False, "/home")

        Terminus2._run_agent_loop = loop_body
        Terminus2._handle_llm_interaction = llm_body
        Terminus2._execute_commands = exec_body

        instrumentor = Terminus2Instrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            agent = Terminus2()
            agent._run_agent_loop("p", None, Chat(), None, "inst")
        finally:
            instrumentor.uninstrument()
            Terminus2._run_agent_loop = original_loop
            Terminus2._handle_llm_interaction = original_llm
            Terminus2._execute_commands = original_exec

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.name == f"invoke_agent {_AGENT_NAME}"
        ]
        step_spans = [s for s in spans if s.name == "react step"]
        tool_spans = [
            s for s in spans if s.name == f"execute_tool {_TERMINAL_TOOL_NAME}"
        ]

        assert len(agent_spans) == 1
        assert len(step_spans) == 1
        assert len(tool_spans) == 1

        agent_span_id = agent_spans[0].context.span_id
        step_span_id = step_spans[0].context.span_id

        # STEP is child of AGENT
        assert step_spans[0].parent is not None
        assert step_spans[0].parent.span_id == agent_span_id

        # TOOL is child of STEP (STEP span stays open as current context)
        assert tool_spans[0].parent is not None
        assert tool_spans[0].parent.span_id == step_span_id


# ═══════════════════════════════════════════════════════════════════════════
# _try_wrap idempotency
# ═══════════════════════════════════════════════════════════════════════════


class TestTryWrapIdempotency:
    """Tests for _try_wrap / _try_unwrap idempotency."""

    def test_double_wrap_does_not_stack(self):
        """Wrapping the same target twice should be a no-op the second time."""
        call_count = 0

        def counting_wrapper(wrapped, instance, args, kwargs):
            nonlocal call_count
            call_count += 1
            return wrapped(*args, **kwargs)

        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
            counting_wrapper,
        )
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
            counting_wrapper,
        )

        agent = Terminus2()
        agent._summarize()

        # Should only be wrapped once
        assert call_count == 1

        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
        )

    def test_try_wrap_nonexistent_module(self):
        """Wrapping a nonexistent module should log warning, not raise."""
        _try_wrap(
            "nonexistent.module.that.does.not.exist",
            "Foo.bar",
            lambda w, i, a, k: w(*a, **k),
        )
        # No exception raised

    def test_try_wrap_nonexistent_attribute(self):
        """Wrapping a nonexistent attribute should log warning, not raise."""
        _try_wrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2.nonexistent_method_xyz",
            lambda w, i, a, k: w(*a, **k),
        )
        # No exception raised

    def test_try_unwrap_not_wrapped(self):
        """Unwrapping a target that was never wrapped is a safe no-op."""
        _try_unwrap(
            "terminal_bench.agents.terminus_2.terminus_2",
            "Terminus2._summarize",
        )
        # No exception raised

    def test_try_unwrap_nonexistent_module(self):
        """Unwrapping a nonexistent module is a safe no-op."""
        _try_unwrap(
            "nonexistent.module.that.does.not.exist",
            "Foo.bar",
        )
        # No exception raised


# ═══════════════════════════════════════════════════════════════════════════
# Constants sanity checks
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify key constants have expected values."""

    def test_framework_name(self):
        assert _FRAMEWORK == "terminal-bench"

    def test_agent_name(self):
        assert _AGENT_NAME == "terminus-2"

    def test_span_kind_values(self):
        assert _SPAN_KIND_ENTRY == "ENTRY"
        assert _SPAN_KIND_AGENT == "AGENT"
        assert _SPAN_KIND_TOOL == "TOOL"
        assert _SPAN_KIND_STEP == "STEP"
        assert _SPAN_KIND_TASK == "TASK"
        assert _SPAN_KIND_CHAIN == "CHAIN"

    def test_operation_values(self):
        assert _OP_ENTER == "enter"
        assert _OP_INVOKE_AGENT == "invoke_agent"
        assert _OP_EXECUTE_TOOL == "execute_tool"
        assert _OP_REACT == "react"
        assert _OP_RUN_TASK == "run_task"
        assert _OP_TASK == "task"
