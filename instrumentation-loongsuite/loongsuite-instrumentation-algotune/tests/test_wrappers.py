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

"""Unit tests for ``opentelemetry.instrumentation.algotune.internal.wrappers``.

Covers every wrapper class and all module-level helper functions.
Uses two complementary strategies:

* **Helper-function tests** -- import and call the helpers directly.
* **Wrapper class tests** -- instantiate each wrapper with a test tracer
  and call ``wrapper(wrapped, instance, args, kwargs)`` directly, which
  avoids the wrapt descriptor-protocol difficulties that prevent swapping
  ``__wrapped__`` at runtime.
* **Integration tests** -- use the ``instrument`` fixture to verify that
  the full wiring (stub module -> wrapt -> wrapper -> span) works for the
  happy path.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import pytest
from AlgoTuner.interfaces.commands.handlers import CommandHandlers

# Import stub classes from the injected AlgoTuner modules.
from AlgoTuner.interfaces.llm_interface import LLMInterface
from AlgoTuner.models.lite_llm_model import LiteLLMModel
from AlgoTuner.models.together_model import TogetherModel
from AlgoTuner.utils.evaluator.baseline_manager import BaselineManager
from AlgoTuner.utils.evaluator.evaluation_orchestrator import (
    EvaluationOrchestrator,
)

from opentelemetry import trace as trace_api

# Import the wrappers module itself (not just its members) so that
# mock.patch.object targets the correct module object even after
# instrumentor tests reimport the package under a new identity.
from opentelemetry.instrumentation.algotune.internal import (
    wrappers as _wrappers_module,
)
from opentelemetry.instrumentation.algotune.internal.utils import (
    INST_LITELLM_ATTEMPTS_ATTR,
    INST_ROUND_ATTR,
    INST_STEP_SPAN_ATTR,
    INST_STEP_TOKEN_ATTR,
)

# Import helpers under test.
from opentelemetry.instrumentation.algotune.internal.wrappers import (
    EvaluateSingleWrapper,
    GetBaselineTimesWrapper,
    GetResponseWrapper,
    HandleCommandWrapper,
    HandleFunctionCallWrapper,
    LiteLLMExecuteQueryWrapper,
    LiteLLMQueryWrapper,
    MainWrapper,
    RunnerEvalDatasetWrapper,
    RunTaskWrapper,
    TogetherModelQueryWrapper,
    _agent_content_attributes,
    _algotune_capture_span_content_enabled,
    _algotune_tool_definitions,
    _clear_step_state,
    _extract_command_name,
    _extract_together_usage,
    _parse_command,
    _publish_agent_content_attributes,
    _safe_get,
    _set_task_input,
    _set_task_output,
    _span_message,
    _task_json_value,
    _text_value,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Private fixture: independent tracer + exporter for direct wrapper tests.
# ---------------------------------------------------------------------------


@pytest.fixture()
def _otel():
    """Return ``(tracer, exporter)`` for tests that call wrappers directly."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace_api.get_tracer("test", tracer_provider=provider)
    return tracer, exporter


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestTextValue:
    def test_none(self):
        assert _text_value(None) == ""

    def test_string(self):
        assert _text_value("hello") == "hello"

    def test_dict(self):
        result = _text_value({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list(self):
        result = _text_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_number(self):
        result = _text_value(42)
        assert result == "42"

    def test_unjsonifiable_object(self):
        class Custom:
            def __repr__(self):
                return "<Custom>"

        result = _text_value(Custom())
        assert "Custom" in result

    def test_json_dumps_fails_falls_back_to_str(self):
        """When json.dumps raises, _text_value should fall back to str()."""
        with mock.patch(
            "opentelemetry.instrumentation.algotune.internal.wrappers.json.dumps",
            side_effect=ValueError("cannot encode"),
        ):
            result = _text_value(42)
            assert result == "42"


class TestSpanMessage:
    def test_basic_message(self):
        msg = _span_message("user", "hello")
        assert msg["role"] == "user"
        assert msg["parts"][0]["type"] == "text"
        assert msg["parts"][0]["content"] == "hello"

    def test_none_role_defaults_to_user(self):
        msg = _span_message(None, "hello")
        assert msg["role"] == "user"

    def test_none_content(self):
        msg = _span_message("assistant", None)
        assert msg["parts"][0]["content"] == ""


class TestExtractCommandName:
    def test_dict_with_command_key(self):
        assert _extract_command_name({"command": "edit"}) == "edit"

    def test_dict_with_name_key(self):
        assert _extract_command_name({"name": "run"}) == "run"

    def test_dict_with_cmd_key(self):
        assert _extract_command_name({"cmd": "test"}) == "test"

    def test_nested_data_dict(self):
        assert _extract_command_name({"data": {"command": "eval"}}) == "eval"

    def test_nested_data_name(self):
        assert _extract_command_name({"data": {"name": "exec"}}) == "exec"

    def test_non_dict(self):
        assert _extract_command_name("not a dict") == ""

    def test_none(self):
        assert _extract_command_name(None) == ""

    def test_empty_dict(self):
        assert _extract_command_name({}) == ""

    def test_non_string_value(self):
        assert _extract_command_name({"command": 42}) == ""

    def test_empty_string_value(self):
        assert _extract_command_name({"command": ""}) == ""

    def test_priority_order(self):
        result = _extract_command_name(
            {"command": "first", "name": "second", "cmd": "third"}
        )
        assert result == "first"


class TestParseCommand:
    def test_dict_command(self):
        cmd_name, args, is_error = _parse_command({"command": "eval"})
        assert cmd_name == "eval"
        assert args is None
        assert is_error is True

    def test_dict_no_command_key(self):
        cmd_name, args, is_error = _parse_command({})
        assert cmd_name == "error_response"
        assert is_error is True

    def test_parsed_command_with_args(self):
        pc = types.SimpleNamespace(command="edit", args={"file": "test.py"})
        cmd_name, args, is_error = _parse_command(pc)
        assert cmd_name == "edit"
        assert args == {"file": "test.py"}
        assert is_error is False

    def test_parsed_command_no_args(self):
        pc = types.SimpleNamespace(command="status", args="non-dict")
        cmd_name, args, is_error = _parse_command(pc)
        assert cmd_name == "status"
        assert args is None
        assert is_error is False

    def test_parsed_command_none_name(self):
        pc = types.SimpleNamespace(command=None, args={"key": "val"})
        cmd_name, args, is_error = _parse_command(pc)
        assert cmd_name == "unknown"

    def test_unknown_type(self):
        cmd_name, args, is_error = _parse_command(12345)
        assert cmd_name == "unknown"
        assert is_error is False


class TestExtractTogetherUsage:
    def test_openai_compatible_keys(self):
        inp, out = _extract_together_usage(
            {"prompt_tokens": 100, "completion_tokens": 50}
        )
        assert inp == 100
        assert out == 50

    def test_alternate_keys(self):
        inp, out = _extract_together_usage(
            {"input_tokens": 200, "output_tokens": 75}
        )
        assert inp == 200
        assert out == 75

    def test_missing_keys(self):
        inp, out = _extract_together_usage({})
        assert inp == 0
        assert out == 0

    def test_none_values(self):
        inp, out = _extract_together_usage(
            {"prompt_tokens": None, "completion_tokens": None}
        )
        assert inp == 0
        assert out == 0

    def test_invalid_values(self):
        inp, out = _extract_together_usage(
            {"prompt_tokens": "bad", "completion_tokens": "data"}
        )
        assert inp == 0
        assert out == 0

    def test_openai_keys_take_priority(self):
        inp, out = _extract_together_usage(
            {
                "prompt_tokens": 10,
                "input_tokens": 99,
                "completion_tokens": 20,
                "output_tokens": 88,
            }
        )
        assert inp == 10
        assert out == 20


class TestSafeGet:
    def test_dict_existing_key(self):
        assert _safe_get({"a": 1}, "a") == 1

    def test_dict_missing_key(self):
        assert _safe_get({"a": 1}, "b") is None

    def test_object_attribute(self):
        obj = types.SimpleNamespace(x=42)
        assert _safe_get(obj, "x") == 42

    def test_object_missing_attribute(self):
        obj = types.SimpleNamespace(x=42)
        assert _safe_get(obj, "y") is None

    def test_none_object(self):
        assert _safe_get(None, "anything") is None


class TestClearStepState:
    def test_clears_step_attributes(self):
        iface = LLMInterface()
        setattr(iface, INST_STEP_SPAN_ATTR, "some_span")
        setattr(iface, INST_STEP_TOKEN_ATTR, "some_token")

        _clear_step_state(iface)

        assert getattr(iface, INST_STEP_SPAN_ATTR) is None
        assert getattr(iface, INST_STEP_TOKEN_ATTR) is None

    def test_handles_readonly_instance(self):
        """If setattr fails on the instance, should not raise."""

        # Frozen instances would fail; simulate with a class that rejects setattr.
        class ReadOnly:
            __slots__ = ()

        obj = ReadOnly()
        # Should not raise.
        _clear_step_state(obj)


class TestTaskJsonValue:
    def test_dict_value(self):
        result = _task_json_value({"key": "val"})
        assert '"key"' in result

    def test_non_serializable_fallback(self):
        result = _task_json_value(42)
        assert "42" in result

    def test_truncation_applied(self):
        """Long values should be truncated."""
        long_val = "x" * 10000
        result = _task_json_value(long_val)
        assert len(result) <= 4096 + 10  # allow for JSON quoting

    def test_json_dumps_failure_fallback(self):
        """When json.dumps raises even with default=str, falls back to str()."""
        with mock.patch(
            "opentelemetry.instrumentation.algotune.internal.wrappers.json.dumps",
            side_effect=TypeError("cannot serialize"),
        ):
            result = _task_json_value({"key": "val"})
            assert "key" in result  # str() fallback still captures content


class TestSetTaskInputOutput:
    def test_set_task_input(self, _otel):
        tracer, exporter = _otel
        with tracer.start_as_current_span("test") as span:
            _set_task_input(span, {"task": "eval"})

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("input.mime_type") == "application/json"
        assert "eval" in spans[0].attributes.get("input.value", "")

    def test_set_task_output(self, _otel):
        tracer, exporter = _otel
        with tracer.start_as_current_span("test") as span:
            _set_task_output(span, {"result": "ok"})

        spans = exporter.get_finished_spans()
        assert (
            spans[0].attributes.get("output.mime_type") == "application/json"
        )
        assert "ok" in spans[0].attributes.get("output.value", "")


class TestAlgotuneCaptureSpanContentEnabled:
    def test_not_set_returns_false(self, monkeypatch):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
        assert _algotune_capture_span_content_enabled() is False

    def test_true(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "TRUE"
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_span_only(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_span_and_event(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "SPAN_AND_EVENT",
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_event_only_not_included(self, monkeypatch):
        """EVENT_ONLY is not in the span-content-enabled set."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "EVENT_ONLY"
        )
        assert _algotune_capture_span_content_enabled() is False

    def test_false_value(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
        )
        assert _algotune_capture_span_content_enabled() is False


class TestAlgotuneToolDefinitions:
    def test_returns_empty_without_types_module(self):
        """When AlgoTuner.interfaces.commands.types is not importable, return []."""
        result = _algotune_tool_definitions()
        assert result == []

    def test_returns_definitions_with_types_module(self):
        """When COMMAND_FORMATS is available, definitions should be extracted."""
        import types as t

        # Create a fake types module with COMMAND_FORMATS.
        types_mod = t.ModuleType("AlgoTuner.interfaces.commands.types")

        fmt = t.SimpleNamespace(
            description="Edit a file", example="edit file.py"
        )
        types_mod.COMMAND_FORMATS = {"edit": fmt}

        sys.modules["AlgoTuner.interfaces.commands.types"] = types_mod
        try:
            result = _algotune_tool_definitions()
            assert len(result) == 1
            assert result[0]["name"] == "edit"
            assert result[0]["type"] == "function"
            assert "Edit a file" in result[0]["description"]
        finally:
            del sys.modules["AlgoTuner.interfaces.commands.types"]


class TestAgentContentAttributes:
    def test_returns_empty_when_capture_disabled(self, monkeypatch):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
        iface = LLMInterface()
        result = _agent_content_attributes(iface)
        assert result == {}

    def test_returns_attributes_when_capture_enabled(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        iface = LLMInterface()
        iface.state.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "You are helpful"},
        ]

        result = _agent_content_attributes(iface)

        assert result["algo.debug.input_messages.count"] == 1
        assert result["algo.debug.output_messages.count"] == 1
        assert result["algo.debug.system_instructions.count"] == 1
        assert "gen_ai.output.messages" in result
        assert "gen_ai.input.messages" in result
        assert "gen_ai.system_instructions" in result

    def test_system_instruction_fallback_from_first_user_msg(
        self, monkeypatch
    ):
        """When no system role messages exist, the first message's content
        is used as a system instruction."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        iface = LLMInterface()
        iface.state.messages = [
            {"role": "user", "content": "Initial instructions here"},
        ]

        result = _agent_content_attributes(iface)
        assert result["algo.debug.system_instructions.count"] == 1
        assert "gen_ai.system_instructions" in result

    def test_empty_messages(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        iface = LLMInterface()
        iface.state.messages = []

        result = _agent_content_attributes(iface)
        assert result["algo.debug.input_messages.count"] == 0
        assert result["algo.debug.output_messages.count"] == 0

    def test_output_value_attribute(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        iface = LLMInterface()
        iface.state.messages = [
            {"role": "assistant", "content": "final answer"},
        ]

        result = _agent_content_attributes(iface)
        assert result.get("output.value") == "final answer"

    def test_non_dict_messages_skipped(self, monkeypatch):
        """Non-dict messages in the list should be skipped."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        iface = LLMInterface()
        iface.state.messages = [
            "not a dict",
            42,
            {"role": "user", "content": "valid"},
        ]

        result = _agent_content_attributes(iface)
        assert result["algo.debug.input_messages.count"] == 1

    def test_none_state_messages(self, monkeypatch):
        """When state.messages is None, should handle gracefully."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        iface = LLMInterface()
        iface.state.messages = None

        result = _agent_content_attributes(iface)
        assert result["algo.debug.input_messages.count"] == 0

    def test_tool_definitions_with_types_module(self, monkeypatch):
        """When tool definitions exist, they should be included."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )

        import types as t

        types_mod = t.ModuleType("AlgoTuner.interfaces.commands.types")
        fmt = t.SimpleNamespace(description="Edit cmd", example="edit f.py")
        types_mod.COMMAND_FORMATS = {"edit": fmt}
        sys.modules["AlgoTuner.interfaces.commands.types"] = types_mod

        try:
            iface = LLMInterface()
            iface.state.messages = [{"role": "user", "content": "hi"}]
            result = _agent_content_attributes(iface)
            assert result["algo.debug.tool_definitions.count"] == 1
            assert "gen_ai.tool.definitions" in result
        finally:
            del sys.modules["AlgoTuner.interfaces.commands.types"]


class TestPublishAgentContentAttributes:
    def test_sets_attributes_on_span(self, monkeypatch, _otel):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        tracer, exporter = _otel

        iface = LLMInterface()
        iface.state.messages = [
            {"role": "user", "content": "Hello"},
        ]

        with tracer.start_as_current_span("test") as span:
            _publish_agent_content_attributes(iface, span)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("algo.debug.input_messages.count") == 1

    def test_skips_none_span(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )

        iface = LLMInterface()
        iface.state.messages = [{"role": "user", "content": "Hello"}]

        # Should not raise.
        _publish_agent_content_attributes(iface, None)

    def test_no_op_when_capture_disabled(self, monkeypatch, _otel):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
        tracer, exporter = _otel

        iface = LLMInterface()
        with tracer.start_as_current_span("test") as span:
            _publish_agent_content_attributes(iface, span)

        spans = exporter.get_finished_spans()
        assert (
            spans[0].attributes.get("algo.debug.input_messages.count") is None
        )


# ---------------------------------------------------------------------------
# Direct wrapper tests -- call wrapper(wrapped, instance, args, kwargs)
# so we can control the ``wrapped`` callable for error paths.
# ---------------------------------------------------------------------------


class TestMainWrapperDirect:
    def test_entry_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        result = wrapper(lambda *a, **k: "ok", None, (), {})
        assert result == "ok"

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        s = entry[0]
        assert s.name == "enter_ai_application_system"
        assert s.attributes["gen_ai.operation.name"] == "enter"
        assert s.attributes["gen_ai.framework"] == "AlgoTune"
        assert "gen_ai.session.id" in s.attributes

    def test_entry_span_captures_argv(self, _otel):
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        original_argv = sys.argv
        sys.argv = ["algotune", "--model", "gpt-4o", "--task", "tsp"]
        try:
            wrapper(lambda *a, **k: None, None, (), {})
        finally:
            sys.argv = original_argv

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        s = entry[0]
        assert s.attributes.get("gen_ai.request.model") == "gpt-4o"
        assert s.attributes.get("algo.task.name") == "tsp"

    def test_entry_span_on_system_exit_nonzero(self, _otel):
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        def raise_exit(*a, **k):
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            wrapper(raise_exit, None, (), {})

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        s = entry[0]
        assert s.attributes.get("algotune.exit_code") == 1
        assert s.status.status_code.name == "ERROR"

    def test_entry_span_on_system_exit_zero(self, _otel):
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        def raise_exit(*a, **k):
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            wrapper(raise_exit, None, (), {})

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        # exit code 0 should NOT set error
        assert entry[0].status.status_code.name != "ERROR"

    def test_entry_span_on_generic_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        def raise_err(*a, **k):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            wrapper(raise_err, None, (), {})

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        assert entry[0].status.status_code.name == "ERROR"

    def test_entry_span_on_memory_error(self, _otel):
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        def raise_oom(*a, **k):
            raise MemoryError()

        with pytest.raises(MemoryError):
            wrapper(raise_oom, None, (), {})

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        assert entry[0].attributes.get("error.type") == "MemoryError"
        assert entry[0].status.status_code.name == "ERROR"


class TestRunTaskWrapperDirect:
    def _make_instance(self, model_name="openai/gpt-4o"):
        return LLMInterface(model_name=model_name)

    def test_agent_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        result = wrapper(lambda *a, **k: "done", iface, (), {})
        assert result == "done"

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        s = agent[0]
        assert s.name == "invoke_agent AlgoTuner"
        assert s.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert s.attributes["gen_ai.framework"] == "AlgoTune"
        assert s.attributes["gen_ai.agent.name"] == "AlgoTuner"
        assert s.attributes["gen_ai.request.model"] == "openai/gpt-4o"
        assert s.attributes["gen_ai.provider.name"] == "openai"

    def test_agent_span_final_status_completed(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert (
            agent[0].attributes.get("algo.agent.final_status") == "completed"
        )

    def test_agent_span_on_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        def raise_err(*a, **k):
            raise RuntimeError("agent error")

        with pytest.raises(RuntimeError, match="agent error"):
            wrapper(raise_err, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        assert agent[0].status.status_code.name == "ERROR"
        assert (
            agent[0].attributes.get("algo.agent.final_status") == "exception"
        )

    def test_agent_span_on_keyboard_interrupt(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        def raise_ki(*a, **k):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            wrapper(raise_ki, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        assert (
            agent[0].attributes.get("algo.agent.final_status")
            == "KeyboardInterrupt"
        )

    def test_agent_span_on_system_exit(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        def raise_se(*a, **k):
            raise SystemExit(2)

        with pytest.raises(SystemExit):
            wrapper(raise_se, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        assert (
            agent[0].attributes.get("algo.agent.final_status") == "SystemExit"
        )
        assert agent[0].status.status_code.name == "ERROR"

    def test_agent_span_records_spend(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()
        iface.state.spend = 1.23

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert agent[0].attributes.get("algo.agent.spend_usd") == 1.23

    def test_agent_span_records_total_rounds(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        # RunTaskWrapper resets INST_ROUND_ATTR to 0 at the start.
        # Simulate rounds occurring during the wrapped function.
        def simulate_rounds(*a, **k):
            setattr(iface, INST_ROUND_ATTR, 5)

        wrapper(simulate_rounds, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert agent[0].attributes.get("algo.agent.total_rounds") == 5

    def test_agent_terminated_by_limit(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()
        iface.check_limits = lambda: True

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert (
            agent[0].attributes.get("algo.agent.final_status")
            == "terminated_by_limit"
        )

    def test_agent_final_eval_success(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()
        iface._final_eval_success = True
        iface._final_eval_metrics = {"mean_speedup": 3.5}

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert agent[0].attributes.get("algo.agent.final_eval_success") is True
        assert agent[0].attributes.get("algo.agent.final_mean_speedup") == 3.5

    def test_agent_no_model_name(self, _otel):
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance(model_name="")

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        # Model-related attributes should not be set if model_name is empty.
        assert agent[0].attributes.get("gen_ai.request.model") is None

    def test_dangling_step_span_closed(self, _otel):
        """A STEP span left open during wrapped fn should be closed in finally."""
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = self._make_instance()

        # RunTaskWrapper resets step attrs at the start, so we need to
        # create a dangling span inside the wrapped function body.
        step_span = tracer.start_span("dangling step")

        def leave_dangling_step(*a, **k):
            setattr(iface, INST_STEP_SPAN_ATTR, step_span)
            setattr(iface, INST_STEP_TOKEN_ATTR, None)

        wrapper(leave_dangling_step, iface, (), {})

        # The dangling step should have been ended by the finally block.
        assert not step_span.is_recording()


class TestGetResponseWrapperDirect:
    def _make_instance(self):
        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)
        setattr(iface, INST_LITELLM_ATTEMPTS_ATTR, 0)
        return iface

    def test_step_span_opened(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()

        result = wrapper(lambda *a, **k: {"content": "hello"}, iface, (), {})
        assert result == {"content": "hello"}

        # STEP span is still open (handle_function_call should close it).
        step_span = getattr(iface, INST_STEP_SPAN_ATTR)
        assert step_span is not None
        assert step_span.is_recording()
        assert step_span.name == "react step"

        attrs = dict(step_span.attributes)
        assert attrs["gen_ai.span.kind"] == "STEP"
        assert attrs["gen_ai.operation.name"] == "react"
        assert attrs["gen_ai.framework"] == "AlgoTune"
        assert attrs["gen_ai.react.round"] == 1

        # Cleanup.
        step_span.end()

    def test_step_span_closed_on_none_response(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()

        result = wrapper(lambda *a, **k: None, iface, (), {})
        assert result is None

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 1
        assert step[0].attributes.get("algo.step.response_empty") is True
        assert (
            step[0].attributes.get("gen_ai.react.finish_reason")
            == "empty_response_retry"
        )

        # Instance STEP state should be cleared.
        assert getattr(iface, INST_STEP_SPAN_ATTR) is None

    def test_step_span_closed_on_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()

        def raise_err(*a, **k):
            raise RuntimeError("llm timeout")

        with pytest.raises(RuntimeError, match="llm timeout"):
            wrapper(raise_err, iface, (), {})

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 1
        assert step[0].status.status_code.name == "ERROR"
        assert (
            step[0].attributes.get("gen_ai.react.finish_reason")
            == "RuntimeError"
        )

    def test_round_counter_increments(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()

        # Each call with None return closes the STEP span.
        wrapper(lambda *a, **k: None, iface, (), {})
        wrapper(lambda *a, **k: None, iface, (), {})
        wrapper(lambda *a, **k: None, iface, (), {})

        assert getattr(iface, INST_ROUND_ATTR) == 3

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 3
        rounds = [s.attributes.get("gen_ai.react.round") for s in step]
        assert rounds == [1, 2, 3]

    def test_previous_step_closed_before_new_one(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()

        # First call returns non-None (STEP stays open).
        wrapper(lambda *a, **k: {"text": "hi"}, iface, (), {})
        step1 = getattr(iface, INST_STEP_SPAN_ATTR)
        assert step1 is not None
        assert step1.is_recording()

        # Second call should close step1 before opening step2.
        wrapper(lambda *a, **k: None, iface, (), {})

        assert not step1.is_recording()

    def test_litellm_attempts_reset(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()
        setattr(iface, INST_LITELLM_ATTEMPTS_ATTR, 5)

        wrapper(lambda *a, **k: None, iface, (), {})

        # Attempts should have been reset to 0 at the start.
        assert getattr(iface, INST_LITELLM_ATTEMPTS_ATTR) == 0

    def test_attempt_count_published_on_error(self, _otel):
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = self._make_instance()
        setattr(iface, INST_LITELLM_ATTEMPTS_ATTR, 3)

        # Call that returns None publishes attempts.
        wrapper(lambda *a, **k: None, iface, (), {})

        # After reset to 0, attempts = 0 so not published (no-op).
        # But if we set it AFTER the wrapper resets it (inside the wrapped fn):
        iface2 = self._make_instance()

        def set_and_return_none(*a, **k):
            setattr(iface2, INST_LITELLM_ATTEMPTS_ATTR, 7)
            return None

        wrapper(set_and_return_none, iface2, (), {})

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        # Find the step for iface2 (last one).
        last = step[-1]
        assert last.attributes.get("algo.llm.retry_count") == 7


class TestHandleFunctionCallWrapperDirect:
    def test_closes_step_span(self, _otel):
        tracer, exporter = _otel
        wrapper_get = GetResponseWrapper(tracer)
        wrapper_hfc = HandleFunctionCallWrapper()

        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)

        # Open a STEP span.
        wrapper_get(lambda *a, **k: {"text": "hi"}, iface, (), {})
        step_span = getattr(iface, INST_STEP_SPAN_ATTR)
        assert step_span is not None
        assert step_span.is_recording()

        # Close it via handle_function_call.
        result = wrapper_hfc(
            lambda *a, **k: {"command": "eval", "success": True},
            iface,
            (),
            {},
        )
        assert result == {"command": "eval", "success": True}
        assert not step_span.is_recording()
        assert getattr(iface, INST_STEP_SPAN_ATTR) is None

    def test_records_command_name_and_finish_reason(self, _otel):
        tracer, exporter = _otel
        wrapper_get = GetResponseWrapper(tracer)
        wrapper_hfc = HandleFunctionCallWrapper()

        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)

        wrapper_get(lambda *a, **k: {"text": "hi"}, iface, (), {})
        wrapper_hfc(
            lambda *a, **k: {"command": "edit", "success": True},
            iface,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 1
        assert step[0].attributes.get("algo.step.command_name") == "edit"
        assert (
            step[0].attributes.get("gen_ai.react.finish_reason")
            == "tool_executed"
        )

    def test_closes_step_on_exception(self, _otel):
        tracer, exporter = _otel
        wrapper_get = GetResponseWrapper(tracer)
        wrapper_hfc = HandleFunctionCallWrapper()

        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)

        wrapper_get(lambda *a, **k: {"text": "hi"}, iface, (), {})

        def raise_err(*a, **k):
            raise RuntimeError("tool crash")

        with pytest.raises(RuntimeError, match="tool crash"):
            wrapper_hfc(raise_err, iface, (), {})

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 1
        assert step[0].status.status_code.name == "ERROR"
        assert (
            step[0].attributes.get("gen_ai.react.finish_reason")
            == "RuntimeError"
        )

    def test_no_step_span_is_noop(self, _otel):
        wrapper_hfc = HandleFunctionCallWrapper()
        iface = LLMInterface()
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)

        result = wrapper_hfc(
            lambda *a, **k: {"command": "edit", "success": True},
            iface,
            (),
            {},
        )
        assert result == {"command": "edit", "success": True}

    def test_publishes_litellm_retry_count(self, _otel):
        tracer, exporter = _otel
        wrapper_get = GetResponseWrapper(tracer)
        wrapper_hfc = HandleFunctionCallWrapper()

        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)

        wrapper_get(lambda *a, **k: {"text": "hi"}, iface, (), {})

        # Simulate LiteLLM retries during the step.
        setattr(iface, INST_LITELLM_ATTEMPTS_ATTR, 4)

        wrapper_hfc(
            lambda *a, **k: {"command": "run", "success": True},
            iface,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert step[0].attributes.get("algo.llm.retry_count") == 4


class TestHandleCommandWrapperDirect:
    def test_tool_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)

        pc = types.SimpleNamespace(command="edit", args={"file": "sol.py"})
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"success": True, "message": "ok"},
            handler,
            (pc,),
            {},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert len(tool) == 1
        s = tool[0]
        assert s.name == "execute_tool edit"
        assert s.attributes["gen_ai.operation.name"] == "execute_tool"
        assert s.attributes["gen_ai.framework"] == "AlgoTune"
        assert s.attributes["gen_ai.tool.name"] == "edit"
        assert s.attributes["gen_ai.tool.type"] == "function"

    def test_tool_span_error_response_dict(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)

        error_cmd = {"command": "bad_cmd", "error": "parse failed"}
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"success": True, "message": "ok"},
            handler,
            (error_cmd,),
            {},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert (
            tool[0].attributes.get("algotune.command.error_response") is True
        )

    def test_tool_span_records_failure(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)

        pc = types.SimpleNamespace(command="bad", args=None)
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"success": False, "message": "command failed"},
            handler,
            (pc,),
            {},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert tool[0].attributes.get("algo.command.success") is False
        assert tool[0].status.status_code.name == "ERROR"

    def test_tool_span_on_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)

        pc = types.SimpleNamespace(command="boom", args=None)
        handler = CommandHandlers()

        def raise_err(*a, **k):
            raise RuntimeError("cmd exploded")

        with pytest.raises(RuntimeError, match="cmd exploded"):
            wrapper(raise_err, handler, (pc,), {})

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert tool[0].status.status_code.name == "ERROR"

    def test_tool_span_unknown_command(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)

        pc = types.SimpleNamespace(command=None, args=None)
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"success": True},
            handler,
            (pc,),
            {},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert tool[0].name == "execute_tool unknown"

    def test_tool_span_snapshot_detected(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)

        pc = types.SimpleNamespace(command="edit", args=None)
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {
                "success": True,
                "data": {"snapshot_saved": True},
            },
            handler,
            (pc,),
            {},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert tool[0].attributes.get("algo.snapshot.saved") is True

    def test_tool_span_with_content_capture(self, _otel, monkeypatch):
        """When content capture is enabled, tool call arguments and results are captured."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )

        # Re-import the module to pick up the changed config.
        for m in list(sys.modules):
            if m.startswith("opentelemetry.instrumentation.algotune"):
                del sys.modules[m]

        from opentelemetry.instrumentation.algotune.internal.wrappers import (
            HandleCommandWrapper as HCW,
        )

        tracer, exporter = _otel
        wrapper = HCW(tracer)

        pc = types.SimpleNamespace(
            command="edit", args={"file": "sol.py", "content": "print('hi')"}
        )
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"success": True, "message": "file edited"},
            handler,
            (pc,),
            {},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert len(tool) == 1
        # Arguments should be captured.
        assert tool[0].attributes.get("gen_ai.tool.call.arguments") is not None
        # Result message should be captured.
        assert tool[0].attributes.get("gen_ai.tool.call.result") is not None

        # Restore module state.
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
        for m in list(sys.modules):
            if m.startswith("opentelemetry.instrumentation.algotune"):
                del sys.modules[m]


class TestRunnerEvalDatasetWrapperDirect:
    def test_task_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        result_obj = types.SimpleNamespace(
            success=True,
            status="ok",
            message="done",
            data={
                "num_evaluated": 5,
                "mean_speedup": 2.0,
                "num_valid": 4,
                "num_invalid": 0,
                "num_timeout": 1,
            },
        )

        wrapper(
            lambda *a, **k: result_obj,
            handler,
            ("train", "agent"),
            {},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert len(task) == 1
        s = task[0]
        assert s.attributes["gen_ai.operation.name"] == "run_task"
        assert s.attributes["gen_ai.framework"] == "AlgoTune"
        assert s.attributes.get("algo.eval.subset") == "train"
        assert s.attributes.get("algo.eval.command_source") == "agent"

    def test_task_span_records_eval_metrics(self, _otel):
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        result_obj = types.SimpleNamespace(
            success=True,
            status="ok",
            message="done",
            data={
                "num_evaluated": 10,
                "mean_speedup": 1.5,
                "num_valid": 8,
                "num_invalid": 1,
                "num_timeout": 1,
            },
        )

        wrapper(lambda *a, **k: result_obj, handler, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        s = task[0]
        assert s.attributes.get("algo.eval.total_problems") == 10
        assert s.attributes.get("algo.eval.mean_speedup") == 1.5
        assert s.attributes.get("algo.eval.num_valid") == 8
        assert s.attributes.get("algo.eval.num_invalid") == 1
        assert s.attributes.get("algo.eval.num_timeout") == 1

    def test_task_span_on_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        def raise_err(*a, **k):
            raise RuntimeError("eval crash")

        with pytest.raises(RuntimeError, match="eval crash"):
            wrapper(raise_err, handler, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert task[0].status.status_code.name == "ERROR"

    def test_task_span_with_kwargs(self, _otel):
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        result_obj = types.SimpleNamespace(
            success=True,
            status="ok",
            message="done",
            data={},
        )

        wrapper(
            lambda *a, **k: result_obj,
            handler,
            (),
            {"data_subset": "test", "command_source": "manual"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert task[0].attributes.get("algo.eval.subset") == "test"
        assert task[0].attributes.get("algo.eval.command_source") == "manual"

    def test_task_span_test_mode(self, _otel):
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()
        handler.interface = types.SimpleNamespace(max_samples=5)

        result_obj = types.SimpleNamespace(
            success=True,
            status="ok",
            message="done",
            data={},
        )

        wrapper(lambda *a, **k: result_obj, handler, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert task[0].attributes.get("algo.eval.test_mode") is True


class TestEvaluateSingleWrapperDirect:
    def test_task_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result_obj = types.SimpleNamespace(
            speedup=2.0,
            solver_time_ms=150.0,
            is_valid=True,
            execution=types.SimpleNamespace(
                timeout_occurred=False, error_type=None
            ),
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {
                "problem_id": "tsp_001",
                "problem_index": 3,
                "baseline_time_ms": 500.0,
            },
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert len(task) == 1
        s = task[0]
        assert s.attributes["gen_ai.operation.name"] == "run_task"
        assert s.attributes.get("algo.problem.id") == "tsp_001"
        assert s.attributes.get("algo.problem.index") == 3
        assert s.attributes.get("algo.problem.baseline_time_ms") == 500.0

    def test_task_span_records_problem_result(self, _otel):
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result_obj = types.SimpleNamespace(
            speedup=2.0,
            solver_time_ms=150.0,
            is_valid=True,
            execution=types.SimpleNamespace(
                timeout_occurred=False, error_type=None
            ),
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {"problem_id": "sort_002", "problem_index": 1},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        s = task[0]
        assert s.attributes.get("algo.problem.speedup") == 2.0
        assert s.attributes.get("algo.problem.solver_time_ms") == 150.0
        assert s.attributes.get("algo.problem.is_valid") is True

    def test_task_span_on_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        def raise_err(*a, **k):
            raise RuntimeError("eval failed")

        with pytest.raises(RuntimeError, match="eval failed"):
            wrapper(raise_err, orch, (), {"problem_id": "fail"})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert task[0].status.status_code.name == "ERROR"

    def test_task_span_with_error_type(self, _otel):
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result_obj = types.SimpleNamespace(
            speedup=0.0,
            solver_time_ms=0.0,
            is_valid=False,
            execution=types.SimpleNamespace(
                timeout_occurred=True, error_type="TIMEOUT"
            ),
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {"problem_id": "timeout_prob"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        s = task[0]
        assert s.attributes.get("algo.problem.is_valid") is False
        assert s.attributes.get("algo.problem.timeout_occurred") is True
        assert s.attributes.get("algo.problem.error_type") == "TIMEOUT"

    def test_task_span_with_enum_error_type(self, _otel):
        """error_type with .value attribute (enum) should use .value."""
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        class ErrorType:
            value = "RUNTIME_ERROR"

        result_obj = types.SimpleNamespace(
            speedup=None,
            solver_time_ms=None,
            is_valid=False,
            execution=types.SimpleNamespace(
                timeout_occurred=False, error_type=ErrorType()
            ),
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {"problem_id": "enum_err"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert (
            task[0].attributes.get("algo.problem.error_type")
            == "RUNTIME_ERROR"
        )

    def test_task_span_dict_result(self, _otel):
        """Result as a dict (not dataclass) should also work via _safe_get."""
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result = {
            "speedup": 1.5,
            "solver_time_ms": 200.0,
            "is_valid": True,
            "execution": {"timeout_occurred": False, "error_type": None},
        }

        wrapper(
            lambda *a, **k: result,
            orch,
            (),
            {"problem_id": "dict_result"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert task[0].attributes.get("algo.problem.speedup") == 1.5


class TestGetBaselineTimesWrapperDirect:
    def test_task_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()

        result = wrapper(
            lambda *a, **k: {"p1": 100.0, "p2": 200.0},
            mgr,
            ("train",),
            {},
        )
        assert result == {"p1": 100.0, "p2": 200.0}

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert len(task) == 1
        s = task[0]
        assert s.attributes["gen_ai.operation.name"] == "run_task"
        assert s.attributes.get("algo.baseline.subset") == "train"
        assert s.attributes.get("algo.baseline.cache_hit") is False
        assert s.attributes.get("algo.baseline.actual_count") == 2

    def test_task_span_cache_hit(self, _otel):
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()
        mgr._cache = {"train": {1: 100}}

        wrapper(
            lambda *a, **k: {"p1": 100.0},
            mgr,
            ("train",),
            {},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert task[0].attributes.get("algo.baseline.cache_hit") is True

    def test_task_span_on_system_exit(self, _otel):
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()

        def raise_exit(*a, **k):
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            wrapper(raise_exit, mgr, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert len(task) == 1
        s = task[0]
        assert s.status.status_code.name == "ERROR"
        fatal = [e for e in s.events if e.name == "baseline.fatal_failure"]
        assert len(fatal) == 1
        assert fatal[0].attributes["exit_code"] == 1

    def test_task_span_on_generic_exception(self, _otel):
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()

        def raise_err(*a, **k):
            raise ValueError("bad baseline")

        with pytest.raises(ValueError, match="bad baseline"):
            wrapper(raise_err, mgr, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert task[0].status.status_code.name == "ERROR"

    def test_task_span_kwarg_subset(self, _otel):
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()

        wrapper(
            lambda *a, **k: {},
            mgr,
            (),
            {"subset": "test_set"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert task[0].attributes.get("algo.baseline.subset") == "test_set"


class TestLiteLLMQueryWrapperDirect:
    def test_no_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = LiteLLMQueryWrapper()
        model = LiteLLMModel()

        result = wrapper(lambda *a, **k: "llm_response", model, (), {})
        assert result == "llm_response"

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 0

    def test_attempt_counter_reset(self, _otel):
        wrapper = LiteLLMQueryWrapper()
        model = LiteLLMModel()
        setattr(model, "_otel_algo_litellm_call_attempts", 5)

        wrapper(lambda *a, **k: "ok", model, (), {})

        assert getattr(model, "_otel_algo_litellm_call_attempts") == 0

    def test_last_call_attempts_published(self, _otel):
        """When wrapped fn sets call_attempts, last_call_attempts is published
        on the active span."""
        tracer, exporter = _otel
        wrapper = LiteLLMQueryWrapper()
        model = LiteLLMModel()

        def simulate_retries(*a, **k):
            setattr(model, "_otel_algo_litellm_call_attempts", 3)
            return "ok"

        # We need an active recording span for publish to work.
        with tracer.start_as_current_span("step"):
            wrapper(simulate_retries, model, (), {})

        spans = exporter.get_finished_spans()
        step = [s for s in spans if s.name == "step"]
        assert step[0].attributes.get("algo.llm.last_call_attempts") == 3


class TestLiteLLMExecuteQueryWrapperDirect:
    def test_attempt_counter_incremented(self, _otel):
        wrapper = LiteLLMExecuteQueryWrapper()
        model = LiteLLMModel()
        setattr(model, "_otel_algo_litellm_call_attempts", 0)

        wrapper(lambda *a, **k: "ok", model, (), {})

        assert getattr(model, "_otel_algo_litellm_call_attempts") == 1

    def test_multiple_attempts(self, _otel):
        wrapper = LiteLLMExecuteQueryWrapper()
        model = LiteLLMModel()
        setattr(model, "_otel_algo_litellm_call_attempts", 0)

        wrapper(lambda *a, **k: "ok", model, (), {})
        wrapper(lambda *a, **k: "ok", model, (), {})
        wrapper(lambda *a, **k: "ok", model, (), {})

        assert getattr(model, "_otel_algo_litellm_call_attempts") == 3

    def test_retry_count_on_active_span(self, _otel):
        tracer, exporter = _otel
        wrapper = LiteLLMExecuteQueryWrapper()
        model = LiteLLMModel()
        setattr(model, "_otel_algo_litellm_call_attempts", 0)

        with tracer.start_as_current_span("step"):
            wrapper(lambda *a, **k: "ok", model, (), {})
            wrapper(lambda *a, **k: "ok", model, (), {})

        spans = exporter.get_finished_spans()
        step = [s for s in spans if s.name == "step"]
        assert step[0].attributes.get("algo.llm.retry_count") == 2


class TestTogetherModelQueryWrapperDirect:
    def test_llm_span_created(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/llama-70b")

        result = wrapper(
            lambda *a, **k: {
                "message": "ok",
                "cost": 0.01,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
            model,
            (),
            {},
        )

        assert result["cost"] == 0.01

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 1
        s = llm[0]
        assert s.name == "chat together/llama-70b"
        assert s.attributes["gen_ai.operation.name"] == "chat"
        assert s.attributes["gen_ai.framework"] == "AlgoTune"
        assert s.attributes["gen_ai.request.model"] == "together/llama-70b"
        assert s.attributes["gen_ai.provider.name"] == "together_ai"

    def test_together_usage_tokens(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/model-y")

        wrapper(
            lambda *a, **k: {
                "cost": 0.02,
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            },
            model,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        s = llm[0]
        assert s.attributes.get("gen_ai.usage.input_tokens") == 200
        assert s.attributes.get("gen_ai.usage.output_tokens") == 100
        assert s.attributes.get("gen_ai.usage.total_tokens") == 300
        assert s.attributes.get("algo.llm.response_cost_usd") == 0.02

    def test_together_request_params(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/test")
        model.default_params = {
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 2048,
        }

        wrapper(lambda *a, **k: {}, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        s = llm[0]
        assert s.attributes.get("gen_ai.request.temperature") == 0.5
        assert s.attributes.get("gen_ai.request.top_p") == 0.8
        assert s.attributes.get("gen_ai.request.max_tokens") == 2048

    def test_together_error_handling(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/fail")

        def raise_err(*a, **k):
            raise ConnectionError("together API down")

        with pytest.raises(ConnectionError, match="together API down"):
            wrapper(raise_err, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert llm[0].status.status_code.name == "ERROR"

    def test_together_no_usage(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/no-usage")

        wrapper(lambda *a, **k: {"message": "ok"}, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        s = llm[0]
        assert s.attributes.get("gen_ai.usage.input_tokens") is None
        assert s.attributes.get("gen_ai.usage.output_tokens") is None

    def test_together_no_default_params(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/bare")
        model.default_params = {}

        wrapper(lambda *a, **k: {}, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        s = llm[0]
        assert s.attributes.get("gen_ai.request.temperature") is None
        assert s.attributes.get("gen_ai.request.top_p") is None

    def test_together_non_dict_result(self, _otel):
        """When query() returns a non-dict, should not crash."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/str-result")

        result = wrapper(lambda *a, **k: "plain string", model, (), {})
        assert result == "plain string"

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 1

    def test_together_with_content_capture(self, _otel, monkeypatch):
        """When content capture is enabled, output messages are captured."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )

        for m in list(sys.modules):
            if m.startswith("opentelemetry.instrumentation.algotune"):
                del sys.modules[m]

        from opentelemetry.instrumentation.algotune.internal.wrappers import (
            TogetherModelQueryWrapper as TMQW,
        )

        tracer, exporter = _otel
        wrapper = TMQW(tracer)
        model = TogetherModel(model_name="together/capture-test")

        wrapper(
            lambda *a, **k: {"message": "hello world", "usage": {}},
            model,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 1
        assert llm[0].attributes.get("gen_ai.output.messages") is not None

        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
        for m in list(sys.modules):
            if m.startswith("opentelemetry.instrumentation.algotune"):
                del sys.modules[m]

    def test_together_no_model_name(self, _otel):
        """When model_name is empty/None, should use 'unknown'."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="")
        model.model_name = ""

        wrapper(lambda *a, **k: {}, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert llm[0].name == "chat unknown"

    def test_together_none_default_params(self, _otel):
        """When default_params is None, should not crash."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/none-params")
        model.default_params = None

        wrapper(lambda *a, **k: {}, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 1

    def test_together_cost_none(self, _otel):
        """When cost is None in result, attribute should not be set."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/no-cost")

        wrapper(
            lambda *a, **k: {"cost": None, "usage": {}},
            model,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert llm[0].attributes.get("algo.llm.response_cost_usd") is None

    def test_together_total_tokens_fallback(self, _otel):
        """When total_tokens is not in usage, should compute from input+output."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/computed-total")

        wrapper(
            lambda *a, **k: {
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            model,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert llm[0].attributes.get("gen_ai.usage.total_tokens") == 15


# ---------------------------------------------------------------------------
# Integration tests (using instrument fixture for full wiring)
# ---------------------------------------------------------------------------


class TestIntegrationMainWrapper:
    def test_entry_span_created(self, span_exporter, instrument):
        from AlgoTuner.main import main

        main()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        s = entry[0]
        assert s.name == "enter_ai_application_system"
        assert s.attributes["gen_ai.framework"] == "AlgoTune"


class TestIntegrationRunTask:
    def test_agent_span_created(self, span_exporter, instrument):
        iface = LLMInterface(model_name="openai/gpt-4o")
        iface.run_task()

        spans = span_exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        s = agent[0]
        assert s.attributes["gen_ai.request.model"] == "openai/gpt-4o"
        assert s.attributes["gen_ai.provider.name"] == "openai"


class TestIntegrationGetResponseAndHandleFunctionCall:
    def test_step_open_and_close(self, span_exporter, instrument):
        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)

        # Open STEP.
        iface.get_response()
        step = getattr(iface, INST_STEP_SPAN_ATTR)
        assert step is not None
        assert step.is_recording()

        # Close STEP.
        iface.handle_function_call()
        assert not step.is_recording()
        assert getattr(iface, INST_STEP_SPAN_ATTR) is None

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step_spans) == 1
        assert (
            step_spans[0].attributes.get("gen_ai.react.finish_reason")
            == "tool_executed"
        )


class TestIntegrationHandleCommand:
    def test_tool_span_created(self, span_exporter, instrument):
        handler = CommandHandlers()
        pc = types.SimpleNamespace(command="edit", args={"file": "sol.py"})
        handler.handle_command(pc)

        spans = span_exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert len(tool) == 1
        assert tool[0].name == "execute_tool edit"


class TestIntegrationEvalDataset:
    def test_task_span_created(self, span_exporter, instrument):
        handler = CommandHandlers()
        handler._runner_eval_dataset("train", "agent")

        spans = span_exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert len(task) == 1
        assert task[0].attributes.get("algo.eval.subset") == "train"


class TestIntegrationEvaluateSingle:
    def test_task_span_created(self, span_exporter, instrument):
        orch = EvaluationOrchestrator()
        orch.evaluate_single(problem_id="tsp_001", problem_index=3)

        spans = span_exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert len(task) == 1
        assert task[0].attributes.get("algo.problem.id") == "tsp_001"


class TestIntegrationBaseline:
    def test_task_span_created(self, span_exporter, instrument):
        mgr = BaselineManager()
        mgr.get_baseline_times("train")

        spans = span_exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert len(task) == 1


class TestIntegrationLiteLLM:
    def test_no_llm_span(self, span_exporter, instrument):
        model = LiteLLMModel()
        model.query()

        spans = span_exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 0

    def test_execute_query_increments_counter(self, span_exporter, instrument):
        model = LiteLLMModel()
        setattr(model, "_otel_algo_litellm_call_attempts", 0)

        model._execute_query()
        model._execute_query()

        assert getattr(model, "_otel_algo_litellm_call_attempts") == 2


class TestIntegrationFullFlow:
    def test_entry_agent_step_hierarchy(self, span_exporter, tracer_provider):
        """Smoke test: ENTRY -> AGENT -> STEP hierarchy with full wiring."""
        from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor

        instrumentor = AlgoTuneInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        try:
            from AlgoTuner.main import main

            # The stub main() just returns a string, creating an ENTRY span.
            main()

            # Also test the agent and step path directly.
            iface = LLMInterface(model_name="openai/gpt-4o")
            iface.run_task()

            spans = span_exporter.get_finished_spans()
            span_kinds = {s.attributes.get("gen_ai.span.kind") for s in spans}
            assert "ENTRY" in span_kinds
            assert "AGENT" in span_kinds

            for s in spans:
                if s.attributes.get("gen_ai.framework"):
                    assert s.attributes["gen_ai.framework"] == "AlgoTune"
        finally:
            instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Additional coverage tests -- content capture, helper functions, edge cases
# ---------------------------------------------------------------------------


class TestAlgotuneCaptureSpanContentEnabled:
    def test_returns_true_for_true(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_returns_true_for_one(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "1"
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_returns_true_for_span_only(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_returns_true_for_span_and_event(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "span_and_event",
        )
        assert _algotune_capture_span_content_enabled() is True

    def test_returns_false_for_empty(self, monkeypatch):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            raising=False,
        )
        assert _algotune_capture_span_content_enabled() is False

    def test_returns_false_for_false(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
        )
        assert _algotune_capture_span_content_enabled() is False


class TestTextValueCircularRef:
    def test_circular_ref_fallback_to_str(self):
        """json.dumps fails on circular refs; should fall back to str()."""
        d: dict[str, Any] = {}
        d["self"] = d
        result = _text_value(d)
        assert isinstance(result, str)
        assert len(result) > 0


class TestTaskJsonValue:
    def test_normal_value(self):
        result = _task_json_value({"key": "value"})
        assert '"key"' in result

    def test_exception_fallback(self):
        """json.dumps fails on circular refs; should fall back to str()."""
        d: dict[str, Any] = {}
        d["self"] = d
        result = _task_json_value(d)
        assert isinstance(result, str)
        assert len(result) > 0


class TestSetTaskInputOutput:
    def test_set_task_input(self, _otel):
        tracer, exporter = _otel
        with tracer.start_as_current_span("test") as span:
            _set_task_input(span, {"key": "val"})
        spans = exporter.get_finished_spans()
        s = spans[0]
        assert s.attributes.get("input.mime_type") == "application/json"
        assert '"key"' in s.attributes.get("input.value", "")

    def test_set_task_output(self, _otel):
        tracer, exporter = _otel
        with tracer.start_as_current_span("test") as span:
            _set_task_output(span, {"result": 42})
        spans = exporter.get_finished_spans()
        s = spans[0]
        assert s.attributes.get("output.mime_type") == "application/json"
        assert "42" in s.attributes.get("output.value", "")


class TestClearStepState:
    def test_clears_step_attrs(self):
        iface = LLMInterface()
        setattr(iface, INST_STEP_SPAN_ATTR, "some_span")
        setattr(iface, INST_STEP_TOKEN_ATTR, "some_token")

        _clear_step_state(iface)

        assert getattr(iface, INST_STEP_SPAN_ATTR) is None
        assert getattr(iface, INST_STEP_TOKEN_ATTR) is None

    def test_handles_setattr_failure(self):
        """If setattr raises, should not propagate."""

        class Frozen:
            __slots__ = ()

        instance = Frozen()
        # Should not raise.
        _clear_step_state(instance)


class TestAlgotuneToolDefinitions:
    def test_returns_definitions_from_command_formats(self):
        fmt1 = types.SimpleNamespace(
            description="Edit a file", example="edit file.py"
        )
        fmt2 = types.SimpleNamespace(description="Run the code", example="")
        command_formats = {"edit": fmt1, "run": fmt2}

        types_mod = types.ModuleType("AlgoTuner.interfaces.commands.types")
        types_mod.COMMAND_FORMATS = command_formats
        sys.modules["AlgoTuner.interfaces.commands.types"] = types_mod

        try:
            result = _algotune_tool_definitions()
            assert len(result) == 2
            names = {d["name"] for d in result}
            assert "edit" in names
            assert "run" in names
            for defn in result:
                assert defn["type"] == "function"
                assert "parameters" in defn
                assert defn["parameters"]["type"] == "object"
                assert "command" in defn["parameters"]["properties"]
        finally:
            del sys.modules["AlgoTuner.interfaces.commands.types"]

    def test_returns_empty_when_import_fails(self):
        sys.modules.pop("AlgoTuner.interfaces.commands.types", None)
        result = _algotune_tool_definitions()
        assert result == []

    def test_with_no_description_or_example(self):
        fmt = types.SimpleNamespace()  # no description or example attrs
        command_formats = {"cmd": fmt}
        types_mod = types.ModuleType("AlgoTuner.interfaces.commands.types")
        types_mod.COMMAND_FORMATS = command_formats
        sys.modules["AlgoTuner.interfaces.commands.types"] = types_mod

        try:
            result = _algotune_tool_definitions()
            assert len(result) == 1
            assert result[0]["name"] == "cmd"
            assert "AlgoTune command cmd" in result[0]["description"]
        finally:
            del sys.modules["AlgoTuner.interfaces.commands.types"]


class TestAgentContentAttributes:
    def test_returns_empty_when_capture_disabled(self, monkeypatch):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            raising=False,
        )
        iface = LLMInterface()
        result = _agent_content_attributes(iface)
        assert result == {}

    def test_with_all_message_roles(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        iface = LLMInterface()
        iface.state.messages = [
            {"role": "system", "content": "You are an optimizer."},
            {"role": "user", "content": "Optimize this algorithm."},
            {"role": "assistant", "content": "I will try approach A."},
        ]
        result = _agent_content_attributes(iface)
        assert result != {}
        assert result["algo.debug.input_messages.count"] == 1
        assert result["algo.debug.output_messages.count"] == 1
        assert result["algo.debug.system_instructions.count"] == 1
        assert "gen_ai.output.messages" in result
        assert "output.value" in result
        assert "gen_ai.input.messages" in result
        assert "gen_ai.system_instructions" in result

    def test_no_system_instructions_fallback(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        iface = LLMInterface()
        iface.state.messages = [
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "Response"},
        ]
        result = _agent_content_attributes(iface)
        # first user message used as fallback system instruction
        assert result["algo.debug.system_instructions.count"] == 1
        assert "gen_ai.system_instructions" in result

    def test_non_dict_messages_skipped(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        iface = LLMInterface()
        iface.state.messages = [
            "not a dict",
            {"role": "user", "content": "hello"},
        ]
        result = _agent_content_attributes(iface)
        assert result["algo.debug.input_messages.count"] == 1

    def test_with_no_state(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        instance = types.SimpleNamespace()  # no state attribute
        result = _agent_content_attributes(instance)
        assert result != {}
        assert result["algo.debug.input_messages.count"] == 0

    def test_with_tool_definitions(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        iface = LLMInterface()
        iface.state.messages = []

        fmt = types.SimpleNamespace(
            description="Run tests", example="run tests"
        )
        types_mod = types.ModuleType("AlgoTuner.interfaces.commands.types")
        types_mod.COMMAND_FORMATS = {"run": fmt}
        sys.modules["AlgoTuner.interfaces.commands.types"] = types_mod

        try:
            result = _agent_content_attributes(iface)
            assert result["algo.debug.tool_definitions.count"] == 1
            assert "gen_ai.tool.definitions" in result
        finally:
            del sys.modules["AlgoTuner.interfaces.commands.types"]

    def test_empty_messages_no_output(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        iface = LLMInterface()
        iface.state.messages = []
        result = _agent_content_attributes(iface)
        assert result["algo.debug.output_messages.count"] == 0
        # No output => output.value should not be present
        assert "output.value" not in result

    def test_none_role_defaults_to_user(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        iface = LLMInterface()
        iface.state.messages = [
            {"role": None, "content": "hello"},
        ]
        result = _agent_content_attributes(iface)
        # None role is treated as "user"
        assert result["algo.debug.input_messages.count"] == 1


class TestPublishAgentContentAttributes:
    def test_publishes_to_recording_span(self, _otel, monkeypatch):
        tracer, exporter = _otel
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )

        iface = LLMInterface()
        iface.state.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        with tracer.start_as_current_span("test") as span:
            _publish_agent_content_attributes(iface, span)

        spans = exporter.get_finished_spans()
        s = spans[0]
        assert s.attributes.get("algo.debug.input_messages.count") == 1

    def test_skips_non_recording_span(self, _otel, monkeypatch):
        tracer, exporter = _otel
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )

        iface = LLMInterface()
        iface.state.messages = [{"role": "user", "content": "test"}]

        span = tracer.start_span("test")
        span.end()
        # Should not raise on non-recording span.
        _publish_agent_content_attributes(iface, span)

    def test_skips_none_span(self, _otel, monkeypatch):
        tracer, exporter = _otel
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )

        iface = LLMInterface()
        iface.state.messages = [{"role": "user", "content": "test"}]
        # Should not raise when None is passed as a span.
        _publish_agent_content_attributes(iface, None)

    def test_noop_when_capture_disabled(self, _otel, monkeypatch):
        tracer, exporter = _otel
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            raising=False,
        )

        iface = LLMInterface()
        iface.state.messages = [{"role": "user", "content": "test"}]

        with tracer.start_as_current_span("test") as span:
            _publish_agent_content_attributes(iface, span)

        spans = exporter.get_finished_spans()
        s = spans[0]
        assert s.attributes.get("algo.debug.input_messages.count") is None


class TestRunTaskContentCapture:
    """Tests RunTaskWrapper with content capture enabled to cover
    _publish_agent_content_attributes call from the finally block."""

    def test_agent_publishes_content_on_success(self, _otel, monkeypatch):
        tracer, exporter = _otel
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        wrapper = RunTaskWrapper(tracer)
        iface = LLMInterface()
        iface.state.messages = [
            {"role": "user", "content": "Optimize TSP"},
            {"role": "assistant", "content": "Here is the solution"},
        ]

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        s = agent[0]
        assert s.attributes.get("algo.debug.input_messages.count") == 1
        assert s.attributes.get("algo.debug.output_messages.count") == 1


class TestHandleCommandContentCapture:
    """Tests HandleCommandWrapper with OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    enabled to cover tool_call_arguments and tool_call_result attribute lines."""

    def test_tool_args_captured(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)
        pc = types.SimpleNamespace(
            command="edit", args={"file": "sol.py", "code": "def f(): pass"}
        )
        handler = CommandHandlers()

        with mock.patch.object(
            _wrappers_module,
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            True,
        ):
            wrapper(
                lambda *a, **k: {
                    "success": True,
                    "message": "File edited successfully",
                },
                handler,
                (pc,),
                {},
            )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert len(tool) == 1
        s = tool[0]
        # Tool call arguments should be captured.
        tool_args = s.attributes.get("gen_ai.tool.call.arguments")
        assert tool_args is not None
        assert "sol.py" in tool_args
        # Tool call result should be captured.
        tool_result = s.attributes.get("gen_ai.tool.call.result")
        assert tool_result is not None
        assert "File edited" in tool_result

    def test_tool_args_not_captured_when_disabled(self, _otel):
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)
        pc = types.SimpleNamespace(command="edit", args={"file": "sol.py"})
        handler = CommandHandlers()

        with mock.patch.object(
            _wrappers_module,
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            False,
        ):
            wrapper(
                lambda *a, **k: {
                    "success": True,
                    "message": "Edited",
                },
                handler,
                (pc,),
                {},
            )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert tool[0].attributes.get("gen_ai.tool.call.arguments") is None
        assert tool[0].attributes.get("gen_ai.tool.call.result") is None

    def test_command_str_kwarg_fallback(self, _otel):
        """When no positional arg, command_str kwarg is used."""
        tracer, exporter = _otel
        wrapper = HandleCommandWrapper(tracer)
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"success": True},
            handler,
            (),
            {"command_str": {"command": "fallback_cmd"}},
        )

        spans = exporter.get_finished_spans()
        tool = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        assert (
            tool[0].attributes.get("algotune.command.error_response") is True
        )


class TestTogetherModelContentCapture:
    """Test TogetherModel wrapper with content capture enabled."""

    def test_message_captured_when_enabled(self, _otel):
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/llama-70b")

        with mock.patch.object(
            _wrappers_module,
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            True,
        ):
            wrapper(
                lambda *a, **k: {
                    "message": "Together model response text",
                    "cost": 0.01,
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    },
                },
                model,
                (),
                {},
            )

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 1
        s = llm[0]
        output_msg = s.attributes.get("gen_ai.output.messages")
        assert output_msg is not None
        assert "Together model response" in output_msg

    def test_total_tokens_from_sum(self, _otel):
        """When total_tokens is absent, uses prompt+completion sum."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/test")

        wrapper(
            lambda *a, **k: {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            },
            model,
            (),
            {},
        )

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        s = llm[0]
        assert s.attributes.get("gen_ai.usage.total_tokens") == 150

    def test_together_none_default_params(self, _otel):
        """When default_params is None, should not crash."""
        tracer, exporter = _otel
        wrapper = TogetherModelQueryWrapper(tracer)
        model = TogetherModel(model_name="together/bare")
        model.default_params = None

        wrapper(lambda *a, **k: {}, model, (), {})

        spans = exporter.get_finished_spans()
        llm = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"
        ]
        assert len(llm) == 1


class TestRunnerEvalDatasetEdgeCases:
    def test_aggregate_metrics_sub_dict(self, _otel):
        """Test _record_eval_attributes when metrics are under aggregate_metrics key."""
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        result_obj = types.SimpleNamespace(
            success=True,
            status="ok",
            message="done",
            data={
                "aggregate_metrics": {
                    "num_evaluated": 20,
                    "mean_speedup": 3.0,
                    "num_valid": 18,
                    "num_invalid": 1,
                    "num_timeout": 1,
                }
            },
        )

        wrapper(lambda *a, **k: result_obj, handler, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        s = task[0]
        assert s.attributes.get("algo.eval.total_problems") == 20
        assert s.attributes.get("algo.eval.mean_speedup") == 3.0
        assert s.attributes.get("algo.eval.num_valid") == 18
        assert s.attributes.get("algo.eval.num_invalid") == 1
        assert s.attributes.get("algo.eval.num_timeout") == 1

    def test_non_dataclass_result(self, _otel):
        """Result as a plain dict (no .data attribute)."""
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        wrapper(
            lambda *a, **k: {"num_evaluated": 5, "mean_speedup": 1.2},
            handler,
            ("test",),
            {},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert len(task) == 1

    def test_none_result_data(self, _otel):
        """Result with non-dict data should not crash."""
        tracer, exporter = _otel
        wrapper = RunnerEvalDatasetWrapper(tracer)
        handler = CommandHandlers()

        result_obj = types.SimpleNamespace(
            success=True, status="ok", message="done", data="not_a_dict"
        )

        wrapper(lambda *a, **k: result_obj, handler, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.dataset_eval"
        ]
        assert len(task) == 1


class TestEvaluateSingleEdgeCases:
    def test_none_speedup(self, _otel):
        """Handles None speedup gracefully."""
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result_obj = types.SimpleNamespace(
            speedup=None,
            solver_time_ms=None,
            is_valid=None,
            execution=None,
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {"problem_id": "null_problem"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert len(task) == 1
        # None values should not be set as attributes.
        assert task[0].attributes.get("algo.problem.speedup") is None

    def test_non_convertible_problem_index(self, _otel):
        """problem_index that cannot be converted to int should not crash."""
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result_obj = types.SimpleNamespace(
            speedup=1.0,
            solver_time_ms=100.0,
            is_valid=True,
            execution=types.SimpleNamespace(
                timeout_occurred=False, error_type=None
            ),
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {"problem_id": "idx_test", "problem_index": "not_an_int"},
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert len(task) == 1

    def test_non_convertible_baseline_time(self, _otel):
        """baseline_time_ms that cannot be converted to float should not crash."""
        tracer, exporter = _otel
        wrapper = EvaluateSingleWrapper(tracer)
        orch = EvaluationOrchestrator()

        result_obj = types.SimpleNamespace(
            speedup=1.0,
            solver_time_ms=100.0,
            is_valid=True,
            execution=types.SimpleNamespace(
                timeout_occurred=False, error_type=None
            ),
        )

        wrapper(
            lambda *a, **k: result_obj,
            orch,
            (),
            {
                "problem_id": "base_test",
                "baseline_time_ms": "not_a_float",
            },
        )

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name") == "benchmark.problem_eval"
        ]
        assert len(task) == 1


class TestGetBaselineTimesEdgeCases:
    def test_non_dict_result(self, _otel):
        """Non-dict result should not crash."""
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()

        result = wrapper(lambda *a, **k: "not_a_dict", mgr, ("train",), {})
        assert result == "not_a_dict"

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert len(task) == 1
        # actual_count should not be set.
        assert task[0].attributes.get("algo.baseline.actual_count") is None

    def test_system_exit_code_none(self, _otel):
        """SystemExit with non-int code defaults to 1."""
        tracer, exporter = _otel
        wrapper = GetBaselineTimesWrapper(tracer)
        mgr = BaselineManager()

        def raise_exit(*a, **k):
            raise SystemExit("fatal error string")

        with pytest.raises(SystemExit):
            wrapper(raise_exit, mgr, ("train",), {})

        spans = exporter.get_finished_spans()
        task = [
            s
            for s in spans
            if s.attributes.get("gen_ai.task.name")
            == "benchmark.baseline_generation"
        ]
        assert task[0].status.status_code.name == "ERROR"
        fatal = [
            e for e in task[0].events if e.name == "baseline.fatal_failure"
        ]
        assert fatal[0].attributes["exit_code"] == 1


class TestMainWrapperEdgeCases:
    def test_system_exit_non_int_code(self, _otel):
        """SystemExit with non-int code treated as 0."""
        tracer, exporter = _otel
        wrapper = MainWrapper(tracer)

        def raise_exit(*a, **k):
            raise SystemExit(None)

        with pytest.raises(SystemExit):
            wrapper(raise_exit, None, (), {})

        spans = exporter.get_finished_spans()
        entry = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "ENTRY"
        ]
        assert len(entry) == 1
        # code=None => not isinstance(None, int) => code=0 => no error status
        assert entry[0].status.status_code.name != "ERROR"


class TestRunTaskWrapperEdgeCases:
    def test_final_eval_metrics_non_dict(self, _otel):
        """_final_eval_metrics that is not a dict should not crash."""
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = LLMInterface()
        iface._final_eval_success = True
        iface._final_eval_metrics = "not_a_dict"

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert agent[0].attributes.get("algo.agent.final_eval_success") is True
        # final_mean_speedup should not be set because metrics is not a dict.
        assert agent[0].attributes.get("algo.agent.final_mean_speedup") is None

    def test_final_eval_metrics_with_non_float_speedup(self, _otel):
        """mean_speedup that can't convert to float should not crash."""
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = LLMInterface()
        iface._final_eval_success = True
        iface._final_eval_metrics = {"mean_speedup": "not_a_number"}

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        # Should not crash; attribute not set due to ValueError.
        assert agent[0].attributes.get("algo.agent.final_mean_speedup") is None

    def test_system_exit_zero_code(self, _otel):
        """SystemExit(0) should not set error status on agent span."""
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = LLMInterface()

        def raise_se(*a, **k):
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            wrapper(raise_se, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent) == 1
        assert agent[0].status.status_code.name != "ERROR"

    def test_infer_termination_check_limits_exception(self, _otel):
        """If check_limits raises, should fall back to 'completed'."""
        tracer, exporter = _otel
        wrapper = RunTaskWrapper(tracer)
        iface = LLMInterface()

        def bad_check():
            raise RuntimeError("broken")

        iface.check_limits = bad_check

        wrapper(lambda *a, **k: None, iface, (), {})

        spans = exporter.get_finished_spans()
        agent = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert (
            agent[0].attributes.get("algo.agent.final_status") == "completed"
        )


class TestGetResponseEdgeCases:
    def test_keyboard_interrupt_closes_step(self, _otel):
        """KeyboardInterrupt during get_response should close the step span."""
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)
        setattr(iface, INST_LITELLM_ATTEMPTS_ATTR, 0)

        def raise_ki(*a, **k):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            wrapper(raise_ki, iface, (), {})

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 1
        assert step[0].status.status_code.name == "ERROR"
        assert (
            step[0].attributes.get("gen_ai.react.finish_reason")
            == "KeyboardInterrupt"
        )

    def test_system_exit_closes_step(self, _otel):
        """SystemExit during get_response should close the step span."""
        tracer, exporter = _otel
        wrapper = GetResponseWrapper(tracer)
        iface = LLMInterface()
        setattr(iface, INST_ROUND_ATTR, 0)
        setattr(iface, INST_STEP_SPAN_ATTR, None)
        setattr(iface, INST_STEP_TOKEN_ATTR, None)
        setattr(iface, INST_LITELLM_ATTEMPTS_ATTR, 0)

        def raise_se(*a, **k):
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            wrapper(raise_se, iface, (), {})

        spans = exporter.get_finished_spans()
        step = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "STEP"
        ]
        assert len(step) == 1
        assert step[0].status.status_code.name == "ERROR"
