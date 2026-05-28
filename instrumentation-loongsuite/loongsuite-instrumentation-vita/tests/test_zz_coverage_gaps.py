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

"""Additional tests to cover missing lines in VitaBench instrumentation.

Covers error/fallback paths in __init__.py, patch.py, and utils.py.
All patch.py tests call wrapper functions directly to avoid framework-level
error handling that swallows exceptions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opentelemetry.instrumentation.vita import VitaInstrumentor
from opentelemetry.instrumentation.vita.patch import (
    _close_active_react_step,
    _in_agent_invoke,
    _react_step_invocation,
    wrap_generate,
    wrap_generate_next_message,
    wrap_get_response,
    wrap_orchestrator_step,
    wrap_run_task,
)
from opentelemetry.instrumentation.vita.utils import (
    _convert_vita_assistant_to_output,
    _convert_vita_messages_to_input,
    _get_tool_definitions,
    _infer_provider,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

# ==================== helpers ====================


def _make_handler(tracer_provider, logger_provider=None, meter_provider=None):
    return ExtendedTelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )


# ==================== utils.py coverage ====================


class TestUtilsCoverage:
    """Cover missing lines in utils.py."""

    def test_convert_empty_messages_returns_empty(self):
        """Line 44: empty/None messages -> return []."""
        assert _convert_vita_messages_to_input(None) == []
        assert _convert_vita_messages_to_input([]) == []

    def test_convert_single_message_not_list(self):
        """Line 47: non-list message -> wrap in list."""
        msg = SimpleNamespace(role="user", content="hello", tool_calls=None)
        result = _convert_vita_messages_to_input(msg)
        assert len(result) == 1
        assert result[0].role == "user"

    def test_convert_message_without_role_skipped(self):
        """Line 54: message with role=None -> skip."""
        msg = SimpleNamespace(role=None, content="data", tool_calls=None)
        result = _convert_vita_messages_to_input([msg])
        assert result == []

    def test_convert_message_exception_logged_and_skipped(self):
        """Lines 87-89: exception during conversion -> log and continue."""

        class BadMessage:
            @property
            def role(self):
                return "user"

            @property
            def content(self):
                raise RuntimeError("boom")

            @property
            def tool_calls(self):
                raise RuntimeError("boom")

        result = _convert_vita_messages_to_input([BadMessage()])
        assert result == []

    def test_convert_assistant_to_output_falsy_msg(self):
        """Line 97: falsy msg -> return []."""
        assert _convert_vita_assistant_to_output(None) == []
        assert _convert_vita_assistant_to_output("") == []

    def test_convert_assistant_to_output_no_content_no_tools(self):
        """Line 121: no content and no tool_calls -> append empty Text."""
        msg = SimpleNamespace(content=None, tool_calls=None)
        result = _convert_vita_assistant_to_output(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].parts[0].content == ""

    def test_infer_provider_empty_model(self):
        """Line 129: empty model_name -> return 'unknown'."""
        assert _infer_provider("") == "unknown"

    def test_infer_provider_deepseek(self):
        """Lines 137-138: 'deepseek' provider."""
        assert _infer_provider("deepseek-chat") == "deepseek"

    def test_infer_provider_gemini(self):
        """Lines 139-140: 'gemini' provider."""
        assert _infer_provider("gemini-pro") == "google"

    def test_infer_provider_unknown_model(self):
        """Line 141: unknown model -> return 'unknown'."""
        assert _infer_provider("some-custom-model") == "unknown"

    def test_get_tool_definitions_tool_without_name(self):
        """Line 154: tool with no name -> skip."""
        tool_no_name = SimpleNamespace(
            name=None,
            short_desc="desc",
            openai_schema={"function": {"name": "test", "parameters": {}}},
        )
        tool_with_name = SimpleNamespace(
            name="good_tool",
            short_desc="desc",
            openai_schema={
                "function": {"name": "good_tool", "parameters": {}}
            },
        )
        result = _get_tool_definitions([tool_no_name, tool_with_name])
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "good_tool"

    def test_get_tool_definitions_all_nameless_returns_none(self):
        """Line 167: all tools nameless -> defs empty -> return None."""
        tool_no_name = SimpleNamespace(
            name=None, short_desc="desc", openai_schema={}
        )
        result = _get_tool_definitions([tool_no_name])
        assert result is None

    def test_get_tool_definitions_exception_returns_none(self):
        """Lines 168-169: exception in tool definitions -> return None."""
        result = _get_tool_definitions(42)  # non-iterable
        assert result is None


# ==================== patch.py coverage (direct wrapper calls) ====================


class TestCloseActiveReactStep:
    """Cover _close_active_react_step exception path."""

    def test_close_active_react_step_exception(self, tracer_provider):
        """Lines 82-83: handler.stop_react_step raises -> log and set None."""
        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        handler = _make_handler(tracer_provider)
        step_inv = ReactStepInvocation(round=1)
        token = _react_step_invocation.set(step_inv)

        try:
            with patch.object(
                handler,
                "stop_react_step",
                side_effect=RuntimeError("stop failed"),
            ):
                _close_active_react_step(handler)

            assert _react_step_invocation.get() is None
        finally:
            _react_step_invocation.reset(token)


class TestWrapRunTaskDirect:
    """Cover wrap_run_task error and reward paths via direct calls."""

    def test_run_task_with_reward_info(self, span_exporter, tracer_provider):
        """Lines 117-119: result has reward_info.reward."""
        handler = _make_handler(tracer_provider)
        task = SimpleNamespace(instructions="test task")
        result = SimpleNamespace(
            termination_reason="agent_stop",
            reward_info=SimpleNamespace(reward=0.95),
        )

        ret = wrap_run_task(
            lambda *a, **k: result, None, ("domain", task), {}, handler=handler
        )

        assert ret.reward_info.reward == 0.95
        spans = span_exporter.get_finished_spans()
        entry_span = next(s for s in spans if "enter" in s.name)
        assert entry_span is not None

    def test_run_task_exception(self, span_exporter, tracer_provider):
        """Lines 131-133: wrapped() raises -> handler.fail_entry."""
        handler = _make_handler(tracer_provider)
        task = SimpleNamespace(instructions="test task")

        def raising(*a, **k):
            raise RuntimeError("run_task failed")

        with pytest.raises(RuntimeError, match="run_task failed"):
            wrap_run_task(raising, None, ("domain", task), {}, handler=handler)

        spans = span_exporter.get_finished_spans()
        entry_span = next(s for s in spans if "enter" in s.name)
        assert entry_span.status.status_code.name == "ERROR"


class TestWrapOrchestratorStepDirect:
    """Cover wrap_orchestrator_step edge cases via direct calls.

    wrap_orchestrator_step uses deferred close: it opens a step span but
    does NOT close it (the span is closed later by _close_active_react_step
    in wrap_orchestrator_run). So after calling wrap_orchestrator_step, we
    must manually close the step to see it in finished spans.
    """

    def test_role_string_fallback(self, span_exporter, tracer_provider):
        """Lines 206-207, 213: Role import fails -> fallback string comparison."""
        handler = _make_handler(tracer_provider)
        instance = SimpleNamespace(
            to_role="Role.AGENT",
            done=True,
            termination_reason=None,
            message=None,
        )

        import sys

        vita_orch_mod = sys.modules["vita.orchestrator.orchestrator"]
        saved_role = vita_orch_mod.Role

        try:
            # Temporarily remove Role from the cached module so the import
            # inside wrap_orchestrator_step fails with ImportError
            delattr(vita_orch_mod, "Role")

            wrap_orchestrator_step(
                lambda *a, **k: "ok", instance, (), {}, handler=handler
            )
        finally:
            # Restore Role
            vita_orch_mod.Role = saved_role

        # Close the deferred step span
        _close_active_react_step(handler)

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if "react step" in s.name]
        assert len(step_spans) == 1

    def test_done_no_termination_reason(self, span_exporter, tracer_provider):
        """Line 242: done=True, termination_reason=None -> 'agent_stop'."""
        handler = _make_handler(tracer_provider)
        from vita.orchestrator.orchestrator import Role

        instance = SimpleNamespace(
            to_role=Role.AGENT,
            done=True,
            termination_reason=None,
            message=None,
        )

        wrap_orchestrator_step(
            lambda *a, **k: "ok", instance, (), {}, handler=handler
        )

        # Close the deferred step span
        _close_active_react_step(handler)

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if "react step" in s.name]
        assert len(step_spans) == 1
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "agent_stop"

    def test_not_done_no_tool_call(self, span_exporter, tracer_provider):
        """Line 248: not done, message is not tool call -> 'assistant_text'."""
        handler = _make_handler(tracer_provider)
        from vita.orchestrator.orchestrator import Role

        message_mock = SimpleNamespace(content="Just text")
        message_mock.is_tool_call = lambda: False

        instance = SimpleNamespace(
            to_role=Role.AGENT,
            done=False,
            termination_reason=None,
            message=message_mock,
        )

        wrap_orchestrator_step(
            lambda *a, **k: "ok", instance, (), {}, handler=handler
        )

        # Close the deferred step span
        _close_active_react_step(handler)

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if "react step" in s.name]
        assert len(step_spans) == 1
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "assistant_text"


class TestWrapGenerateNextMessageDirect:
    """Cover wrap_generate_next_message reentrancy and error paths."""

    def test_reentrancy_guard(self, span_exporter, tracer_provider):
        """Line 269: _in_agent_invoke already True -> bypass, just call wrapped."""
        handler = _make_handler(tracer_provider)
        token = _in_agent_invoke.set(True)
        try:
            instance = SimpleNamespace(
                __class__=type("FakeAgent", (), {}),
                llm="model",
                tools=[],
            )
            result = wrap_generate_next_message(
                lambda *a, **k: "bypassed", instance, (), {}, handler=handler
            )
            assert result == "bypassed"
            # No spans should be created due to reentrancy guard
            spans = span_exporter.get_finished_spans()
            agent_spans = [s for s in spans if "invoke_agent" in s.name]
            assert len(agent_spans) == 0
        finally:
            _in_agent_invoke.reset(token)

    def test_exception_path(self, span_exporter, tracer_provider):
        """Lines 319-321: wrapped() raises -> handler.fail_invoke_agent."""
        handler = _make_handler(tracer_provider)
        instance = SimpleNamespace(
            __class__=type("FakeAgent", (), {}),
            llm="model",
            tools=[],
        )

        def raising(*a, **k):
            raise RuntimeError("agent failed")

        with pytest.raises(RuntimeError, match="agent failed"):
            wrap_generate_next_message(
                raising, instance, (), {}, handler=handler
            )

        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if "invoke_agent" in s.name)
        assert agent_span.status.status_code.name == "ERROR"


class TestWrapGenerateDirect:
    """Cover wrap_generate exception path."""

    def test_exception_path(self, span_exporter, tracer_provider):
        """Lines 379-381: wrapped() raises -> handler.fail_llm."""
        handler = _make_handler(tracer_provider)

        def raising(*a, **k):
            raise ConnectionError("network error")

        with pytest.raises(ConnectionError, match="network error"):
            wrap_generate(
                raising,
                None,
                ("test-model", [], None),
                {"temperature": 0.5},
                handler=handler,
            )

        spans = span_exporter.get_finished_spans()
        llm_span = next(s for s in spans if "chat" in s.name)
        assert llm_span.status.status_code.name == "ERROR"


class TestWrapGetResponseDirect:
    """Cover wrap_get_response error and fallback paths."""

    def test_json_dumps_fallback(self, span_exporter, tracer_provider):
        """Lines 408-409: json.dumps raises -> fallback to str()."""
        handler = _make_handler(tracer_provider)
        message = SimpleNamespace(
            id="tc_1", name="my_tool", arguments={"key": "value"}
        )
        result = SimpleNamespace(content="tool output", error=False)

        with patch(
            "opentelemetry.instrumentation.vita.patch.json.dumps",
            side_effect=TypeError("not serializable"),
        ):
            ret = wrap_get_response(
                lambda *a, **k: result, None, (message,), {}, handler=handler
            )

        assert ret.content == "tool output"
        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if "execute_tool" in s.name)
        assert tool_span is not None

    def test_exception_path(self, span_exporter, tracer_provider):
        """Lines 430-432: wrapped() raises -> handler.fail_execute_tool."""
        handler = _make_handler(tracer_provider)
        message = SimpleNamespace(
            id="tc_crash", name="crash_tool", arguments={"key": "val"}
        )

        def raising(*a, **k):
            raise RuntimeError("wrapped get_response crashed")

        with pytest.raises(RuntimeError, match="wrapped get_response crashed"):
            wrap_get_response(raising, None, (message,), {}, handler=handler)

        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if "execute_tool" in s.name)
        assert tool_span.status.status_code.name == "ERROR"


# ==================== __init__.py coverage ====================


class TestInstrumentExceptionHandling:
    """Cover exception paths in _instrument and _uninstrument.

    Uses mock patches to avoid actually wrapping/unwrapping real functions,
    preventing state leakage between tests.
    """

    def test_instrument_wrapping_failures(
        self, tracer_provider, logger_provider, meter_provider
    ):
        """Lines 103-104, 116-117, 129-130, 142-143, 155-156, 168-169, 181-182:
        wrap_function_wrapper raises for each module -> logged and skipped."""
        instrumentor = VitaInstrumentor()

        with patch(
            "opentelemetry.instrumentation.vita.wrap_function_wrapper",
            side_effect=Exception("cannot wrap"),
        ):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                skip_dep_check=True,
            )

        assert instrumentor._handler is not None

        # Real uninstrument to properly clean up any previously wrapped functions
        instrumentor.uninstrument()

    def test_uninstrument_failures(
        self, tracer_provider, logger_provider, meter_provider
    ):
        """Lines 190-191, 198-199, 206-207, 213-214, 220-221:
        uninstrument fails for each module -> logged and skipped."""
        instrumentor = VitaInstrumentor()

        # Instrument first with real wrapping
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )

        # Now uninstrument with all imports failing — exercises the except paths
        import builtins

        _real_import = builtins.__import__
        fail_modules = {
            "vita.run",
            "vita.orchestrator.orchestrator",
            "vita.agent.llm_agent",
            "vita.utils.llm_utils",
            "vita.environment.environment",
        }

        def mock_import(name, *args, **kwargs):
            if name in fail_modules:
                raise ImportError(f"mocked: cannot import {name}")
            return _real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            instrumentor.uninstrument()

        assert instrumentor._handler is None

        # Re-instrument and properly uninstrument to clean up wrapped functions
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        instrumentor.uninstrument()
