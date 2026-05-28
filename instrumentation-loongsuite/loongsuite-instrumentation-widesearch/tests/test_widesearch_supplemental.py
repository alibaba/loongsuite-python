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

"""Supplemental tests for WideSearch instrumentation.

Covers uncovered error-handling branches, edge cases, and fallback paths
in __init__.py, patch.py, and utils.py to bring coverage above 90%.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import patch

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from .conftest import (
    ActionStep,
    Agent,
    ErrorMarker,
    InternalResponse,
    LLMOutputItem,
    MemoryAgent,
    ModelResponse,
    StepStatus,
    ToolCall,
    ToolCallResult,
)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_handler():
    """Create a fresh handler with its own exporter for direct wrapper tests."""
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    handler = ExtendedTelemetryHandler(tracer_provider=tp)
    return handler, exporter


# ---------------------------------------------------------------------------
# __init__.py: _instrument error handling (lines 83-84, 96-97, 109-110,
#              122-123, 137-138)
# ---------------------------------------------------------------------------


class TestInstrumentErrorHandling:
    """Test that _instrument gracefully handles wrap_function_wrapper failures."""

    def test_instrument_survives_missing_run_module(
        self, tracer_provider, meter_provider
    ):
        """If src.agent.run is missing, instrumentation should warn but not crash."""
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        saved = {}
        keys_to_remove = [
            "src.agent.run",
            "src.agent.multi_agent_tools",
        ]
        for k in keys_to_remove:
            if k in sys.modules:
                saved[k] = sys.modules.pop(k)

        try:
            instrumentor = WideSearchInstrumentor()
            # Should not raise even though modules are missing
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                skip_dep_check=True,
            )
            assert instrumentor._handler is not None
            instrumentor.uninstrument()
        finally:
            sys.modules.update(saved)

    def test_instrument_individual_wrap_failures(
        self, tracer_provider, meter_provider
    ):
        """Each wrap_function_wrapper call is independently try/excepted."""
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        call_count = 0

        def failing_wrap(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Wrap failure #{call_count}")

        with patch(
            "opentelemetry.instrumentation.widesearch.wrap_function_wrapper",
            side_effect=failing_wrap,
        ):
            instrumentor = WideSearchInstrumentor()
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                skip_dep_check=True,
            )
            # All 5 wraps should have been attempted (and failed gracefully)
            assert call_count == 5
            assert instrumentor._handler is not None
            instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# __init__.py: _uninstrument error handling (lines 151-152, 159-160)
# ---------------------------------------------------------------------------


class TestUninstrumentErrorHandling:
    def test_uninstrument_survives_import_failure(
        self, tracer_provider, meter_provider
    ):
        """_uninstrument should gracefully handle missing modules."""
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        instrumentor = WideSearchInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )

        # Remove modules so uninstrument's import fails
        saved = {}
        for k in ["src.agent.run", "src.agent.multi_agent_tools"]:
            if k in sys.modules:
                saved[k] = sys.modules.pop(k)

        try:
            # Should not raise
            instrumentor.uninstrument()
            assert instrumentor._handler is None
        finally:
            sys.modules.update(saved)

    def test_uninstrument_survives_unwrap_failure(
        self, tracer_provider, meter_provider
    ):
        """_uninstrument should gracefully handle unwrap exceptions."""
        from opentelemetry.instrumentation.widesearch import (
            WideSearchInstrumentor,
        )

        instrumentor = WideSearchInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )

        with patch(
            "opentelemetry.instrumentation.widesearch.unwrap",
            side_effect=RuntimeError("unwrap boom"),
        ):
            instrumentor.uninstrument()
            assert instrumentor._handler is None


# ---------------------------------------------------------------------------
# patch.py: wrap_run_single_query reentrant guard (line 43)
# ---------------------------------------------------------------------------


class TestRunSingleQueryReentrant:
    def test_reentrant_call_skips_span(self, span_exporter, instrument):
        """Nested run_single_query calls should skip instrumentation."""
        from opentelemetry.instrumentation.widesearch.patch import (
            _in_run_single_query,
        )

        token = _in_run_single_query.set(True)
        try:
            from src.agent.run import run_single_query

            result = _run_async(run_single_query("nested query"))
            # Should still return a result
            assert result is not None
            # Should NOT create a new ENTRY span (beyond any from parent)
            spans = span_exporter.get_finished_spans()
            entry_spans = [
                s for s in spans if s.name == "enter_ai_application_system"
            ]
            assert len(entry_spans) == 0
        finally:
            _in_run_single_query.reset(token)


# ---------------------------------------------------------------------------
# patch.py: wrap_run_single_query invocation creation failure (lines 57-60)
# ---------------------------------------------------------------------------


class TestEntrySingleQueryInvocationFailure:
    def test_entry_invocation_creation_failure_falls_back(
        self, span_exporter, instrument
    ):
        """If _create_entry_invocation raises, the call should proceed without span."""
        from src.agent.run import run_single_query

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_entry_invocation",
            side_effect=RuntimeError("creation failed"),
        ):
            result = _run_async(run_single_query("fallback test"))
            assert result is not None

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        # No ENTRY span should be created when invocation creation fails
        assert len(entry_spans) == 0


# ---------------------------------------------------------------------------
# patch.py: wrap_runner_run invocation creation failure (lines 89-93)
# ---------------------------------------------------------------------------


class TestAgentInvocationCreationFailure:
    def test_agent_invocation_creation_failure_falls_back(
        self, span_exporter, instrument
    ):
        """If _create_agent_invocation raises, Runner.run should still yield steps."""
        from src.agent.run import Runner

        agent = Agent(name="fallback-agent")

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_agent_invocation",
            side_effect=RuntimeError("agent inv creation failed"),
        ):

            async def _run():
                results = []
                async for step in Runner.run(agent, "Hello"):
                    results.append(step)
                return results

            results = _run_async(_run())
            assert len(results) >= 1

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 0


# ---------------------------------------------------------------------------
# patch.py: GeneratorExit in wrap_runner_run (lines 108-111)
# ---------------------------------------------------------------------------


class TestAgentGeneratorExit:
    def test_agent_span_on_generator_exit(self, span_exporter, instrument):
        """AGENT span should record error when GeneratorExit is raised."""
        from src.agent.run import Runner

        agent = Agent(name="gen-exit-agent")

        async def _partial_consume():
            gen = Runner.run(agent, "Hello")
            # Get first step, then close the generator
            async for step in gen:
                await gen.aclose()
                break

        _run_async(_partial_consume())

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1
        assert agent_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# patch.py: wrap_runner_step start failure (lines 133-135)
# ---------------------------------------------------------------------------


class TestStepStartFailure:
    def test_step_start_failure_falls_back(self, span_exporter, instrument):
        """If handler.start_react_step raises, _step should still execute."""
        from opentelemetry.instrumentation.widesearch.patch import (
            wrap_runner_step,
        )

        handler, exporter = _make_handler()

        agent = Agent(name="step-fail-agent")
        memory = MemoryAgent(system_instructions="test")

        # Patch start_react_step to raise
        with patch.object(
            handler,
            "start_react_step",
            side_effect=RuntimeError("start failed"),
        ):
            # Call the wrapper directly with the step function
            async def mock_step(*, agent, memory):
                return ActionStep(
                    step_status=StepStatus.FINISHED, content="Done"
                )

            result = _run_async(
                wrap_runner_step(
                    mock_step,
                    None,
                    (),
                    {"agent": agent, "memory": memory},
                    handler=handler,
                )
            )

        # Step should still return a result despite start failure
        assert result is not None
        assert result.content == "Done"


# ---------------------------------------------------------------------------
# patch.py: step finish_reason branches (lines 151-154)
# ---------------------------------------------------------------------------


class TestStepFinishReasonBranches:
    def test_step_finish_reason_error_marker(self, span_exporter, instrument):
        """STEP should set finish_reason='error' when error_marker is present."""
        from src.agent.run import Runner

        async def step_with_error_marker(*, agent, memory):
            return ActionStep(
                step_status=StepStatus.CONTINUE,
                content="partial",
                error_marker=ErrorMarker(message="some error"),
            )

        Runner._step_override = step_with_error_marker
        agent = Agent(name="error-marker-agent")

        try:

            async def _run():
                async for _ in Runner.run(agent, "test"):
                    pass

            _run_async(_run())
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) >= 1
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "error"

    def test_step_finish_reason_continue(self, span_exporter, instrument):
        """STEP should set finish_reason='continue' for intermediate steps."""
        from src.agent.run import Runner

        async def continuing_step(*, agent, memory):
            return ActionStep(
                step_status=StepStatus.CONTINUE,
                content="thinking...",
                error_marker=None,
            )

        Runner._step_override = continuing_step
        agent = Agent(name="continue-agent")

        try:

            async def _run():
                async for _ in Runner.run(agent, "test"):
                    pass

            _run_async(_run())
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) >= 1
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "continue"

    def test_step_exception_sets_error_finish_reason(
        self, span_exporter, instrument
    ):
        """STEP should set finish_reason='error' and re-raise on exception."""
        from src.agent.run import Runner

        async def raising_step(*, agent, memory):
            raise ValueError("step exploded")

        Runner._step_override = raising_step
        agent = Agent(name="raise-agent")

        try:

            async def _run():
                async for _ in Runner.run(agent, "test"):
                    pass

            with pytest.raises(ValueError, match="step exploded"):
                _run_async(_run())
        finally:
            Runner._step_override = None

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) >= 1
        attrs = dict(step_spans[0].attributes)
        assert attrs.get("gen_ai.react.finish_reason") == "error"


# ---------------------------------------------------------------------------
# patch.py: wrap_invoke_tool_call empty outputs/tool_calls (lines 174, 178)
# ---------------------------------------------------------------------------


class TestToolCallEmptyPaths:
    def test_no_outputs_returns_original(self, span_exporter, instrument):
        """If model_response.outputs is empty, return wrapped result directly."""
        from src.agent.run import Runner

        agent = Agent(name="no-output-agent")
        model_resp = ModelResponse(outputs=[])

        result = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert result == []

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 0

    def test_no_tool_calls_returns_original(self, span_exporter, instrument):
        """If outputs[0].tool_calls is empty, return wrapped result directly."""
        from src.agent.run import Runner

        agent = Agent(name="no-tc-agent")
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[])])

        result = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert result == []

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 0


# ---------------------------------------------------------------------------
# patch.py: _create_tool_invocation failure -> _call_original (lines 185-187,
#           256-276)
# ---------------------------------------------------------------------------


class TestToolInvocationCreationFailure:
    def test_tool_invocation_creation_failure_uses_fallback(
        self, span_exporter, instrument
    ):
        """If _create_tool_invocation raises, _call_original is used as fallback."""
        from src.agent.run import Runner

        async def mock_tool(**kwargs):
            return InternalResponse(data="fallback result")

        agent = Agent(
            name="fallback-tool-agent",
            tools={"my_tool": mock_tool},
        )

        tc = ToolCall(
            tool_name="my_tool",
            arguments='{"key": "val"}',
            tool_call_id="call_fb",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_tool_invocation",
            side_effect=RuntimeError("invocation boom"),
        ):
            results = _run_async(Runner._invoke_tool_call(agent, model_resp))

        assert len(results) == 1
        assert results[0].content == "fallback result"

        # No tool span should be created
        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 0

    def test_call_original_tool_not_found(self, span_exporter, instrument):
        """_call_original should handle tool-not-found case."""
        from src.agent.run import Runner

        agent = Agent(name="no-tool-agent", tools={})

        tc = ToolCall(
            tool_name="missing_tool",
            arguments="{}",
            tool_call_id="call_miss",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_tool_invocation",
            side_effect=RuntimeError("invocation boom"),
        ):
            results = _run_async(Runner._invoke_tool_call(agent, model_resp))

        assert len(results) == 1
        assert results[0].error_marker is not None

    def test_call_original_tool_raises(self, span_exporter, instrument):
        """_call_original should handle tool exceptions."""
        from src.agent.run import Runner

        async def exploding_tool(**kwargs):
            raise ConnectionError("timeout")

        agent = Agent(
            name="explode-agent",
            tools={"bomb": exploding_tool},
        )

        tc = ToolCall(
            tool_name="bomb",
            arguments='{"x": 1}',
            tool_call_id="call_boom",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_tool_invocation",
            side_effect=RuntimeError("invocation boom"),
        ):
            results = _run_async(Runner._invoke_tool_call(agent, model_resp))

        assert len(results) == 1
        assert results[0].error_marker is not None
        assert "timeout" in results[0].error_marker.message

    def test_call_original_with_string_arguments_json_valid(
        self, span_exporter, instrument
    ):
        """_call_original should parse string arguments as JSON."""
        from src.agent.run import Runner

        async def echo_tool(**kwargs):
            return InternalResponse(data=json.dumps(kwargs))

        agent = Agent(
            name="echo-agent",
            tools={"echo": echo_tool},
        )

        tc = ToolCall(
            tool_name="echo",
            arguments='{"msg": "hello"}',
            tool_call_id="call_echo",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_tool_invocation",
            side_effect=RuntimeError("invocation boom"),
        ):
            results = _run_async(Runner._invoke_tool_call(agent, model_resp))

        assert len(results) == 1
        assert results[0].content is not None

    def test_call_original_with_invalid_json_arguments(
        self, span_exporter, instrument
    ):
        """_call_original should handle invalid JSON arguments string."""
        from src.agent.run import Runner

        async def any_tool(**kwargs):
            return InternalResponse(data="ok")

        agent = Agent(
            name="bad-json-agent",
            tools={"do_thing": any_tool},
        )

        tc = ToolCall(
            tool_name="do_thing",
            arguments="not valid json {{{",
            tool_call_id="call_bad",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_tool_invocation",
            side_effect=RuntimeError("invocation boom"),
        ):
            results = _run_async(Runner._invoke_tool_call(agent, model_resp))

        assert len(results) == 1
        # Should succeed since arguments fall back to {}
        assert results[0].content == "ok"

    def test_call_original_with_error_response(
        self, span_exporter, instrument
    ):
        """_call_original should propagate error/system_error from tool response."""
        from src.agent.run import Runner

        async def error_tool(**kwargs):
            return InternalResponse(
                data="partial",
                error="application error",
                system_error=None,
                extra={"key": "val"},
            )

        agent = Agent(
            name="error-resp-agent",
            tools={"err_tool": error_tool},
        )

        tc = ToolCall(
            tool_name="err_tool",
            arguments="{}",
            tool_call_id="call_err_resp",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        with patch(
            "opentelemetry.instrumentation.widesearch.patch._create_tool_invocation",
            side_effect=RuntimeError("invocation boom"),
        ):
            results = _run_async(Runner._invoke_tool_call(agent, model_resp))

        assert len(results) == 1
        assert results[0].error_marker is not None


# ---------------------------------------------------------------------------
# patch.py: JSON decode error in tool arguments (lines 211-212)
# ---------------------------------------------------------------------------


class TestToolArgumentsJsonDecode:
    def test_tool_arguments_invalid_json_falls_back_to_empty(
        self, span_exporter, instrument
    ):
        """When tool_call.arguments is invalid JSON string, fall back to {}."""
        from src.agent.run import Runner

        async def mock_tool(**kwargs):
            return InternalResponse(data="ok with empty args")

        agent = Agent(
            name="bad-args-agent",
            tools={"my_tool": mock_tool},
        )

        tc = ToolCall(
            tool_name="my_tool",
            arguments="not-json!!!",
            tool_call_id="call_badjson",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 1
        assert results[0].content == "ok with empty args"


# ---------------------------------------------------------------------------
# patch.py: tool response with error/system_error (lines 239-240)
# ---------------------------------------------------------------------------


class TestToolResponseErrors:
    def test_tool_response_with_error_marker(self, span_exporter, instrument):
        """Tool response with .error should trigger fail_execute_tool."""
        from src.agent.run import Runner

        async def tool_with_error(**kwargs):
            return InternalResponse(
                data="partial data",
                error="application-level error",
            )

        agent = Agent(
            name="err-tool-agent",
            tools={"err_tool": tool_with_error},
            tools_desc=[
                {
                    "type": "function",
                    "function": {
                        "name": "err_tool",
                        "description": "A tool that errors",
                    },
                }
            ],
        )

        tc = ToolCall(
            tool_name="err_tool",
            arguments="{}",
            tool_call_id="call_err_marker",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 1
        assert results[0].error_marker is not None

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR

    def test_tool_response_with_system_error(self, span_exporter, instrument):
        """Tool response with .system_error should trigger fail_execute_tool."""
        from src.agent.run import Runner

        async def tool_with_sys_err(**kwargs):
            return InternalResponse(
                data="partial",
                system_error="internal system failure",
            )

        agent = Agent(
            name="sys-err-agent",
            tools={"sys_tool": tool_with_sys_err},
        )

        tc = ToolCall(
            tool_name="sys_tool",
            arguments="{}",
            tool_call_id="call_sys_err",
        )
        model_resp = ModelResponse(outputs=[LLMOutputItem(tool_calls=[tc])])

        results = _run_async(Runner._invoke_tool_call(agent, model_resp))
        assert len(results) == 1
        assert results[0].system_error_marker is not None

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# patch.py: task span safe_input JSON failure (lines 325-326) and
#           output truncation (line 338)
# ---------------------------------------------------------------------------


class TestTaskSpanEdgeCases:
    def test_task_span_safe_input_json_failure(
        self, span_exporter, instrument
    ):
        """TASK span should survive if json.dumps for input fails."""
        from opentelemetry.instrumentation.widesearch.patch import (
            wrap_create_sub_agents_factory,
        )

        handler, exporter = _make_handler()

        async def original_closure(sub_agents):
            return InternalResponse(data="done")

        def factory(*args, **kwargs):
            return original_closure

        wrapped = wrap_create_sub_agents_factory(
            factory, None, (), {}, handler=handler
        )

        # Pass sub_agents that will cause json.dumps to fail via sa.get()
        # The except Exception: pass on lines 325-326 should swallow it
        class BadObj:
            pass

        sub_agents = [BadObj()]  # Not a dict, .get() will raise

        # The closure should still complete despite input serialization failure
        result = _run_async(wrapped(sub_agents))
        assert result is not None

        spans = exporter.get_finished_spans()
        task_spans = [
            s for s in spans if s.name == "run_task create_sub_agents"
        ]
        assert len(task_spans) == 1
        # input.value should NOT be set since serialization failed
        attrs = dict(task_spans[0].attributes)
        assert "input.value" not in attrs

    def test_task_span_output_truncation(self, span_exporter, instrument):
        """TASK span should truncate output > 4096 chars."""
        from opentelemetry.instrumentation.widesearch.patch import (
            wrap_create_sub_agents_factory,
        )

        handler, exporter = _make_handler()

        long_data = "x" * 5000

        async def original_closure(sub_agents):
            return InternalResponse(data=long_data)

        def factory(*args, **kwargs):
            return original_closure

        wrapped = wrap_create_sub_agents_factory(
            factory, None, (), {}, handler=handler
        )

        result = _run_async(wrapped([{"index": 0, "prompt": "test"}]))
        assert result is not None

        spans = exporter.get_finished_spans()
        task_spans = [
            s for s in spans if s.name == "run_task create_sub_agents"
        ]
        assert len(task_spans) == 1
        output_val = dict(task_spans[0].attributes).get("output.value", "")
        assert output_val.endswith("...(truncated)")
        assert len(output_val) < 5000

    def test_task_span_output_non_string_data(self, span_exporter, instrument):
        """TASK span should JSON-serialize non-string data."""
        from opentelemetry.instrumentation.widesearch.patch import (
            wrap_create_sub_agents_factory,
        )

        handler, exporter = _make_handler()

        async def original_closure(sub_agents):
            return InternalResponse(data={"result": "structured"})

        def factory(*args, **kwargs):
            return original_closure

        wrapped = wrap_create_sub_agents_factory(
            factory, None, (), {}, handler=handler
        )

        result = _run_async(wrapped([{"index": 0, "prompt": "test"}]))
        assert result is not None

        spans = exporter.get_finished_spans()
        task_spans = [
            s for s in spans if s.name == "run_task create_sub_agents"
        ]
        assert len(task_spans) == 1
        output_val = dict(task_spans[0].attributes).get("output.value", "")
        assert "structured" in output_val


# ---------------------------------------------------------------------------
# utils.py: _create_agent_invocation model_config import failure (lines 67-68)
# ---------------------------------------------------------------------------


class TestUtilsAgentInvocation:
    def test_model_config_import_failure(self):
        """_create_agent_invocation should handle model_config import failure."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _create_agent_invocation,
        )

        agent = Agent(
            name="test-agent",
            model_config_name="some-model",
        )

        # Remove the config module so import fails
        saved = sys.modules.get("src.utils.config")
        sys.modules["src.utils.config"] = None  # Will cause import to fail

        try:
            # Override the import path to raise
            with patch.dict(sys.modules, {"src.utils.config": None}):
                # Need to make the import actually fail
                saved_mod = sys.modules.pop("src.utils.config", None)
                bad_mod = types.ModuleType("src.utils.config")
                # Don't set model_config attr so getattr raises
                sys.modules["src.utils.config"] = bad_mod

                try:
                    inv = _create_agent_invocation(agent, "test input")
                    # Should fall back to model_config_name
                    assert inv.request_model == "some-model"
                finally:
                    if saved_mod is not None:
                        sys.modules["src.utils.config"] = saved_mod
        finally:
            if saved is not None:
                sys.modules["src.utils.config"] = saved

    def test_agent_no_model_config_name(self):
        """_create_agent_invocation with no model_config_name."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _create_agent_invocation,
        )

        agent = Agent(name="test-agent", model_config_name="")

        inv = _create_agent_invocation(agent, "hello")
        assert inv.agent_name == "test-agent"


# ---------------------------------------------------------------------------
# utils.py: _create_tool_invocation JSON decode error (lines 101-102)
# ---------------------------------------------------------------------------


class TestUtilsToolInvocation:
    def test_tool_invocation_invalid_json_args(self):
        """_create_tool_invocation should handle invalid JSON in arguments."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _create_tool_invocation,
        )

        tc = ToolCall(
            tool_name="my_tool",
            arguments="not valid {json}",
            tool_call_id="call_1",
        )
        agent = Agent(name="test-agent")

        inv = _create_tool_invocation(tc, agent)
        assert inv.tool_name == "my_tool"
        assert inv.tool_call_arguments == {"raw": "not valid {json}"}

    def test_tool_invocation_with_description_match(self):
        """_create_tool_invocation should find matching tool description."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _create_tool_invocation,
        )

        tc = ToolCall(
            tool_name="search",
            arguments={"q": "test"},
            tool_call_id="call_2",
        )
        agent = Agent(
            name="test-agent",
            tools_desc=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                    },
                }
            ],
        )

        inv = _create_tool_invocation(tc, agent)
        assert inv.tool_description == "Search the web"

    def test_tool_invocation_no_matching_description(self):
        """_create_tool_invocation with no matching tool in tools_desc."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _create_tool_invocation,
        )

        tc = ToolCall(
            tool_name="other_tool",
            arguments={"q": "test"},
            tool_call_id="call_3",
        )
        agent = Agent(
            name="test-agent",
            tools_desc=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                    },
                }
            ],
        )

        inv = _create_tool_invocation(tc, agent)
        assert inv.tool_description is None


# ---------------------------------------------------------------------------
# utils.py: _extract_output_messages edge cases (lines 126, 133-134)
# ---------------------------------------------------------------------------


class TestExtractOutputMessages:
    def test_empty_messages_returns_empty(self):
        """_extract_output_messages with empty/None input returns []."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _extract_output_messages,
        )

        assert _extract_output_messages(None) == []
        assert _extract_output_messages([]) == []

    def test_string_content(self):
        """_extract_output_messages with string content (not dict)."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _extract_output_messages,
        )

        messages = [{"role": "assistant", "content": "direct string answer"}]
        result = _extract_output_messages(messages)
        assert len(result) == 1
        assert result[0].parts[0].content == "direct string answer"

    def test_dict_content(self):
        """_extract_output_messages with dict content containing 'content' key."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _extract_output_messages,
        )

        messages = [
            {"role": "assistant", "content": {"content": "nested answer"}}
        ]
        result = _extract_output_messages(messages)
        assert len(result) == 1
        assert result[0].parts[0].content == "nested answer"

    def test_non_dict_non_string_content(self):
        """_extract_output_messages with unexpected content type."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _extract_output_messages,
        )

        messages = [{"role": "assistant", "content": 12345}]
        result = _extract_output_messages(messages)
        assert len(result) == 1
        # content should be empty string (no branch matches)
        assert result[0].parts[0].content == ""

    def test_non_dict_last_message(self):
        """_extract_output_messages with non-dict last message."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _extract_output_messages,
        )

        messages = ["just a string"]
        result = _extract_output_messages(messages)
        assert len(result) == 1
        assert result[0].parts[0].content == ""


# ---------------------------------------------------------------------------
# utils.py: _step_to_output_messages edge cases (lines 152-158, 167-170)
# ---------------------------------------------------------------------------


class TestStepToOutputMessages:
    def test_step_with_tool_calls_invalid_json_args(self):
        """_step_to_output_messages should handle invalid JSON in tool_call args."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _step_to_output_messages,
        )

        tc = ToolCall(
            tool_name="my_tool",
            arguments="invalid json {{{",
            tool_call_id="tc_1",
        )

        step = ActionStep(
            step_status=StepStatus.CONTINUE,
            content="thinking",
            tool_calls=[tc],
        )

        result = _step_to_output_messages(step)
        assert len(result) == 1
        assert result[0].finish_reason == "tool_calls"
        # Should have Text part + ToolCall part
        assert len(result[0].parts) == 2

    def test_step_with_tool_call_results(self):
        """_step_to_output_messages should handle tool_call_results."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _step_to_output_messages,
        )

        tcr = ToolCallResult(
            tool_call_id="tc_1",
            content="search results here",
        )

        step = ActionStep(
            step_status=StepStatus.CONTINUE,
            content="response",
            tool_call_results=[tcr],
        )

        result = _step_to_output_messages(step)
        assert len(result) == 1
        # Should have Text part + ToolCallResponse part
        assert len(result[0].parts) == 2

    def test_step_with_tool_call_results_error_marker(self):
        """_step_to_output_messages should handle tool_call_results with error_marker."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _step_to_output_messages,
        )

        # Use a dict for error_marker since the source code calls .get("message")
        tcr = ToolCallResult(
            tool_call_id="tc_err",
            content=None,
            error_marker={"message": "tool failed"},
        )

        step = ActionStep(
            step_status=StepStatus.CONTINUE,
            content=None,
            tool_call_results=[tcr],
        )

        result = _step_to_output_messages(step)
        assert len(result) == 1
        # Should have ToolCallResponse part (with error as response)
        # No content, so parts should contain the ToolCallResponse
        found_response_part = False
        for part in result[0].parts:
            if hasattr(part, "response") and part.response == "tool failed":
                found_response_part = True
        assert found_response_part

    def test_step_no_content_no_tool_calls(self):
        """_step_to_output_messages with empty step produces default Text part."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _step_to_output_messages,
        )

        step = ActionStep(
            step_status=StepStatus.FINISHED,
            content=None,
            tool_calls=[],
            tool_call_results=[],
        )

        result = _step_to_output_messages(step)
        assert len(result) == 1
        assert result[0].finish_reason == "stop"
        # Should have a default Text(content="") part
        assert len(result[0].parts) == 1

    def test_step_with_dict_tool_call_arguments(self):
        """_step_to_output_messages should pass through dict arguments."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _step_to_output_messages,
        )

        tc = ToolCall(
            tool_name="my_tool",
            arguments={"key": "value"},
            tool_call_id="tc_dict",
        )

        step = ActionStep(
            step_status=StepStatus.CONTINUE,
            tool_calls=[tc],
        )

        result = _step_to_output_messages(step)
        assert len(result) == 1
        assert result[0].finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# utils.py: _convert_tools_desc edge cases
# ---------------------------------------------------------------------------


class TestConvertToolsDesc:
    def test_empty_tools_desc_returns_none(self):
        """_convert_tools_desc with no function-type entries returns None."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _convert_tools_desc,
        )

        result = _convert_tools_desc([{"type": "other"}])
        assert result is None

    def test_mixed_tools_desc(self):
        """_convert_tools_desc filters to function type only."""
        from opentelemetry.instrumentation.widesearch.utils import (
            _convert_tools_desc,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                },
            },
            {"type": "retrieval"},  # not function
        ]

        result = _convert_tools_desc(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "search"
