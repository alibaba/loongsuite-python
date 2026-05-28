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

"""Tests for agent_wrappers.py -- DefaultAgentRunWrapper and DefaultAgentStepWrapper __call__.

Covers the main AGENT and STEP span-creation paths (lines 45-53, 71-139, 172-203)
that are the biggest coverage gap.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.minisweagent.config import ENTRY_SPAN_ACTIVE


def _wrappers():
    """Lazy import so stub modules from conftest are in place."""
    from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
        DefaultAgentRunWrapper,
        DefaultAgentStepWrapper,
        _populate_invoke_from_agent,
    )

    return (
        DefaultAgentRunWrapper,
        DefaultAgentStepWrapper,
        _populate_invoke_from_agent,
    )


def _make_agent(
    messages=None, model_name="gpt-4o", step_limit=0, cost_limit=0.0
):
    """Create a minimal agent stub for wrapper tests."""
    cfg = type("Cfg", (), {"model_name": model_name})()
    model = type("Model", (), {"config": cfg})()
    config = type(
        "Config", (), {"step_limit": step_limit, "cost_limit": cost_limit}
    )()

    agent = type(
        "Agent",
        (),
        {
            "__module__": "minisweagent.agents.default",
            "messages": messages or [],
            "model": model,
            "config": config,
            "n_calls": 0,
            "cost": 0.0,
        },
    )()
    return agent


# =====================================================================
# _populate_invoke_from_agent  (lines 45-53)
# =====================================================================


class TestPopulateInvokeFromAgent:
    """Tests for the _populate_invoke_from_agent helper."""

    def test_success_path(self):
        _, _, populate = _wrappers()

        agent = _make_agent(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Fix bug"},
                {"role": "assistant", "content": "Done."},
            ]
        )

        inv = type(
            "Inv",
            (),
            {
                "system_instruction": None,
                "input_messages": None,
                "output_messages": None,
                "tool_definitions": None,
            },
        )()

        populate(inv, agent)

        assert inv.system_instruction is not None
        assert inv.input_messages is not None
        assert inv.output_messages is not None
        assert inv.tool_definitions is not None

    def test_exception_path_returns_early(self):
        _, _, populate = _wrappers()

        inv = type(
            "Inv",
            (),
            {
                "system_instruction": None,
                "input_messages": None,
                "output_messages": None,
                "tool_definitions": None,
            },
        )()

        with patch(
            "opentelemetry.instrumentation.minisweagent.internal.agent_wrappers.build_invoke_agent_payload",
            side_effect=Exception("payload build failed"),
        ):
            populate(inv, _make_agent())

        # Fields remain None because exception was caught
        assert inv.system_instruction is None


# =====================================================================
# DefaultAgentRunWrapper.__call__  (lines 71-139)
# =====================================================================


class TestDefaultAgentRunWrapperCall:
    """Tests for the AGENT invoke_agent span wrapper."""

    def _make_wrapper(self):
        RunWrapper, _, _ = _wrappers()
        tracer = trace_api.get_tracer("test")
        return RunWrapper(tracer)

    # --- success paths ---

    def test_success_with_task_in_args(self):
        wrapper = self._make_wrapper()
        agent = _make_agent(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Fix bug"},
            ]
        )

        def run(task="", **kw):
            return {"exit_status": "submitted", "submission": "done"}

        result = wrapper(run, agent, ("Fix bug",), {})
        assert result == {"exit_status": "submitted", "submission": "done"}

    def test_success_with_task_in_kwargs(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            return {"exit_status": "ok"}

        result = wrapper(run, agent, (), {"task": "My task"})
        assert result == {"exit_status": "ok"}

    def test_success_with_empty_task(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(**kw):
            return {}

        result = wrapper(run, agent, (), {})
        assert result == {}

    def test_success_with_none_task_kwarg(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task=None, **kw):
            return {"ok": True}

        result = wrapper(run, agent, (), {"task": None})
        assert result == {"ok": True}

    def test_success_non_dict_result(self):
        """When wrapped returns a non-dict, the exit_status/submission block is skipped."""
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            return "string result"

        result = wrapper(run, agent, ("task",), {})
        assert result == "string result"

    def test_success_dict_without_exit_status(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            return {"some_key": "value"}

        result = wrapper(run, agent, ("task",), {})
        assert result == {"some_key": "value"}

    def test_success_with_long_task_preview(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            return {}

        long_task = "x" * 300
        result = wrapper(run, agent, (long_task,), {})
        assert result == {}

    def test_success_with_long_submission(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        long_sub = "y" * 300

        def run(task="", **kw):
            return {"exit_status": "ok", "submission": long_sub}

        result = wrapper(run, agent, ("task",), {})
        assert result["submission"] == long_sub

    # --- ENTRY_SPAN_ACTIVE behaviour ---

    def test_no_entry_created_when_already_active(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        token = ENTRY_SPAN_ACTIVE.set(True)
        try:

            def run(task="", **kw):
                return {"exit_status": "ok"}

            result = wrapper(run, agent, ("task",), {})
            assert result == {"exit_status": "ok"}
        finally:
            ENTRY_SPAN_ACTIVE.reset(token)

    def test_entry_span_reset_after_success(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        assert ENTRY_SPAN_ACTIVE.get() is False

        def run(task="", **kw):
            # During execution, ENTRY_SPAN_ACTIVE should be True
            assert ENTRY_SPAN_ACTIVE.get() is True
            return {}

        wrapper(run, agent, ("task",), {})
        assert ENTRY_SPAN_ACTIVE.get() is False

    def test_entry_span_reset_after_exception(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        assert ENTRY_SPAN_ACTIVE.get() is False

        def run(task="", **kw):
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            wrapper(run, agent, ("task",), {})

        assert ENTRY_SPAN_ACTIVE.get() is False

    # --- exception paths ---

    def test_exception_triggers_fail(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            raise RuntimeError("run failed")

        with pytest.raises(RuntimeError, match="run failed"):
            wrapper(run, agent, ("task",), {})

    def test_base_exception_triggers_stop_not_fail(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            wrapper(run, agent, ("task",), {})

    def test_exception_with_entry_already_active(self):
        """Exception when ENTRY_SPAN_ACTIVE is True -- no entry span to fail."""
        wrapper = self._make_wrapper()
        agent = _make_agent()

        token = ENTRY_SPAN_ACTIVE.set(True)
        try:

            def run(task="", **kw):
                raise ValueError("fail with entry active")

            with pytest.raises(ValueError, match="fail with entry active"):
                wrapper(run, agent, ("task",), {})
        finally:
            ENTRY_SPAN_ACTIVE.reset(token)

    def test_base_exception_with_entry_already_active(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()

        token = ENTRY_SPAN_ACTIVE.set(True)
        try:

            def run(task="", **kw):
                raise KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                wrapper(run, agent, ("task",), {})
        finally:
            ENTRY_SPAN_ACTIVE.reset(token)

    def test_exception_populate_failure_is_suppressed(self):
        """If _populate_invoke_from_agent fails during exception handling, it is suppressed."""
        wrapper = self._make_wrapper()
        agent = _make_agent()

        def run(task="", **kw):
            raise RuntimeError("run failed")

        with patch(
            "opentelemetry.instrumentation.minisweagent.internal.agent_wrappers._populate_invoke_from_agent",
            side_effect=Exception("populate error"),
        ):
            with pytest.raises(RuntimeError, match="run failed"):
                wrapper(run, agent, ("task",), {})

    # --- model extraction ---

    def test_request_model_populated(self):
        wrapper = self._make_wrapper()
        agent = _make_agent(model_name="claude-3-opus")

        def run(task="", **kw):
            return {}

        result = wrapper(run, agent, ("task",), {})
        assert result == {}


# =====================================================================
# DefaultAgentStepWrapper.__call__  (lines 172-203)
# =====================================================================


class TestDefaultAgentStepWrapperCall:
    """Tests for the ReAct STEP span wrapper."""

    def _make_wrapper(self):
        _, StepWrapper, _ = _wrappers()
        tracer = trace_api.get_tracer("test")
        return StepWrapper(tracer)

    def test_normal_step_execution(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = 0

        call_log = []

        def step():
            call_log.append("called")

        wrapper(step, agent, (), {})
        assert call_log == ["called"]
        assert agent._otel_msw_round == 1

    def test_round_counter_increments(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = 5

        wrapper(lambda: None, agent, (), {})
        assert agent._otel_msw_round == 6

    def test_round_counter_defaults_when_missing(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        # _otel_msw_round not set at all

        wrapper(lambda: None, agent, (), {})
        assert agent._otel_msw_round == 1

    def test_round_counter_handles_none(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = None

        wrapper(lambda: None, agent, (), {})
        assert agent._otel_msw_round == 1

    def test_interrupt_agent_flow(self):
        from minisweagent.exceptions import InterruptAgentFlow

        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = 0

        def step():
            raise InterruptAgentFlow("flow interrupted")

        with pytest.raises(InterruptAgentFlow):
            wrapper(step, agent, (), {})

    def test_regular_exception(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = 0

        def step():
            raise RuntimeError("step failed")

        with pytest.raises(RuntimeError, match="step failed"):
            wrapper(step, agent, (), {})

    def test_base_exception(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = 0

        def step():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            wrapper(step, agent, (), {})

    def test_limits_exceeded_bypasses_span(self):
        """When limits are exceeded, wrapped is called directly without span."""
        wrapper = self._make_wrapper()
        agent = _make_agent(step_limit=5)
        agent.n_calls = 10  # exceeded

        call_log = []

        def step():
            call_log.append("called")

        wrapper(step, agent, (), {})
        assert call_log == ["called"]

    def test_step_returns_result(self):
        wrapper = self._make_wrapper()
        agent = _make_agent()
        agent._otel_msw_round = 0

        def step():
            return "step result"

        result = wrapper(step, agent, (), {})
        assert result == "step result"
