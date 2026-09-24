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

"""Tests for V0 (Legacy CodeAct) wrappers.

We exercise the four V0 patches (``run_controller``, ``run_agent_until_done``,
``AgentController._step``, ``Runtime.run_action``) and assert that:

* The ``ENTRY → AGENT → STEP → TOOL`` span tree is produced.
* Parent-child linkage is correct.
* Per-action ``gen_ai.tool.name`` is mapped from the V0 ``action`` field.
"""

from __future__ import annotations

import asyncio

import pytest


def _spans_by_kind_attr(exporter, kind: str):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("gen_ai.span.kind") == kind
    ]


@pytest.fixture
def instrumented_v0(tracer_provider, stub_openhands_v0_modules):
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor
    from opentelemetry.instrumentation.openhands.internal import (
        session_context,
    )

    session_context.clear_all()
    inst = OpenHandsInstrumentor()
    inst.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    try:
        yield inst, tracer_provider._exporter  # type: ignore[attr-defined]
    finally:
        try:
            inst.uninstrument()
        except Exception:
            pass
        session_context.clear_all()


def test_v0_full_span_tree(instrumented_v0):
    inst, exporter = instrumented_v0

    import openhands.controller.agent_controller as ctrl_mod
    import openhands.runtime.base as rt_base

    ctrl = ctrl_mod.AgentController()
    runtime = rt_base.Runtime()
    action = rt_base.Action(action_type="run", command="ls /")

    async def _scenario():
        for _ in range(2):
            await ctrl._step()
            runtime.run_action(action)
        await ctrl.close()

    asyncio.run(_scenario())

    entry = _spans_by_kind_attr(exporter, "ENTRY")
    agent = _spans_by_kind_attr(exporter, "AGENT")
    step = _spans_by_kind_attr(exporter, "STEP")
    tool = _spans_by_kind_attr(exporter, "TOOL")

    assert len(entry) == 1, f"unexpected ENTRY count: {len(entry)}"
    assert len(agent) == 1, f"unexpected AGENT count: {len(agent)}"
    assert len(step) == 2, f"unexpected STEP count: {len(step)}"
    assert len(tool) == 2, f"unexpected TOOL count: {len(tool)}"

    e = entry[0]
    a = agent[0]
    assert e.name == "enter openhands"
    assert e.attributes.get("gen_ai.framework") == "openhands"
    assert e.attributes.get("gen_ai.session.id") == "sid-test"
    # ENTRY span no longer carries OpenInference input.value/output.value;
    # the same payload lives on gen_ai.input.messages / gen_ai.output.messages.
    assert "input.value" not in e.attributes
    assert "output.value" not in e.attributes

    assert a.name.startswith("invoke_agent ")
    assert a.attributes.get("gen_ai.agent.name") == "CodeActAgent"
    assert a.attributes.get("gen_ai.request.model") == "qwen3-coder-plus"
    assert "gen_ai.system_instruction" not in a.attributes
    assert "input.value" not in a.attributes
    assert "output.value" not in a.attributes

    # All STEP spans share the AGENT as parent.
    for s in step:
        assert s.parent is not None
        assert s.parent.span_id == a.context.span_id
        assert s.attributes.get("gen_ai.operation.name") == "react"
        assert s.attributes.get("gen_ai.react.round") in (1, 2)

    # TOOL spans carry the expected attributes.
    for t in tool:
        assert t.attributes.get("gen_ai.tool.name") == "bash"
        assert t.attributes.get("openhands.action.type") == "run"
        assert t.attributes.get("openhands.action.exit_code") == 0


def test_v0_step_round_increments_per_controller(instrumented_v0):
    inst, exporter = instrumented_v0
    import openhands.controller.agent_controller as ctrl_mod

    ctrl_a = ctrl_mod.AgentController(sid="A")
    ctrl_b = ctrl_mod.AgentController(sid="B")

    async def _go():
        await ctrl_a._step()
        await ctrl_a._step()
        await ctrl_b._step()
        await ctrl_a.close()
        await ctrl_b.close()

    asyncio.run(_go())

    step_spans = _spans_by_kind_attr(exporter, "STEP")
    assert len(step_spans) == 3
    rounds_a = sorted(
        s.attributes.get("gen_ai.react.round")
        for s in step_spans
        if s.attributes.get("gen_ai.session.id") == "A"
    )
    rounds_b = sorted(
        s.attributes.get("gen_ai.react.round")
        for s in step_spans
        if s.attributes.get("gen_ai.session.id") == "B"
    )
    assert rounds_a == [1, 2]
    assert rounds_b == [1]


def test_v0_runtime_error_observation_marks_span(instrumented_v0):
    inst, exporter = instrumented_v0
    import openhands.runtime.base as rt_base

    runtime = rt_base.Runtime()

    class _ErrAction:
        action = "run"
        command = "false"

    # Use the conftest hook to make the next run_action return an error obs.
    err_obs = rt_base.Observation(exit_code=2)
    runtime._next_observation = err_obs

    runtime.run_action(_ErrAction())

    tool_spans = _spans_by_kind_attr(exporter, "TOOL")
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes.get("openhands.action.exit_code") == 2
    assert span.status.status_code.name == "ERROR"


def test_v0_run_controller_cancelled_marks_span_error(instrumented_v0):
    """``asyncio.CancelledError`` is a BaseException and marks ENTRY as ERROR."""
    _, exporter = instrumented_v0
    import openhands.core.main as main_mod

    main_mod._test_raise_cancelled = True
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                main_mod.run_controller(
                    config=None,
                    initial_user_action=type(
                        "Msg", (), {"content": "hello"}
                    )(),
                    sid="sid-cancel",
                )
            )
    finally:
        main_mod._test_raise_cancelled = False

    entry = _spans_by_kind_attr(exporter, "ENTRY")
    assert len(entry) == 1
    assert entry[0].status.status_code.name == "ERROR"
