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

"""Cross-thread / cross-loop trace continuity tests for V0 wrappers.

These tests model the *real* OpenHands V0 runtime behaviour: events are
delivered by ``EventStream`` via a ``ThreadPoolExecutor`` and the controller
processes them with ``asyncio.get_event_loop().run_until_complete(...)`` —
which spins a brand-new asyncio loop in the worker thread. Without our
session-context bridge, STEP / TOOL spans would start fresh root traces.

We assert:

* All ENTRY / AGENT / STEP / TOOL spans share the **same** ``trace_id``.
* Parent-child wiring is correct (STEP is parented under AGENT, TOOL too).
* The session-context store is cleaned up after the entry returns.
* GenAI semantic-convention I/O attributes are populated when content
  capture is enabled.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

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


def _drive_step_in_worker_thread(controller, runtime, action) -> None:
    """Reproduce the V0 EventStream → ThreadPoolExecutor → run_until_complete path.

    The worker thread (a) has no shared asyncio loop with the caller and
    (b) has a *fresh* ``contextvars.Context`` (Python copies the snapshot
    at submit-time, but the snapshot is from this test thread — the same
    fresh context the real EventStream queue thread would have).
    """
    barrier = threading.Event()
    err: list[BaseException] = []

    def _worker():
        try:
            # New event loop per worker — exactly what V0 does.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(controller._step())
                # Run_action is sync — call it directly inside the worker.
                runtime.run_action(action)
            finally:
                loop.close()
        except BaseException as exc:  # pragma: no cover - surfaced via err
            err.append(exc)
        finally:
            barrier.set()

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_worker)
    fut.result(timeout=5)
    pool.shutdown(wait=True)
    barrier.wait(timeout=5)
    if err:
        raise err[0]


def test_all_spans_share_one_trace_id_across_threads(instrumented_v0):
    """The whole V0 trace must collapse onto a single trace_id even when
    STEP / TOOL run in fresh worker threads with fresh asyncio loops."""
    inst, exporter = instrumented_v0

    import openhands.controller.agent_controller as ctrl_mod
    import openhands.runtime.base as rt_base

    ctrl = ctrl_mod.AgentController(sid="bench-001")
    runtime = rt_base.Runtime(sid="bench-001")
    action = rt_base.Action(action_type="run", command="ls /")

    async def _scenario():
        for _ in range(2):
            _drive_step_in_worker_thread(ctrl, runtime, action)
        await ctrl.close()

    asyncio.run(_scenario())

    spans = exporter.get_finished_spans()
    by_kind = {
        kind: _spans_by_kind_attr(exporter, kind)
        for kind in ("ENTRY", "AGENT", "STEP", "TOOL")
    }

    assert len(by_kind["ENTRY"]) == 1
    assert len(by_kind["AGENT"]) == 1
    assert len(by_kind["STEP"]) == 2
    assert len(by_kind["TOOL"]) == 2

    entry = by_kind["ENTRY"][0]
    agent = by_kind["AGENT"][0]
    trace_id = entry.context.trace_id

    # Same trace_id for every span
    for s in spans:
        assert s.context.trace_id == trace_id, (
            f"span {s.name!r} (kind={s.attributes.get('gen_ai.span.kind')}) "
            f"has trace_id {s.context.trace_id} but expected {trace_id}"
        )

    # Parent-child links: AGENT under ENTRY, STEP under AGENT
    assert (
        agent.parent is not None
        and agent.parent.span_id == entry.context.span_id
    )
    for s in by_kind["STEP"]:
        assert (
            s.parent is not None and s.parent.span_id == agent.context.span_id
        )
    # TOOL spans are children of their respective STEP spans
    step_span_ids = {s.context.span_id for s in by_kind["STEP"]}
    for t in by_kind["TOOL"]:
        assert t.parent is not None and t.parent.span_id in step_span_ids


def test_session_context_cleared_after_entry(instrumented_v0):
    """The per-sid stash must not leak across runs."""
    inst, exporter = instrumented_v0

    import openhands.core.main as main_mod

    from opentelemetry.instrumentation.openhands.internal import (
        session_context,
    )

    async def _scenario():
        await main_mod.run_controller(
            config=None,
            initial_user_action=type(
                "Msg", (), {"content": "x", "source": "user"}
            )(),
            sid="ephemeral-sid",
        )

    asyncio.run(_scenario())
    assert session_context.get_context("ephemeral-sid") is None


def test_io_attributes_on_entry_agent_step(instrumented_v0):
    """Verify GenAI / OpenInference I/O attributes are populated."""
    inst, exporter = instrumented_v0

    import openhands.controller.agent_controller as ctrl_mod
    import openhands.runtime.base as rt_base

    # Seed history with a *MessageAction*-named instance — that's the type
    # name the AGENT wrapper looks for when computing input.messages.
    class MessageAction:
        content = "do the thing"
        source = "user"

    ctrl = ctrl_mod.AgentController(sid="io-sid")
    runtime = rt_base.Runtime(sid="io-sid")
    action = rt_base.Action(action_type="run", command="cat /etc/hosts")

    ctrl.state.history = [MessageAction()]

    async def _scenario():
        await ctrl._step()
        runtime.run_action(action)
        await ctrl.close()

    asyncio.run(_scenario())

    entry = _spans_by_kind_attr(exporter, "ENTRY")[0]
    agent = _spans_by_kind_attr(exporter, "AGENT")[0]
    step = _spans_by_kind_attr(exporter, "STEP")[0]
    tool = _spans_by_kind_attr(exporter, "TOOL")[0]

    # ENTRY
    assert entry.attributes.get("gen_ai.framework") == "openhands"
    assert entry.attributes.get("gen_ai.system") == "openhands"
    assert entry.attributes.get("gen_ai.session.id") == "io-sid"
    # ENTRY no longer mirrors OpenInference input.value/output.value;
    # the same payload is still available via gen_ai.input.messages.
    assert "input.value" not in entry.attributes
    assert "output.value" not in entry.attributes
    assert entry.attributes.get("gen_ai.input.messages")
    assert "do the thing" in entry.attributes.get("gen_ai.input.messages")

    # AGENT
    agent_input = agent.attributes.get("gen_ai.input.messages")
    assert agent_input
    assert "do the thing" in agent_input
    # AGENT messages must be valid JSON (not Python repr with single quotes).
    agent_msgs = json.loads(agent_input)
    assert isinstance(agent_msgs, list) and agent_msgs
    for msg in agent_msgs:
        assert isinstance(msg, dict)
        for part in msg.get("parts", []):
            assert isinstance(part, dict), (
                "AGENT message parts must be JSON objects, not stringified dicts"
            )
    assert "gen_ai.system_instruction" not in agent.attributes
    assert "input.value" not in agent.attributes
    assert "output.value" not in agent.attributes
    assert agent.attributes.get("gen_ai.session.id") == "io-sid"

    # STEP
    assert step.attributes.get("input.value")
    assert step.attributes.get("output.value")
    assert step.attributes.get("gen_ai.output.messages")
    assert step.attributes.get("openhands.action.type") == "run"
    out = step.attributes.get("output.value")
    assert "tool_calls" in out and "echo step" in out

    # TOOL spans: arguments only via gen_ai.tool.call.arguments; no input/output.value.
    assert tool.attributes.get("gen_ai.tool.name") == "bash"
    assert "input.value" not in tool.attributes
    assert "output.value" not in tool.attributes
    args = json.loads(tool.attributes["gen_ai.tool.call.arguments"])
    assert args.get("command") == "cat /etc/hosts"
    result = tool.attributes.get("gen_ai.tool.call.result")
    assert result
    assert "exit_code" in result
