"""ARMS GenAI semconv §Tool conformance tests for the V0 TOOL wrapper.

I/O capture is always on (no env-var gating, no truncation), so the
TOOL span must carry every attribute the spec calls out — both
required and recommended — on every run.
"""

from __future__ import annotations

import asyncio
import json

import pytest


def _spans_by_kind(exporter, kind: str):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("gen_ai.span.kind") == kind
    ]


@pytest.fixture
def instrumented(tracer_provider, stub_openhands_v0_modules):
    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor
    from opentelemetry.instrumentation.openhands.internal import session_context

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


def _run_one_tool_call(rt_base, ctrl_mod, loop_mod, main_mod):
    """Drive a single ENTRY → AGENT → STEP → TOOL flow."""
    ctrl = ctrl_mod.AgentController(sid="tool-sid")
    runtime = rt_base.Runtime(sid="tool-sid")

    tcm = rt_base.ToolCallMetadata(
        function_name="execute_bash",
        tool_call_id="call_abc123",
        arguments={"command": "ls /tmp", "thought": "list temp"},
    )
    action = rt_base.Action(
        action_type="run",
        command="ls /tmp",
        tool_call_metadata=tcm,
    )

    class MessageAction:
        content = "list /tmp"
        source = "user"

    async def _inner(_c, _r):
        await ctrl._step()
        runtime.run_action(action)

    loop_mod._test_inner_callback = _inner
    main_mod._test_inner_args = (ctrl, runtime)

    async def _scenario():
        await main_mod.run_controller(
            config=None,
            initial_user_action=MessageAction(),
            sid="tool-sid",
        )

    try:
        asyncio.run(_scenario())
    finally:
        loop_mod._test_inner_callback = None
        main_mod._test_inner_args = None


def test_tool_span_carries_all_arms_required_attributes(instrumented):
    inst, exporter = instrumented

    import openhands.controller.agent_controller as ctrl_mod
    import openhands.core.loop as loop_mod
    import openhands.core.main as main_mod
    import openhands.runtime.base as rt_base

    _run_one_tool_call(rt_base, ctrl_mod, loop_mod, main_mod)

    tools = _spans_by_kind(exporter, "TOOL")
    assert len(tools) == 1
    tool = tools[0]
    attrs = tool.attributes

    # Required
    assert attrs["gen_ai.span.kind"] == "TOOL"
    assert attrs["gen_ai.operation.name"] == "execute_tool"

    # Span name should be `execute_tool {tool_name}`
    assert tool.name == "execute_tool execute_bash"

    # Recommended attributes
    assert attrs["gen_ai.tool.name"] == "execute_bash"
    assert attrs["gen_ai.tool.type"] == "function"
    assert attrs["gen_ai.tool.call.id"] == "call_abc123"
    assert attrs.get("gen_ai.tool.description") == (
        "Run a bash command on the runtime sandbox."
    )

    # Arguments should be the BARE JSON dict, not the wrapping
    # {"tool": ..., "arguments": ...} envelope.
    args_json = attrs.get("gen_ai.tool.call.arguments")
    assert args_json is not None
    args = json.loads(args_json)
    assert args == {"command": "ls /tmp", "thought": "list temp"}

    # Result should reflect the observation.
    result_json = attrs.get("gen_ai.tool.call.result")
    assert result_json is not None
    result = json.loads(result_json)
    assert result.get("exit_code") == 0
    assert "observation" in result


def test_tool_span_falls_back_to_action_field_when_no_tool_call_metadata(
    instrumented,
):
    """If the action wasn't generated from an LLM tool call (e.g. a
    user-initiated agent.action), the wrapper should still produce a
    sensible ``gen_ai.tool.name`` derived from the action type."""
    inst, exporter = instrumented

    import openhands.controller.agent_controller as ctrl_mod
    import openhands.core.loop as loop_mod
    import openhands.core.main as main_mod
    import openhands.runtime.base as rt_base

    ctrl = ctrl_mod.AgentController(sid="tool-fallback-sid")
    runtime = rt_base.Runtime(sid="tool-fallback-sid")
    action = rt_base.Action(action_type="run", command="echo hi")

    class MessageAction:
        content = "say hi"
        source = "user"

    async def _inner(_c, _r):
        await ctrl._step()
        runtime.run_action(action)

    loop_mod._test_inner_callback = _inner
    main_mod._test_inner_args = (ctrl, runtime)

    async def _scenario():
        await main_mod.run_controller(
            config=None,
            initial_user_action=MessageAction(),
            sid="tool-fallback-sid",
        )

    try:
        asyncio.run(_scenario())
    finally:
        loop_mod._test_inner_callback = None
        main_mod._test_inner_args = None

    tool = _spans_by_kind(exporter, "TOOL")[0]
    attrs = tool.attributes

    # Action.action == "run" → tool name "bash"
    assert attrs["gen_ai.tool.name"] == "bash"
    assert tool.name == "execute_tool bash"
    # No tool-call id when the action wasn't from an LLM call
    assert attrs.get("gen_ai.tool.call.id", "") == ""
    # Arguments still produced from the action's fields
    args = json.loads(attrs["gen_ai.tool.call.arguments"])
    assert args.get("command") == "echo hi"


def test_agent_span_emits_tool_definitions(instrumented):
    """AGENT span should advertise the agent's available tools per the
    ARMS GenAI semconv §Agent → ``gen_ai.tool.definitions``."""
    inst, exporter = instrumented

    import openhands.controller.agent_controller as ctrl_mod
    import openhands.core.loop as loop_mod
    import openhands.core.main as main_mod
    import openhands.runtime.base as rt_base

    _run_one_tool_call(rt_base, ctrl_mod, loop_mod, main_mod)

    agent = _spans_by_kind(exporter, "AGENT")[0]
    defs_json = agent.attributes.get("gen_ai.tool.definitions")
    assert defs_json, "AGENT span should set gen_ai.tool.definitions"
    defs = json.loads(defs_json)
    assert isinstance(defs, list) and defs
    assert defs[0]["type"] == "function"
    assert defs[0]["name"] == "execute_bash"
    assert "description" in defs[0]
