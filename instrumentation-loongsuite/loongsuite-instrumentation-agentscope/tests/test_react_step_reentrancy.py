# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# -*- coding: utf-8 -*-
"""Unit tests for React step hook reentrancy (nested _reasoning / _acting)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.agentscope._wrapper import (
    _REACT_STATE,
    _make_post_acting_hook,
    _make_post_reasoning_hook,
    _make_pre_acting_hook,
    _make_pre_reasoning_hook,
    _ReactStepState,
)


class _MsgWithToolBlocks:
    def __init__(self, n_tools: int):
        self._n = n_tools

    def get_content_blocks(self, kind: str):
        if kind != "tool_use":
            return []
        return [{"type": "tool_use"}] * self._n


@pytest.fixture
def react_state():
    """Per-test ReactStepState bound to the ``_REACT_STATE`` ContextVar.

    Hooks read state via ``_REACT_STATE.get()`` and validate ``state.owner
    is agent_self``. Tests must set ``react_state.owner = agent`` before
    firing hooks so the owner check passes.
    """
    state = _ReactStepState(original_context=None)
    token = _REACT_STATE.set(state)
    try:
        yield state
    finally:
        _REACT_STATE.reset(token)


def test_nested_reasoning_only_outermost_opens_step_and_counts_tools(
    react_state,
):
    handler = MagicMock()
    state = react_state
    agent = SimpleNamespace()
    react_state.owner = agent
    pre = _make_pre_reasoning_hook(handler)
    post = _make_post_reasoning_hook(handler)

    pre(agent, {})
    assert state.reasoning_nesting == 1
    assert handler.start_react_step.call_count == 1
    assert state.react_round == 1

    pre(agent, {})
    assert state.reasoning_nesting == 2
    assert handler.start_react_step.call_count == 1

    out = _MsgWithToolBlocks(2)
    post(agent, {}, out)
    assert state.reasoning_nesting == 1
    assert state.pending_acting_count == 0

    post(agent, {}, out)
    assert state.reasoning_nesting == 0
    assert state.pending_acting_count == 2


def test_nested_acting_only_outermost_decrements_pending(react_state):
    handler = MagicMock()
    state = react_state
    state.pending_acting_count = 2
    state.active_step = MagicMock()
    agent = SimpleNamespace()
    react_state.owner = agent
    pre_a = _make_pre_acting_hook()
    post_a = _make_post_acting_hook(handler)

    pre_a(agent, {})
    assert state.acting_nesting == 1
    pre_a(agent, {})
    assert state.acting_nesting == 2

    post_a(agent, {}, None)
    assert state.acting_nesting == 1
    assert state.pending_acting_count == 2

    post_a(agent, {}, None)
    assert state.acting_nesting == 0
    assert state.pending_acting_count == 1
    assert handler.stop_react_step.call_count == 0

    pre_a(agent, {})
    pre_a(agent, {})
    post_a(agent, {}, None)
    post_a(agent, {}, None)
    assert state.pending_acting_count == 0
    assert handler.stop_react_step.call_count == 1


def test_hook_no_ops_when_react_state_not_set():
    """When ``_REACT_STATE`` is unset (default None), hooks must not crash
    or call the handler. This is the safety valve that lets the same
    hook callable serve concurrent contexts: only contexts that opt in
    via ``_REACT_STATE.set(...)`` see telemetry.
    """
    handler = MagicMock()
    pre = _make_pre_reasoning_hook(handler)
    post = _make_post_reasoning_hook(handler)
    pre_a = _make_pre_acting_hook()
    post_a = _make_post_acting_hook(handler)

    agent = SimpleNamespace()
    # _REACT_STATE intentionally not set in this context.
    assert _REACT_STATE.get() is None

    pre(agent, {})
    post(agent, {}, _MsgWithToolBlocks(1))
    pre_a(agent, {})
    post_a(agent, {}, None)

    assert handler.start_react_step.call_count == 0
    assert handler.stop_react_step.call_count == 0


def test_hook_no_ops_when_owner_mismatch():
    """Hooks must ignore state owned by a different agent instance.

    Regression: when a child agent is called from within a parent agent's
    execution context, both share the same ContextVar value.  Before the
    owner check, the child hook would read and mutate the parent's
    _ReactStepState (clearing pending_acting_count, bumping react_round),
    causing the parent step to close prematurely.
    """
    handler = MagicMock()
    parent_agent = SimpleNamespace()
    child_agent = SimpleNamespace()

    parent_state = _ReactStepState(original_context=None, owner=parent_agent)
    token = _REACT_STATE.set(parent_state)
    try:
        pre = _make_pre_reasoning_hook(handler)
        post = _make_post_reasoning_hook(handler)
        pre_a = _make_pre_acting_hook()
        post_a = _make_post_acting_hook(handler)

        # Child hooks fire in the parent's context — must all no-op.
        pre(child_agent, {})
        post(child_agent, {}, _MsgWithToolBlocks(3))
        pre_a(child_agent, {})
        post_a(child_agent, {}, None)

        assert parent_state.react_round == 0, (
            "child hook mutated parent react_round"
        )
        assert parent_state.reasoning_nesting == 0
        assert parent_state.acting_nesting == 0
        assert parent_state.pending_acting_count == 0
        assert handler.start_react_step.call_count == 0
        assert handler.stop_react_step.call_count == 0
    finally:
        _REACT_STATE.reset(token)
