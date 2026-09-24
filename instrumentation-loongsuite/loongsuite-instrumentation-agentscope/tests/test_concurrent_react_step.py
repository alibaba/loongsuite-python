# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# -*- coding: utf-8 -*-
"""Concurrency regression tests for AgentScope ReAct step instrumentation.

Background
----------
Before the ContextVar fix, per-call ``_ReactStepState`` was attached to the
agent instance (``call_self._react_step_state = state``) and ReAct hooks
were registered with a per-call uuid suffix. Two concurrent
``await agent(...)`` invocations on the same instance therefore:

1. Overwrote each other's ``active_step`` reference, leaking a step span
   that was never closed by ``stop_react_step``.
2. Each registered four hooks under unique names, so AgentScope's framework
   fired *both* hook sets on every ``_reasoning`` / ``_acting`` callback,
   doubling the start_react_step / stop_react_step counts and corrupting
   the round counter.
3. ``del call_self._react_step_state`` in the first invocation's ``finally``
   block removed the *other* invocation's state mid-flight.

These tests exercise the same hook plumbing that
``AgentScopeAgentWrapper.async_wrapped_call`` uses, but drive it directly so
the test does not depend on a real AgentScope ``ReActAgent``. They lock in
the ContextVar isolation + reference-counted hook registration contract.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from opentelemetry.instrumentation.agentscope._wrapper import (
    _REACT_HOOK_NAME,
    _REACT_HOOK_REGISTRY,
    _REACT_STATE,
    _acquire_react_hooks,
    _ReactStepState,
    _release_react_hooks,
)


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------
class _ToolUseBlock:
    pass


class _Reasoning:
    """Minimal ``ChatResponse``-like stub for ``post_reasoning`` hooks."""

    def __init__(self, n_tools: int) -> None:
        self._n = n_tools

    def get_content_blocks(self, kind: str):
        return [_ToolUseBlock()] * self._n if kind == "tool_use" else []


class _FakeAgent:
    """Mimics the slice of ``AgentBase`` used by the hook registration path.

    Real ``AgentBase`` exposes ``register_instance_hook`` /
    ``remove_instance_hook`` and an ``_instance_pre_reasoning_hooks`` attribute
    used for ReAct duck-typing. The hook registry stores one callable per
    ``(hook_type, name)`` pair; firing a hook type invokes every registered
    callable in registration order.
    """

    def __init__(self) -> None:
        self._instance_pre_reasoning_hooks: dict = {}
        self._hooks: "dict[str, dict[str, callable]]" = {
            "pre_reasoning": {},
            "post_reasoning": {},
            "pre_acting": {},
            "post_acting": {},
        }

    def register_instance_hook(self, hook_type, name, fn) -> None:
        self._hooks[hook_type][name] = fn

    def remove_instance_hook(self, hook_type, name) -> None:
        self._hooks[hook_type].pop(name)

    def fire(self, hook_type: str, *args) -> None:
        for fn in list(self._hooks[hook_type].values()):
            fn(self, *args)


class _RecordingHandler:
    """Thread-safe handler stub recording start/stop_react_step round numbers."""

    def __init__(self) -> None:
        self.start_calls: list = []
        self.stop_calls: list = []
        self._lock = threading.Lock()

    def start_react_step(self, inv, context=None) -> None:
        with self._lock:
            self.start_calls.append(inv.round)

    def stop_react_step(self, inv) -> None:
        with self._lock:
            self.stop_calls.append(inv.round)

    def fail_react_step(self, inv, err) -> None:
        with self._lock:
            self.stop_calls.append(("fail", inv.round))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
async def _drive_one_invocation(
    agent: _FakeAgent,
    handler: _RecordingHandler,
    barrier: asyncio.Event,
    n_arrived: list,
    n_arrived_lock: threading.Lock,
) -> tuple:
    """Drive a single fake ``__call__`` lifecycle through the ContextVar.

    Mirrors the production wrapper:

    - ``_REACT_STATE.set(state)`` opens an isolated state slot.
    - ``_acquire_react_hooks`` registers the shared hook callables (or
      bumps the refcount when another concurrent call already did so).
    - Two ReAct rounds (one tool round, one stop round) run between the
      acquire / release boundaries.
    - ``_release_react_hooks`` + ``_REACT_STATE.reset(token)`` mirror the
      wrapper's ``finally`` cleanup.
    """
    state = _ReactStepState(owner=agent, original_context=None)
    token = _REACT_STATE.set(state)
    _acquire_react_hooks(agent, handler)
    try:
        # ----- ReAct round 1: one tool_call -> one acting iteration -----
        agent.fire("pre_reasoning", {})
        agent.fire("post_reasoning", {}, _Reasoning(n_tools=1))
        agent.fire("pre_acting", {})
        agent.fire("post_acting", {}, None)

        # Wait until both concurrent invocations finish round 1, so
        # round 2 also runs while the registry refcount is at 2 (the
        # critical window where the buggy implementation leaked spans).
        with n_arrived_lock:
            n_arrived[0] += 1
        while n_arrived[0] < 2:
            await asyncio.sleep(0)
        barrier.set()
        await barrier.wait()

        # ----- ReAct round 2: finish_reason == "stop" -----
        agent.fire("pre_reasoning", {})
        agent.fire("post_reasoning", {}, _Reasoning(n_tools=0))

        # The wrapper closes a leftover step (no acting -> finish_reason "stop").
        if state.active_step is not None:
            handler.stop_react_step(state.active_step)
            state.active_step = None

        return (
            state.react_round,
            list(handler.start_calls),
            list(handler.stop_calls),
        )
    finally:
        _release_react_hooks(agent)
        _REACT_STATE.reset(token)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_state_isolated_across_concurrent_invocations():
    """Regression: concurrent invocations must produce 4 starts / 4 stops.

    With the old instance-attribute state, the second invocation's
    ``call_self._react_step_state = state`` assignment would clobber the
    first invocation's ``active_step``, leaving its round-1 step span
    open forever (3 starts, 3 stops, broken parent/child topology).
    """
    agent = _FakeAgent()
    handler = _RecordingHandler()
    barrier = asyncio.Event()
    n_arrived = [0]
    n_arrived_lock = threading.Lock()

    results = await asyncio.gather(
        _drive_one_invocation(
            agent, handler, barrier, n_arrived, n_arrived_lock
        ),
        _drive_one_invocation(
            agent, handler, barrier, n_arrived, n_arrived_lock
        ),
    )

    # Each invocation observed its own round counter advancing 1 -> 2,
    # untouched by the sibling.
    for round_, _, _ in results:
        assert round_ == 2, (
            "react_round counter leaked across concurrent invocations: "
            f"got {round_}, expected 2"
        )

    # Two invocations * two rounds = exactly 4 starts and 4 stops, no leak.
    assert len(handler.start_calls) == 4, (
        "expected 4 start_react_step (2 invocations x 2 rounds), got "
        f"{len(handler.start_calls)}: {handler.start_calls}"
    )
    assert len(handler.stop_calls) == 4, (
        "expected 4 stop_react_step (2 invocations x 2 rounds), got "
        f"{len(handler.stop_calls)}: {handler.stop_calls} "
        "- a leaked step span indicates broken concurrency isolation"
    )

    # round=1 must fire exactly twice (once per invocation). With the old
    # uuid-suffixed double-registration bug, AgentScope would invoke both
    # hook sets and round=1 would appear 4 times.
    assert handler.start_calls.count(1) == 2, (
        f"round=1 fired {handler.start_calls.count(1)} times, expected 2 "
        "- duplicate hook registration is double-firing _reasoning"
    )
    assert handler.start_calls.count(2) == 2


@pytest.mark.asyncio
async def test_hook_registry_releases_after_last_invocation():
    """After the last concurrent invocation unwinds, hooks are removed
    and the registry entry for that agent is gone."""
    agent = _FakeAgent()
    handler = _RecordingHandler()
    barrier = asyncio.Event()
    n_arrived = [0]
    n_arrived_lock = threading.Lock()

    await asyncio.gather(
        _drive_one_invocation(
            agent, handler, barrier, n_arrived, n_arrived_lock
        ),
        _drive_one_invocation(
            agent, handler, barrier, n_arrived, n_arrived_lock
        ),
    )

    assert agent not in _REACT_HOOK_REGISTRY, (
        "hook refcount entry leaked after last invocation"
    )
    for hook_type in (
        "pre_reasoning",
        "post_reasoning",
        "pre_acting",
        "post_acting",
    ):
        assert _REACT_HOOK_NAME not in agent._hooks[hook_type], (
            f"{hook_type} hook still registered after last invocation"
        )


@pytest.mark.asyncio
async def test_hook_registry_keeps_hooks_during_overlapping_calls():
    """Hooks must stay registered while a sibling concurrent call still
    needs them; they are removed only after the **last** call unwinds.

    Note: in production, concurrent calls run in separate asyncio tasks
    with independent ContextVar copies.  This test exercises the refcount
    logic in a single context, so tokens must be reset in LIFO order
    (token_b before token_a).  The "first/second unwind" labels refer to
    the refcount transitions, not to specific production call identities.
    """
    agent = _FakeAgent()
    handler = _RecordingHandler()

    # First call enters the critical region.
    state_a = _ReactStepState(owner=agent, original_context=None)
    token_a = _REACT_STATE.set(state_a)
    _acquire_react_hooks(agent, handler)
    assert _REACT_HOOK_REGISTRY.get(agent) == 1

    # Second concurrent call enters; refcount must bump, hooks unchanged.
    state_b = _ReactStepState(owner=agent, original_context=None)
    token_b = _REACT_STATE.set(state_b)
    _acquire_react_hooks(agent, handler)
    assert _REACT_HOOK_REGISTRY.get(agent) == 2
    for hook_type in agent._hooks:
        assert _REACT_HOOK_NAME in agent._hooks[hook_type]

    # First unwind (LIFO: token_b is the most recent .set()) -> refcount
    # drops from 2 to 1, hooks must remain for the other outstanding call.
    _release_react_hooks(agent)
    _REACT_STATE.reset(token_b)
    assert _REACT_HOOK_REGISTRY.get(agent) == 1
    for hook_type in agent._hooks:
        assert _REACT_HOOK_NAME in agent._hooks[hook_type], (
            f"{hook_type} hook removed prematurely while another call "
            "was still in flight"
        )

    # Second unwind (token_a) -> refcount drops to 0, hooks removed.
    _release_react_hooks(agent)
    _REACT_STATE.reset(token_a)
    assert agent not in _REACT_HOOK_REGISTRY
    for hook_type in agent._hooks:
        assert _REACT_HOOK_NAME not in agent._hooks[hook_type]


@pytest.mark.asyncio
async def test_acquire_rolls_back_on_partial_registration_failure():
    """When AgentScope rejects one of the four hook registrations, the
    already-installed hooks must be rolled back and the refcount must
    stay at zero so the agent ends up un-instrumented and the exception
    propagates to the caller (i.e. ``acquire`` is all-or-nothing)."""

    class _FailingAgent(_FakeAgent):
        def register_instance_hook(self, hook_type, name, fn):
            if hook_type == "pre_acting":
                raise RuntimeError("simulated registration failure")
            super().register_instance_hook(hook_type, name, fn)

    agent = _FailingAgent()
    handler = _RecordingHandler()

    with pytest.raises(RuntimeError, match="simulated registration failure"):
        _acquire_react_hooks(agent, handler)

    # No partial state left behind.
    assert agent not in _REACT_HOOK_REGISTRY
    for hook_type in agent._hooks:
        assert _REACT_HOOK_NAME not in agent._hooks[hook_type], (
            f"{hook_type} hook leaked through after acquire rollback"
        )

    # A subsequent successful acquire on a fresh agent must work fine -
    # i.e. the failed one did not poison module-level state.
    fresh_agent = _FakeAgent()
    _acquire_react_hooks(fresh_agent, handler)
    try:
        assert _REACT_HOOK_REGISTRY.get(fresh_agent) == 1
    finally:
        _release_react_hooks(fresh_agent)


@pytest.mark.asyncio
async def test_release_without_prior_acquire_is_safe_noop():
    """If a caller's ``finally`` invokes release after an acquire failed
    (and therefore never bumped the refcount), the release must not
    raise or underflow the registry entry."""
    agent = _FakeAgent()

    # No prior acquire on this agent.
    _release_react_hooks(agent)
    _release_react_hooks(agent)  # idempotent

    assert agent not in _REACT_HOOK_REGISTRY
