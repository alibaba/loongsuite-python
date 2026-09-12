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

"""Lifecycle state shared across WebArena wrappers.

WebArena's ``run.py:test()`` is a single function with a *for* loop over
config files (one task each) and a nested *while* loop (one ReAct round
each). It exposes no per-task hook, so we synthesise ENTRY / CHAIN / STEP
spans by latching on to the boundaries that *do* exist:

* ``ScriptBrowserEnv.reset(...)`` — first call after a task starts
* ``ScriptBrowserEnv.close(...)`` — end of the whole batch
* ``PromptAgent.next_action(...)`` — start of a new ReAct round
* ``ScriptBrowserEnv.step(...)`` — execution of the picked action

This module owns the ``ContextVar`` slots used to thread span handles
between those wrappers in a single process / thread, and the helpers
that close any spans that may still be open when an outer boundary
fires.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from opentelemetry import context as otel_context

# Whether we are currently inside a WebArena task (between an env.reset
# and the next env.reset / env.close). Used by the AGENT(invoke_agent)
# wrapper to decide whether STEP rotation is meaningful.
_in_task: ContextVar[bool] = ContextVar("webarena_in_task", default=False)

# ENTRY span handle + its attached context token.
_entry_span: ContextVar[Any] = ContextVar("webarena_entry_span", default=None)
_entry_token: ContextVar[Any] = ContextVar(
    "webarena_entry_token", default=None
)

# CHAIN(workflow) span handle + token (always nested inside ENTRY).
_chain_span: ContextVar[Any] = ContextVar("webarena_chain_span", default=None)
_chain_token: ContextVar[Any] = ContextVar(
    "webarena_chain_token", default=None
)

# Currently active STEP span handle + token.
_step_span: ContextVar[Any] = ContextVar("webarena_step_span", default=None)
_step_token: ContextVar[Any] = ContextVar("webarena_step_token", default=None)

# Per-task counters, used to populate STEP attributes / CHAIN summaries.
_step_counter: ContextVar[int] = ContextVar("webarena_step_counter", default=0)
_tool_counter: ContextVar[int] = ContextVar("webarena_tool_counter", default=0)
_parsing_failure_counter: ContextVar[int] = ContextVar(
    "webarena_parsing_failure_counter", default=0
)


def _detach_token(token: Any) -> None:
    """Detach an OTel context token, swallowing already-detached errors."""
    if token is None:
        return
    try:
        otel_context.detach(token)
    except Exception:  # noqa: BLE001
        pass


def end_step() -> int:
    """Close the active STEP span (if any) and return the round number it had.

    Returns ``0`` when no STEP was active.
    """
    span = _step_span.get(None)
    token = _step_token.get(None)
    round_no = 0
    if span is not None:
        try:
            round_no = int(span.attributes.get("gen_ai.react.round", 0))  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            round_no = 0
        try:
            span.end()
        except Exception:  # noqa: BLE001
            pass
    _step_span.set(None)
    _detach_token(token)
    _step_token.set(None)
    return round_no


def end_chain() -> None:
    """Close the active CHAIN span (if any) and detach its token."""
    span = _chain_span.get(None)
    token = _chain_token.get(None)
    if span is not None:
        try:
            span.end()
        except Exception:  # noqa: BLE001
            pass
    _chain_span.set(None)
    _detach_token(token)
    _chain_token.set(None)


def end_entry() -> None:
    """Close the active ENTRY span (if any) and detach its token."""
    span = _entry_span.get(None)
    token = _entry_token.get(None)
    if span is not None:
        try:
            span.end()
        except Exception:  # noqa: BLE001
            pass
    _entry_span.set(None)
    _detach_token(token)
    _entry_token.set(None)


def end_task_spans() -> None:
    """Close STEP → CHAIN → ENTRY in order (most-nested first)."""
    end_step()
    end_chain()
    end_entry()
    _in_task.set(False)
    _step_counter.set(0)
    _tool_counter.set(0)
    _parsing_failure_counter.set(0)


def in_task() -> bool:
    return bool(_in_task.get(False))


def mark_in_task(value: bool) -> None:
    _in_task.set(value)


def set_entry(span: Any, token: Any) -> None:
    _entry_span.set(span)
    _entry_token.set(token)


def set_chain(span: Any, token: Any) -> None:
    _chain_span.set(span)
    _chain_token.set(token)


def set_step(span: Any, token: Any) -> None:
    _step_span.set(span)
    _step_token.set(token)


def get_chain_span() -> Any:
    return _chain_span.get(None)


def get_entry_span() -> Any:
    return _entry_span.get(None)


def get_step_span() -> Any:
    return _step_span.get(None)


def increment_step() -> int:
    n = int(_step_counter.get(0)) + 1
    _step_counter.set(n)
    return n


def increment_tool() -> int:
    n = int(_tool_counter.get(0)) + 1
    _tool_counter.set(n)
    return n


def increment_parsing_failure() -> int:
    n = int(_parsing_failure_counter.get(0)) + 1
    _parsing_failure_counter.set(n)
    return n


def step_count() -> int:
    return int(_step_counter.get(0))


def tool_count() -> int:
    return int(_tool_counter.get(0))


def parsing_failure_count() -> int:
    return int(_parsing_failure_counter.get(0))
