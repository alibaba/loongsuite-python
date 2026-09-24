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

"""Per-thread ReAct state for the BFCL v4 instrumentation.

We use ``contextvars.ContextVar`` so that each worker thread spawned by the
BFCL ``ThreadPoolExecutor`` gets its own copy.  ``_ContextPropagatingExecutor``
in :mod:`threading_propagation` makes sure ENTRY-time context is copied into
the worker thread; the BaseHandler.inference wrapper then initializes a fresh
state on top of that copy.
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional

_REACT_STATE: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("bfclv4_react_state", default=None)
)


def init_state() -> contextvars.Token:
    """Initialise per-AGENT state and return the reset token."""
    state: Dict[str, Any] = {
        # ``turn_idx`` is incremented by the wrapper around
        # ``_add_next_turn_user_message_*``; it stays ``0`` for single-turn
        # tests.
        "turn_idx": 0,
        # ``fc_round`` is the ReAct round counter.  We bump it on every STEP
        # entry so the first STEP within a turn ends up with ``round=1``.
        "fc_round": 0,
        # Counter of executed tool calls within the current AGENT - useful for
        # the TOOL span ``tool_call_id`` synthesis.
        "tool_index": 0,
    }
    return _REACT_STATE.set(state)


def reset_state(token: contextvars.Token) -> None:
    try:
        _REACT_STATE.reset(token)
    except (LookupError, ValueError):
        # Token may have already been reset (e.g. nested error path).
        pass


def get_state() -> Optional[Dict[str, Any]]:
    return _REACT_STATE.get()


def bump_round() -> int:
    state = _REACT_STATE.get()
    if state is None:
        return 1
    state["fc_round"] = state.get("fc_round", 0) + 1
    return state["fc_round"]


def reset_round_for_turn() -> None:
    state = _REACT_STATE.get()
    if state is None:
        return
    state["fc_round"] = 0


def bump_turn() -> int:
    state = _REACT_STATE.get()
    if state is None:
        return 0
    state["turn_idx"] = state.get("turn_idx", 0) + 1
    state["fc_round"] = 0
    return state["turn_idx"]


def next_tool_index() -> int:
    state = _REACT_STATE.get()
    if state is None:
        return 0
    idx = state.get("tool_index", 0)
    state["tool_index"] = idx + 1
    return idx
