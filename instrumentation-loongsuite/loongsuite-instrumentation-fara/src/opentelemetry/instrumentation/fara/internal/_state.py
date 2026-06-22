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

"""Lifecycle state shared across Fara wrappers.

Fara's ``FaraAgent.run()`` is a single async method with an internal
for-loop over ReAct rounds. ``run_fara_agent`` is the CLI entry. We
synthesise ENTRY / AGENT / STEP / TOOL spans by latching on to:

* ``run_fara_agent(...)`` — ENTRY span (one per task)
* ``FaraAgent.run(...)`` — AGENT span (one per agent invocation)
* ``FaraAgent.generate_model_call(...)`` — STEP rotation (one per round)
* ``FaraAgent.execute_action(...)`` — TOOL span (one per action)

Per-task state is threaded via ``ContextVar`` so concurrent async runs
are isolated.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional


# ENTRY span handle + session_id (for AGENT's conversation_id).
_entry_session_id: ContextVar[Optional[str]] = ContextVar(
    "fara_entry_session_id", default=None
)

# Currently active STEP invocation (ReactStepInvocation). Set by
# GenerateModelCallWrapper when a new round starts; cleared when the
# STEP is closed.
_step_invocation: ContextVar[Any] = ContextVar("fara_step_inv", default=None)

# Per-run STEP counter (1-based round number).
_step_counter: ContextVar[int] = ContextVar("fara_step_counter", default=0)


def get_entry_session_id() -> Optional[str]:
    return _entry_session_id.get(None)


def set_entry_session_id(value: Optional[str]) -> Any:
    return _entry_session_id.set(value)


def reset_entry_session_id(token: Any) -> None:
    _entry_session_id.reset(token)


def get_step_invocation() -> Any:
    return _step_invocation.get(None)


def set_step_invocation(value: Any) -> Any:
    return _step_invocation.set(value)


def reset_step_invocation(token: Any) -> None:
    _step_invocation.reset(token)


def get_step_counter() -> int:
    return int(_step_counter.get(0))


def set_step_counter(value: int) -> Any:
    return _step_counter.set(value)


def reset_step_counter(token: Any) -> None:
    _step_counter.reset(token)


def increment_step_counter() -> int:
    n = int(_step_counter.get(0)) + 1
    _step_counter.set(n)
    return n
