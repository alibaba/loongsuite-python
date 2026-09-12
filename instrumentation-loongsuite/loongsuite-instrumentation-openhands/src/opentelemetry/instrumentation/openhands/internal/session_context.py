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

"""Cross-thread / cross-loop OTel context bridge keyed by OpenHands session id.

Why this exists
---------------

OpenHands V0's ``EventStream`` delivers events to subscribers via a
``ThreadPoolExecutor``. The ``AgentController.on_event`` callback then runs

.. code:: python

    asyncio.get_event_loop().run_until_complete(self._on_event(event))

inside a *worker thread*, which spins up a brand-new asyncio loop with a
fresh ``contextvars.Context``. This means none of the OTel context (tracer
spans / baggage) attached on the main coroutine in ``run_controller`` is
visible inside ``AgentController._step`` or ``Runtime.run_action`` — every
STEP / TOOL span starts at the **trace root**, fragmenting the trace into
many disconnected pieces.

This module bridges that gap. We snapshot the OTel context at entry-time
(``run_controller`` / ``run_agent_until_done``) under the controller's
session id, and the STEP / TOOL wrappers re-attach the snapshot before
opening their spans so every span shares a single ``trace_id`` rooted at
the ENTRY span.

The store is keyed by **session id (sid)** so concurrent benchmark
sessions stay isolated.
"""

from __future__ import annotations

import threading
from typing import Optional

from opentelemetry import context as otel_context

_lock = threading.Lock()
# Map session id -> OTel Context object. The Context contains the active
# Span (and any baggage / suppression flags). Re-attaching it makes the
# stored span the *current* span for whatever thread/loop attaches it.
_session_contexts: dict[str, otel_context.Context] = {}

# Map session id -> { tool_name: tool_definition_dict }. Captured at
# AGENT span open from ``controller.agent.tools`` and consumed by the
# TOOL wrapper to populate ``gen_ai.tool.description`` and friends — the
# Runtime instance does not have direct access to the agent's tool list.
_session_tool_registry: dict[str, dict[str, dict]] = {}

# Tracks the most-recent sid we stored a context for. Used as a fallback
# when a hook point (typically ``Runtime.run_action``) cannot locate the
# session id from its arguments — in single-session CLI runs this is
# always the right answer.
_last_sid: Optional[str] = None


def store_context(sid: Optional[str], ctx: otel_context.Context) -> None:
    """Stash ``ctx`` under ``sid``. Updates ``_last_sid``."""
    if not sid:
        return
    global _last_sid
    with _lock:
        _session_contexts[sid] = ctx
        _last_sid = sid


def get_context(sid: Optional[str]) -> Optional[otel_context.Context]:
    """Return the stashed context for ``sid``, falling back to the last sid."""
    with _lock:
        if sid and sid in _session_contexts:
            return _session_contexts[sid]
        if _last_sid and _last_sid in _session_contexts:
            return _session_contexts[_last_sid]
        return None


def clear_context(sid: Optional[str]) -> None:
    if not sid:
        return
    global _last_sid
    with _lock:
        _session_contexts.pop(sid, None)
        _session_tool_registry.pop(sid, None)
        if _last_sid == sid:
            _last_sid = None


def clear_all() -> None:
    """Drop everything (only used by tests)."""
    global _last_sid
    with _lock:
        _session_contexts.clear()
        _session_tool_registry.clear()
        _last_sid = None


# ---------------------------------------------------------------------------
# Tool registry (per-sid)
# ---------------------------------------------------------------------------


def store_tool_registry(sid: Optional[str], tools: object) -> None:
    """Index ``tools`` by name and stash under ``sid``.

    ``tools`` is whatever ``controller.agent.tools`` exposes — typically a
    list of LiteLLM ``ChatCompletionToolParam`` dicts of the form
    ``{"type": "function", "function": {"name": ..., "description": ..., ...}}``.
    Anything that doesn't fit that shape is best-effort skipped.
    """
    if not sid or not tools:
        return
    registry: dict[str, dict] = {}
    try:
        for t in tools:  # type: ignore[union-attr]
            try:
                if isinstance(t, dict):
                    fn = t.get("function") or {}
                    name = fn.get("name") if isinstance(fn, dict) else None
                else:
                    fn = getattr(t, "function", None)
                    name = (
                        getattr(fn, "name", None) if fn is not None else None
                    )
                    # Normalize to a dict so the consumer doesn't need type-knowledge.
                    if name and not isinstance(t, dict):
                        t = {
                            "type": getattr(t, "type", "function"),
                            "function": {
                                "name": name,
                                "description": getattr(fn, "description", "")
                                or "",
                                "parameters": getattr(fn, "parameters", None)
                                or {},
                            },
                        }
                if name:
                    registry[str(name)] = t
            except Exception:
                continue
    except TypeError:
        return
    if not registry:
        return
    with _lock:
        _session_tool_registry[sid] = registry


def get_tool_definition(
    sid: Optional[str], name: Optional[str]
) -> Optional[dict]:
    """Look up a single tool's definition (dict) by name, sid-scoped."""
    if not name:
        return None
    with _lock:
        if sid and sid in _session_tool_registry:
            return _session_tool_registry[sid].get(name)
        # Fallback to the most-recent session — single-CLI-run case.
        if _last_sid and _last_sid in _session_tool_registry:
            return _session_tool_registry[_last_sid].get(name)
        return None


def get_tool_registry(sid: Optional[str]) -> Optional[dict[str, dict]]:
    """Return the full ``{name: definition}`` registry for ``sid``."""
    with _lock:
        if sid and sid in _session_tool_registry:
            return dict(_session_tool_registry[sid])
        if _last_sid and _last_sid in _session_tool_registry:
            return dict(_session_tool_registry[_last_sid])
        return None


class AttachedSession:
    """Context manager that attaches the stashed context for ``sid``.

    Usage::

        with AttachedSession(sid):
            span = tracer.start_span(...)
            # span is parented under whatever the stashed context contains

    No-op when no stash exists for the given sid.
    """

    __slots__ = ("_sid", "_token")

    def __init__(self, sid: Optional[str]):
        self._sid = sid
        self._token = None

    def __enter__(self) -> "AttachedSession":
        ctx = get_context(self._sid)
        if ctx is not None:
            self._token = otel_context.attach(ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            try:
                otel_context.detach(self._token)
            except Exception:
                pass
            self._token = None
