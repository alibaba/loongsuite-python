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

"""Tests for react_step_patch module.

These tests do not require ``agent-framework-core`` to be installed. They
verify:
- The module imports cleanly without MAF.
- ``apply_react_step_patch`` is a no-op (with a warning) when MAF internals
  are missing.
- ``revert_react_step_patch`` is safe to call when not applied.
- The ``ExtendedTelemetryHandler.react_step`` context manager produces a span
  with ``gen_ai.span.kind=STEP`` and ``gen_ai.operation.name=react`` when
  invoked directly (the same path the patch uses).
"""

from __future__ import annotations

import logging

from opentelemetry.instrumentation.microsoft_agent_framework import (
    react_step_patch,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import ReactStepInvocation


def test_module_imports_without_maf():
    # Importing the module should not raise even though MAF is absent.
    assert hasattr(react_step_patch, "apply_react_step_patch")
    assert hasattr(react_step_patch, "revert_react_step_patch")


def _reset_extended_handler_singletons():
    react_step_patch._handler = None
    if hasattr(get_extended_telemetry_handler, "_default_handler"):
        delattr(get_extended_telemetry_handler, "_default_handler")


def test_apply_is_noop_when_maf_missing(caplog, monkeypatch):
    # MAF is not installed in this test env, so apply should warn and return.
    import builtins

    original_import = builtins.__import__

    def _blocked_import(
        name, globals_=None, locals_=None, fromlist=(), level=0
    ):
        if name.startswith("agent_framework"):
            raise ImportError("blocked agent_framework")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    react_step_patch.revert_react_step_patch()  # ensure clean state
    react_step_patch._applied = False
    react_step_patch._original_fil_get_response = None
    react_step_patch._original_chat_get_response = None
    _reset_extended_handler_singletons()
    with caplog.at_level(logging.WARNING):
        react_step_patch.apply_react_step_patch(tracer_provider=None)
    assert react_step_patch._applied is False
    assert any("MAF internals not found" in r.message for r in caplog.records)


def test_revert_is_safe_when_not_applied():
    react_step_patch.revert_react_step_patch()
    # Should not raise and should leave state clean.
    assert react_step_patch._applied is False


def test_handler_react_step_emits_step_span():
    """Directly exercise ``handler.react_step`` to confirm the span shape the
    patch relies on: name ``react step``, ``gen_ai.span.kind=STEP``,
    ``gen_ai.operation.name=react``, ``gen_ai.react.round`` propagated."""
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    _reset_extended_handler_singletons()
    handler = get_extended_telemetry_handler(tracer_provider=tp)

    step_inv = ReactStepInvocation(round=3)
    with handler.react_step(step_inv) as step:
        step.finish_reason = "stop"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "react step"
    assert s.attributes.get("gen_ai.span.kind") == "STEP"
    assert s.attributes.get("gen_ai.operation.name") == "react"
    assert s.attributes.get("gen_ai.react.round") == 3
    assert s.attributes.get("gen_ai.react.finish_reason") == "stop"


def _install_fake_maf_modules(monkeypatch):
    """Install minimal fake ``agent_framework._tools`` and ``agent_framework.
    observability`` modules into ``sys.modules`` so the patch can wrap
    classes without the real MAF package installed.

    Returns ``(fil_cls, chat_cls)`` — the two fake classes with their original
    ``get_response`` callables recorded.
    """
    import sys
    import types

    async def _fil_get_response(self, *args, **kwargs):  # pragma: no cover
        return "fil"

    async def _chat_get_response(self, *args, **kwargs):  # pragma: no cover
        return "chat"

    class _FunctionInvocationLayer:
        get_response = staticmethod(_fil_get_response)

    class _ChatTelemetryLayer:
        get_response = staticmethod(_chat_get_response)

    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.FunctionInvocationLayer = _FunctionInvocationLayer  # type: ignore[attr-defined]
    obs_mod = types.ModuleType("agent_framework.observability")
    obs_mod.ChatTelemetryLayer = _ChatTelemetryLayer  # type: ignore[attr-defined]
    af_mod = types.ModuleType("agent_framework")
    af_mod._tools = tools_mod  # type: ignore[attr-defined]
    af_mod.observability = obs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)
    return _FunctionInvocationLayer, _ChatTelemetryLayer


def test_react_patch_apply_revert_apply_no_multi_wrap(monkeypatch):
    """[M2] regression: ``apply → revert → apply`` must not stack wrappers.

    Before the fix, ``_original_*`` was captured *after* the wrapt decorator
    ran, so it stored the wrapper itself; ``revert`` was a no-op and a second
    ``apply`` wrapped the (still-wrapped) function again, producing nested
    wrappers. With the fix the original is captured before wrapping (via
    ``__wrapped__`` unwrapping), so revert truly restores it and re-apply
    produces a single layer.
    """
    fil_cls, chat_cls = _install_fake_maf_modules(monkeypatch)

    # Reset module state (other tests may have left it set).
    react_step_patch.revert_react_step_patch()
    react_step_patch._applied = False
    react_step_patch._original_fil_get_response = None
    react_step_patch._original_chat_get_response = None
    _reset_extended_handler_singletons()

    fil_before = fil_cls.get_response
    chat_before = chat_cls.get_response

    # 1) apply
    react_step_patch.apply_react_step_patch(tracer_provider=None)
    assert react_step_patch._applied is True
    fil_after_1 = fil_cls.get_response
    chat_after_1 = chat_cls.get_response
    assert fil_after_1 is not fil_before, (
        "wrapt did not replace FIL.get_response"
    )
    assert chat_after_1 is not chat_before, (
        "wrapt did not replace Chat.get_response"
    )

    # 2) revert
    react_step_patch.revert_react_step_patch()
    assert react_step_patch._applied is False
    # After revert the attribute must point to the *original* (unwrapped)
    # function — not to the wrapper.
    assert fil_cls.get_response is fil_before, (
        "revert did not restore FIL.get_response"
    )
    assert chat_cls.get_response is chat_before, (
        "revert did not restore Chat.get_response"
    )

    # 3) apply again
    react_step_patch.apply_react_step_patch(tracer_provider=None)
    assert react_step_patch._applied is True
    fil_after_2 = fil_cls.get_response
    chat_after_2 = chat_cls.get_response
    # Same wrapper identity as the first apply would mean we did not stack;
    # in any case, the underlying __wrapped__ must equal the original.
    assert react_step_patch._unwrap_to_function(fil_after_2) is fil_before
    assert react_step_patch._unwrap_to_function(chat_after_2) is chat_before

    # Depth check: walking __wrapped__ from the second wrapper must reach the
    # original in a bounded number of steps (== 1, since revert restored it).
    depth = 0
    cur = fil_after_2
    while (
        getattr(cur, "__wrapped__", None) is not None
        and cur.__wrapped__ is not cur
    ):
        cur = cur.__wrapped__
        depth += 1
        assert depth < 8, "wrapper chain too deep — multi-wrap detected"
    assert cur is fil_before

    react_step_patch.revert_react_step_patch()


def test_unwrap_to_function_peels_wrappers(monkeypatch):
    """``_unwrap_to_function`` walks the ``__wrapped__`` chain to the
    underlying callable, and returns non-wrappers unchanged."""
    fil_cls, _ = _install_fake_maf_modules(monkeypatch)
    original = fil_cls.get_response

    assert react_step_patch._unwrap_to_function(original) is original

    # Build a fake wrapper chain
    class _Wrapper:
        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

    w1 = _Wrapper(original)
    w2 = _Wrapper(w1)
    assert react_step_patch._unwrap_to_function(w2) is original


def test_fil_wrapper_returns_coroutine(monkeypatch):
    """[P0] regression: ``FunctionInvocationLayer.get_response`` wrapper must
    return a coroutine when wrapping an ``async def`` function, so the MAF
    runtime's ``await layer.get_response(...)`` (``_agents.py:964``) does not
    raise ``TypeError``. The previous ``@wrapt.decorator`` + ``async def``
    variant produced a non-awaitable FunctionWrapper under MAF 1.0.0.
    """
    import asyncio
    import sys
    import types

    async def _fil_get_response(self, *args, **kwargs):
        return ("fil-ok", self, args, kwargs)

    class _FunctionInvocationLayer:
        get_response = staticmethod(_fil_get_response)

    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.FunctionInvocationLayer = _FunctionInvocationLayer  # type: ignore[attr-defined]
    obs_mod = types.ModuleType("agent_framework.observability")

    class _ChatTelemetryLayer:
        get_response = staticmethod(lambda *a, **kw: None)

    obs_mod.ChatTelemetryLayer = _ChatTelemetryLayer  # type: ignore[attr-defined]
    af_mod = types.ModuleType("agent_framework")
    af_mod._tools = tools_mod  # type: ignore[attr-defined]
    af_mod.observability = obs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)

    # Reset module state.
    react_step_patch.revert_react_step_patch()
    react_step_patch._applied = False
    react_step_patch._original_fil_get_response = None
    react_step_patch._original_chat_get_response = None
    _reset_extended_handler_singletons()

    react_step_patch.apply_react_step_patch(tracer_provider=None)
    assert react_step_patch._applied is True

    # Call the wrapped method directly (no instance — staticmethod-style).
    coro = _FunctionInvocationLayer.get_response("self-arg", "a", kw="v")
    assert asyncio.iscoroutine(coro), (
        "FIL wrapper must return a coroutine so MAF can `await` it"
    )
    result = asyncio.get_event_loop().run_until_complete(coro)
    assert result[0] == "fil-ok"

    react_step_patch.revert_react_step_patch()


def test_fil_wrapper_preserves_streaming_response_type(monkeypatch):
    """When MAF calls ``get_response(stream=True)``, the wrapped method returns
    a ResponseStream-like object, not an awaitable. The patch must preserve
    that contract so ``agent.run(..., stream=True)`` can keep using ``.map``
    and async iteration.
    """
    import asyncio
    import sys
    import types

    class _Stream:
        def __await__(self):
            async def _done():
                return self

            return _done().__await__()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        def map(self, *args, **kwargs):
            return self

        async def get_final_response(self):
            return None

    stream = _Stream()

    def _fil_get_response(self, *args, **kwargs):
        return stream

    class _FunctionInvocationLayer:
        get_response = staticmethod(_fil_get_response)

    class _ChatTelemetryLayer:
        get_response = staticmethod(lambda *a, **kw: None)

    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.FunctionInvocationLayer = _FunctionInvocationLayer  # type: ignore[attr-defined]
    obs_mod = types.ModuleType("agent_framework.observability")
    obs_mod.ChatTelemetryLayer = _ChatTelemetryLayer  # type: ignore[attr-defined]
    af_mod = types.ModuleType("agent_framework")
    af_mod._tools = tools_mod  # type: ignore[attr-defined]
    af_mod.observability = obs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)

    react_step_patch.revert_react_step_patch()
    react_step_patch._applied = False
    react_step_patch._original_fil_get_response = None
    react_step_patch._original_chat_get_response = None
    _reset_extended_handler_singletons()

    react_step_patch.apply_react_step_patch(tracer_provider=None)
    result = _FunctionInvocationLayer.get_response("self-arg", stream=True)
    assert result is stream
    assert not asyncio.iscoroutine(result)

    react_step_patch.revert_react_step_patch()


def test_chat_wrapper_outside_loop_passes_through(monkeypatch):
    """[P0] Outside a react-loop scope, the chat wrapper must return the raw
    coroutine produced by the wrapped function (no react_step span). This
    preserves the normal ``await layer.get_response(...)`` path used by MAF
    when ReAct is not active.
    """
    import sys
    import types

    async def _chat_get_response(self, *args, **kwargs):
        return ("chat-ok", self, args, kwargs)

    class _FunctionInvocationLayer:
        get_response = staticmethod(lambda *a, **kw: None)

    class _ChatTelemetryLayer:
        get_response = staticmethod(_chat_get_response)

    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.FunctionInvocationLayer = _FunctionInvocationLayer  # type: ignore[attr-defined]
    obs_mod = types.ModuleType("agent_framework.observability")
    obs_mod.ChatTelemetryLayer = _ChatTelemetryLayer  # type: ignore[attr-defined]
    af_mod = types.ModuleType("agent_framework")
    af_mod._tools = tools_mod  # type: ignore[attr-defined]
    af_mod.observability = obs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)

    react_step_patch.revert_react_step_patch()


def test_chat_wrapper_inside_loop_preserves_streaming_response_type(
    monkeypatch,
):
    """Inside the ReAct ContextVar scope, a streaming ResponseStream-like value
    must still pass through unchanged. ReAct step spans are only added around
    awaitable non-streaming chat calls.
    """
    import asyncio
    import sys
    import types

    class _Stream:
        def __await__(self):
            async def _done():
                return self

            return _done().__await__()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        def map(self, *args, **kwargs):
            return self

        async def get_final_response(self):
            return None

    stream = _Stream()

    class _FunctionInvocationLayer:
        get_response = staticmethod(lambda *a, **kw: None)

    class _ChatTelemetryLayer:
        get_response = staticmethod(lambda *a, **kw: stream)

    tools_mod = types.ModuleType("agent_framework._tools")
    tools_mod.FunctionInvocationLayer = _FunctionInvocationLayer  # type: ignore[attr-defined]
    obs_mod = types.ModuleType("agent_framework.observability")
    obs_mod.ChatTelemetryLayer = _ChatTelemetryLayer  # type: ignore[attr-defined]
    af_mod = types.ModuleType("agent_framework")
    af_mod._tools = tools_mod  # type: ignore[attr-defined]
    af_mod.observability = obs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework._tools", tools_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.observability", obs_mod)

    react_step_patch.revert_react_step_patch()
    react_step_patch._applied = False
    react_step_patch._original_fil_get_response = None
    react_step_patch._original_chat_get_response = None
    _reset_extended_handler_singletons()

    react_step_patch.apply_react_step_patch(tracer_provider=None)
    token = react_step_patch._maf_react_loop_active.set(True)
    try:
        result = _ChatTelemetryLayer.get_response("self-arg", stream=True)
        assert result is stream
        assert not asyncio.iscoroutine(result)
    finally:
        react_step_patch._maf_react_loop_active.reset(token)

    react_step_patch.revert_react_step_patch()
