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

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
from opentelemetry.util.genai.extended_types import ReactStepInvocation

from opentelemetry.instrumentation.microsoft_agent_framework import react_step_patch


def test_module_imports_without_maf():
    # Importing the module should not raise even though MAF is absent.
    assert hasattr(react_step_patch, "apply_react_step_patch")
    assert hasattr(react_step_patch, "revert_react_step_patch")


def test_apply_is_noop_when_maf_missing(caplog):
    # MAF is not installed in this test env, so apply should warn and return.
    react_step_patch.revert_react_step_patch()  # ensure clean state
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
    import asyncio
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
    monkeypatch.setitem(
        sys.modules, "agent_framework.observability", obs_mod
    )
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
    react_step_patch._handler = None

    fil_before = fil_cls.get_response
    chat_before = chat_cls.get_response

    # 1) apply
    react_step_patch.apply_react_step_patch(tracer_provider=None)
    assert react_step_patch._applied is True
    fil_after_1 = fil_cls.get_response
    chat_after_1 = chat_cls.get_response
    assert fil_after_1 is not fil_before, "wrapt did not replace FIL.get_response"
    assert chat_after_1 is not chat_before, "wrapt did not replace Chat.get_response"

    # 2) revert
    react_step_patch.revert_react_step_patch()
    assert react_step_patch._applied is False
    # After revert the attribute must point to the *original* (unwrapped)
    # function — not to the wrapper.
    assert fil_cls.get_response is fil_before, "revert did not restore FIL.get_response"
    assert chat_cls.get_response is chat_before, "revert did not restore Chat.get_response"

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
    while getattr(cur, "__wrapped__", None) is not None and cur.__wrapped__ is not cur:
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
