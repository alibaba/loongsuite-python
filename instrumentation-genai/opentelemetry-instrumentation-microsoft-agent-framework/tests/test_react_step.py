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
