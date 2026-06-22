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

"""End-to-end smoke tests for the Agent-Reach instrumentor.

These tests import agent_reach submodules AFTER calling ``instrument()`` so
that wrapt's ``wrap_function_wrapper`` patches are visible through the module
attribute lookups. (If the test does ``from agent_reach.doctor import
check_all`` before instrumenting, the local name keeps pointing at the
original unwrapped function.)
"""

import importlib

import pytest

pytest.importorskip("agent_reach")

from opentelemetry.instrumentation.agent_reach import (  # noqa: E402
    AgentReachInstrumentor,
)
from opentelemetry.instrumentation.agent_reach import (
    semconv as sc,  # noqa: E402
)
from opentelemetry.util.genai.extended_handler import (  # noqa: E402
    ExtendedTelemetryHandler,
)
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,  # noqa: E402
)


def _attr(span, key):
    for k, v in span.attributes.items():
        if k == key:
            return v
    return None


def _entry_invocation(handler: ExtendedTelemetryHandler) -> EntryInvocation:
    """Build an EntryInvocation that mirrors what EntryWrapper produces when
    wrapping ``agent_reach.cli.main``. Tests call ``handler.entry(inv)`` with
    this object to obtain an ENTRY span carrying the standard attributes."""
    inv = EntryInvocation()
    inv.attributes.update(
        {
            sc.GEN_AI_SPAN_KIND: sc.SPAN_KIND_ENTRY,
            sc.GEN_AI_OPERATION_NAME: sc.OPERATION_ENTER,
            sc.GEN_AI_FRAMEWORK_ATTR: sc.GEN_AI_FRAMEWORK,
        }
    )
    return inv


def test_doctor_emits_entry_and_tool_spans(span_exporter):
    instr = AgentReachInstrumentor()
    try:
        instr.instrument()
        # Re-import AFTER instrument so the local name binds to the patched
        # function. (Module-level `from ... import ...` would shadow the
        # patched attribute.)
        doctor = importlib.import_module("agent_reach.doctor")
        config_cls = importlib.import_module("agent_reach.config").Config
        with instr._handler.entry(_entry_invocation(instr._handler)):
            results = doctor.check_all(config_cls())
        assert isinstance(results, dict)
        assert len(results) >= 10
    finally:
        instr.uninstrument()

    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "enter_ai_application_system" in names
    assert "execute_tool agent-reach-doctor" in names

    doctor_span = next(
        s for s in spans if s.name == "execute_tool agent-reach-doctor"
    )
    assert _attr(doctor_span, "gen_ai.span.kind") == "TOOL"
    assert _attr(doctor_span, "gen_ai.operation.name") == "execute_tool"
    assert _attr(doctor_span, "gen_ai.tool.name") == "agent-reach-doctor"
    assert _attr(doctor_span, "gen_ai.framework") == "agent-reach"

    entry_span = next(
        s for s in spans if s.name == "enter_ai_application_system"
    )
    assert _attr(entry_span, "gen_ai.span.kind") == "ENTRY"
    assert _attr(entry_span, "gen_ai.operation.name") == "enter"
    assert _attr(entry_span, "gen_ai.framework") == "agent-reach"

    channel_spans = [
        s for s in spans if s.name.startswith("execute_tool agent-reach-channel-")
    ]
    assert len(channel_spans) >= 10


def test_channel_check_span_attributes(span_exporter):
    instr = AgentReachInstrumentor()
    try:
        instr.instrument()
        channels = importlib.import_module("agent_reach.channels")
        config_cls = importlib.import_module("agent_reach.config").Config
        with instr._handler.entry(_entry_invocation(instr._handler)):
            for ch in channels.get_all_channels():
                ch.check(config_cls())
    finally:
        instr.uninstrument()

    spans = span_exporter.get_finished_spans()
    channel_spans = [
        s for s in spans if s.name.startswith("execute_tool agent-reach-channel-")
    ]
    assert channel_spans, [s.name for s in spans]
    for s in channel_spans:
        assert _attr(s, "gen_ai.span.kind") == "TOOL"
        assert _attr(s, "gen_ai.operation.name") == "execute_tool"
        assert _attr(s, "gen_ai.framework") == "agent-reach"
        tool_name = _attr(s, "gen_ai.tool.name") or ""
        assert tool_name.startswith("agent-reach-channel-")
        tier = _attr(s, "agent_reach.channel.tier")
        assert tier is not None
        assert isinstance(tier, (int, float))


def test_probe_span(span_exporter):
    instr = AgentReachInstrumentor()
    try:
        instr.instrument()
        probe = importlib.import_module("agent_reach.probe")
        with instr._handler.entry(_entry_invocation(instr._handler)):
            probe.probe_command("python", ["--version"], timeout=5)
    finally:
        instr.uninstrument()

    spans = span_exporter.get_finished_spans()
    probe_spans = [
        s for s in spans if s.name.startswith("execute_tool agent-reach-probe")
    ]
    assert probe_spans, [s.name for s in spans]
    span = probe_spans[0]
    assert _attr(span, "gen_ai.span.kind") == "TOOL"
    assert _attr(span, "gen_ai.operation.name") == "execute_tool"
    assert _attr(span, "gen_ai.tool.name") == "agent-reach-probe"
    assert _attr(span, "agent_reach.probe.cmd") == "python"
    status = _attr(span, "agent_reach.probe.status")
    assert status in {"ok", "missing", "broken", "timeout", "error"}


def test_uninstrument_restores_originals(span_exporter):
    instr = AgentReachInstrumentor()
    instr.instrument()
    probe = importlib.import_module("agent_reach.probe")
    wrapped = probe.probe_command
    assert wrapped is not probe.probe_command.__wrapped__  # has a wrapper
    instr.uninstrument()
    # After uninstrument, calling probe_command should produce no spans.
    with instr._handler.entry(_entry_invocation(instr._handler)):
        probe.probe_command("python", ["--version"], timeout=5)

    spans = span_exporter.get_finished_spans()
    ar_spans = [
        s
        for s in spans
        if s.name.startswith("execute_tool agent-reach-probe")
        or s.name.startswith("execute_tool agent-reach-channel-")
        or s.name.startswith("execute_tool agent-reach-doctor")
    ]
    assert ar_spans == [], [s.name for s in ar_spans]


def test_content_capture_disabled_by_default(span_exporter):
    """No gen_ai.tool.call.arguments / .result attributes unless
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is set."""
    instr = AgentReachInstrumentor()
    try:
        instr.instrument()
        probe = importlib.import_module("agent_reach.probe")
        with instr._handler.entry(_entry_invocation(instr._handler)):
            probe.probe_command("python", ["--version"], timeout=5)
    finally:
        instr.uninstrument()

    spans = span_exporter.get_finished_spans()
    probe_spans = [
        s for s in spans if s.name.startswith("execute_tool agent-reach-probe")
    ]
    assert probe_spans
    for s in probe_spans:
        assert _attr(s, "gen_ai.tool.call.arguments") is None
        assert _attr(s, "gen_ai.tool.call.result") is None


def test_content_capture_enabled(span_exporter, enable_capture):
    instr = AgentReachInstrumentor()
    try:
        instr.instrument()
        probe = importlib.import_module("agent_reach.probe")
        with instr._handler.entry(_entry_invocation(instr._handler)):
            probe.probe_command("python", ["--version"], timeout=5)
    finally:
        instr.uninstrument()

    spans = span_exporter.get_finished_spans()
    probe_spans = [
        s for s in spans if s.name.startswith("execute_tool agent-reach-probe")
    ]
    assert probe_spans
    for s in probe_spans:
        args = _attr(s, "gen_ai.tool.call.arguments")
        assert args is not None  # captured under SPAN_ONLY


def test_repeated_instrument_uninstrument(span_exporter):
    instr = AgentReachInstrumentor()
    for _ in range(3):
        instr.instrument()
        try:
            probe = importlib.import_module("agent_reach.probe")
            with instr._handler.entry(_entry_invocation(instr._handler)):
                probe.probe_command("python", ["--version"], timeout=5)
        finally:
            instr.uninstrument()
