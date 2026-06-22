# Copyright The OpenTelemetry Authors
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

"""Behaviour tests for the Fara wrappers.

Covers ENTRY / AGENT / STEP / TOOL span emission, attributes, parent /
child relationships, finish_reason semantics, error handling, and
content-capture gating.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import fara.fara_agent as fara_agent_mod
import fara.run_fara as run_fara_mod
import pytest


def span_attr(span, key):
    return span.attributes.get(key)


def spans_by_name(spans, name):
    return [s for s in spans if name in s.name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _await(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _build_agent(**kwargs):
    return fara_agent_mod.FaraAgent(
        client_config={
            "model": "microsoft/Fara-7B",
            "base_url": "http://localhost:5000/v1",
            "api_key": "not-needed",
        },
        max_rounds=kwargs.pop("max_rounds", 3),
        actions=kwargs.pop("actions", ["left_click", "type", "terminate"]),
    )


# ---------------------------------------------------------------------------
# ENTRY span
# ---------------------------------------------------------------------------


def test_entry_span_emitted(instrument, span_exporter):
    _await(run_fara_mod.run_fara_agent(initial_task="open bing"))
    spans = span_exporter.get_finished_spans()
    entry = spans_by_name(spans, "enter_ai_application_system")
    assert len(entry) == 1
    span = entry[0]
    assert span_attr(span, "gen_ai.span.kind") == "ENTRY"
    assert span_attr(span, "gen_ai.operation.name") == "enter"
    assert span_attr(span, "gen_ai.framework") == "fara"
    session_id = span_attr(span, "gen_ai.session.id")
    assert session_id
    # Must be a valid UUID.
    uuid.UUID(str(session_id))


def test_entry_span_does_not_capture_content_by_default(instrument, span_exporter):
    _await(run_fara_mod.run_fara_agent(initial_task="secret task text"))
    spans = span_exporter.get_finished_spans()
    entry = spans_by_name(spans, "enter_ai_application_system")[0]
    assert span_attr(entry, "gen_ai.input.messages") is None


def test_entry_span_captures_content_when_enabled(instrument_with_content, span_exporter):
    _await(run_fara_mod.run_fara_agent(initial_task="open bing"))
    spans = span_exporter.get_finished_spans()
    entry = spans_by_name(spans, "enter_ai_application_system")[0]
    messages = span_attr(entry, "gen_ai.input.messages")
    assert messages is not None
    assert "open bing" in messages


# ---------------------------------------------------------------------------
# AGENT span
# ---------------------------------------------------------------------------


def test_agent_span_emitted(instrument, span_exporter):
    agent = _build_agent()
    _await(agent.run("click the search box"))
    spans = span_exporter.get_finished_spans()
    agent_spans = spans_by_name(spans, "invoke_agent FaraAgent")
    assert len(agent_spans) == 1
    span = agent_spans[0]
    assert span_attr(span, "gen_ai.span.kind") == "AGENT"
    assert span_attr(span, "gen_ai.operation.name") == "invoke_agent"
    assert span_attr(span, "gen_ai.agent.name") == "FaraAgent"
    assert span_attr(span, "gen_ai.request.model") == "microsoft/Fara-7B"
    # conversation.id should match the ENTRY session.id.
    entry = spans_by_name(spans, "enter_ai_application_system")
    if entry:
        assert span_attr(span, "gen_ai.conversation.id") == span_attr(
            entry[0], "gen_ai.session.id"
        )


def test_agent_span_error_on_exception(instrument, span_exporter):
    # generate_model_call raises on round 1; the AGENT span should be
    # marked ERROR.
    agent = _build_agent(actions=["RAISE:agent crashed"])
    with pytest.raises(RuntimeError, match="agent crashed"):
        _await(agent.run("anything"))
    spans = span_exporter.get_finished_spans()
    agent_spans = spans_by_name(spans, "invoke_agent FaraAgent")
    assert len(agent_spans) == 1
    assert agent_spans[0].status.is_ok is False


def _canonical_tool_names():
    return [
        "key",
        "type",
        "mouse_move",
        "left_click",
        "scroll",
        "visit_url",
        "web_search",
        "history_back",
        "pause_and_memorize_fact",
        "wait",
        "terminate",
    ]


def test_agent_span_tool_definitions_default(instrument, span_exporter):
    # capture disabled: AGENT span must still carry gen_ai.tool.definitions
    # as a JSON array of 11 {type, name} entries (no description).
    agent = _build_agent()
    _await(agent.run("click the search box"))
    spans = span_exporter.get_finished_spans()
    agent_span = spans_by_name(spans, "invoke_agent FaraAgent")[0]
    raw = span_attr(agent_span, "gen_ai.tool.definitions")
    assert raw is not None
    defs = json.loads(raw)
    assert isinstance(defs, list)
    assert len(defs) == 11
    assert [d["name"] for d in defs] == _canonical_tool_names()
    for d in defs:
        assert d["type"] == "function"
        assert "description" not in d


def test_agent_span_tool_definitions_with_content(instrument_with_content, span_exporter):
    # capture enabled: each entry should also carry a description.
    agent = _build_agent()
    _await(agent.run("click the search box"))
    spans = span_exporter.get_finished_spans()
    agent_span = spans_by_name(spans, "invoke_agent FaraAgent")[0]
    raw = span_attr(agent_span, "gen_ai.tool.definitions")
    assert raw is not None
    defs = json.loads(raw)
    assert len(defs) == 11
    assert [d["name"] for d in defs] == _canonical_tool_names()
    for d in defs:
        assert d["type"] == "function"
        assert d.get("description")


# ---------------------------------------------------------------------------
# STEP span
# ---------------------------------------------------------------------------


def test_step_spans_one_per_round(instrument, span_exporter):
    agent = _build_agent(actions=["left_click", "type", "terminate"])
    _await(agent.run("do something"))
    spans = span_exporter.get_finished_spans()
    step_spans = spans_by_name(spans, "react step")
    assert len(step_spans) == 3
    rounds = sorted(
        span_attr(s, "gen_ai.react.round") for s in step_spans
    )
    assert rounds == [1, 2, 3]
    for s in step_spans:
        assert span_attr(s, "gen_ai.span.kind") == "STEP"
        assert span_attr(s, "gen_ai.operation.name") == "react"


def test_step_finish_reason_terminate(instrument, span_exporter):
    agent = _build_agent(actions=["left_click", "terminate"])
    _await(agent.run("task"))
    spans = span_exporter.get_finished_spans()
    step_spans = sorted(
        spans_by_name(spans, "react step"),
        key=lambda s: span_attr(s, "gen_ai.react.round"),
    )
    assert span_attr(step_spans[-1], "gen_ai.react.finish_reason") == "terminate"
    # Earlier STEPs should be action_complete.
    assert span_attr(step_spans[0], "gen_ai.react.finish_reason") == "action_complete"


def test_step_finish_reason_max_rounds(instrument, span_exporter):
    agent = _build_agent(
        actions=["scroll", "scroll", "scroll"],
        max_rounds=3,
    )
    _await(agent.run("scroll forever"))
    spans = span_exporter.get_finished_spans()
    step_spans = spans_by_name(spans, "react step")
    assert len(step_spans) == 3
    last = sorted(
        step_spans, key=lambda s: span_attr(s, "gen_ai.react.round")
    )[-1]
    assert span_attr(last, "gen_ai.react.finish_reason") == "max_rounds"


def test_step_finish_reason_on_exception(instrument, span_exporter):
    # execute_action raises ValueError on round 2 (after a normal
    # round 1). STEP 2 should carry finish_reason="ValueError".
    agent = _build_agent(actions=["left_click", "EXEC_RAISE:bad tool"])
    with pytest.raises(ValueError, match="bad tool"):
        _await(agent.run("fail please"))
    spans = span_exporter.get_finished_spans()
    step_spans = spans_by_name(spans, "react step")
    assert step_spans
    last = sorted(
        step_spans, key=lambda s: span_attr(s, "gen_ai.react.round")
    )[-1]
    assert span_attr(last, "gen_ai.react.finish_reason") == "ValueError"


# ---------------------------------------------------------------------------
# TOOL span
# ---------------------------------------------------------------------------


def test_tool_span_emitted_per_action(instrument, span_exporter):
    agent = _build_agent(actions=["left_click", "type", "terminate"])
    _await(agent.run("do things"))
    spans = span_exporter.get_finished_spans()
    tool_spans = spans_by_name(spans, "execute_tool")
    assert len(tool_spans) == 3
    actions = sorted(
        span_attr(s, "gen_ai.tool.name") for s in tool_spans
    )
    assert actions == ["left_click", "terminate", "type"]
    for s in tool_spans:
        assert span_attr(s, "gen_ai.span.kind") == "TOOL"
        assert span_attr(s, "gen_ai.operation.name") == "execute_tool"
        assert span_attr(s, "gen_ai.tool.type") == "browser_action"
        assert span_attr(s, "gen_ai.tool.call.id") == "dummy"


def test_tool_span_description(instrument, span_exporter):
    agent = _build_agent(actions=["visit_url", "terminate"])
    _await(agent.run("go places"))
    spans = span_exporter.get_finished_spans()
    visit = [
        s for s in spans if span_attr(s, "gen_ai.tool.name") == "visit_url"
    ][0]
    assert (
        span_attr(visit, "gen_ai.tool.description") == "Navigate to a URL"
    )


def test_tool_span_arguments_not_captured_by_default(instrument, span_exporter):
    agent = _build_agent(actions=["visit_url", "terminate"])
    _await(agent.run("no capture"))
    spans = span_exporter.get_finished_spans()
    visit = [
        s for s in spans if span_attr(s, "gen_ai.tool.name") == "visit_url"
    ][0]
    assert span_attr(visit, "gen_ai.tool.call.arguments") is None


def test_tool_span_arguments_captured_when_enabled(instrument_with_content, span_exporter):
    agent = _build_agent(actions=["visit_url", "terminate"])
    _await(agent.run("capture please"))
    spans = span_exporter.get_finished_spans()
    visit = [
        s for s in spans if span_attr(s, "gen_ai.tool.name") == "visit_url"
    ][0]
    args = span_attr(visit, "gen_ai.tool.call.arguments")
    assert args is not None
    assert "visit_url" in args


# ---------------------------------------------------------------------------
# Parent / child relationships
# ---------------------------------------------------------------------------


def test_hierarchy_entry_agent_step_tool(instrument, span_exporter):
    # Use run_fara_agent so ENTRY is also emitted; then ENTRY > AGENT
    # > STEP > TOOL hierarchy can be verified end-to-end.
    _await(
        run_fara_mod.run_fara_agent(
            initial_task="hierarchy check",
            max_rounds=3,
        )
    )
    spans = span_exporter.get_finished_spans()
    entry = spans_by_name(spans, "enter_ai_application_system")[0]
    agent_span = spans_by_name(spans, "invoke_agent FaraAgent")[0]
    step_spans = sorted(
        spans_by_name(spans, "react step"),
        key=lambda s: span_attr(s, "gen_ai.react.round"),
    )
    tool_spans = spans_by_name(spans, "execute_tool")

    # AGENT parent should be ENTRY.
    assert agent_span.parent.span_id == entry.context.span_id

    # STEP parent should be AGENT.
    for step in step_spans:
        assert step.parent.span_id == agent_span.context.span_id

    # TOOL parent should be a STEP.
    step_ids = {s.context.span_id for s in step_spans}
    for tool in tool_spans:
        assert tool.parent.span_id in step_ids


def test_step_count_matches_actions(instrument, span_exporter):
    agent = _build_agent(
        actions=["left_click", "type", "scroll", "terminate"],
        max_rounds=4,
    )
    _await(agent.run("many actions"))
    spans = span_exporter.get_finished_spans()
    assert len(spans_by_name(spans, "react step")) == 4
    assert len(spans_by_name(spans, "execute_tool")) == 4
