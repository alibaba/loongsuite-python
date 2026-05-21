# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import sys
import types

import pytest

from opentelemetry.instrumentation.deepagents.internal._entry_patch import (
    instrument_entry_patch,
    uninstrument_entry_patch,
)
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


class FakeGraph:
    def __init__(self, name: str = "supervisor") -> None:
        self.name = name
        self.last_config = None
        self.config = {
            "metadata": {
                "ls_integration": "deepagents",
                "versions": {"deepagents": "0.6.2"},
                "lc_agent_name": name,
            }
        }

    def invoke(self, value, config=None):
        self.last_config = config
        return {"messages": [*value["messages"], {"role": "assistant", "content": "done"}]}

    async def ainvoke(self, value, config=None):
        self.last_config = config
        return {"messages": [*value["messages"], {"role": "assistant", "content": "done"}]}

    def stream(self, value, config=None):
        self.last_config = config
        yield {"messages": [*value["messages"], {"role": "assistant", "content": "part"}]}
        yield {"messages": [*value["messages"], {"role": "assistant", "content": "done"}]}

    async def astream(self, value, config=None):
        self.last_config = config
        yield {"messages": [*value["messages"], {"role": "assistant", "content": "done"}]}


@pytest.fixture(name="fake_deepagents_graph")
def fixture_fake_deepagents_graph(monkeypatch):
    deepagents_module = types.ModuleType("deepagents")
    deepagents_module.__version__ = "0.6.2"
    graph_module = types.ModuleType("deepagents.graph")

    def create_deep_agent(*, name="supervisor", subagents=None):
        del subagents
        return FakeGraph(name)

    graph_module.create_deep_agent = create_deep_agent
    deepagents_module.create_deep_agent = create_deep_agent
    deepagents_module.graph = graph_module
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_module)
    monkeypatch.setitem(sys.modules, "deepagents.graph", graph_module)
    try:
        yield graph_module
    finally:
        uninstrument_entry_patch()


def _entry_spans(span_exporter):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.span.kind") == "ENTRY"
    ]


def test_invoke_creates_one_deepagents_entry_span(
    fake_deepagents_graph,
    tracer_provider,
    span_exporter,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    instrument_entry_patch(handler)

    graph = fake_deepagents_graph.create_deep_agent(
        name="supervisor",
        subagents=[
            {"name": "researcher", "description": "Research agent"},
        ],
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}]},
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert result["messages"][-1]["content"] == "done"
    assert getattr(graph, "_loongsuite_react_agent") is True
    assert graph.last_config["metadata"]["_loongsuite_react_agent"] is True
    [entry_span] = _entry_spans(span_exporter)
    attributes = entry_span.attributes
    assert attributes["gen_ai.operation.name"] == "invoke"
    assert attributes["gen_ai.framework"] == "deepagents"
    assert attributes["gen_ai.framework.version"] == "0.6.2"
    assert attributes["gen_ai.agent.name"] == "supervisor"
    assert attributes["gen_ai.session.id"] == "thread-1"


def test_top_level_create_deep_agent_export_is_wrapped_and_restored(
    fake_deepagents_graph,
    tracer_provider,
    span_exporter,
):
    import deepagents  # noqa: PLC0415

    original_top_level = deepagents.create_deep_agent
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    instrument_entry_patch(handler)

    from deepagents import create_deep_agent  # noqa: PLC0415

    assert create_deep_agent is deepagents.create_deep_agent
    assert create_deep_agent is fake_deepagents_graph.create_deep_agent
    assert create_deep_agent is not original_top_level

    graph = create_deep_agent(name="supervisor")
    graph.invoke({"messages": [{"role": "user", "content": "hi"}]})

    [entry_span] = _entry_spans(span_exporter)
    assert entry_span.attributes["gen_ai.operation.name"] == "invoke"

    uninstrument_entry_patch()
    assert deepagents.create_deep_agent is original_top_level


def test_entry_is_skipped_inside_existing_agent_span(
    fake_deepagents_graph,
    tracer_provider,
    span_exporter,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    instrument_entry_patch(handler)
    graph = fake_deepagents_graph.create_deep_agent(name="supervisor")
    tracer = get_tracer(__name__, tracer_provider=tracer_provider)

    with tracer.start_as_current_span("invoke_agent parent") as span:
        span.set_attribute("gen_ai.span.kind", "AGENT")
        graph.invoke({"messages": [{"role": "user", "content": "nested"}]})

    assert _entry_spans(span_exporter) == []


def test_stream_keeps_entry_open_until_iteration_finishes(
    fake_deepagents_graph,
    tracer_provider,
    span_exporter,
):
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    instrument_entry_patch(handler)
    graph = fake_deepagents_graph.create_deep_agent(name="supervisor")

    chunks = list(graph.stream({"messages": [{"role": "user", "content": "hi"}]}))

    assert chunks[-1]["messages"][-1]["content"] == "done"
    [entry_span] = _entry_spans(span_exporter)
    assert entry_span.attributes["gen_ai.operation.name"] == "stream"
