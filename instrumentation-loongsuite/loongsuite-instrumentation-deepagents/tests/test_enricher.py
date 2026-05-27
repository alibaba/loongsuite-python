# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from uuid import uuid4

from opentelemetry.instrumentation.deepagents.internal import _enricher
from opentelemetry.instrumentation.deepagents.internal._enricher import (
    DeepAgentsEnricherCallbackHandler,
)
from opentelemetry.instrumentation.deepagents.internal._utils import (
    reset_current_subagent_registry,
    set_current_subagent_registry,
)


def test_enricher_sets_framework_and_subagent_attributes(
    tracer_provider,
    span_exporter,
):
    handler = DeepAgentsEnricherCallbackHandler()
    tracer = tracer_provider.get_tracer(__name__)
    token = set_current_subagent_registry({"researcher": "Research agent"})

    with tracer.start_as_current_span("invoke_agent researcher"):
        handler.on_chain_start(
            {},
            {},
            run_id=uuid4(),
            metadata={
                "ls_integration": "deepagents",
                "versions": {"deepagents": "0.6.2"},
                "lc_agent_name": "researcher",
                "ls_agent_type": "subagent",
            },
        )

    reset_current_subagent_registry(token)
    [span] = span_exporter.get_finished_spans()
    attributes = span.attributes
    assert attributes["gen_ai.framework"] == "deepagents"
    assert attributes["gen_ai.framework.version"] == "0.6.2"
    assert attributes["gen_ai.agent.name"] == "researcher"
    assert attributes["gen_ai.agent.type"] == "subagent"
    assert attributes["gen_ai.agent.description"] == "Research agent"


def test_enricher_marks_task_tool_as_agent_tool(
    tracer_provider,
    span_exporter,
):
    handler = DeepAgentsEnricherCallbackHandler()
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("execute_tool task"):
        handler.on_tool_start({"name": "task"}, "", run_id=uuid4())

    [span] = span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.tool.name"] == "task"
    assert span.attributes["gen_ai.tool.type"] == "agent"


def test_enricher_targets_loongsuite_run_span_and_uses_registry_fallback(
    tracer_provider,
    span_exporter,
    monkeypatch,
):
    handler = DeepAgentsEnricherCallbackHandler()
    tracer = tracer_provider.get_tracer(__name__)
    run_id = uuid4()
    token = set_current_subagent_registry({"researcher": "Research agent"})
    run_span = tracer.start_span("invoke_agent researcher")

    monkeypatch.setattr(
        _enricher,
        "_span_for_run",
        lambda seen_run_id: run_span if seen_run_id == run_id else None,
    )

    with tracer.start_as_current_span("current chain"):
        handler.on_chain_start(
            {},
            {},
            run_id=run_id,
            metadata={
                "ls_integration": "deepagents",
                "versions": {"deepagents": "0.6.2"},
                "lc_agent_name": "researcher",
            },
        )

    run_span.end()
    reset_current_subagent_registry(token)
    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    run_attributes = spans["invoke_agent researcher"].attributes
    assert run_attributes["gen_ai.agent.type"] == "subagent"
    assert run_attributes["gen_ai.agent.description"] == "Research agent"
    assert "gen_ai.agent.type" not in spans["current chain"].attributes
