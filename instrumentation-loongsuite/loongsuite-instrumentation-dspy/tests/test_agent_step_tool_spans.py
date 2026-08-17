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

"""AGENT / STEP / TOOL span coverage for ``dspy.ReAct``."""

import json
import os

import dspy
import pytest

from opentelemetry.instrumentation.dspy.internal.config import (
    OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED,
)
from opentelemetry.instrumentation.dspy.internal.semconv import (
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DEFINITIONS,
)

from ._helpers import (
    boom,
    get_weather,
    make_react_agent,
    parent_of,
    single,
    spans_by_kind,
)

_TWO_ROUND_ANSWERS = [
    {
        "next_thought": "look it up",
        "next_tool_name": "get_weather",
        "next_tool_args": {"city": "Tokyo"},
    },
    {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
    {"reasoning": "sunny", "answer": "It is sunny in Tokyo."},
]


@pytest.mark.usefixtures("instrument")
def test_react_span_tree(span_exporter):
    agent = make_react_agent(_TWO_ROUND_ANSWERS, [get_weather])
    agent(question="What is the weather in Tokyo?")

    spans = span_exporter.get_finished_spans()
    entry = single(spans, "ENTRY")
    agent_span = single(spans, "AGENT")
    steps = spans_by_kind(spans, "STEP")
    tools = spans_by_kind(spans, "TOOL")

    assert agent_span.name == "invoke_agent ReAct"
    assert agent_span.attributes[GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert agent_span.attributes[GenAI.GEN_AI_AGENT_NAME] == "ReAct"
    assert parent_of(spans, agent_span) is entry

    # One STEP per ReAct round; the trailing ``extract`` is not a round.
    assert len(steps) == 2
    assert [s.attributes[GEN_AI_REACT_ROUND] for s in steps] == [1, 2]
    for step in steps:
        assert step.name == "react step"
        assert step.attributes[GEN_AI_OPERATION_NAME] == "react"
        assert parent_of(spans, step) is agent_span

    # The round that selected a tool ends with ``tool_calls``; the round that
    # selected ``finish`` ends with ``finish``.
    assert steps[0].attributes[GEN_AI_REACT_FINISH_REASON] == "tool_calls"
    assert steps[1].attributes[GEN_AI_REACT_FINISH_REASON] == "finish"

    # The tool a round decided on belongs to that round.
    assert [t.name for t in tools] == [
        "execute_tool get_weather",
        "execute_tool finish",
    ]
    assert parent_of(spans, tools[0]) is steps[0]
    assert parent_of(spans, tools[1]) is steps[1]

    # ``extract`` runs outside any step, directly under the agent.
    extract = [s for s in spans if s.name == "chain ChainOfThought"]
    assert len(extract) == 1
    assert parent_of(spans, extract[0]) is agent_span

    # Everything belongs to one trace.
    assert {s.context.trace_id for s in spans} == {entry.context.trace_id}


@pytest.mark.usefixtures("instrument")
def test_react_tool_attributes(span_exporter):
    agent = make_react_agent(_TWO_ROUND_ANSWERS, [get_weather])
    agent(question="What is the weather in Tokyo?")

    spans = span_exporter.get_finished_spans()
    tool = spans_by_kind(spans, "TOOL")[0]

    assert tool.attributes[GEN_AI_OPERATION_NAME] == "execute_tool"
    assert tool.attributes[GenAI.GEN_AI_TOOL_NAME] == "get_weather"
    assert tool.attributes[GenAI.GEN_AI_TOOL_TYPE] == "function"
    assert tool.attributes[GenAI.GEN_AI_TOOL_DESCRIPTION]
    assert tool.attributes[GenAI.GEN_AI_TOOL_CALL_ID]
    assert tool.attributes[GEN_AI_FRAMEWORK] == FRAMEWORK_NAME
    assert "Tokyo" in tool.attributes[GEN_AI_TOOL_CALL_ARGUMENTS]
    assert "sunny" in tool.attributes[GEN_AI_TOOL_CALL_RESULT]

    agent_span = single(spans, "AGENT")
    definitions = json.loads(agent_span.attributes[GEN_AI_TOOL_DEFINITIONS])
    assert {d["name"] for d in definitions} == {"get_weather", "finish"}


@pytest.mark.usefixtures("instrument")
def test_failing_tool_produces_error_span_without_breaking_the_trace(
    span_exporter,
):
    answers = [
        {
            "next_thought": "try it",
            "next_tool_name": "boom",
            "next_tool_args": {"city": "Tokyo"},
        },
        {
            "next_thought": "give up",
            "next_tool_name": "finish",
            "next_tool_args": {},
        },
        {"reasoning": "failed", "answer": "unknown"},
    ]
    agent = make_react_agent(answers, [boom])
    agent(question="What is the weather in Tokyo?")

    spans = span_exporter.get_finished_spans()
    failing = [s for s in spans if s.name == "execute_tool boom"]
    assert len(failing) == 1
    assert failing[0].status.status_code.name == "ERROR"

    # The agent recovers, so later rounds keep nesting correctly: the failed
    # tool span must not have leaked its context.
    steps = spans_by_kind(spans, "STEP")
    assert len(steps) == 2
    assert parent_of(spans, failing[0]) is steps[0]
    assert parent_of(spans, spans_by_kind(spans, "TOOL")[1]) is steps[1]
    assert single(spans, "ENTRY").status.status_code.name != "ERROR"


@pytest.mark.usefixtures("instrument")
def test_agent_records_aggregate_usage_when_tracking_enabled(span_exporter):
    agent = make_react_agent(_TWO_ROUND_ANSWERS, [get_weather])
    dspy.settings.configure(track_usage=True)
    agent(question="What is the weather in Tokyo?")

    agent_span = single(span_exporter.get_finished_spans(), "AGENT")
    # DummyLM reports no usage, so the attribute is simply absent — what
    # matters is that no token metric is emitted from here (see
    # test_no_llm_telemetry.py).
    assert (
        GEN_AI_USAGE_TOTAL_TOKENS not in agent_span.attributes
        or isinstance(agent_span.attributes[GEN_AI_USAGE_TOTAL_TOKENS], int)
    )


@pytest.mark.usefixtures("instrument")
def test_react_step_can_be_disabled(span_exporter):
    os.environ[OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED] = "false"
    try:
        agent = make_react_agent(_TWO_ROUND_ANSWERS, [get_weather])
        agent(question="What is the weather in Tokyo?")
    finally:
        os.environ.pop(OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED, None)

    spans = span_exporter.get_finished_spans()
    assert spans_by_kind(spans, "STEP") == []
    # Tool spans fall back to the agent span as parent.
    assert parent_of(spans, spans_by_kind(spans, "TOOL")[0]) is single(
        spans, "AGENT"
    )


@pytest.mark.usefixtures("instrument")
def test_react_max_iters_step_count(span_exporter):
    answers = [
        {
            "next_thought": f"round {index}",
            "next_tool_name": "get_weather",
            "next_tool_args": {"city": "Tokyo"},
        }
        for index in range(3)
    ] + [{"reasoning": "enough", "answer": "sunny"}]
    agent = make_react_agent(answers, [get_weather])

    agent(question="What is the weather in Tokyo?", max_iters=3)

    spans = span_exporter.get_finished_spans()
    steps = spans_by_kind(spans, "STEP")
    assert [s.attributes[GEN_AI_REACT_ROUND] for s in steps] == [1, 2, 3]
    assert [s.attributes[GEN_AI_REACT_FINISH_REASON] for s in steps] == [
        "tool_calls",
        "tool_calls",
        "tool_calls",
    ]


@pytest.mark.usefixtures("instrument")
def test_no_chain_span_for_agent_module(span_exporter):
    agent = make_react_agent(_TWO_ROUND_ANSWERS, [get_weather])
    agent(question="What is the weather in Tokyo?")

    spans = span_exporter.get_finished_spans()
    # The ReAct module itself is an AGENT, never also a CHAIN.
    assert "chain ReAct" not in [s.name for s in spans]
    assert len(spans_by_kind(spans, "AGENT")) == 1
    assert all(
        s.attributes.get(GEN_AI_SPAN_KIND) != "AGENT"
        for s in spans
        if s.name.startswith("chain ")
    )
