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

"""The ``acall`` / ``aforward`` path produces the same span tree."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.dspy.internal.semconv import (
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
)

from ._helpers import parent_of, single, spans_by_kind

_ANSWERS = [
    {
        "next_thought": "look it up",
        "next_tool_name": "get_weather",
        "next_tool_args": {"city": "Tokyo"},
    },
    {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
    {"reasoning": "sunny", "answer": "sunny"},
]


async def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"The weather in {city} is sunny."


@pytest.mark.asyncio
@pytest.mark.usefixtures("instrument")
async def test_async_predict_span_tree(span_exporter):
    # ``dspy.configure`` is owned by one async task; overrides belong in
    # ``dspy.context`` inside a coroutine.
    with dspy.context(lm=DummyLM([{"answer": "blue"}])):
        await dspy.Predict("question->answer").acall(
            question="What color is the sky?"
        )

    spans = span_exporter.get_finished_spans()
    chain = single(spans, "CHAIN")
    entry = single(spans, "ENTRY")
    assert parent_of(spans, chain) is entry
    assert otel_context.get_current() == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("instrument")
async def test_async_react_step_tree(span_exporter):
    agent = dspy.ReAct("question->answer", tools=[get_weather])

    with dspy.context(lm=DummyLM(_ANSWERS)):
        await agent.acall(question="What is the weather in Tokyo?")

    spans = span_exporter.get_finished_spans()
    steps = spans_by_kind(spans, "STEP")
    tools = spans_by_kind(spans, "TOOL")
    agent_span = single(spans, "AGENT")

    assert [s.attributes[GEN_AI_REACT_ROUND] for s in steps] == [1, 2]
    assert [s.attributes[GEN_AI_REACT_FINISH_REASON] for s in steps] == [
        "tool_calls",
        "finish",
    ]
    assert parent_of(spans, tools[0]) is steps[0]
    assert parent_of(spans, steps[0]) is agent_span
    assert otel_context.get_current() == {}
