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

"""Cross-plugin context propagation.

LLM spans are produced by ``loongsuite-instrumentation-litellm`` deeper in the
DSPy call stack. They only join the DSPy trace if every DSPy framework span is
attached to the OpenTelemetry context, so these tests stand in for the LiteLLM
instrumentation by creating an LLM span from inside the LM call.
"""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.dspy.internal.semconv import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import LLMInvocation

from ._helpers import (
    get_weather,
    make_react_agent,
    parent_of,
    single,
    spans_by_kind,
)


class _SpanEmittingLM(DummyLM):
    """A ``DummyLM`` that emits an LLM span the way the litellm probe would."""

    otel_handler = None

    # ``__call__`` is the entry point every supported DSPy version routes
    # through; ``DummyLM.forward`` only exists in newer ones.
    def __call__(self, *args, **kwargs):
        handler = self.otel_handler
        invocation = LLMInvocation(request_model="dummy", provider="dashscope")
        handler.start_llm(invocation)
        try:
            response = super().__call__(*args, **kwargs)
        except Exception:
            handler.abandon_llm(invocation)
            raise
        invocation.input_tokens = 7
        invocation.output_tokens = 11
        handler.stop_llm(invocation)
        return response


def _llm_span_emitting_lm(answers, tracer_provider):
    lm = _SpanEmittingLM(answers)
    lm.otel_handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    return lm


@pytest.mark.usefixtures("instrument")
def test_llm_span_nests_under_chain(span_exporter, tracer_provider):
    dspy.settings.configure(
        lm=_llm_span_emitting_lm([{"answer": "blue"}], tracer_provider)
    )

    dspy.Predict("question->answer")(question="What color is the sky?")

    spans = span_exporter.get_finished_spans()
    llm = single(spans, "LLM")
    chain = single(spans, "CHAIN")
    entry = single(spans, "ENTRY")

    assert parent_of(spans, llm) is chain
    assert llm.context.trace_id == entry.context.trace_id


@pytest.mark.usefixtures("instrument")
def test_llm_span_nests_under_react_step(span_exporter, tracer_provider):
    answers = [
        {
            "next_thought": "look it up",
            "next_tool_name": "get_weather",
            "next_tool_args": {"city": "Tokyo"},
        },
        {
            "next_thought": "done",
            "next_tool_name": "finish",
            "next_tool_args": {},
        },
        {"reasoning": "sunny", "answer": "It is sunny in Tokyo."},
    ]
    agent = make_react_agent(answers, [get_weather])
    dspy.settings.configure(lm=_llm_span_emitting_lm(answers, tracer_provider))

    agent(question="What is the weather in Tokyo?")

    spans = span_exporter.get_finished_spans()
    llms = spans_by_kind(spans, "LLM")
    steps = spans_by_kind(spans, "STEP")
    assert len(llms) == 3

    # Reasoning LLM calls sit under CHAIN -> STEP -> AGENT.
    for llm in llms[:2]:
        chain = parent_of(spans, llm)
        assert chain.attributes[GEN_AI_SPAN_KIND] == "CHAIN"
        assert parent_of(spans, chain) in steps

    assert {s.context.trace_id for s in spans} == {
        single(spans, "ENTRY").context.trace_id
    }


@pytest.mark.usefixtures("instrument")
def test_context_is_restored_after_the_run(span_exporter):
    before = otel_context.get_current()

    agent = make_react_agent(
        [
            {
                "next_thought": "look it up",
                "next_tool_name": "get_weather",
                "next_tool_args": {"city": "Tokyo"},
            },
            {
                "next_thought": "done",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {"reasoning": "sunny", "answer": "sunny"},
        ],
        [get_weather],
    )
    agent(question="What is the weather in Tokyo?")

    assert otel_context.get_current() == before


@pytest.mark.usefixtures("instrument")
def test_context_is_restored_after_a_failure(span_exporter):
    before = otel_context.get_current()

    class Failing(dspy.Module):
        def forward(self, **kwargs):
            raise ValueError("nope")

    with pytest.raises(ValueError):
        Failing()(question="anything")

    assert otel_context.get_current() == before


@pytest.mark.usefixtures("instrument")
def test_sequential_runs_do_not_share_parents(span_exporter):
    dspy.settings.configure(lm=DummyLM([{"answer": "a"}, {"answer": "b"}]))
    predict = dspy.Predict("question->answer")

    predict(question="first")
    predict(question="second")

    spans = span_exporter.get_finished_spans()
    entries = spans_by_kind(spans, "ENTRY")
    assert len(entries) == 2
    # Two independent traces, each rooted at its own ENTRY.
    assert all(entry.parent is None for entry in entries)
    assert len({entry.context.trace_id for entry in entries}) == 2
