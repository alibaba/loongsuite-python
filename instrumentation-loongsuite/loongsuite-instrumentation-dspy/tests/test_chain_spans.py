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

"""ENTRY and CHAIN span coverage."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from opentelemetry.instrumentation.dspy.internal.semconv import (
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SPAN_KIND,
    INPUT_VALUE,
    OUTPUT_VALUE,
)

from ._helpers import parent_of, single, span_kinds


@pytest.mark.usefixtures("instrument")
def test_predict_emits_entry_and_chain(span_exporter):
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))

    prediction = dspy.Predict("question->answer")(
        question="What color is the sky?"
    )

    assert prediction.answer == "blue"
    spans = span_exporter.get_finished_spans()
    assert span_kinds(spans) == ["CHAIN", "ENTRY"]

    chain = single(spans, "CHAIN")
    entry = single(spans, "ENTRY")

    assert chain.name == "chain Predict"
    assert chain.attributes[GEN_AI_OPERATION_NAME] == "task"
    assert chain.attributes[GEN_AI_FRAMEWORK] == FRAMEWORK_NAME
    assert chain.attributes[GEN_AI_REQUEST_MODEL] == "dummy"

    assert entry.name == "enter_ai_application_system"
    assert entry.attributes[GEN_AI_OPERATION_NAME] == "enter"
    assert entry.attributes[GEN_AI_FRAMEWORK] == FRAMEWORK_NAME

    # The chain hangs off the entry, and the entry is the trace root.
    assert parent_of(spans, chain) is entry
    assert entry.parent is None
    assert chain.context.trace_id == entry.context.trace_id


@pytest.mark.usefixtures("instrument")
def test_nested_module_builds_chain_tree(span_exporter):
    class Pipeline(dspy.Module):
        def __init__(self):
            super().__init__()
            self.step = dspy.Predict("question->answer")

        def forward(self, question):
            return self.step(question=question)

    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))
    Pipeline()(question="What color is the sky?")

    spans = span_exporter.get_finished_spans()
    chains = [
        s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "CHAIN"
    ]
    assert [s.name for s in chains] == ["chain Predict", "chain Pipeline"]

    predict_span, pipeline_span = chains
    # A composite module is a workflow, a leaf predictor is a task.
    assert pipeline_span.attributes[GEN_AI_OPERATION_NAME] == "workflow"
    assert predict_span.attributes[GEN_AI_OPERATION_NAME] == "task"
    assert parent_of(spans, predict_span) is pipeline_span
    assert parent_of(spans, pipeline_span) is single(spans, "ENTRY")


@pytest.mark.usefixtures("instrument")
def test_chain_captures_input_and_output(span_exporter):
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))
    dspy.Predict("question->answer")(question="What color is the sky?")

    chain = single(span_exporter.get_finished_spans(), "CHAIN")
    assert "What color is the sky?" in chain.attributes[INPUT_VALUE]
    assert "blue" in chain.attributes[OUTPUT_VALUE]


@pytest.mark.usefixtures("instrument_no_content")
def test_chain_omits_content_when_disabled(span_exporter):
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))
    dspy.Predict("question->answer")(question="What color is the sky?")

    chain = single(span_exporter.get_finished_spans(), "CHAIN")
    assert INPUT_VALUE not in chain.attributes
    assert OUTPUT_VALUE not in chain.attributes


@pytest.mark.usefixtures("instrument")
def test_module_exception_marks_chain_and_entry_as_error(span_exporter):
    class Failing(dspy.Module):
        def forward(self, **kwargs):
            raise ValueError("nope")

    with pytest.raises(ValueError):
        Failing()(question="anything")

    spans = span_exporter.get_finished_spans()
    chain = single(spans, "CHAIN")
    entry = single(spans, "ENTRY")
    assert chain.status.status_code.name == "ERROR"
    assert entry.status.status_code.name == "ERROR"
