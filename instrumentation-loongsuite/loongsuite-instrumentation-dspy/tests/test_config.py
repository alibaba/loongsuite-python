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

"""Configuration switches."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from opentelemetry.instrumentation.dspy.internal.config import (
    OTEL_INSTRUMENTATION_DSPY_CAPTURE_ENTRY_SPAN,
    OTEL_INSTRUMENTATION_DSPY_CAPTURE_MODEL_NAME,
    OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO,
    root_sample_ratio,
)
from opentelemetry.instrumentation.dspy.internal.semconv import (
    GEN_AI_REQUEST_MODEL,
)

from ._helpers import get_weather, make_react_agent, single, span_kinds


@pytest.fixture(name="env")
def fixture_env(monkeypatch):
    return monkeypatch


@pytest.mark.usefixtures("instrument")
def test_entry_span_can_be_disabled(env, span_exporter):
    env.setenv(OTEL_INSTRUMENTATION_DSPY_CAPTURE_ENTRY_SPAN, "false")
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))

    dspy.Predict("question->answer")(question="What color is the sky?")

    spans = span_exporter.get_finished_spans()
    assert span_kinds(spans) == ["CHAIN"]
    assert spans[0].parent is None


@pytest.mark.usefixtures("instrument")
def test_model_name_can_be_disabled(env, span_exporter):
    env.setenv(OTEL_INSTRUMENTATION_DSPY_CAPTURE_MODEL_NAME, "false")
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))

    dspy.Predict("question->answer")(question="What color is the sky?")

    chain = single(span_exporter.get_finished_spans(), "CHAIN")
    assert GEN_AI_REQUEST_MODEL not in chain.attributes


@pytest.mark.usefixtures("instrument")
def test_zero_sample_ratio_drops_the_whole_subtree(env, span_exporter):
    env.setenv(OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO, "0")
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

    result = agent(question="What is the weather in Tokyo?")

    # No partial tree: nothing at all, and the program still ran.
    assert result.answer == "sunny"
    assert span_exporter.get_finished_spans() == ()


@pytest.mark.usefixtures("instrument")
def test_sampling_recovers_after_a_dropped_run(env, span_exporter):
    predict = dspy.Predict("question->answer")
    dspy.settings.configure(lm=DummyLM([{"answer": "a"}, {"answer": "b"}]))

    env.setenv(OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO, "0")
    predict(question="first")
    env.setenv(OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO, "1")
    predict(question="second")

    assert span_kinds(span_exporter.get_finished_spans()) == ["CHAIN", "ENTRY"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 1.0),
        ("", 1.0),
        ("0.25", 0.25),
        ("-1", 0.0),
        ("7", 1.0),
        ("nope", 1.0),
    ],
)
def test_root_sample_ratio_parsing(env, raw, expected):
    if raw is None:
        env.delenv(OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO, raising=False)
    else:
        env.setenv(OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO, raw)
    assert root_sample_ratio() == expected
