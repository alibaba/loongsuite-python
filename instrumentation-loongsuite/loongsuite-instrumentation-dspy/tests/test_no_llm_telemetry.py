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

"""LLM / EMBEDDING telemetry stays out of scope.

``loongsuite-instrumentation-litellm`` is the single source of LLM and
EMBEDDING spans and of every token measurement. This instrumentation must not
duplicate either, otherwise traces show two spans per model call and token
dashboards double-count.
"""

import dspy
import pytest
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM

from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.dspy.internal.callback import (
    OTelDSPyCallback,
)

from ._helpers import get_weather, make_react_agent, span_kinds

_ANSWERS = [
    {
        "next_thought": "look it up",
        "next_tool_name": "get_weather",
        "next_tool_args": {"city": "Tokyo"},
    },
    {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
    {"reasoning": "sunny", "answer": "sunny"},
]


@pytest.mark.usefixtures("instrument")
def test_no_llm_or_embedding_spans(span_exporter):
    agent = make_react_agent(_ANSWERS, [get_weather])
    agent(question="What is the weather in Tokyo?")

    kinds = set(span_kinds(span_exporter.get_finished_spans()))
    assert kinds == {"ENTRY", "AGENT", "STEP", "CHAIN", "TOOL"}
    assert "LLM" not in kinds
    assert "EMBEDDING" not in kinds


@pytest.mark.usefixtures("instrument")
def test_no_lm_callbacks_are_implemented():
    # Leaving the lm hooks at their base no-op implementation is what keeps
    # DSPy from producing a second LLM span per model call.
    for hook in ("on_lm_start", "on_lm_end"):
        assert getattr(OTelDSPyCallback, hook) is getattr(BaseCallback, hook)


@pytest.mark.usefixtures("instrument")
def test_framework_spans_emit_no_token_metrics(span_exporter, metric_reader):
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]), track_usage=True)

    dspy.Predict("question->answer")(question="What color is the sky?")

    metrics_data = metric_reader.get_metrics_data()
    recorded = []
    for resource_metric in (
        metrics_data.resource_metrics if metrics_data else []
    ):
        for scope_metric in resource_metric.scope_metrics:
            recorded.extend(metric.name for metric in scope_metric.metrics)

    assert not any("token" in name for name in recorded), recorded


@pytest.mark.usefixtures("instrument")
def test_instrumentation_does_not_suppress_downstream_probes(span_exporter):
    captured = []

    class _ProbingLM(DummyLM):
        def __call__(self, *args, **kwargs):
            captured.append(
                otel_context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
            )
            return super().__call__(*args, **kwargs)

    dspy.settings.configure(lm=_ProbingLM([{"answer": "blue"}]))
    dspy.Predict("question->answer")(question="What color is the sky?")

    # Suppressing instrumentation here would silence the litellm probe that
    # this plugin depends on for LLM spans and tokens.
    assert captured == [None]
