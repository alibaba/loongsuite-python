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

"""Instrumentor lifecycle: registration, re-registration and clean removal."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from opentelemetry.instrumentation.dspy import DSPyInstrumentor
from opentelemetry.instrumentation.dspy.internal.callback import (
    OTelDSPyCallback,
)


def _registered_callbacks():
    return [
        item
        for item in (dspy.settings.get("callbacks") or [])
        if isinstance(item, OTelDSPyCallback)
    ]


def test_instrumentation_dependencies():
    assert DSPyInstrumentor().instrumentation_dependencies() == (
        "dspy >= 3.0.0",
    )


@pytest.mark.usefixtures("instrument")
def test_callback_and_patch_are_installed():
    assert len(_registered_callbacks()) == 1
    wrapper = dspy.ReAct._call_with_potential_trajectory_truncation
    assert wrapper.__module__.endswith("instrumentation.dspy.internal.patch")


def test_uninstrument_removes_callback_and_restores_react(
    tracer_provider, span_exporter
):
    original = dspy.ReAct._call_with_potential_trajectory_truncation
    original_async = (
        dspy.ReAct._async_call_with_potential_trajectory_truncation
    )

    instrumentor = DSPyInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    assert len(_registered_callbacks()) == 1
    instrumentor.uninstrument()

    assert _registered_callbacks() == []
    assert dspy.ReAct._call_with_potential_trajectory_truncation is original
    assert (
        dspy.ReAct._async_call_with_potential_trajectory_truncation
        is original_async
    )

    span_exporter.clear()
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))
    dspy.Predict("question->answer")(question="What color is the sky?")
    assert span_exporter.get_finished_spans() == ()


@pytest.mark.usefixtures("instrument")
def test_user_configure_does_not_drop_the_callback():
    class _UserCallback(dspy.utils.callback.BaseCallback):
        pass

    dspy.configure(callbacks=[_UserCallback()])

    assert len(_registered_callbacks()) == 1
    assert any(
        isinstance(item, _UserCallback)
        for item in dspy.settings.get("callbacks")
    )


def test_instrumenting_twice_registers_one_callback(
    tracer_provider, span_exporter
):
    instrumentor = DSPyInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        assert len(_registered_callbacks()) == 1
    finally:
        instrumentor.uninstrument()
        span_exporter.clear()
