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

"""Shared helpers for DSPy instrumentation tests."""

from __future__ import annotations

from typing import Any

import dspy
from dspy.utils.dummies import DummyLM

from opentelemetry.instrumentation.dspy.internal.semconv import (
    GEN_AI_SPAN_KIND,
)


def spans_by_kind(spans: Any, kind: str) -> list[Any]:
    return [s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == kind]


def span_kinds(spans: Any) -> list[str]:
    return [s.attributes.get(GEN_AI_SPAN_KIND) for s in spans]


def single(spans: Any, kind: str) -> Any:
    matches = spans_by_kind(spans, kind)
    assert len(matches) == 1, (
        f"expected exactly one {kind} span, got {len(matches)}"
    )
    return matches[0]


def parent_of(spans: Any, span: Any) -> Any:
    if span.parent is None:
        return None
    for candidate in spans:
        if candidate.context.span_id == span.parent.span_id:
            return candidate
    return None


def make_react_agent(answers: list[dict[str, Any]], tools: list[Any]):
    """Configure a DummyLM and return a ReAct agent driven by *answers*."""
    dspy.settings.configure(lm=DummyLM(answers))
    return dspy.ReAct("question->answer", tools=tools)


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"The weather in {city} is sunny."


def boom(city: str) -> str:
    """Always fail."""
    raise RuntimeError("tool exploded")
