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

"""Data extraction helpers."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from opentelemetry.instrumentation.dspy.internal._utils import (
    aggregate_lm_usage,
    build_retrieval_documents,
    extract_query,
    extract_tool_definitions,
    extract_top_k,
    normalize_callback_inputs,
    resolve_request_model,
    safe_json,
    to_plain,
)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({"args": (), "kwargs": {"question": "q"}}, {"question": "q"}),
        ({"kwargs": {"city": "Tokyo"}}, {"city": "Tokyo"}),
        ({"args": ("q",), "kwargs": {}}, ["q"]),
        ({"args": ("q",), "kwargs": {"k": 3}}, {"args": ["q"], "k": 3}),
        ({"question": "q"}, {"question": "q"}),
        ({}, {}),
        ("raw", "raw"),
    ],
)
def test_normalize_callback_inputs(inputs, expected):
    assert normalize_callback_inputs(inputs) == expected


def test_to_plain_unwraps_dspy_containers():
    prediction = dspy.Prediction(answer="blue")
    assert to_plain(prediction) == {"answer": "blue"}
    assert to_plain([prediction]) == [{"answer": "blue"}]
    assert to_plain({"pred": prediction}) == {"pred": {"answer": "blue"}}


def test_safe_json_truncates_and_never_raises():
    class Unserializable:
        def __repr__(self):
            return "x" * 100

    assert safe_json({"a": Unserializable()}, max_len=20).endswith(
        "...[truncated]"
    )
    assert safe_json({"answer": "blue"}) == '{"answer": "blue"}'


def test_resolve_request_model_strips_the_provider_prefix():
    # The litellm instrumentation records ``gen_ai.request.model`` without the
    # provider prefix; the framework spans must match it exactly.
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
    assert resolve_request_model() == "gpt-4o-mini"

    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}]))
    assert resolve_request_model() == "dummy"

    dspy.settings.configure(lm=None)
    assert resolve_request_model() is None


def test_extract_tool_definitions():
    def get_weather(city: str) -> str:
        """Return the weather."""
        return "sunny"

    agent = dspy.ReAct("question->answer", tools=[get_weather])
    definitions = extract_tool_definitions(agent)

    assert {d.name for d in definitions} == {"get_weather", "finish"}
    weather = next(d for d in definitions if d.name == "get_weather")
    assert weather.type == "function"
    assert weather.description
    assert "city" in weather.parameters


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [({"query": "q"}, "q"), (["q"], "q"), ({}, None), ({"query": 3}, None)],
)
def test_extract_query(inputs, expected):
    assert extract_query(inputs) == expected


def test_extract_top_k_prefers_the_call_over_the_instance():
    retrieve = dspy.Retrieve(k=3)
    assert extract_top_k({"k": 7}, retrieve) == 7.0
    assert extract_top_k({}, retrieve) == 3.0
    # A nonsensical value is omitted rather than coerced.
    assert extract_top_k({"k": True}, retrieve) is None


def test_build_retrieval_documents_synthesizes_ordinal_ids():
    documents = build_retrieval_documents(
        dspy.Prediction(passages=["first", "second"])
    )
    assert [d.id for d in documents] == ["0", "1"]
    assert all(d.score is None for d in documents)


def test_build_retrieval_documents_keeps_real_scores():
    documents = build_retrieval_documents(
        [{"long_text": "first", "score": 0.9}, {"text": "second"}]
    )
    assert [d.score for d in documents] == [0.9, None]


def test_aggregate_lm_usage_sums_across_models():
    prediction = dspy.Prediction(answer="blue")
    prediction.set_lm_usage(
        {
            "openai/gpt-4o": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            "openai/gpt-4o-mini": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        }
    )

    assert aggregate_lm_usage(prediction) == {
        "prompt_tokens": 15,
        "completion_tokens": 3,
        "total_tokens": 18,
    }


def test_aggregate_lm_usage_without_tracking():
    assert aggregate_lm_usage(dspy.Prediction(answer="blue")) == {}
    assert aggregate_lm_usage(None) == {}
