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

"""Unit tests for the framework-agnostic helpers."""

import contextvars

import pytest


def test_state_lifecycle():
    from opentelemetry.instrumentation.bfclv4.internal.state import (
        bump_round,
        bump_turn,
        get_state,
        init_state,
        next_tool_index,
        reset_state,
    )

    token = init_state()
    try:
        state = get_state()
        assert state == {"turn_idx": 0, "fc_round": 0, "tool_index": 0}

        assert bump_round() == 1
        assert bump_round() == 2
        assert bump_turn() == 1
        # bump_turn resets fc_round
        state = get_state()
        assert state["turn_idx"] == 1
        assert state["fc_round"] == 0
        assert next_tool_index() == 0
        assert next_tool_index() == 1
    finally:
        reset_state(token)

    # After reset the state should be gone (None default).
    assert get_state() is None


def test_context_propagating_executor_carries_contextvars():
    from opentelemetry.instrumentation.bfclv4.internal.threading_propagation import (
        ContextPropagatingExecutor,
    )

    cv: contextvars.ContextVar[str] = contextvars.ContextVar(
        "bfclv4_test_cv", default="default"
    )
    cv.set("from_main_thread")

    def _read():
        return cv.get()

    with ContextPropagatingExecutor(max_workers=2) as pool:
        future = pool.submit(_read)
        assert future.result() == "from_main_thread"


def test_extract_tool_name_and_arguments():
    from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
        _extract_tool_arguments,
        _extract_tool_name,
        _parse_python_call_arguments,
    )

    assert _extract_tool_name("calc.add(1, 2)") == "add"
    assert _extract_tool_name("list_files()") == "list_files"
    assert _extract_tool_name("not a call") == "unknown"
    assert _extract_tool_arguments("foo(a=1, b=2)") == "a=1, b=2"
    assert _extract_tool_arguments("foo()") is None
    assert _parse_python_call_arguments("foo(a=1, b='x')") == {
        "a": 1,
        "b": "x",
    }


def test_infer_finish_reason_heuristic():
    from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
        _infer_finish_reason,
    )

    assert _infer_finish_reason([]) == "empty_response"
    assert _infer_finish_reason([[]]) == "empty_response"
    assert _infer_finish_reason([{"name": "x"}]) == "tool_calls"
    assert _infer_finish_reason("plain string") == "stop"
    assert _infer_finish_reason(None) == "unknown"


def test_test_entry_to_messages_extracts_genai_content():
    from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
        _test_entry_to_messages,
    )

    test_entry = {
        "id": "simple_001",
        "system_prompt": "Use the provided tools.",
        "question": [
            [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            [{"role": "assistant", "content": "I will check."}],
        ],
    }

    inputs, system_instructions = _test_entry_to_messages(test_entry)

    assert [message.role for message in inputs] == ["user", "assistant"]
    assert inputs[0].parts[0].content == "What is the weather in Paris?"
    assert inputs[1].parts[0].content == "I will check."
    assert [part.content for part in system_instructions] == [
        "Use the provided tools.",
        "Answer concisely.",
    ]


def test_test_entry_to_tool_definitions_extracts_bfcl_functions():
    from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
        _test_entry_to_tool_definitions,
        _tool_description_map,
    )

    test_entry = {
        "id": "simple_001",
        "function": [
            {
                "name": "get_weather",
                "description": "Get weather information.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a flight.",
                    "parameters": {"type": "object"},
                },
            },
        ],
        "missed_function": {
            "1": [
                {
                    "name": "cancel_booking",
                    "description": "Cancel a booking.",
                    "parameters": {"type": "object"},
                }
            ]
        },
    }

    definitions = _test_entry_to_tool_definitions(test_entry)

    assert [definition.name for definition in definitions] == [
        "get_weather",
        "book_flight",
        "cancel_booking",
    ]
    assert definitions[0].type == "function"
    assert definitions[0].parameters["required"] == ["location"]
    assert _tool_description_map(test_entry)["get_weather"] == (
        "Get weather information."
    )


def test_result_to_output_messages_extracts_last_inference_log_output():
    from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
        _result_to_output_messages,
    )

    outputs = _result_to_output_messages(
        {
            "inference_log": {
                "step_0": {"inference_output": {"content": "intermediate"}},
                "step_1": {"inference_output": {"content": "final"}},
            }
        }
    )

    assert len(outputs) == 1
    assert outputs[0].role == "assistant"
    assert outputs[0].parts[0].content == '{"content": "final"}'
    assert outputs[0].finish_reason == "stop"


def test_provider_mapping_without_bfcl(monkeypatch):
    from opentelemetry.instrumentation.bfclv4.internal.provider import (
        infer_provider,
    )

    pytest.importorskip(
        "opentelemetry.util.genai.extended_types",
    )

    class _Dummy:
        model_style = None

    name, extras = infer_provider(_Dummy())
    # If bfcl-eval is not installed, ``ModelStyle`` import fails and we get
    # ``unknown``; otherwise we still get ``unknown`` because ``model_style``
    # is None.
    assert name == "unknown"
    assert extras == {}
