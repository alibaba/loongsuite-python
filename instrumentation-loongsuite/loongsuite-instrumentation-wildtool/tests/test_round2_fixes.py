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

"""Round 2 regression tests covering the H1 / H2 / M1 / M2 / M3 fixes.

See ``llm-dev/execute.md`` § "修订记录 (Round 2 fix)" and
``example-deploy/validation/SUMMARY.md`` for the original validation gaps
addressed by these tests.
"""

import json

import pytest
from wtb.model_handler.base_handler import BaseHandler

from opentelemetry.trace import StatusCode


class _StubHandler(BaseHandler):
    """Minimal handler with controllable LLM responses (no real network)."""

    def __init__(self):
        super().__init__("test-model", 0.0)
        self._step_responses = []
        self._step_idx = 0

    def _request_tool_call(self, inference_data):
        resp = self._step_responses[self._step_idx]
        self._step_idx += 1
        if isinstance(resp, Exception):
            raise resp
        return resp, 0.05

    def _parse_api_response(self, api_response):
        data = json.loads(api_response.json())
        choice = data["choices"][0]
        message = choice["message"]
        return {
            "reasoning_content": None,
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "input_token": data["usage"]["prompt_tokens"],
            "output_token": data["usage"]["completion_tokens"],
        }


def _spans_by_kind(spans, kind):
    return [
        s
        for s in spans
        if (s.attributes or {}).get("gen_ai.span.kind") == kind
    ]


def _spans_named(spans, name):
    return [s for s in spans if s.name == name]


def _step_for_round(spans, round_num):
    for s in _spans_named(spans, "react step"):
        attrs = s.attributes or {}
        if attrs.get("gen_ai.react.round") == round_num:
            return s
    raise AssertionError(f"no STEP span found for round={round_num}")


# ============================================================================
# H1: TOOL span parent_span_id == STEP span_id (was CHAIN before fix)
# ============================================================================


class TestToolParentIsStep:
    def test_single_tool_parent_is_step_round_one(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """The single TOOL span in simple_test_entry should be a child of the
        first STEP span (round=1), not the CHAIN span."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        tool_spans = _spans_by_kind(spans, "TOOL")
        assert len(tool_spans) == 1, [s.name for s in spans]

        tool = tool_spans[0]
        step_round1 = _step_for_round(spans, 1)
        chain = _spans_by_kind(spans, "CHAIN")[0]

        # H1 core assertion: parent is STEP, not CHAIN.
        assert tool.parent is not None
        assert tool.parent.span_id == step_round1.context.span_id, (
            "TOOL parent should be STEP round=1, got "
            f"{tool.parent.span_id} (STEP={step_round1.context.span_id}, "
            f"CHAIN={chain.context.span_id})"
        )
        assert tool.parent.span_id != chain.context.span_id

        # And trace_id of course remains consistent.
        assert tool.context.trace_id == step_round1.context.trace_id

    def test_multi_step_each_tool_parented_to_correct_step(
        self,
        span_exporter,
        instrument,
        tool_call_response_factory,
        text_response_factory,
    ):
        """multi-step scenario: 2 successful tool steps + 1 prepare_to_answer.

        Each TOOL span must be parented to the STEP span of its own round,
        not to the CHAIN or to a different round's STEP.
        """
        handler = _StubHandler()
        # Test entry with 2 tool steps (search, lookup) then prepare_to_answer.
        test_entry = {
            "id": "wild_tool_bench_multi_001",
            "english_env_info": "2025-01-01",
            "english_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search items",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up details",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                            "required": ["id"],
                        },
                    },
                },
            ],
            "english_tasks": ["Find and summarize item X"],
            "english_answer_list": [
                [
                    {
                        "action": {"name": "search", "arguments": {"q": "X"}},
                        "observation": "found:item_42",
                        "dependency_list": [],
                    },
                    {
                        "action": {
                            "name": "lookup",
                            "arguments": {"id": "item_42"},
                        },
                        "observation": "details:hello",
                        "dependency_list": [0],
                    },
                    {
                        "action": {
                            "name": "prepare_to_answer",
                            "arguments": {},
                        },
                        "observation": "Item X is hello.",
                        "dependency_list": [1],
                    },
                ]
            ],
        }

        resp_step1 = tool_call_response_factory(
            "search", {"q": "X"}, "call_search_1"
        )
        resp_step2 = tool_call_response_factory(
            "lookup", {"id": "item_42"}, "call_lookup_1"
        )
        resp_step3 = text_response_factory("Item X is hello.")
        handler._step_responses = [resp_step1, resp_step2, resp_step3]

        handler.inference_multi_turn(test_entry)

        spans = span_exporter.get_finished_spans()
        tool_spans = sorted(
            _spans_by_kind(spans, "TOOL"),
            key=lambda s: (s.attributes or {}).get("gen_ai.tool.name") or "",
        )
        assert len(tool_spans) == 2, [s.name for s in spans]

        step_round1 = _step_for_round(spans, 1)
        step_round2 = _step_for_round(spans, 2)
        chain = _spans_by_kind(spans, "CHAIN")[0]

        lookup_tool = next(
            t
            for t in tool_spans
            if (t.attributes or {}).get("gen_ai.tool.name") == "lookup"
        )
        search_tool = next(
            t
            for t in tool_spans
            if (t.attributes or {}).get("gen_ai.tool.name") == "search"
        )

        # search → STEP round=1, lookup → STEP round=2
        assert search_tool.parent.span_id == step_round1.context.span_id
        assert lookup_tool.parent.span_id == step_round2.context.span_id
        # Neither parented on CHAIN (the regression we are fixing)
        for t in tool_spans:
            assert t.parent.span_id != chain.context.span_id
            assert t.context.trace_id == chain.context.trace_id


# ============================================================================
# M1: CHAIN span carries input.value and output.value
# ============================================================================


class TestChainInputOutputValue:
    def test_chain_input_value_and_output_value(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        chain_spans = _spans_by_kind(spans, "CHAIN")
        assert len(chain_spans) == 1
        attrs = dict(chain_spans[0].attributes or {})

        # input.value: last user message of the chain (prepared by wtb's
        # _pre_messages_processing which appends the current task as user).
        assert "input.value" in attrs, attrs
        assert attrs["input.value"] == "What is the weather in Beijing?"

        # output.value: JSON containing action_name_label, task_idx, is_optimal.
        assert "output.value" in attrs, attrs
        out = json.loads(attrs["output.value"])
        assert out["action_name_label"] == "correct"
        assert out["task_idx"] == 0
        assert out["is_optimal"] is True

    def test_chain_input_value_truncated_when_long(
        self,
        span_exporter,
        instrument,
        tool_call_response_factory,
        text_response_factory,
    ):
        """Very long user content should be truncated to keep span attribute small."""
        handler = _StubHandler()
        long_text = "x" * 5000
        test_entry = {
            "id": "wild_tool_bench_long_001",
            "english_env_info": "2025-01-01",
            "english_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "noop",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "english_tasks": [long_text],
            "english_answer_list": [
                [
                    {
                        "action": {
                            "name": "prepare_to_answer",
                            "arguments": {},
                        },
                        "observation": "ok",
                        "dependency_list": [],
                    }
                ]
            ],
        }
        handler._step_responses = [text_response_factory("ok")]

        handler.inference_multi_turn(test_entry)

        spans = span_exporter.get_finished_spans()
        chain = _spans_by_kind(spans, "CHAIN")[0]
        attrs = dict(chain.attributes or {})
        assert "input.value" in attrs
        # Default cap is 4096; truncated form must be <= cap + suffix length.
        assert len(attrs["input.value"]) <= 4096 + len("...(truncated)")
        assert attrs["input.value"].startswith("xxx")


# ============================================================================
# M2: STEP span carries gen_ai.react.finish_reason on error paths
# ============================================================================


class TestStepFinishReason:
    def test_finish_reason_action_name_mismatch(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
    ):
        handler = _StubHandler()
        # wrong tool name → wtb's "action name not in candidate" branch
        handler._step_responses = [
            tool_call_response_factory("wrong_tool", {"x": 1}, "call_bad")
        ]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        steps = _spans_named(spans, "react step")
        assert len(steps) == 1
        attrs = dict(steps[0].attributes or {})
        assert (
            attrs.get("gen_ai.react.finish_reason") == "action_name_mismatch"
        )

    def test_finish_reason_empty_response(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        make_completion,
    ):
        """Empty content + no tool_calls → STEP gets finish_reason=empty_response."""
        from tests.conftest import (
            FakeChatCompletion,
            _make_chat_completion_response,
        )

        handler = _StubHandler()
        handler._step_responses = [
            FakeChatCompletion(
                _make_chat_completion_response(content="", tool_calls=None)
            )
        ]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        steps = _spans_named(spans, "react step")
        assert len(steps) == 1
        attrs = dict(steps[0].attributes or {})
        assert attrs.get("gen_ai.react.finish_reason") == "empty_response"

    def test_finish_reason_request_exception(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
    ):
        """Exception in _request_tool_call → STEP ERROR + finish_reason=error."""
        handler = _StubHandler()
        handler._step_responses = [RuntimeError("Boom")]

        with pytest.raises(RuntimeError):
            handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        steps = _spans_named(spans, "react step")
        assert len(steps) == 1
        attrs = dict(steps[0].attributes or {})
        assert steps[0].status.status_code == StatusCode.ERROR
        assert attrs.get("gen_ai.react.finish_reason") == "error"

    def test_finish_reason_omitted_on_success(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """Successful steps should NOT have a finish_reason (per execute.md)."""
        handler = _StubHandler()
        handler._step_responses = [
            tool_call_response_factory(
                "get_weather", {"city": "Beijing"}, "call_001"
            ),
            text_response_factory("OK"),
        ]
        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        for s in _spans_named(spans, "react step"):
            attrs = dict(s.attributes or {})
            assert "gen_ai.react.finish_reason" not in attrs, (
                f"unexpected finish_reason on success step round="
                f"{attrs.get('gen_ai.react.round')}: {attrs.get('gen_ai.react.finish_reason')}"
            )


# ============================================================================
# M3: TOOL span carries gen_ai.tool.call.arguments / result / description
#     (and keeps wildtool.tool.execution_mode)
# ============================================================================


class TestToolSensitiveAttributes:
    def test_tool_args_result_description_and_execution_mode(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("Sunny day")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        tool_spans = _spans_by_kind(spans, "TOOL")
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes or {})

        # M3 explicit attrs.
        args_attr = attrs.get("gen_ai.tool.call.arguments")
        assert args_attr is not None
        assert json.loads(args_attr) == {"city": "Beijing"}

        # observation comes from the appended {"role": "tool", ...} message
        # written by wtb after the call matches the answer; it's a string.
        result_attr = attrs.get("gen_ai.tool.call.result")
        assert result_attr == "Sunny, 25°C", attrs

        # description sourced from inference_data["tools"][i].function.description
        assert attrs.get("gen_ai.tool.description") == "Get weather for a city"

        # Existing custom attribute must still be present.
        assert (
            attrs.get("wildtool.tool.execution_mode") == "ground_truth_replay"
        )


# ============================================================================
# H2: STEP span carries gen_ai.system / gen_ai.provider.name fallback
# ============================================================================


class TestStepProviderFallback:
    def test_step_has_provider_name_fallback(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        handler = _StubHandler()
        handler._step_responses = [
            tool_call_response_factory(
                "get_weather", {"city": "Beijing"}, "call_001"
            ),
            text_response_factory("OK"),
        ]
        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        steps = _spans_named(spans, "react step")
        assert len(steps) == 2
        for s in steps:
            attrs = dict(s.attributes or {})
            assert attrs.get("gen_ai.system") == "openai", attrs
            assert attrs.get("gen_ai.provider.name") == "openai", attrs
