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

"""Tests for CHAIN / STEP / TOOL spans (P3, P4, P5)."""

import json

from wtb.model_handler.base_handler import BaseHandler


class _StubHandler(BaseHandler):
    """Minimal handler subclass with controllable responses."""

    def __init__(self):
        super().__init__("test-model", 0.0)
        self._step_responses = []
        self._step_idx = 0

    def _request_tool_call(self, inference_data):
        resp = self._step_responses[self._step_idx]
        self._step_idx += 1
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


class TestChainSpan:
    def test_chain_span_per_task(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """Each task should produce one CHAIN span with correct attributes."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("workflow")]
        assert len(chain_spans) == 1

        chain = chain_spans[0]
        assert chain.name == "workflow task_0"
        attrs = dict(chain.attributes or {})
        assert attrs.get("gen_ai.span.kind") == "CHAIN"
        assert attrs.get("gen_ai.operation.name") == "workflow"
        assert attrs.get("gen_ai.framework") == "wildtool"
        assert attrs.get("wildtool.task_idx") == 0
        assert (
            attrs.get("wildtool.test_entry_id") == "wild_tool_bench_test_001"
        )
        assert attrs.get("wildtool.action_name_label") == "correct"
        assert attrs.get("wildtool.is_optimal") is True

    def test_chain_parent_is_agent(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """CHAIN span should be child of AGENT span."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        chain_spans = [s for s in spans if s.name.startswith("workflow")]

        assert len(agent_spans) == 1
        assert len(chain_spans) == 1

        agent = agent_spans[0]
        chain = chain_spans[0]
        assert chain.context.trace_id == agent.context.trace_id
        assert chain.parent is not None
        assert chain.parent.span_id == agent.context.span_id


class TestStepSpans:
    def test_step_spans_per_chain(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """Each _request_tool_call invocation should produce a STEP span."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) == 2

        attrs0 = dict(step_spans[0].attributes or {})
        attrs1 = dict(step_spans[1].attributes or {})
        rounds = sorted(
            [
                attrs0.get("gen_ai.react.round"),
                attrs1.get("gen_ai.react.round"),
            ]
        )
        assert rounds == [1, 2]

        for ss in step_spans:
            a = dict(ss.attributes or {})
            assert a.get("gen_ai.span.kind") == "STEP"
            assert a.get("gen_ai.operation.name") == "react"

    def test_step_parent_is_chain(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """STEP spans should be children of CHAIN span."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("workflow")]
        step_spans = [s for s in spans if s.name == "react step"]

        assert len(chain_spans) == 1
        chain = chain_spans[0]

        for ss in step_spans:
            assert ss.context.trace_id == chain.context.trace_id
            assert ss.parent is not None
            assert ss.parent.span_id == chain.context.span_id

    def test_step_token_attributes(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """STEP span should have gen_ai.usage.input_tokens and output_tokens."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory(
            "The weather in Beijing is Sunny, 25°C",
            input_tokens=25,
            output_tokens=12,
        )
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        step_spans = sorted(
            [s for s in spans if s.name == "react step"],
            key=lambda s: s.attributes.get("gen_ai.react.round", 0),
        )
        assert len(step_spans) == 2

        # First step: default 10 input, 5 output from make_completion defaults
        a0 = dict(step_spans[0].attributes or {})
        assert a0.get("gen_ai.usage.input_tokens") == 10
        assert a0.get("gen_ai.usage.output_tokens") == 5

        # Second step: 25 input, 12 output
        a1 = dict(step_spans[1].attributes or {})
        assert a1.get("gen_ai.usage.input_tokens") == 25
        assert a1.get("gen_ai.usage.output_tokens") == 12


class TestToolSpans:
    def test_tool_span_attributes(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """TOOL span should have correct attributes including execution_mode."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1

        tool = tool_spans[0]
        assert tool.name == "execute_tool get_weather"
        attrs = dict(tool.attributes or {})
        assert attrs.get("gen_ai.span.kind") == "TOOL"
        assert attrs.get("gen_ai.operation.name") == "execute_tool"
        assert attrs.get("gen_ai.tool.name") == "get_weather"
        assert attrs.get("gen_ai.tool.type") == "function"
        assert (
            attrs.get("wildtool.tool.execution_mode") == "ground_truth_replay"
        )

    def test_tool_span_parent_is_chain(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """TOOL spans share the CHAIN trace_id (parent is STEP after Round 2)."""
        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("workflow")]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        assert len(chain_spans) == 1
        assert len(tool_spans) >= 1

        chain = chain_spans[0]
        for ts in tool_spans:
            assert ts.context.trace_id == chain.context.trace_id


class TestSpanHierarchy:
    def test_full_hierarchy(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """Verify ENTRY → AGENT → CHAIN → STEP hierarchy and consistent trace_id."""
        from wtb._llm_response_generation import multi_threaded_inference

        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        test_case = simple_test_entry.copy()
        multi_threaded_inference(handler, "test-model", test_case)

        spans = span_exporter.get_finished_spans()

        entry = [s for s in spans if s.name == "enter_ai_application_system"]
        agent = [s for s in spans if "invoke_agent" in s.name]
        chain = [s for s in spans if s.name.startswith("workflow")]
        step = [s for s in spans if s.name == "react step"]
        tool = [s for s in spans if "execute_tool" in s.name]

        assert len(entry) == 1
        assert len(agent) == 1
        assert len(chain) == 1
        assert len(step) == 2
        assert len(tool) >= 1

        trace_id = entry[0].context.trace_id
        for s in spans:
            assert s.context.trace_id == trace_id

        # AGENT parent = ENTRY
        assert agent[0].parent.span_id == entry[0].context.span_id
        # CHAIN parent = AGENT
        assert chain[0].parent.span_id == agent[0].context.span_id
        # STEP parent = CHAIN
        for s in step:
            assert s.parent.span_id == chain[0].context.span_id
