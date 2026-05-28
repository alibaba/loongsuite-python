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

"""Tests for AGENT span (P2: inference_multi_turn)."""

import json

from wtb.model_handler.base_handler import BaseHandler


class _StubHandler(BaseHandler):
    """Minimal handler subclass for testing AGENT span."""

    def __init__(self):
        super().__init__("test-model", 0.0)
        self._step_responses = []
        self._step_idx = 0

    def _request_tool_call(self, inference_data):
        resp = self._step_responses[self._step_idx]
        self._step_idx += 1
        return resp, 0.1

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


class TestAgentSpan:
    def test_agent_span_attributes(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        make_completion,
        tool_call_response_factory,
        text_response_factory,
    ):
        """AGENT span should exist with correct attributes and token aggregation."""
        handler = _StubHandler()

        # Step 0: model returns tool call for get_weather
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        # Step 1: model returns text (prepare_to_answer match)
        resp1 = text_response_factory(
            "The weather in Beijing is Sunny, 25°C",
            input_tokens=20,
            output_tokens=15,
        )
        handler._step_responses = [resp0, resp1]

        result = handler.inference_multi_turn(simple_test_entry)
        assert result is not None

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1

        span = agent_spans[0]
        assert span.name == "invoke_agent _StubHandler"
        attrs = dict(span.attributes or {})
        assert attrs.get("gen_ai.span.kind") == "AGENT"
        assert attrs.get("gen_ai.operation.name") == "invoke_agent"
        assert attrs.get("gen_ai.framework") == "wildtool"
        assert attrs.get("gen_ai.agent.name") == "_StubHandler"
        assert (
            attrs.get("gen_ai.conversation.id") == "wild_tool_bench_test_001"
        )
        assert attrs.get("gen_ai.request.model") == "test-model"
        assert attrs.get("wildtool.turn_count") == 1

        assert attrs.get("gen_ai.usage.input_tokens") == 30
        assert attrs.get("gen_ai.usage.output_tokens") == 20

    def test_agent_span_captures_input_and_output_messages(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """AGENT span should always carry GenAI input/output messages."""

        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        agent_span = [s for s in spans if "invoke_agent" in s.name][0]
        attrs = dict(agent_span.attributes or {})
        input_messages = json.loads(attrs["gen_ai.input.messages"])
        output_messages = json.loads(attrs["gen_ai.output.messages"])

        assert input_messages[0]["role"] == "user"
        assert (
            input_messages[0]["parts"][0]["content"]
            == "What is the weather in Beijing?"
        )
        assert output_messages[0]["role"] == "assistant"
        assert (
            output_messages[0]["parts"][0]["content"]
            == "The weather in Beijing is Sunny, 25°C"
        )

    def test_agent_parent_is_entry(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
        text_response_factory,
    ):
        """When called via multi_threaded_inference, AGENT span should be child of ENTRY."""
        from wtb._llm_response_generation import multi_threaded_inference  # noqa: I001, PLC0415

        handler = _StubHandler()
        resp0 = tool_call_response_factory(
            "get_weather", {"city": "Beijing"}, "call_001"
        )
        resp1 = text_response_factory("The weather in Beijing is Sunny, 25°C")
        handler._step_responses = [resp0, resp1]

        test_case = simple_test_entry.copy()
        multi_threaded_inference(handler, "test-model", test_case)

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        agent_spans = [s for s in spans if "invoke_agent" in s.name]

        assert len(entry_spans) == 1
        assert len(agent_spans) == 1

        entry = entry_spans[0]
        agent = agent_spans[0]
        assert agent.context.trace_id == entry.context.trace_id
        assert agent.parent is not None
        assert agent.parent.span_id == entry.context.span_id
