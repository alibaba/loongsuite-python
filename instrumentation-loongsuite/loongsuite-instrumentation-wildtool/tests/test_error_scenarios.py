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

"""Tests for error/edge-case scenarios."""

import json

import pytest
from wtb.model_handler.base_handler import BaseHandler

from opentelemetry.trace import StatusCode


class _StubHandler(BaseHandler):
    """Handler with controllable step responses."""

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


class TestErrorScenarios:
    def test_action_name_mismatch(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        tool_call_response_factory,
    ):
        """When model calls wrong tool, CHAIN span should still be OK with error label."""
        handler = _StubHandler()
        # Model calls wrong_tool instead of get_weather
        resp0 = tool_call_response_factory("wrong_tool", {"x": 1}, "call_bad")
        handler._step_responses = [resp0]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("workflow")]
        assert len(chain_spans) == 1

        chain = chain_spans[0]
        attrs = dict(chain.attributes or {})
        assert attrs.get("wildtool.action_name_label") == "error"
        assert chain.status.status_code == StatusCode.OK

    def test_empty_response(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
        make_completion,
    ):
        """When model returns no content and no tool_calls, process terminates gracefully."""
        from tests.conftest import (
            FakeChatCompletion,
            _make_chat_completion_response,
        )

        handler = _StubHandler()
        resp = FakeChatCompletion(
            _make_chat_completion_response(content="", tool_calls=None)
        )
        handler._step_responses = [resp]

        handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("workflow")]
        assert len(chain_spans) == 1
        attrs = dict(chain_spans[0].attributes or {})
        assert attrs.get("wildtool.action_name_label") == "error"

    def test_request_tool_call_exception_sets_error(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
    ):
        """Exception in _request_tool_call should produce ERROR on STEP span and propagate."""
        handler = _StubHandler()
        handler._step_responses = [RuntimeError("Connection timeout")]

        with pytest.raises(RuntimeError, match="Connection timeout"):
            handler.inference_multi_turn(simple_test_entry)

        spans = span_exporter.get_finished_spans()
        step_spans = [s for s in spans if s.name == "react step"]
        assert len(step_spans) == 1
        assert step_spans[0].status.status_code == StatusCode.ERROR

        chain_spans = [s for s in spans if s.name.startswith("workflow")]
        assert len(chain_spans) == 1
        assert chain_spans[0].status.status_code == StatusCode.ERROR

    def test_entry_span_captures_retry_error_path(
        self,
        span_exporter,
        instrument,
    ):
        """multi_threaded_inference catches non-rate-limit errors and returns error dict.
        ENTRY span should still complete successfully (not raise)."""
        from wtb._llm_response_generation import multi_threaded_inference

        handler = _StubHandler()

        def failing_inference(test_entry):
            raise ValueError("Invalid JSON from model")

        handler.inference = failing_inference

        test_case = {
            "id": "wild_tool_bench_err_001",
            "english_tasks": ["task1"],
        }

        # multi_threaded_inference catches non-rate-limit errors
        result = multi_threaded_inference(handler, "test-model", test_case)
        assert "Error during inference" in result["result"]

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) == 1
        # multi_threaded_inference's own try/except converts the error into a
        # normal return, so the ENTRY wrapper observes a successful call and
        # leaves the span at the default UNSET status (definitely not ERROR).
        span = entry_spans[0]
        assert span.status.status_code != StatusCode.ERROR
