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

"""Tests for ENTRY span (P1: multi_threaded_inference).

Module-level imports of ``wtb._llm_response_generation.multi_threaded_inference``
must be avoided: ``wrapt.wrap_function_wrapper`` patches the attribute on the
module, but a pre-imported local binding still references the original
unwrapped function. All tests therefore import the symbol lazily after the
``instrument`` fixture has run.
"""

import json

import pytest
from wtb.model_handler.base_handler import BaseHandler

from opentelemetry.trace import StatusCode


class _StubHandler(BaseHandler):
    """Minimal handler subclass for testing.

    Overrides ``inference`` so the multi_threaded_inference wrapper invokes a
    deterministic, side-effect-free body that returns a fake result dict and
    therefore exercises only the ENTRY span codepath.
    """

    def __init__(self):
        super().__init__("test-model", 0.0)

    def _request_tool_call(self, inference_data):
        raise NotImplementedError

    def _parse_api_response(self, api_response):
        raise NotImplementedError

    def inference(self, test_entry):
        return [
            {
                "action_name_label": "correct",
                "is_optimal": True,
                "inference_log": {},
                "latency": [0.1],
                "input_token_count": [10],
                "output_token_count": [5],
            }
        ]


class TestEntrySpan:
    def test_entry_span_created(self, span_exporter, instrument):
        """ENTRY span should be created with correct attributes."""
        from wtb._llm_response_generation import multi_threaded_inference  # noqa: I001, PLC0415

        handler = _StubHandler()
        test_case = {
            "id": "wild_tool_bench_test_001",
            "english_tasks": ["task1", "task2"],
        }

        result = multi_threaded_inference(handler, "gpt-4o", test_case)

        assert result is not None
        assert result["id"] == "wild_tool_bench_test_001"

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) == 1

        span = entry_spans[0]
        attrs = dict(span.attributes or {})
        assert attrs.get("gen_ai.span.kind") == "ENTRY"
        assert attrs.get("gen_ai.operation.name") == "enter"
        assert attrs.get("gen_ai.framework") == "wildtool"
        assert attrs.get("gen_ai.session.id") == "wild_tool_bench_test_001"
        assert attrs.get("gen_ai.request.model") == "gpt-4o"
        assert attrs.get("wildtool.turn_count") == 2
        # ENTRY spans rely on default OTel status semantics: success leaves
        # the span UNSET, failures explicitly mark it ERROR.
        assert span.status.status_code != StatusCode.ERROR

    def test_entry_span_captures_input_and_output_messages(
        self,
        span_exporter,
        instrument,
    ):
        """ENTRY span should always carry GenAI input/output messages."""

        from opentelemetry.instrumentation.wildtool._wrappers import (  # noqa: PLC0415
            WildToolEntryWrapper,
        )

        wrapper = WildToolEntryWrapper(instrument._handler)
        test_case = {
            "id": "wild_tool_bench_test_messages",
            "english_tasks": ["Search for the capital of France"],
        }

        def _success(handler, model_name, test_case):
            return [
                {
                    "action_name_label": "correct",
                    "is_optimal": True,
                    "inference_log": {
                        "step_0": {
                            "inference_output": {
                                "content": "Paris is the capital of France."
                            }
                        }
                    },
                }
            ]

        wrapper(_success, None, (_StubHandler(), "gpt-4o", test_case), {})

        spans = span_exporter.get_finished_spans()
        entry_span = [
            s for s in spans if s.name == "enter_ai_application_system"
        ][0]
        attrs = dict(entry_span.attributes or {})
        input_messages = json.loads(attrs["gen_ai.input.messages"])
        output_messages = json.loads(attrs["gen_ai.output.messages"])

        assert input_messages[0]["role"] == "user"
        assert (
            input_messages[0]["parts"][0]["content"]
            == "Search for the capital of France"
        )
        assert output_messages[0]["role"] == "assistant"
        assert (
            output_messages[0]["parts"][0]["content"]
            == "Paris is the capital of France."
        )

    def test_entry_span_error_path(self, span_exporter, instrument):
        """The ENTRY wrapper marks the span ERROR when the wrapped callable
        raises an unhandled exception.

        ``multi_threaded_inference`` swallows non-rate-limit errors itself
        (see test_error_scenarios.test_entry_span_captures_retry_error_path
        for that path). To exercise the wrapper's failure branch directly we
        invoke the underlying ``WildToolEntryWrapper`` with a callable that
        deliberately raises, bypassing ``multi_threaded_inference``'s own
        error handling.
        """
        from opentelemetry.instrumentation.wildtool._wrappers import (  # noqa: PLC0415
            WildToolEntryWrapper,
        )

        wrapper = WildToolEntryWrapper(instrument._handler)

        def _raising(handler, model_name, test_case):
            raise RuntimeError("API connection failed")

        handler = _StubHandler()
        test_case = {
            "id": "wild_tool_bench_test_002",
            "english_tasks": ["task1"],
        }

        with pytest.raises(RuntimeError, match="API connection failed"):
            wrapper(_raising, None, (handler, "gpt-4o", test_case), {})

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) == 1
        span = entry_spans[0]
        assert span.status.status_code == StatusCode.ERROR
