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

"""Supplemental tests to increase code coverage for uncovered branches.

Covers:
- utils.py: safe_json_dumps
- _wrappers.py helper functions: _stringify, _truncate, _tasks_to_input_messages,
  _task_results_to_output_messages, _extract_task_results, _extract_task_result_output,
  _extract_output_from_inference_log, _step_log_sort_key, _extract_finish_reason,
  _extract_input_value, _derive_step_finish_reason
- _wrappers.py edge-case branches in wrapper classes
- __init__.py: ensure_handler_class_patched, uninstrument edge cases
"""

import json
from unittest.mock import MagicMock

from wtb.model_handler.base_handler import BaseHandler

# ============================================================================
# utils.py
# ============================================================================


class TestSafeJsonDumps:
    def test_none_returns_none(self):
        from opentelemetry.instrumentation.wildtool.utils import (
            safe_json_dumps,
        )

        assert safe_json_dumps(None) is None

    def test_dict_serialized(self):
        from opentelemetry.instrumentation.wildtool.utils import (
            safe_json_dumps,
        )

        result = safe_json_dumps({"key": "value"})
        assert result == '{"key": "value"}'

    def test_long_string_truncated(self):
        from opentelemetry.instrumentation.wildtool.utils import (
            safe_json_dumps,
        )

        obj = {"data": "x" * 20000}
        result = safe_json_dumps(obj, max_length=100)
        assert len(result) <= 100 + len("...(truncated)")
        assert result.endswith("...(truncated)")

    def test_non_serializable_returns_str(self):
        from opentelemetry.instrumentation.wildtool.utils import (
            safe_json_dumps,
        )

        class Unserializable:
            def __repr__(self):
                return "Unserializable()"

        result = safe_json_dumps(Unserializable())
        assert "Unserializable" in result

    def test_short_string_not_truncated(self):
        from opentelemetry.instrumentation.wildtool.utils import (
            safe_json_dumps,
        )

        result = safe_json_dumps({"a": 1}, max_length=10000)
        assert result == '{"a": 1}'


# ============================================================================
# _wrappers.py helper functions
# ============================================================================


class TestStringify:
    def test_string_passthrough(self):
        from opentelemetry.instrumentation.wildtool._wrappers import _stringify

        assert _stringify("hello") == "hello"

    def test_dict_serialized(self):
        from opentelemetry.instrumentation.wildtool._wrappers import _stringify

        assert _stringify({"a": 1}) == '{"a": 1}'

    def test_non_serializable_uses_str(self):
        from opentelemetry.instrumentation.wildtool._wrappers import _stringify

        class BadObj:
            def __repr__(self):
                return "BadObj()"

        result = _stringify(BadObj())
        assert "BadObj" in result


class TestTruncate:
    def test_short_text_unchanged(self):
        from opentelemetry.instrumentation.wildtool._wrappers import _truncate

        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        from opentelemetry.instrumentation.wildtool._wrappers import _truncate

        result = _truncate("a" * 200, 50)
        assert len(result) == 50 + len("...(truncated)")
        assert result.endswith("...(truncated)")


class TestTasksToInputMessages:
    def test_non_dict_returns_empty(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _tasks_to_input_messages,
        )

        assert _tasks_to_input_messages("not a dict") == []

    def test_non_list_tasks_returns_empty(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _tasks_to_input_messages,
        )

        assert _tasks_to_input_messages({"english_tasks": "not a list"}) == []

    def test_skips_empty_tasks(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _tasks_to_input_messages,
        )

        result = _tasks_to_input_messages(
            {"english_tasks": [None, "", [], {}, "real task"]}
        )
        assert len(result) == 1
        assert result[0].parts[0].content == "real task"

    def test_valid_tasks(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _tasks_to_input_messages,
        )

        result = _tasks_to_input_messages(
            {"english_tasks": ["task1", "task2"]}
        )
        assert len(result) == 2
        assert result[0].role == "user"


class TestTaskResultsToOutputMessages:
    def test_empty_content_skipped(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _task_results_to_output_messages,
        )

        # Task result with no extractable output
        result = _task_results_to_output_messages([{}])
        assert len(result) == 0

    def test_with_final_answer(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _task_results_to_output_messages,
        )

        result = _task_results_to_output_messages(
            [{"final_answer": "The answer is 42"}]
        )
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].parts[0].content == "The answer is 42"

    def test_error_finish_reason(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _task_results_to_output_messages,
        )

        result = _task_results_to_output_messages(
            [{"action_name_label": "error", "final_answer": "something"}]
        )
        assert len(result) == 1
        assert result[0].finish_reason == "error"


class TestExtractTaskResults:
    def test_list_passthrough(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        data = [{"a": 1}]
        assert _extract_task_results(data) is data

    def test_non_dict_non_list_returns_empty(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        assert _extract_task_results(42) == []
        assert _extract_task_results("string") == []

    def test_dict_with_result_list(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        result = _extract_task_results({"result": [{"a": 1}]})
        assert result == [{"a": 1}]

    def test_dict_with_result_dict(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        result = _extract_task_results({"result": {"a": 1}})
        assert result == [{"a": 1}]

    def test_dict_with_result_scalar(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        result = _extract_task_results({"result": "some string"})
        assert result == ["some string"]

    def test_dict_with_action_name_label(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        data = {"action_name_label": "correct", "is_optimal": True}
        result = _extract_task_results(data)
        assert result == [data]

    def test_dict_with_inference_log(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        data = {"inference_log": {"step_0": {}}}
        result = _extract_task_results(data)
        assert result == [data]

    def test_dict_with_inference_output(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        data = {"inference_output": {"content": "hi"}}
        result = _extract_task_results(data)
        assert result == [data]

    def test_dict_with_final_answer(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        data = {"final_answer": "42"}
        result = _extract_task_results(data)
        assert result == [data]

    def test_empty_dict_returns_empty(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        assert _extract_task_results({}) == []

    def test_dict_with_answers_key(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        result = _extract_task_results({"answers": [{"a": 1}]})
        assert result == [{"a": 1}]

    def test_dict_with_none_result_falls_through(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_results,
        )

        # All standard keys are None or empty; but has is_optimal
        data = {"result": None, "results": None, "is_optimal": True}
        result = _extract_task_results(data)
        assert result == [data]


class TestExtractTaskResultOutput:
    def test_non_dict_returns_itself(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        assert _extract_task_result_output("raw string") == "raw string"
        assert _extract_task_result_output(42) == 42

    def test_dict_with_final_answer(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        assert _extract_task_result_output({"final_answer": "42"}) == "42"

    def test_dict_with_answer(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        assert _extract_task_result_output({"answer": "hello"}) == "hello"

    def test_dict_with_output(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        assert _extract_task_result_output({"output": "data"}) == "data"

    def test_label_is_optimal_fallback(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        result = _extract_task_result_output(
            {"action_name_label": "correct", "is_optimal": True}
        )
        assert result == {"action_name_label": "correct", "is_optimal": True}

    def test_no_extractable_output_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        assert _extract_task_result_output({"unrelated_key": 123}) is None

    def test_only_is_optimal(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_task_result_output,
        )

        result = _extract_task_result_output({"is_optimal": False})
        assert result == {"action_name_label": None, "is_optimal": False}


class TestExtractOutputFromInferenceLog:
    def test_non_dict_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        assert _extract_output_from_inference_log(None) is None
        assert _extract_output_from_inference_log("string") is None
        assert _extract_output_from_inference_log([]) is None

    def test_step_data_not_dict_skipped(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log({"step_0": "not a dict"})
        assert result is None

    def test_extracts_content_from_output(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log(
            {"step_0": {"inference_output": {"content": "Hello world"}}}
        )
        assert result == "Hello world"

    def test_extracts_reasoning_content(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log(
            {
                "step_0": {
                    "inference_output": {"reasoning_content": "I think..."}
                }
            }
        )
        assert result == "I think..."

    def test_extracts_error_reason(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log(
            {
                "step_0": {
                    "inference_output": {
                        "error_reason": "parse tool_calls failed"
                    }
                }
            }
        )
        assert result == "parse tool_calls failed"

    def test_extracts_observation_from_answer(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log(
            {
                "step_0": {
                    "inference_output": {},
                    "inference_answer": {
                        "candidate_0_answer_function_list": {
                            "observation": "Sunny, 25C"
                        }
                    },
                }
            }
        )
        assert result == "Sunny, 25C"

    def test_returns_answer_dict_if_no_observation(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log(
            {
                "step_0": {
                    "inference_output": {},
                    "inference_answer": {"some_key": "some_value"},
                }
            }
        )
        assert result == {"some_key": "some_value"}

    def test_prefers_latest_step(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_output_from_inference_log,
        )

        result = _extract_output_from_inference_log(
            {
                "step_0": {"inference_output": {"content": "first"}},
                "step_1": {"inference_output": {"content": "second"}},
            }
        )
        assert result == "second"


class TestStepLogSortKey:
    def test_valid_key(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _step_log_sort_key,
        )

        assert _step_log_sort_key("step_0") == 0
        assert _step_log_sort_key("step_42") == 42

    def test_invalid_key_returns_negative(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _step_log_sort_key,
        )

        assert _step_log_sort_key("step_abc") == -1
        assert _step_log_sort_key("step_") == -1


class TestExtractFinishReason:
    def test_error_label_returns_error(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_finish_reason,
        )

        assert (
            _extract_finish_reason({"action_name_label": "error"}) == "error"
        )

    def test_correct_label_returns_stop(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_finish_reason,
        )

        assert (
            _extract_finish_reason({"action_name_label": "correct"}) == "stop"
        )

    def test_non_dict_returns_stop(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_finish_reason,
        )

        assert _extract_finish_reason("not a dict") == "stop"

    def test_no_label_returns_stop(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _extract_finish_reason,
        )

        assert _extract_finish_reason({}) == "stop"


class TestDeriveStepFinishReason:
    def test_non_error_label_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        assert (
            WildToolChainWrapper._derive_step_finish_reason("correct", "")
            is None
        )

    def test_parse_tool_calls_failed(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._derive_step_finish_reason(
            "error", "parse tool_calls failed in response"
        )
        assert result == "parse_tool_calls_failed"

    def test_action_name_mismatch(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._derive_step_finish_reason(
            "error", "action name not in candidate"
        )
        assert result == "action_name_mismatch"

    def test_empty_response(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._derive_step_finish_reason(
            "error", "tool_calls and content are None"
        )
        assert result == "empty_response"

    def test_generic_error(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._derive_step_finish_reason(
            "error", "something else went wrong"
        )
        assert result == "error"


class TestExtractInputValue:
    def test_non_dict_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        assert WildToolChainWrapper._extract_input_value("not a dict") is None

    def test_non_list_messages_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        assert (
            WildToolChainWrapper._extract_input_value({"messages": "bad"})
            is None
        )

    def test_no_user_messages_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._extract_input_value(
            {"messages": [{"role": "assistant", "content": "hi"}]}
        )
        assert result is None

    def test_skips_non_dict_messages(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._extract_input_value(
            {"messages": ["not a dict", {"role": "user", "content": "hello"}]}
        )
        assert result == "hello"

    def test_skips_none_content(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._extract_input_value(
            {"messages": [{"role": "user", "content": None}]}
        )
        assert result is None

    def test_returns_last_user_message(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        result = WildToolChainWrapper._extract_input_value(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ]
            }
        )
        assert result == "second"

    def test_no_messages_key_returns_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        assert WildToolChainWrapper._extract_input_value({}) is None


# ============================================================================
# Wrapper edge cases
# ============================================================================


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


class TestRequestWrapperOutsideChain:
    """Test that the request wrapper is a no-op when not inside a chain."""

    def test_request_outside_chain_is_passthrough(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolRequestWrapper,
        )

        wrapper = WildToolRequestWrapper(instrument._handler)
        called = []

        def fake_request(*args, **kwargs):
            called.append(True)
            return ("response", 0.1)

        # _in_chain defaults to False
        result = wrapper(fake_request, None, (), {})
        assert result == ("response", 0.1)
        assert len(called) == 1


class TestChainWrapperEdgeCases:
    def test_chain_with_non_dict_inference_data(
        self,
        span_exporter,
        instrument,
    ):
        """Chain wrapper should handle non-dict inference_data gracefully."""
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, instrument)

        def fake_wrapped(*args, **kwargs):
            return {"action_name_label": "correct", "is_optimal": True}

        # Pass non-dict inference_data
        result = wrapper(fake_wrapped, None, ("not a dict",), {})
        assert result["action_name_label"] == "correct"

    def test_chain_with_none_instance(
        self,
        span_exporter,
        instrument,
    ):
        """Chain wrapper should handle None instance (no subclass patching)."""
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, instrument)

        def fake_wrapped(*args, **kwargs):
            return {"action_name_label": "correct", "is_optimal": True}

        result = wrapper(fake_wrapped, None, ({},), {})
        assert result is not None


class TestEnsureHandlerClassPatched:
    def test_skip_already_patched(self, tracer_provider):
        """ensure_handler_class_patched should skip classes already patched."""
        from opentelemetry.instrumentation.wildtool import WildToolInstrumentor

        instrumentor = WildToolInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        handler = _StubHandler()
        cls = type(handler)

        # First call patches
        instrumentor.ensure_handler_class_patched(cls)
        assert cls in instrumentor._patched_handler_classes

        # Second call should be a no-op (already patched)
        instrumentor.ensure_handler_class_patched(cls)
        assert cls in instrumentor._patched_handler_classes

        instrumentor.uninstrument()

    def test_skip_method_not_in_dict(self, tracer_provider):
        """ensure_handler_class_patched should skip methods not overridden."""
        from opentelemetry.instrumentation.wildtool import WildToolInstrumentor

        instrumentor = WildToolInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )

        # Create a handler class that does NOT override _request_tool_call
        class MinimalHandler(BaseHandler):
            pass  # No overrides, methods inherited from BaseHandler

        instrumentor.ensure_handler_class_patched(MinimalHandler)
        assert MinimalHandler in instrumentor._patched_handler_classes

        instrumentor.uninstrument()


class TestParseToolCallsFailed:
    """Test the parse_tool_calls_failed finish_reason path."""

    def test_finish_reason_parse_tool_calls_failed(
        self,
        span_exporter,
        instrument,
        simple_test_entry,
    ):
        """Simulate a parse_tool_calls_failed scenario by directly calling
        the chain wrapper with an inference_log containing that error_reason."""
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
            _step_invocation,
        )
        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        wrapper = WildToolChainWrapper(instrument._handler, None)

        step_inv = ReactStepInvocation(round=1)
        instrument._handler.start_react_step(step_inv)

        token = _step_invocation.set(step_inv)
        try:
            wrapper._apply_last_step_finish_reason(
                {
                    "step_0": {
                        "inference_output": {
                            "current_action_name_label": "error",
                            "error_reason": "parse tool_calls failed in model output",
                        }
                    }
                }
            )
            assert step_inv.finish_reason == "parse_tool_calls_failed"
        finally:
            _step_invocation.reset(token)
            instrument._handler.stop_react_step(step_inv)


class TestApplyLastStepFinishReasonEdgeCases:
    def test_non_dict_inference_log(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        # Should not raise
        wrapper._apply_last_step_finish_reason("not a dict")
        wrapper._apply_last_step_finish_reason(None)

    def test_no_current_step(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
            _step_invocation,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        token = _step_invocation.set(None)
        try:
            # Should not raise
            wrapper._apply_last_step_finish_reason({"step_0": {}})
        finally:
            _step_invocation.reset(token)

    def test_step_data_not_dict(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
            _step_invocation,
        )
        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        wrapper = WildToolChainWrapper(instrument._handler, None)
        step_inv = ReactStepInvocation(round=1)
        instrument._handler.start_react_step(step_inv)
        token = _step_invocation.set(step_inv)
        try:
            wrapper._apply_last_step_finish_reason({"step_0": "not a dict"})
            assert not hasattr(step_inv, "_finish_reason_set")
        finally:
            _step_invocation.reset(token)
            instrument._handler.stop_react_step(step_inv)

    def test_output_not_dict(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
            _step_invocation,
        )
        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        wrapper = WildToolChainWrapper(instrument._handler, None)
        step_inv = ReactStepInvocation(round=1)
        instrument._handler.start_react_step(step_inv)
        token = _step_invocation.set(step_inv)
        try:
            wrapper._apply_last_step_finish_reason(
                {"step_0": {"inference_output": "not a dict"}}
            )
        finally:
            _step_invocation.reset(token)
            instrument._handler.stop_react_step(step_inv)


class TestCreateToolSpansEdgeCases:
    def test_non_dict_inference_log(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        # Should not raise
        wrapper._create_tool_spans_from_log("not a dict", {}, [])

    def test_step_data_not_dict(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        wrapper._create_tool_spans_from_log({"step_0": "bad"}, {}, [])

    def test_output_not_dict(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        wrapper._create_tool_spans_from_log(
            {"step_0": {"inference_output": "bad"}}, {}, []
        )

    def test_label_not_correct_skipped(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        wrapper._create_tool_spans_from_log(
            {
                "step_0": {
                    "inference_output": {
                        "tool_calls": [{"function": {"name": "foo"}}],
                        "current_action_name_label": "error",
                    }
                }
            },
            {},
            [],
        )

    def test_non_dict_tool_call_skipped(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        wrapper._create_tool_spans_from_log(
            {
                "step_0": {
                    "inference_output": {
                        "tool_calls": ["not a dict"],
                        "current_action_name_label": "correct",
                    }
                }
            },
            {},
            [],
        )

    def test_non_dict_func_in_tool_call(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        wrapper._create_tool_spans_from_log(
            {
                "step_0": {
                    "inference_output": {
                        "tool_calls": [{"function": "not a dict", "id": "c1"}],
                        "current_action_name_label": "correct",
                    }
                }
            },
            {},
            [],
        )

    def test_non_dict_tool_in_tools_list(self, instrument):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)
        wrapper._create_tool_spans_from_log(
            {
                "step_0": {
                    "inference_output": {
                        "tool_calls": [
                            {
                                "function": {"name": "t", "arguments": "{}"},
                                "id": "c1",
                            }
                        ],
                        "current_action_name_label": "correct",
                    }
                }
            },
            {
                "tools": [
                    "not a dict",
                    {"function": {"name": "t", "description": "desc"}},
                ]
            },
            [],
        )

    def test_candidate_observation_fallback(
        self,
        span_exporter,
        instrument,
    ):
        """When no tool_call_id match in messages, use candidate_observation."""
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )

        wrapper = WildToolChainWrapper(instrument._handler, None)

        wrapper._create_tool_spans_from_log(
            {
                "step_0": {
                    "inference_output": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "test_tool",
                                    "arguments": "{}",
                                },
                                "id": "no_match_id",
                            }
                        ],
                        "current_action_name_label": "correct",
                    },
                    "inference_answer": {
                        "candidate_0_answer_function_list": {
                            "observation": "candidate obs"
                        }
                    },
                }
            },
            {"tools": [], "messages": []},
            [],
        )

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes or {})
        assert attrs.get("gen_ai.tool.call.result") == "candidate obs"

    def test_step_inv_with_none_span(self, instrument):
        """When step_inv.span is None, skip context building."""
        from opentelemetry.instrumentation.wildtool._wrappers import (
            WildToolChainWrapper,
        )
        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        wrapper = WildToolChainWrapper(instrument._handler, None)

        step_inv = ReactStepInvocation(round=1)
        # Don't start the step, so span stays None
        wrapper._create_tool_spans_from_log(
            {
                "step_0": {
                    "inference_output": {
                        "tool_calls": [
                            {
                                "function": {"name": "foo", "arguments": "{}"},
                                "id": "c1",
                            }
                        ],
                        "current_action_name_label": "correct",
                    }
                }
            },
            {},
            [step_inv],
        )


class TestGetMessageAttributes:
    def test_empty_messages(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _get_message_attributes,
        )

        result = _get_message_attributes([], [])
        assert result == {}

    def test_none_messages(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _get_message_attributes,
        )

        result = _get_message_attributes(None, None)
        assert result == {}


class TestSetMessageAttributes:
    def test_no_attributes_returns_early(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _set_message_attributes,
        )

        invocation = MagicMock()
        invocation.input_messages = []
        invocation.output_messages = []
        _set_message_attributes(invocation)
        # Should return early without trying to update attributes

    def test_span_is_none(self):
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _set_message_attributes,
        )
        from opentelemetry.util.genai.types import InputMessage, Text

        invocation = MagicMock()
        invocation.input_messages = [
            InputMessage(role="user", parts=[Text(content="hi")])
        ]
        invocation.output_messages = []
        invocation.span = None
        invocation.attributes = {}
        _set_message_attributes(invocation)
        assert "gen_ai.input.messages" in invocation.attributes


class TestUninstrumentEdgeCases:
    def test_double_uninstrument(self, tracer_provider):
        """Calling uninstrument twice should not raise."""
        from opentelemetry.instrumentation.wildtool import WildToolInstrumentor

        instrumentor = WildToolInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        instrumentor.uninstrument()
        # Second call should not raise
        instrumentor.uninstrument()


class TestCloseActiveStepException:
    def test_close_active_step_exception_handling(self, instrument):
        """_close_active_step should catch exceptions from stop_react_step."""
        from opentelemetry.instrumentation.wildtool._wrappers import (
            _close_active_step,
            _step_invocation,
        )
        from opentelemetry.util.genai.extended_types import ReactStepInvocation

        step_inv = ReactStepInvocation(round=1)
        # Start step so it has a span
        instrument._handler.start_react_step(step_inv)

        # Mock stop_react_step to raise
        original = instrument._handler.stop_react_step
        instrument._handler.stop_react_step = MagicMock(
            side_effect=RuntimeError("boom")
        )
        token = _step_invocation.set(step_inv)
        try:
            _close_active_step(instrument._handler)
            # Should not raise, should clear _step_invocation
            assert _step_invocation.get() is None
        finally:
            _step_invocation.reset(token)
            instrument._handler.stop_react_step = original
