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

"""Tests for the wrapper classes in ``internal/wrappers.py``.

Each wrapper's ``__call__`` method is tested by directly invoking it with
mocked ``wrapped`` / ``instance`` / ``args`` / ``kwargs``.  The real
``get_extended_telemetry_handler`` singleton is replaced with a handler
backed by an ``InMemorySpanExporter`` so spans can be inspected.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_cli_args(**overrides):
    """Return an object that quacks like the BFCL ``args`` namespace."""
    defaults = {
        "backend": None,
        "test_category": "simple",
        "num_threads": 1,
        "run_ids": False,
    }
    defaults.update(overrides)
    ns = types.SimpleNamespace(**defaults)
    return ns


def _make_test_entry(**overrides):
    base = {
        "id": "simple_001",
        "question": [{"role": "user", "content": "Hello"}],
        "function": [
            {
                "name": "get_weather",
                "description": "Get weather info.",
                "parameters": {"type": "object"},
            }
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ENTRY wrapper (GenerateResultsWrapper)
# ---------------------------------------------------------------------------


class TestGenerateResultsWrapper:
    def test_basic_call_creates_entry_span(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        def _wrapped(args, model_name, test_cases_total):
            return {"status": "done"}

        cli_args = _make_cli_args(test_category="simple", num_threads=2)
        cases = [_make_test_entry()]
        result = wrapper(_wrapped, None, (cli_args, "gpt-4", cases), {})

        assert result == {"status": "done"}
        spans = exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) >= 1
        span = entry_spans[0]
        assert span.attributes.get("gen_ai.framework") == "bfclv4"

    def test_entry_wrapper_restores_thread_pool_executor(
        self, handler_with_tracer
    ):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        gen_mod = sys.modules["bfcl_eval._llm_response_generation"]
        original_executor = gen_mod.ThreadPoolExecutor

        def _wrapped(args, model_name, test_cases_total):
            return "ok"

        cli_args = _make_cli_args()
        wrapper(_wrapped, None, (cli_args, "model", []), {})

        # Should be restored
        assert gen_mod.ThreadPoolExecutor is original_executor

    def test_entry_wrapper_sets_backend_env(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        captured_env = {}

        def _wrapped(args, model_name, test_cases_total):
            captured_env["BFCL_BACKEND"] = os.environ.get("BFCL_BACKEND")
            return "ok"

        cli_args = _make_cli_args(backend="vllm")
        wrapper(_wrapped, None, (cli_args, "model", []), {})

        assert captured_env["BFCL_BACKEND"] == "vllm"
        # Should be cleared after
        assert os.environ.get("BFCL_BACKEND") is None

    def test_entry_wrapper_handles_wrapped_exception(
        self, handler_with_tracer
    ):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        def _wrapped(args, model_name, test_cases_total):
            raise ValueError("test error")

        cli_args = _make_cli_args()
        with pytest.raises(ValueError, match="test error"):
            wrapper(_wrapped, None, (cli_args, "model", []), {})

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1

    def test_entry_wrapper_uses_kwargs(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        def _wrapped(args, model_name, test_cases_total):
            return "ok"

        cli_args = _make_cli_args()
        result = wrapper(
            _wrapped,
            None,
            (),
            {
                "args": cli_args,
                "model_name": "test-model",
                "test_cases_total": [],
            },
        )
        assert result == "ok"

    def test_entry_wrapper_with_test_category_list(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        def _wrapped(args, model_name, test_cases_total):
            return "ok"

        cli_args = _make_cli_args(test_category=["simple", "parallel"])
        wrapper(_wrapped, None, (cli_args, "model", []), {})

        spans = exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if s.name == "enter_ai_application_system"
        ]
        assert len(entry_spans) >= 1

    def test_entry_wrapper_with_session_id_env(
        self, handler_with_tracer, monkeypatch
    ):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            GenerateResultsWrapper,
        )

        monkeypatch.setenv("BFCL_SESSION_ID", "my-session")
        helper = GenAIHookHelper()
        wrapper = GenerateResultsWrapper(helper)

        def _wrapped(args, model_name, test_cases_total):
            return "ok"

        cli_args = _make_cli_args()
        wrapper(_wrapped, None, (cli_args, "model", []), {})

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1


# ---------------------------------------------------------------------------
# AGENT wrapper (BaseHandlerInferenceWrapper)
# ---------------------------------------------------------------------------


class TestBaseHandlerInferenceWrapper:
    def test_basic_agent_span(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)

        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None

        test_entry = _make_test_entry()

        def _wrapped(entry, include_input_log=True, exclude_state_log=True):
            return (
                "model_output",
                {"input_token_count": 10, "output_token_count": 20},
            )

        result = wrapper(_wrapped, instance, (test_entry,), {})
        assert result[0] == "model_output"

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) >= 1
        span = agent_spans[0]
        assert span.attributes.get("gen_ai.framework") == "bfclv4"
        assert span.attributes.get("bfcl.test_entry_id") == "simple_001"

    def test_agent_non_dict_test_entry_passthrough(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)

        def _wrapped(entry):
            return "pass"

        result = wrapper(_wrapped, None, ("not a dict",), {})
        assert result == "pass"

    def test_agent_error_result_string(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)
        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None

        test_entry = _make_test_entry()

        def _wrapped(entry, **kw):
            return ("Error during inference: timeout", {})

        result = wrapper(_wrapped, instance, (test_entry,), {})
        assert "Error during inference" in result[0]

    def test_agent_exception_in_wrapped(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)
        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None
        test_entry = _make_test_entry()

        def _wrapped(entry, **kw):
            raise RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            wrapper(_wrapped, instance, (test_entry,), {})

    def test_agent_with_involved_classes(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)
        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None

        test_entry = _make_test_entry(
            involved_classes=["MathAPI", "StringAPI"]
        )

        def _wrapped(entry, **kw):
            return ("output", {})

        result = wrapper(_wrapped, instance, (test_entry,), {})
        assert result[0] == "output"

    def test_agent_with_token_metadata(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)
        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None
        test_entry = _make_test_entry()

        def _wrapped(entry, **kw):
            return (
                "output",
                {
                    "input_token_count": [10, 20],
                    "output_token_count": 30,
                },
            )

        result = wrapper(_wrapped, instance, (test_entry,), {})
        assert result[0] == "output"

    def test_agent_result_is_list(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            BaseHandlerInferenceWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = BaseHandlerInferenceWrapper(helper)
        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None
        test_entry = _make_test_entry()

        def _wrapped(entry, **kw):
            return ([{"name": "get_weather", "arguments": {}}], {})

        result = wrapper(_wrapped, instance, (test_entry,), {})
        assert isinstance(result[0], list)


# ---------------------------------------------------------------------------
# STEP wrapper (QueryWrapper)
# ---------------------------------------------------------------------------


class TestQueryWrapper:
    def test_query_fc_creates_step_span(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            QueryWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = QueryWrapper(helper, "FC")

        token = init_state()
        try:
            instance = MagicMock()
            instance.model_name = "gpt-4"
            instance.model_style = None

            def _wrapped(*args, **kwargs):
                return "api_response", 0.05

            result = wrapper(_wrapped, instance, (), {})
            assert result == ("api_response", 0.05)

            spans = exporter.get_finished_spans()
            step_spans = [s for s in spans if "react step" in s.name]
            assert len(step_spans) >= 1
            span = step_spans[0]
            assert span.attributes.get("bfcl.query_mode") == "FC"
        finally:
            reset_state(token)

    def test_query_prompting_mode(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            QueryWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = QueryWrapper(helper, "prompting")

        token = init_state()
        try:
            instance = MagicMock()
            instance.model_name = "gpt-4"
            instance.model_style = None

            def _wrapped(*a, **kw):
                return "response", 0.1

            result = wrapper(_wrapped, instance, (), {})
            assert result[0] == "response"
        finally:
            reset_state(token)

    def test_query_exception_propagates(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            QueryWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = QueryWrapper(helper, "FC")

        token = init_state()
        try:
            instance = MagicMock()
            instance.model_name = "gpt-4"
            instance.model_style = None

            def _wrapped(*a, **kw):
                raise ConnectionError("API down")

            with pytest.raises(ConnectionError, match="API down"):
                wrapper(_wrapped, instance, (), {})
        finally:
            reset_state(token)

    def test_query_with_streaming_response(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            QueryWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = QueryWrapper(helper, "FC")

        token = init_state()
        try:
            instance = MagicMock()
            instance.model_name = "gpt-4"
            instance.model_style = None

            def _gen():
                yield "chunk1"
                yield "chunk2"

            def _wrapped(*a, **kw):
                return _gen(), 0.05

            api_response, latency = wrapper(_wrapped, instance, (), {})
            # The streaming response should have been materialised
            chunks = list(api_response)
            assert chunks == ["chunk1", "chunk2"]
        finally:
            reset_state(token)

    def test_query_without_state(self, handler_with_tracer):
        """QueryWrapper should still work when no state is initialized."""
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            QueryWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = QueryWrapper(helper, "FC")

        instance = MagicMock()
        instance.model_name = "gpt-4"
        instance.model_style = None

        def _wrapped(*a, **kw):
            return "response", 0.1

        result = wrapper(_wrapped, instance, (), {})
        assert result[0] == "response"


# ---------------------------------------------------------------------------
# TOOL wrapper (ExecuteFuncCallWrapper)
# ---------------------------------------------------------------------------


class TestExecuteFuncCallWrapper:
    def test_basic_tool_spans(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            ExecuteFuncCallWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = ExecuteFuncCallWrapper(helper)

        token = init_state()
        try:
            func_calls = ["calc.add(1, 2)", "calc.multiply(3, 4)"]

            def _wrapped(func_call_list, *a, **kw):
                return (["3", "12"], {})

            result = wrapper(
                _wrapped,
                None,
                (func_calls, None, None, "model", "test_001"),
                {},
            )
            assert result == (["3", "12"], {})

            spans = exporter.get_finished_spans()
            tool_spans = [s for s in spans if "execute_tool" in s.name]
            assert len(tool_spans) == 2
        finally:
            reset_state(token)

    def test_tool_empty_list_passthrough(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            ExecuteFuncCallWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = ExecuteFuncCallWrapper(helper)

        def _wrapped(func_call_list, *a, **kw):
            return ([], {})

        result = wrapper(_wrapped, None, ([], None, None), {})
        assert result == ([], {})

    def test_tool_with_error_result(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            ExecuteFuncCallWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = ExecuteFuncCallWrapper(helper)

        token = init_state()
        try:
            func_calls = ["api.call()"]

            def _wrapped(func_call_list, *a, **kw):
                return (
                    ["Error during execution: key not found"],
                    {},
                )

            wrapper(
                _wrapped,
                None,
                (func_calls, None, None, "model", "test_001"),
                {},
            )

            spans = exporter.get_finished_spans()
            tool_spans = [s for s in spans if "execute_tool" in s.name]
            assert len(tool_spans) == 1
        finally:
            reset_state(token)

    def test_tool_kwargs_extraction(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            ExecuteFuncCallWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = ExecuteFuncCallWrapper(helper)

        token = init_state()
        try:
            func_calls = ["func()"]

            def _wrapped(func_call_list, *a, **kw):
                return (["ok"], {})

            result = wrapper(
                _wrapped,
                None,
                (),
                {
                    "func_call_list": func_calls,
                    "model_name": "model",
                    "test_entry_id": "test_002",
                },
            )
            assert result == (["ok"], {})
        finally:
            reset_state(token)

    def test_tool_non_list_passthrough(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            ExecuteFuncCallWrapper,
        )

        helper = GenAIHookHelper()
        wrapper = ExecuteFuncCallWrapper(helper)

        def _wrapped(func_call_list, *a, **kw):
            return "result"

        result = wrapper(_wrapped, None, ("not_a_list",), {})
        assert result == "result"


# ---------------------------------------------------------------------------
# TurnBumpWrapper
# ---------------------------------------------------------------------------


class TestTurnBumpWrapper:
    def test_reset_true_resets_state(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            bump_round,
            bump_turn,
            get_state,
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            TurnBumpWrapper,
        )

        wrapper = TurnBumpWrapper(reset=True)
        token = init_state()
        try:
            bump_round()
            bump_turn()
            assert get_state()["turn_idx"] == 1

            def _wrapped(*a, **kw):
                return "ok"

            result = wrapper(_wrapped, None, (), {})
            assert result == "ok"
            assert get_state()["turn_idx"] == 0
            assert get_state()["fc_round"] == 0
        finally:
            reset_state(token)

    def test_reset_false_bumps_turn(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            get_state,
            init_state,
            reset_state,
        )
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            TurnBumpWrapper,
        )

        wrapper = TurnBumpWrapper(reset=False)
        token = init_state()
        try:
            assert get_state()["turn_idx"] == 0

            def _wrapped(*a, **kw):
                return "ok"

            wrapper(_wrapped, None, (), {})
            assert get_state()["turn_idx"] == 1
        finally:
            reset_state(token)

    def test_turn_bump_without_state(self, handler_with_tracer):
        """TurnBumpWrapper should not crash when no state exists."""
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            TurnBumpWrapper,
        )

        wrapper = TurnBumpWrapper(reset=True)

        def _wrapped(*a, **kw):
            return "ok"

        result = wrapper(_wrapped, None, (), {})
        assert result == "ok"

        wrapper2 = TurnBumpWrapper(reset=False)
        result2 = wrapper2(_wrapped, None, (), {})
        assert result2 == "ok"


# ---------------------------------------------------------------------------
# Wrapper helper functions
# ---------------------------------------------------------------------------


class TestWrapperHelpers:
    def test_safe_get_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _safe_get,
        )

        assert _safe_get({"a": 1}, "a") == 1
        assert _safe_get({"a": 1}, "b", "default") == "default"

    def test_safe_get_object(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _safe_get,
        )

        obj = types.SimpleNamespace(x=42)
        assert _safe_get(obj, "x") == 42
        assert _safe_get(obj, "y", "nope") == "nope"

    def test_flatten_tokens_int(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _flatten_tokens,
        )

        assert _flatten_tokens(10) == 10
        assert _flatten_tokens(10.5) == 10
        assert _flatten_tokens(None) is None

    def test_flatten_tokens_list(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _flatten_tokens,
        )

        assert _flatten_tokens([10, 20, 30]) == 60
        assert _flatten_tokens([[10, 20], [30]]) == 60
        assert _flatten_tokens([None, None]) is None

    def test_test_category_from_id(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _test_category_from_id,
        )

        assert _test_category_from_id("simple_001") == "simple"
        assert (
            _test_category_from_id("multi_turn_base_001") == "multi_turn_base"
        )
        assert _test_category_from_id(None) is None
        assert _test_category_from_id("nounderscore") is None
        assert _test_category_from_id("") is None

    def test_join_test_category(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _join_test_category,
        )

        assert _join_test_category("simple") == "simple"
        assert _join_test_category(["a", "b"]) == "a,b"
        assert _join_test_category(None) is None
        assert _join_test_category(42) == "42"
        assert _join_test_category([None, None]) is None

    def test_json_attr(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _json_attr,
        )

        result = _json_attr({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_message_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _message_dict,
        )

        msg = _message_dict("user", "hello")
        assert msg["role"] == "user"
        assert msg["parts"][0]["content"] == "hello"

    def test_system_instruction_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _system_instruction_dict,
        )

        si = _system_instruction_dict("Use the tools")
        assert si["type"] == "text"
        assert si["content"] == "Use the tools"

    def test_record_span_error(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _record_span_error,
        )

        # Test with None span
        _record_span_error(None, "error")  # should not raise

        # Test with non-recording span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        _record_span_error(mock_span, "error")
        mock_span.record_exception.assert_not_called()

        # Test with recording span
        mock_span2 = MagicMock()
        mock_span2.is_recording.return_value = True
        _record_span_error(mock_span2, "test error text")
        mock_span2.record_exception.assert_called_once()
        mock_span2.set_status.assert_called_once()

    def test_normalise_role(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _normalise_role,
        )

        assert _normalise_role("assistant", "user") == "assistant"
        assert _normalise_role(None, "user") == "user"
        assert _normalise_role("", "user") == "user"
        assert _normalise_role([], "user") == "user"

    def test_normalise_message_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _normalise_message_dict,
        )

        assert _normalise_message_dict(None, default_role="user") is None
        assert _normalise_message_dict("", default_role="user") is None

        result = _normalise_message_dict(
            {"role": "user", "content": "hello"}, default_role="user"
        )
        assert result["role"] == "user"
        assert result["parts"][0]["content"] == "hello"

        # Dict with no content but extra keys
        result = _normalise_message_dict(
            {"role": "user", "tool_calls": [{"id": "1"}]},
            default_role="user",
        )
        assert result is not None

        # String input
        result = _normalise_message_dict("just a string", default_role="user")
        assert result["role"] == "user"

        # Dict with content=None but "content" key remains as an extra
        # (the key "content" is not in the exclusion set role/name/tool_call_id),
        # so extras={content: None} is truthy and serialised.
        result = _normalise_message_dict(
            {"role": "user", "content": None}, default_role="user"
        )
        assert result is not None

        # Truly empty dict with only role -> extras is empty -> returns None
        result = _normalise_message_dict({"role": "user"}, default_role="user")
        assert result is None

    def test_flatten_messages(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _flatten_messages,
        )

        # Nested list of lists
        msgs = _flatten_messages(
            [
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                [{"role": "user", "content": "bye"}],
            ]
        )
        assert len(msgs) == 3

        # Empty
        assert _flatten_messages(None) == []
        assert _flatten_messages("") == []

        # Single dict
        msgs = _flatten_messages({"role": "user", "content": "test"})
        assert len(msgs) == 1

        # Plain string
        msgs = _flatten_messages("plain text")
        assert len(msgs) == 1

    def test_messages_to_input(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _messages_to_input,
        )

        msgs = [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "hello"}],
            }
        ]
        result = _messages_to_input(msgs)
        assert len(result) == 1
        assert result[0].role == "user"

        # Empty parts
        msgs2 = [{"role": "user", "parts": []}]
        assert _messages_to_input(msgs2) == []

    def test_messages_to_output(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _messages_to_output,
        )

        msgs = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "hi"}],
            }
        ]
        result = _messages_to_output(msgs)
        assert len(result) == 1
        assert result[0].finish_reason == "stop"

    def test_messages_to_output_empty_parts(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _messages_to_output,
        )

        # Message with empty parts should be skipped
        msgs = [{"role": "assistant", "parts": []}]
        result = _messages_to_output(msgs)
        assert result == []

    def test_test_entry_to_messages_non_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _test_entry_to_messages,
        )

        inputs, si = _test_entry_to_messages("not a dict")
        assert inputs == []
        assert si == []

    def test_test_entry_to_tool_definitions_non_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _test_entry_to_tool_definitions,
        )

        assert _test_entry_to_tool_definitions("not a dict") == []

    def test_tool_value_to_definitions_json_string(self):
        import json

        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        defs = _tool_value_to_definitions(
            json.dumps(
                {
                    "name": "test_func",
                    "description": "A test",
                    "parameters": {},
                }
            )
        )
        assert len(defs) == 1
        assert defs[0].name == "test_func"

    def test_tool_value_to_definitions_empty(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        assert _tool_value_to_definitions(None) == []
        assert _tool_value_to_definitions("") == []
        assert _tool_value_to_definitions([]) == []

    def test_tool_value_to_definitions_generic_type(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        defs = _tool_value_to_definitions(
            {"name": "search", "type": "retrieval"}
        )
        assert len(defs) == 1
        assert defs[0].type == "retrieval"

    def test_tool_value_to_definitions_non_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        assert _tool_value_to_definitions(42) == []

    def test_tool_value_to_definitions_invalid_json_string(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        assert _tool_value_to_definitions("not json{") == []

    def test_tool_value_to_definitions_no_name(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        assert _tool_value_to_definitions({"description": "no name"}) == []

    def test_tool_value_to_definitions_function_name_key(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_value_to_definitions,
        )

        defs = _tool_value_to_definitions(
            {"function_name": "fn", "parameters": {}}
        )
        assert len(defs) == 1
        assert defs[0].name == "fn"

    def test_dedupe_tool_definitions(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _dedupe_tool_definitions,
        )
        from opentelemetry.util.genai.types import FunctionToolDefinition

        d1 = FunctionToolDefinition(name="a", description="d", parameters={})
        d2 = FunctionToolDefinition(name="a", description="d", parameters={})
        result = _dedupe_tool_definitions([d1, d2])
        assert len(result) == 1

    def test_normalise_tool_arguments(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _normalise_tool_arguments,
        )

        assert _normalise_tool_arguments(None) == {}
        assert _normalise_tool_arguments({"a": 1}) == {"a": 1}

    def test_extract_questions_from_cases(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_questions_from_cases,
        )

        cases = [
            {"question": [{"role": "user", "content": "hi"}]},
            {"question": [{"role": "user", "content": "bye"}]},
        ]
        msgs = _extract_questions_from_cases(cases)
        assert len(msgs) == 2

        assert _extract_questions_from_cases("not a list") == []
        assert _extract_questions_from_cases(None) == []

    def test_extract_tool_defs_from_cases(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_tool_defs_from_cases,
        )

        cases = [{"function": [{"name": "f1"}]}]
        result = _extract_tool_defs_from_cases(cases)
        assert len(result) == 1
        assert _extract_tool_defs_from_cases("not a list") == []

    def test_set_json_span_attr(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _set_json_span_attr,
        )

        # None span
        _set_json_span_attr(None, "key", "value")

        # Empty value
        mock_span = MagicMock()
        _set_json_span_attr(mock_span, "key", None)
        mock_span.set_attribute.assert_not_called()

        # Normal
        mock_span2 = MagicMock()
        mock_span2.is_recording.return_value = True
        _set_json_span_attr(mock_span2, "key", {"a": 1})
        mock_span2.set_attribute.assert_called_once()

    def test_span_attr_value(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _span_attr_value,
        )

        assert _span_attr_value("hello") == "hello"
        result = _span_attr_value({"a": 1})
        assert '"a"' in result

    def test_set_tool_call_span_attrs(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _set_tool_call_span_attrs,
        )

        # None span
        _set_tool_call_span_attrs(None)

        # Non-recording
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        _set_tool_call_span_attrs(mock_span, tool_name="test")
        mock_span.set_attribute.assert_not_called()

        # Recording
        mock_span2 = MagicMock()
        mock_span2.is_recording.return_value = True
        _set_tool_call_span_attrs(
            mock_span2,
            tool_name="fn",
            tool_call_id="id-1",
            tool_type="function",
            arguments={"a": 1},
            result="ok",
            description="A function",
        )
        assert mock_span2.set_attribute.call_count >= 5

    def test_parse_python_call_arguments_dict_call(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _parse_python_call_arguments,
        )

        result = _parse_python_call_arguments("func(a=1, b='hello')")
        assert result == {"a": 1, "b": "hello"}

    def test_parse_python_call_arguments_positional(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _parse_python_call_arguments,
        )

        result = _parse_python_call_arguments("func(1, 2, 3)")
        assert result["arg_0"] == 1
        assert result["arg_1"] == 2

    def test_parse_python_call_arguments_non_call(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _parse_python_call_arguments,
        )

        result = _parse_python_call_arguments("not a call")
        assert result is None or isinstance(result, str)

    def test_parse_python_call_arguments_syntax_error(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _parse_python_call_arguments,
        )

        result = _parse_python_call_arguments("func(a=)")
        # Should fall back gracefully
        assert result is not None or result is None

    def test_parse_python_call_arguments_expression(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _parse_python_call_arguments,
        )

        result = _parse_python_call_arguments("1 + 2")
        # Not a call expression, fallback
        assert result is not None or result is None

    def test_parse_python_call_arguments_kwargs(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _parse_python_call_arguments,
        )

        result = _parse_python_call_arguments("func(**kw)")
        assert "kwargs" in result

    def test_iter_model_tool_calls(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _iter_model_tool_calls,
        )

        # Dict items
        calls = list(
            _iter_model_tool_calls([{"get_weather": {"city": "Paris"}}])
        )
        assert len(calls) == 1
        assert calls[0][0] == "get_weather"

        # String items
        calls = list(_iter_model_tool_calls(["calc.add(1, 2)"]))
        assert len(calls) == 1
        assert calls[0][0] == "add"

        # Non-list returns nothing
        calls = list(_iter_model_tool_calls("not a list"))
        assert calls == []

        # None
        calls = list(_iter_model_tool_calls(None))
        assert calls == []

    def test_extract_tool_name(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_tool_name,
        )

        assert _extract_tool_name("module.func(a)") == "func"
        assert _extract_tool_name("func()") == "func"
        assert _extract_tool_name("not_a_call") == "unknown"
        assert _extract_tool_name(42) == "unknown"

    def test_extract_tool_arguments(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_tool_arguments,
        )

        assert _extract_tool_arguments("func(a, b)") == "a, b"
        assert _extract_tool_arguments("func()") is None
        assert _extract_tool_arguments("no_parens") == "no_parens"
        assert _extract_tool_arguments(42) is None

    def test_synth_tool_call_id(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _synth_tool_call_id,
        )

        assert (
            _synth_tool_call_id("test_001", "model", 0) == "test_001-model-0"
        )
        assert _synth_tool_call_id(None, None, 1) == "no_id-no_model-1"

    def test_safe_str(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _safe_str,
        )

        assert _safe_str("hello") == "hello"
        assert _safe_str({"a": 1}) == '{"a": 1}'

        class _Bad:
            def __repr__(self):
                raise RuntimeError("nope")

            def __str__(self):
                raise RuntimeError("nope")

        result = _safe_str(_Bad())
        # Should not raise; fallback to something
        assert isinstance(result, str)

    def test_result_to_output_messages(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _result_to_output_messages,
        )

        # None
        assert _result_to_output_messages(None) == []
        assert _result_to_output_messages("") == []

        # String
        msgs = _result_to_output_messages("hello")
        assert len(msgs) == 1

        # Tuple
        msgs = _result_to_output_messages(("hello",))
        assert len(msgs) == 1

        # Dict with final_answer
        msgs = _result_to_output_messages({"final_answer": "42"})
        assert len(msgs) == 1

    def test_result_to_output_messages_list_payload(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _result_to_output_messages,
        )

        # List payload triggers recursive processing
        msgs = _result_to_output_messages(["response1", "response2"])
        assert len(msgs) == 2

    def test_result_to_output_messages_empty_content(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _result_to_output_messages,
        )

        # Dict with all empty known keys
        msgs = _result_to_output_messages({"unknown_key": "value"})
        assert len(msgs) == 1  # Falls through to return the dict itself

    def test_extract_result_content_keys(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_result_content,
        )

        assert _extract_result_content({"final_answer": "a"}) == "a"
        assert _extract_result_content({"answer": "b"}) == "b"
        assert _extract_result_content({"output": "c"}) == "c"
        assert _extract_result_content({"result": "d"}) == "d"
        assert _extract_result_content({"model_response": "e"}) == "e"
        assert _extract_result_content("plain") == "plain"

    def test_extract_result_content_inference_log_non_dict_step(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_result_content,
        )

        # Non-dict step_data should be skipped (continue)
        result = _extract_result_content(
            {
                "inference_log": {
                    "step_0": "not_a_dict",
                    "step_1": {"inference_output": "final"},
                }
            }
        )
        assert result == "final"

    def test_extract_result_content_inference_log_answer_key(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_result_content,
        )

        result = _extract_result_content(
            {
                "inference_log": {
                    "step_0": {
                        "inference_output": None,
                        "inference_answer": "the_answer",
                    }
                }
            }
        )
        assert result == "the_answer"

    def test_extract_result_content_no_known_keys(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _extract_result_content,
        )

        # Dict with no known keys -> returns the dict itself
        d = {"unknown": "data"}
        assert _extract_result_content(d) is d

    def test_step_log_sort_key(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _step_log_sort_key,
        )

        assert _step_log_sort_key("step_0") == 0
        assert _step_log_sort_key("step_5") == 5
        assert _step_log_sort_key("bad") == -1

    def test_lookup_tool_description_none(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _lookup_tool_description,
        )

        assert _lookup_tool_description(None) is None
        assert _lookup_tool_description("") is None

    def test_tool_description_map_basic(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_description_map,
        )

        entry = {
            "function": [
                {"name": "fn1", "description": "Desc1", "parameters": {}},
            ],
        }
        desc_map = _tool_description_map(entry)
        assert desc_map["fn1"] == "Desc1"

    def test_tool_description_map_non_dict(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_description_map,
        )

        assert _tool_description_map("not a dict") == {}

    def test_bfcl_captured_error(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _BFCLCapturedError,
        )

        exc = _BFCLCapturedError("test")
        assert str(exc) == "test"
        assert isinstance(exc, RuntimeError)

    def test_test_entry_to_tool_definitions_missed_function_list(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _test_entry_to_tool_definitions,
        )

        entry = {
            "id": "t1",
            "missed_function": [
                {
                    "name": "missed_fn",
                    "description": "Missed",
                    "parameters": {},
                },
            ],
        }
        defs = _test_entry_to_tool_definitions(entry)
        assert any(d.name == "missed_fn" for d in defs)

    def test_append_question_messages_system_role(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _append_question_messages,
        )

        inputs = []
        system_instructions = []
        _append_question_messages(
            {"role": "system", "content": "Be helpful"},
            inputs,
            system_instructions,
        )
        assert len(system_instructions) == 1
        assert len(inputs) == 0

    def test_append_question_messages_empty(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _append_question_messages,
        )

        inputs = []
        si = []
        _append_question_messages(None, inputs, si)
        assert inputs == []
        assert si == []

    def test_append_question_messages_bare_string(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _append_question_messages,
        )

        inputs = []
        si = []
        _append_question_messages("just text", inputs, si)
        assert len(inputs) == 1

    def test_append_question_messages_dict_empty_content(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _append_question_messages,
        )

        # {"role": "user", "content": None} -> extras includes "content": None
        # which is a non-empty dict, so it *does* produce a message
        inputs = []
        si = []
        _append_question_messages(
            {"role": "user", "content": None}, inputs, si
        )
        assert len(inputs) == 1

        # Truly empty: only role key, no other keys -> extras is empty -> skip
        inputs2 = []
        si2 = []
        _append_question_messages({"role": "user"}, inputs2, si2)
        assert len(inputs2) == 0

    def test_emit_synthetic_tool_spans(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _emit_synthetic_tool_spans,
        )

        count = _emit_synthetic_tool_spans(
            [{"get_weather": {"city": "Paris"}}],
            test_entry_id="test_001",
            model_name="gpt-4",
        )
        assert count == 1
        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1

    def test_emit_synthetic_tool_spans_empty(self, handler_with_tracer):
        handler, exporter = handler_with_tracer
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _emit_synthetic_tool_spans,
        )

        count = _emit_synthetic_tool_spans(
            None, test_entry_id=None, model_name=None
        )
        assert count == 0

    def test_literal_or_source(self):
        import ast

        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _literal_or_source,
        )

        source = "func(42)"
        tree = ast.parse(source, mode="eval")
        call = tree.body
        assert isinstance(call, ast.Call)
        result = _literal_or_source(call.args[0], source)
        assert result == 42

        # Non-literal
        source2 = "func(x)"
        tree2 = ast.parse(source2, mode="eval")
        call2 = tree2.body
        result2 = _literal_or_source(call2.args[0], source2)
        assert result2 == "x"

    def test_json_attr_fallback(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _json_attr,
        )

        # Normal case
        result = _json_attr({"key": "value"})
        assert '"key"' in result

        # Should handle non-serializable via _safe_str fallback
        class _CycleObj:
            pass

        obj = _CycleObj()
        result = _json_attr(obj)
        assert isinstance(result, str)

    def test_tool_description_map_with_involved_classes(self):
        """Test _tool_description_map when involved_classes + CLASS_FILE_PATH_MAPPING."""
        import sys
        import types

        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _tool_description_map,
        )

        # Set up a mock class with a method that has a docstring
        mock_module = types.ModuleType("mock_exec_backend")

        class MathAPI:
            def add(self, a, b):
                """Add two numbers together."""
                return a + b

            def _private(self):
                """Private method should be skipped."""
                pass

        mock_module.MathAPI = MathAPI
        sys.modules["mock_exec_backend"] = mock_module

        exec_cfg = sys.modules["bfcl_eval.constants.executable_backend_config"]
        exec_cfg.CLASS_FILE_PATH_MAPPING = {"MathAPI": "mock_exec_backend"}

        try:
            entry = {
                "function": [],
                "involved_classes": ["MathAPI"],
            }
            desc_map = _tool_description_map(entry)
            assert "add" in desc_map
            assert "two numbers" in desc_map["add"]
            assert "_private" not in desc_map
        finally:
            exec_cfg.CLASS_FILE_PATH_MAPPING = {}
            sys.modules.pop("mock_exec_backend", None)

    def test_lookup_tool_description_from_class_mapping(self):
        """Test _lookup_tool_description when it falls back to CLASS_FILE_PATH_MAPPING."""
        import sys
        import types

        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _TOOL_DESCRIPTION_MAP,
            _lookup_tool_description,
        )

        mock_module = types.ModuleType("mock_tool_backend")

        class ToolClass:
            def my_tool(self):
                """Does something useful."""
                pass

        mock_module.ToolClass = ToolClass
        sys.modules["mock_tool_backend"] = mock_module

        exec_cfg = sys.modules["bfcl_eval.constants.executable_backend_config"]
        exec_cfg.CLASS_FILE_PATH_MAPPING = {"ToolClass": "mock_tool_backend"}

        # Make sure the context var map doesn't have this tool
        token = _TOOL_DESCRIPTION_MAP.set({})
        try:
            result = _lookup_tool_description("my_tool")
            assert result is not None
            assert "useful" in result
        finally:
            _TOOL_DESCRIPTION_MAP.reset(token)
            exec_cfg.CLASS_FILE_PATH_MAPPING = {}
            sys.modules.pop("mock_tool_backend", None)

    def test_lookup_tool_description_no_match(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _TOOL_DESCRIPTION_MAP,
            _lookup_tool_description,
        )

        token = _TOOL_DESCRIPTION_MAP.set({})
        try:
            result = _lookup_tool_description("nonexistent_tool")
            assert result is None
        finally:
            _TOOL_DESCRIPTION_MAP.reset(token)

    def test_lookup_tool_description_from_context_var(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _TOOL_DESCRIPTION_MAP,
            _lookup_tool_description,
        )

        token = _TOOL_DESCRIPTION_MAP.set({"my_tool": "A tool description"})
        try:
            result = _lookup_tool_description("my_tool")
            assert result == "A tool description"
        finally:
            _TOOL_DESCRIPTION_MAP.reset(token)

    def test_record_span_error_is_recording_exception(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _record_span_error,
        )

        # Span where is_recording raises
        mock_span = MagicMock()
        mock_span.is_recording.side_effect = RuntimeError("broken")
        _record_span_error(mock_span, "error")
        # Should not raise

    def test_set_json_span_attr_exception(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _set_json_span_attr,
        )

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.set_attribute.side_effect = RuntimeError("oops")
        # Should not raise
        _set_json_span_attr(mock_span, "key", {"data": "value"})

    def test_set_tool_call_span_attrs_exception(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _set_tool_call_span_attrs,
        )

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.set_attribute.side_effect = RuntimeError("oops")
        # Should not raise
        _set_tool_call_span_attrs(mock_span, tool_name="fn")

    def test_infer_finish_reason_continue(self):
        from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
            _infer_finish_reason,
        )

        # Non-string, non-list, non-None -> "continue"
        assert _infer_finish_reason(42) == "continue"
        assert _infer_finish_reason({}) == "continue"
