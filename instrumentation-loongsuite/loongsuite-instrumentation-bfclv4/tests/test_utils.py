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

"""Tests for ``utils.py`` -- GenAIHookHelper, to_text_input, to_text_output,
_to_safe_str, and truncate_text."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestGenAIHookHelper:
    def test_on_completion_non_recording_span(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper

        helper = GenAIHookHelper()
        span = MagicMock()
        span.is_recording.return_value = False
        # Should not raise or set attributes
        helper.on_completion(span, inputs=[], outputs=[])
        span.set_attribute.assert_not_called()

    def test_on_completion_with_attributes(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper

        helper = GenAIHookHelper()
        span = MagicMock()
        span.is_recording.return_value = True

        attrs = {"custom.key": "value", "none_key": None}
        helper.on_completion(span, attributes=attrs)
        # None values should be skipped
        span.set_attribute.assert_called_once_with("custom.key", "value")

    def test_on_completion_with_content_capturing(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper
        from opentelemetry.util.genai.types import (
            InputMessage,
            OutputMessage,
            Text,
        )

        helper = GenAIHookHelper(capture_content=True)
        span = MagicMock()
        span.is_recording.return_value = True

        inputs = [InputMessage(role="user", parts=[Text(content="hello")])]
        outputs = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="hi")],
                finish_reason="stop",
            )
        ]
        system = [Text(content="Be helpful")]

        with (
            patch(
                "opentelemetry.instrumentation.bfclv4.utils.is_experimental_mode",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.bfclv4.utils.get_content_capturing_mode",
            ) as mock_mode,
        ):
            from opentelemetry.util.genai.types import ContentCapturingMode

            mock_mode.return_value = ContentCapturingMode.SPAN_ONLY
            helper.on_completion(
                span,
                inputs=inputs,
                outputs=outputs,
                system_instructions=system,
            )

        assert span.set_attribute.call_count >= 3

    def test_on_completion_no_capture_content(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper
        from opentelemetry.util.genai.types import InputMessage, Text

        helper = GenAIHookHelper(capture_content=False)
        span = MagicMock()
        span.is_recording.return_value = True

        inputs = [InputMessage(role="user", parts=[Text(content="hello")])]
        helper.on_completion(span, inputs=inputs)
        # Should not set input messages (no experimental mode check)
        # Only attributes if provided

    def test_on_completion_span_and_event_mode(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper
        from opentelemetry.util.genai.types import InputMessage, Text

        helper = GenAIHookHelper(capture_content=True)
        span = MagicMock()
        span.is_recording.return_value = True

        inputs = [InputMessage(role="user", parts=[Text(content="hello")])]

        with (
            patch(
                "opentelemetry.instrumentation.bfclv4.utils.is_experimental_mode",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.bfclv4.utils.get_content_capturing_mode",
            ) as mock_mode,
        ):
            from opentelemetry.util.genai.types import ContentCapturingMode

            mock_mode.return_value = ContentCapturingMode.SPAN_AND_EVENT
            helper.on_completion(span, inputs=inputs)

        assert span.set_attribute.call_count >= 1

    def test_on_completion_event_only_mode_no_span_write(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper
        from opentelemetry.util.genai.types import InputMessage, Text

        helper = GenAIHookHelper(capture_content=True)
        span = MagicMock()
        span.is_recording.return_value = True

        inputs = [InputMessage(role="user", parts=[Text(content="hello")])]

        with (
            patch(
                "opentelemetry.instrumentation.bfclv4.utils.is_experimental_mode",
                return_value=True,
            ),
            patch(
                "opentelemetry.instrumentation.bfclv4.utils.get_content_capturing_mode",
            ) as mock_mode,
        ):
            from opentelemetry.util.genai.types import ContentCapturingMode

            mock_mode.return_value = ContentCapturingMode.EVENT_ONLY
            helper.on_completion(span, inputs=inputs)

        # EVENT_ONLY should not write to span
        span.set_attribute.assert_not_called()

    def test_on_completion_attribute_set_failure(self):
        from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper

        helper = GenAIHookHelper()
        span = MagicMock()
        span.is_recording.return_value = True
        span.set_attribute.side_effect = RuntimeError("oops")

        # Should not raise
        helper.on_completion(span, attributes={"key": "val"})


class TestToTextInput:
    def test_basic(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_input

        result = to_text_input("user", "hello")
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].parts[0].content == "hello"

    def test_empty_content(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_input

        assert to_text_input("user", None) == []
        assert to_text_input("user", "") == []
        assert to_text_input("user", []) == []
        assert to_text_input("user", {}) == []

    def test_non_string_content(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_input

        result = to_text_input("user", {"key": "value"})
        assert len(result) == 1


class TestToTextOutput:
    def test_basic(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_output

        result = to_text_output("assistant", "response")
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].finish_reason == "stop"

    def test_empty_content(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_output

        assert to_text_output("assistant", None) == []
        assert to_text_output("assistant", "") == []

    def test_custom_finish_reason(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_output

        result = to_text_output("assistant", "text", "tool_calls")
        assert result[0].finish_reason == "tool_calls"

    def test_non_string_content(self):
        from opentelemetry.instrumentation.bfclv4.utils import to_text_output

        result = to_text_output("assistant", {"answer": 42})
        assert len(result) == 1


class TestToSafeStr:
    def test_serializable(self):
        from opentelemetry.instrumentation.bfclv4.utils import _to_safe_str

        result = _to_safe_str({"a": 1})
        assert "a" in result

    def test_unserializable(self):
        from opentelemetry.instrumentation.bfclv4.utils import _to_safe_str

        class _Bad:
            pass

        result = _to_safe_str(_Bad())
        assert isinstance(result, str)

    def test_completely_unserializable(self):
        from opentelemetry.instrumentation.bfclv4.utils import _to_safe_str

        class _Terrible:
            def __repr__(self):
                raise RuntimeError("nope")

            def __str__(self):
                raise RuntimeError("nope")

        # Should fall back to "<unserialisable>" or similar
        result = _to_safe_str(_Terrible())
        assert isinstance(result, str)


class TestTruncateText:
    def test_short_text_unchanged(self):
        from opentelemetry.instrumentation.bfclv4.utils import truncate_text

        assert truncate_text("hello") == "hello"

    def test_long_text_truncated(self):
        from opentelemetry.instrumentation.bfclv4.utils import truncate_text

        long_text = "x" * 5000
        result = truncate_text(long_text, limit=100)
        assert len(result) < 5000
        assert "truncated" in result

    def test_exact_limit(self):
        from opentelemetry.instrumentation.bfclv4.utils import truncate_text

        text = "a" * 4096
        assert truncate_text(text) == text

    def test_custom_limit(self):
        from opentelemetry.instrumentation.bfclv4.utils import truncate_text

        result = truncate_text("hello world", limit=5)
        assert result.startswith("hello")
        assert "truncated" in result
