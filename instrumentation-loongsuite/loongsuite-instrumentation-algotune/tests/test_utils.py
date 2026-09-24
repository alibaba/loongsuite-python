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

"""Unit tests for ``opentelemetry.instrumentation.algotune.internal.utils``
and ``opentelemetry.instrumentation.algotune.config``.

Covers ``truncate()``, ``provider_from_model()``, ``safe_close_step()``,
the module-level constants, and the config helper functions
``_bool_env``, ``_int_env``, ``_float_env``, ``_genai_capture_enabled``.
"""

from __future__ import annotations

import types
from unittest import mock

from opentelemetry.instrumentation.algotune.internal.utils import (
    ALGOTUNE_FRAMEWORK_VALUE,
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    INST_LITELLM_ATTEMPTS_ATTR,
    INST_ROUND_ATTR,
    INST_STEP_SPAN_ATTR,
    INST_STEP_TOKEN_ATTR,
    provider_from_model,
    safe_close_step,
    truncate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_framework_value(self):
        assert ALGOTUNE_FRAMEWORK_VALUE == "AlgoTune"

    def test_span_kind_attr_name(self):
        assert GEN_AI_SPAN_KIND == "gen_ai.span.kind"

    def test_framework_attr_name(self):
        assert GEN_AI_FRAMEWORK == "gen_ai.framework"

    def test_total_tokens_attr_name(self):
        assert GEN_AI_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"

    def test_instance_attribute_names(self):
        assert INST_STEP_SPAN_ATTR == "_otel_algo_step_span"
        assert INST_STEP_TOKEN_ATTR == "_otel_algo_step_token"
        assert INST_ROUND_ATTR == "_otel_algo_round"
        assert INST_LITELLM_ATTEMPTS_ATTR == "_otel_algo_litellm_attempts"


# ---------------------------------------------------------------------------
# truncate()
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_none_returns_empty(self):
        assert truncate(None) == ""

    def test_short_string_unchanged(self):
        assert truncate("hello") == "hello"

    def test_exact_max_len(self):
        text = "a" * 4096
        assert truncate(text) == text

    def test_long_string_truncated_with_ellipsis(self):
        text = "a" * 5000
        result = truncate(text)
        assert len(result) == 4096
        assert result.endswith("...")

    def test_custom_max_len(self):
        result = truncate("abcdefghij", max_len=7)
        assert result == "abcd..."
        assert len(result) == 7

    def test_max_len_3_no_ellipsis(self):
        result = truncate("abcdefghij", max_len=3)
        assert result == "abc"

    def test_max_len_2_no_ellipsis(self):
        result = truncate("abcdefghij", max_len=2)
        assert result == "ab"

    def test_max_len_1(self):
        result = truncate("abcdefghij", max_len=1)
        assert result == "a"

    def test_max_len_0(self):
        result = truncate("abcdefghij", max_len=0)
        assert result == ""

    def test_non_string_int(self):
        assert truncate(42) == "42"

    def test_non_string_list(self):
        result = truncate([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_non_string_dict(self):
        result = truncate({"a": 1})
        assert "a" in result

    def test_non_string_unconvertible(self):
        """Object whose str() raises should return empty string."""

        class BadStr:
            def __str__(self):
                raise RuntimeError("cannot convert")

        assert truncate(BadStr()) == ""

    def test_empty_string(self):
        assert truncate("") == ""

    def test_unicode_string(self):
        text = "hello world"
        assert truncate(text) == text

    def test_long_unicode_truncated(self):
        text = "x" * 5000
        result = truncate(text, max_len=10)
        assert result == "xxxxxxx..."
        assert len(result) == 10


# ---------------------------------------------------------------------------
# provider_from_model()
# ---------------------------------------------------------------------------


class TestProviderFromModel:
    """Test all explicit prefix mappings and substring heuristics."""

    def test_empty_string(self):
        assert provider_from_model("") == "unknown"

    def test_none_like_empty(self):
        # The function checks ``if not model_name``; empty string is falsy.
        assert provider_from_model("") == "unknown"

    # -- Explicit prefix mappings --

    def test_openai_prefix(self):
        assert provider_from_model("openai/gpt-4o") == "openai"

    def test_anthropic_prefix(self):
        assert (
            provider_from_model("anthropic/claude-3-5-sonnet") == "anthropic"
        )

    def test_vertex_ai_prefix(self):
        assert provider_from_model("vertex_ai/gemini-1.5-pro") == "google"

    def test_gemini_prefix(self):
        assert provider_from_model("gemini/gemini-1.5-flash") == "google"

    def test_google_prefix(self):
        assert provider_from_model("google/gemini-pro") == "google"

    def test_mistral_prefix(self):
        assert provider_from_model("mistral/mistral-large") == "mistral"

    def test_azure_prefix(self):
        assert provider_from_model("azure/gpt-4") == "azure"

    def test_azure_ai_prefix(self):
        assert provider_from_model("azure_ai/model-x") == "azure"

    def test_bedrock_prefix(self):
        assert provider_from_model("bedrock/anthropic.claude-v2") == "bedrock"

    def test_groq_prefix(self):
        assert provider_from_model("groq/llama-3-70b") == "groq"

    def test_deepseek_prefix(self):
        assert provider_from_model("deepseek/deepseek-coder") == "deepseek"

    def test_openrouter_prefix(self):
        assert (
            provider_from_model("openrouter/meta-llama/llama-3")
            == "openrouter"
        )

    def test_together_ai_prefix(self):
        assert (
            provider_from_model("together_ai/meta-llama/Meta-Llama-3")
            == "together_ai"
        )

    # -- Case insensitivity --

    def test_case_insensitive_prefix(self):
        assert provider_from_model("OpenAI/GPT-4o") == "openai"

    # -- Substring heuristics (no prefix) --

    def test_claude_substring(self):
        assert provider_from_model("claude-3-5-sonnet-20241022") == "anthropic"

    def test_anthropic_substring(self):
        assert provider_from_model("anthropic-model-v2") == "anthropic"

    def test_gemini_substring(self):
        assert provider_from_model("gemini-1.5-pro") == "google"

    def test_vertex_substring(self):
        assert provider_from_model("vertex-gemini-pro") == "google"

    def test_google_substring(self):
        assert provider_from_model("google-bison-001") == "google"

    def test_mistral_substring(self):
        assert provider_from_model("mistral-7b-instruct") == "mistral"

    def test_deepseek_substring(self):
        assert provider_from_model("deepseek-coder-v2") == "deepseek"

    def test_qwen_substring(self):
        assert provider_from_model("qwen-72b-chat") == "dashscope"

    def test_dashscope_substring(self):
        assert provider_from_model("dashscope-turbo") == "dashscope"

    def test_gpt_substring(self):
        assert provider_from_model("gpt-4-turbo") == "openai"

    def test_openai_substring(self):
        assert provider_from_model("my-openai-model") == "openai"

    def test_o1_substring(self):
        assert provider_from_model("o1-mini") == "openai"

    def test_o3_substring(self):
        assert provider_from_model("o3-mini") == "openai"

    def test_unknown_model(self):
        assert provider_from_model("some-random-model") == "unknown"

    def test_unknown_prefix(self):
        assert provider_from_model("custom_provider/model-v1") == "unknown"

    # -- Priority: prefix before substring --

    def test_prefix_takes_priority(self):
        # Even though "claude" is in the name, the prefix "openai" wins.
        assert provider_from_model("openai/claude-lookalike") == "openai"


# ---------------------------------------------------------------------------
# safe_close_step()
# ---------------------------------------------------------------------------


class TestSafeCloseStep:
    def test_recording_span_gets_ended(self):
        """A recording span should be ended by safe_close_step."""
        span = mock.MagicMock()
        span.is_recording.return_value = True

        instance = types.SimpleNamespace(
            **{
                INST_STEP_SPAN_ATTR: span,
                INST_STEP_TOKEN_ATTR: mock.MagicMock(),
            }
        )

        safe_close_step(instance)

        span.end.assert_called_once()
        assert getattr(instance, INST_STEP_SPAN_ATTR) is None
        assert getattr(instance, INST_STEP_TOKEN_ATTR) is None

    def test_non_recording_span_not_ended(self):
        """A non-recording span (already ended) should not be ended again."""
        span = mock.MagicMock()
        span.is_recording.return_value = False

        instance = types.SimpleNamespace(
            **{
                INST_STEP_SPAN_ATTR: span,
                INST_STEP_TOKEN_ATTR: None,
            }
        )

        safe_close_step(instance)

        span.end.assert_not_called()
        assert getattr(instance, INST_STEP_SPAN_ATTR) is None

    def test_none_span(self):
        """When span is None, safe_close_step should be a no-op."""
        instance = types.SimpleNamespace(
            **{
                INST_STEP_SPAN_ATTR: None,
                INST_STEP_TOKEN_ATTR: None,
            }
        )

        # Should not raise.
        safe_close_step(instance)

        assert getattr(instance, INST_STEP_SPAN_ATTR) is None

    def test_no_attributes_on_instance(self):
        """Instance without OTEL attributes should not raise."""
        instance = types.SimpleNamespace()

        # Should not raise.
        safe_close_step(instance)

    def test_token_gets_detached(self):
        """The OTel context token should be detached."""
        span = mock.MagicMock()
        span.is_recording.return_value = True
        token = mock.MagicMock()

        instance = types.SimpleNamespace(
            **{
                INST_STEP_SPAN_ATTR: span,
                INST_STEP_TOKEN_ATTR: token,
            }
        )

        with mock.patch("opentelemetry.context.detach") as mock_detach:
            safe_close_step(instance)
            mock_detach.assert_called_once_with(token)

    def test_span_end_raises_exception(self):
        """If span.end() raises, safe_close_step should still clear state."""
        span = mock.MagicMock()
        span.is_recording.return_value = True
        span.end.side_effect = RuntimeError("end failed")

        instance = types.SimpleNamespace(
            **{
                INST_STEP_SPAN_ATTR: span,
                INST_STEP_TOKEN_ATTR: None,
            }
        )

        # Should not raise.
        safe_close_step(instance)

        assert getattr(instance, INST_STEP_SPAN_ATTR) is None


# ---------------------------------------------------------------------------
# config.py: _bool_env, _int_env, _float_env, _genai_capture_enabled
# ---------------------------------------------------------------------------


class TestBoolEnv:
    def test_default_true(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _bool_env

        monkeypatch.delenv("TEST_BOOL_ENV_VAR", raising=False)
        assert _bool_env("TEST_BOOL_ENV_VAR", True) is True

    def test_default_false(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _bool_env

        monkeypatch.delenv("TEST_BOOL_ENV_VAR", raising=False)
        assert _bool_env("TEST_BOOL_ENV_VAR", False) is False

    def test_true_values(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _bool_env

        for val in ("true", "1", "yes", "on", "True", "YES", "ON"):
            monkeypatch.setenv("TEST_BOOL_ENV_VAR", val)
            assert _bool_env("TEST_BOOL_ENV_VAR", False) is True, (
                f"Failed for {val}"
            )

    def test_false_values(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _bool_env

        for val in ("false", "0", "no", "off", "random"):
            monkeypatch.setenv("TEST_BOOL_ENV_VAR", val)
            assert _bool_env("TEST_BOOL_ENV_VAR", True) is False, (
                f"Failed for {val}"
            )


class TestIntEnv:
    def test_default_value(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _int_env

        monkeypatch.delenv("TEST_INT_ENV_VAR", raising=False)
        assert _int_env("TEST_INT_ENV_VAR", "42") == 42

    def test_custom_value(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _int_env

        monkeypatch.setenv("TEST_INT_ENV_VAR", "100")
        assert _int_env("TEST_INT_ENV_VAR", "42") == 100

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _int_env

        monkeypatch.setenv("TEST_INT_ENV_VAR", "not_a_number")
        assert _int_env("TEST_INT_ENV_VAR", "42") == 42


class TestFloatEnv:
    def test_default_value(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _float_env

        monkeypatch.delenv("TEST_FLOAT_ENV_VAR", raising=False)
        assert _float_env("TEST_FLOAT_ENV_VAR", "3.14") == 3.14

    def test_custom_value(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _float_env

        monkeypatch.setenv("TEST_FLOAT_ENV_VAR", "2.71")
        assert _float_env("TEST_FLOAT_ENV_VAR", "3.14") == 2.71

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import _float_env

        monkeypatch.setenv("TEST_FLOAT_ENV_VAR", "not_a_float")
        assert _float_env("TEST_FLOAT_ENV_VAR", "3.14") == 3.14


class TestGenaiCaptureEnabled:
    def test_not_set_returns_false(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import (
            _genai_capture_enabled,
        )

        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
        assert _genai_capture_enabled() is False

    def test_true_value(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import (
            _genai_capture_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "TRUE"
        )
        assert _genai_capture_enabled() is True

    def test_span_only(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import (
            _genai_capture_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        assert _genai_capture_enabled() is True

    def test_span_and_event(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import (
            _genai_capture_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "SPAN_AND_EVENT",
        )
        assert _genai_capture_enabled() is True

    def test_event_only(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import (
            _genai_capture_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "EVENT_ONLY"
        )
        assert _genai_capture_enabled() is True

    def test_invalid_value_returns_false(self, monkeypatch):
        from opentelemetry.instrumentation.algotune.config import (
            _genai_capture_enabled,
        )

        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "invalid"
        )
        assert _genai_capture_enabled() is False
