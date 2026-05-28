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

"""Unit tests for pure helper functions in the terminus2 instrumentation."""

from __future__ import annotations

import json

import pytest

from opentelemetry.instrumentation.terminus2 import (
    _GEN_AI_REACT_FINISH_REASON,
    _commands_to_arguments_json,
    _current_step_span,
    _current_step_token,
    _end_current_step,
    _infer_provider_name,
    _text_messages_json,
)

from .conftest import Command

# ═══════════════════════════════════════════════════════════════════════════
# _commands_to_arguments_json
# ═══════════════════════════════════════════════════════════════════════════


class TestCommandsToArgumentsJson:
    """Tests for _commands_to_arguments_json."""

    def test_empty_list(self):
        result = _commands_to_arguments_json([])
        assert json.loads(result) == []

    def test_single_command(self):
        cmds = [Command(keystrokes="ls -la", duration_sec=5.0)]
        result = _commands_to_arguments_json(cmds)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["keystrokes"] == "ls -la"
        assert parsed[0]["duration_sec"] == 5.0

    def test_multiple_commands(self):
        cmds = [
            Command(keystrokes="cd /tmp", duration_sec=1.0),
            Command(keystrokes="echo hello", duration_sec=2.0),
            Command(keystrokes="exit", duration_sec=None),
        ]
        result = _commands_to_arguments_json(cmds)
        parsed = json.loads(result)
        assert len(parsed) == 3
        assert parsed[0]["keystrokes"] == "cd /tmp"
        assert parsed[0]["duration_sec"] == 1.0
        assert parsed[1]["keystrokes"] == "echo hello"
        assert parsed[2]["duration_sec"] is None

    def test_command_without_attributes(self):
        """Object with missing keystrokes/duration_sec attributes."""

        class BareObj:
            pass

        cmds = [BareObj()]
        result = _commands_to_arguments_json(cmds)
        parsed = json.loads(result)
        assert parsed[0]["keystrokes"] == ""
        assert parsed[0]["duration_sec"] is None

    def test_unicode_keystrokes(self):
        cmds = [Command(keystrokes="echo 你好", duration_sec=1.0)]
        result = _commands_to_arguments_json(cmds)
        # ensure_ascii=False means Chinese characters are preserved
        assert "你好" in result
        parsed = json.loads(result)
        assert parsed[0]["keystrokes"] == "echo 你好"


# ═══════════════════════════════════════════════════════════════════════════
# _text_messages_json
# ═══════════════════════════════════════════════════════════════════════════


class TestTextMessagesJson:
    """Tests for _text_messages_json."""

    def test_user_role(self):
        result = _text_messages_json("user", "hello world")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"
        assert parsed[0]["parts"][0]["type"] == "text"
        assert parsed[0]["parts"][0]["content"] == "hello world"

    def test_assistant_role(self):
        result = _text_messages_json("assistant", "I will help")
        parsed = json.loads(result)
        assert parsed[0]["role"] == "assistant"
        assert parsed[0]["parts"][0]["content"] == "I will help"

    def test_non_string_content(self):
        """Content should be str()-ified."""
        result = _text_messages_json("system", 42)
        parsed = json.loads(result)
        assert parsed[0]["parts"][0]["content"] == "42"

    def test_empty_content(self):
        result = _text_messages_json("user", "")
        parsed = json.loads(result)
        assert parsed[0]["parts"][0]["content"] == ""

    def test_compact_separators(self):
        """Verify the output uses compact JSON separators (no spaces)."""
        result = _text_messages_json("user", "x")
        # separators=(",", ":") means no space after : or ,
        assert ": " not in result
        assert ", " not in result

    def test_none_content(self):
        result = _text_messages_json("user", None)
        parsed = json.loads(result)
        assert parsed[0]["parts"][0]["content"] == "None"


# ═══════════════════════════════════════════════════════════════════════════
# _infer_provider_name
# ═══════════════════════════════════════════════════════════════════════════


class TestInferProviderName:
    """Tests for _infer_provider_name."""

    # -- OpenAI variants --
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-3.5-turbo",
            "GPT-4-turbo",
            "o1-mini",
            "o3-preview",
            "o4-mini",
        ],
    )
    def test_openai_models(self, model):
        assert _infer_provider_name(model) == "openai"

    # -- Anthropic variants --
    @pytest.mark.parametrize(
        "model",
        ["claude-3-opus-20240229", "claude-3-5-sonnet", "anthropic/claude-3"],
    )
    def test_anthropic_models(self, model):
        assert _infer_provider_name(model) == "anthropic"

    # -- Google --
    @pytest.mark.parametrize("model", ["gemini-pro", "gemini-1.5-flash"])
    def test_google_models(self, model):
        assert _infer_provider_name(model) == "google"

    # -- Meta --
    @pytest.mark.parametrize("model", ["llama-3-70b", "meta-llama/Llama-2-7b"])
    def test_meta_models(self, model):
        assert _infer_provider_name(model) == "meta"

    # -- Mistral --
    @pytest.mark.parametrize("model", ["mistral-large-latest", "mistral-7b"])
    def test_mistral_models(self, model):
        assert _infer_provider_name(model) == "mistral"

    # -- Alibaba --
    @pytest.mark.parametrize(
        "model", ["qwen-72b", "qwen-turbo", "Qwen2.5-Coder"]
    )
    def test_alibaba_models(self, model):
        assert _infer_provider_name(model) == "alibaba"

    # -- DeepSeek --
    @pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-coder-v2"])
    def test_deepseek_models(self, model):
        assert _infer_provider_name(model) == "deepseek"

    # -- Prefix/model pattern --
    def test_prefix_model_slash(self):
        assert _infer_provider_name("together/llama-3-70b") == "meta"

    def test_prefix_model_slash_unknown_keyword(self):
        """prefix/model where keyword is not recognized -> prefix."""
        assert _infer_provider_name("groq/some-custom-model") == "groq"

    # -- Edge cases --
    def test_empty_string(self):
        assert _infer_provider_name("") == "unknown"

    def test_unknown_model(self):
        assert _infer_provider_name("my-custom-model") == "unknown"

    def test_case_insensitivity(self):
        assert _infer_provider_name("Claude-3.5-Sonnet") == "anthropic"
        assert _infer_provider_name("GEMINI-PRO") == "google"
        assert _infer_provider_name("GPT-4O") == "openai"


# ═══════════════════════════════════════════════════════════════════════════
# _end_current_step
# ═══════════════════════════════════════════════════════════════════════════


class TestEndCurrentStep:
    """Tests for _end_current_step."""

    def test_no_active_span(self):
        """Calling with no active span should be a safe no-op."""
        _current_step_span.set(None)
        _current_step_token.set(None)
        _end_current_step()  # should not raise

    def test_no_active_span_with_reason(self):
        """Calling with finish_reason and no span should be a safe no-op."""
        _current_step_span.set(None)
        _current_step_token.set(None)
        _end_current_step(finish_reason="loop_end")  # should not raise

    def test_ends_active_span(self, tracer_provider, span_exporter):
        """Should end the span and clear the ContextVar."""
        from opentelemetry import trace as trace_api

        tracer = trace_api.get_tracer("test", tracer_provider=tracer_provider)
        span = tracer.start_span("test-step")
        _current_step_span.set(span)
        _current_step_token.set(None)  # token is optional for this test

        _end_current_step(finish_reason="complete")

        assert _current_step_span.get() is None
        # Span should have been exported
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test-step"
        assert (
            spans[0].attributes.get(_GEN_AI_REACT_FINISH_REASON) == "complete"
        )

    def test_ends_span_without_finish_reason(
        self, tracer_provider, span_exporter
    ):
        """When finish_reason is None, no finish_reason attribute is set."""
        from opentelemetry import trace as trace_api

        tracer = trace_api.get_tracer("test", tracer_provider=tracer_provider)
        span = tracer.start_span("test-step-no-reason")
        _current_step_span.set(span)
        _current_step_token.set(None)

        _end_current_step()

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert _GEN_AI_REACT_FINISH_REASON not in spans[0].attributes

    def test_detaches_context_token(self, tracer_provider):
        """Should detach the context token and clear the ContextVar."""
        from opentelemetry import context as context_api
        from opentelemetry import trace as trace_api

        tracer = trace_api.get_tracer("test", tracer_provider=tracer_provider)
        span = tracer.start_span("test-step-token")
        ctx = trace_api.set_span_in_context(span)
        token = context_api.attach(ctx)

        _current_step_span.set(span)
        _current_step_token.set(token)

        _end_current_step(finish_reason="loop_end")

        assert _current_step_span.get() is None
        assert _current_step_token.get() is None
