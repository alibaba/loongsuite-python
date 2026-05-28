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

"""Comprehensive wrapper tests for WebArena instrumentation spans.

Each test class covers one wrapper / span type. Tests verify:
- Span names and span kind attributes
- gen_ai.span.kind values (ENTRY, CHAIN, STEP, AGENT, TOOL, TASK, LLM)
- gen_ai.framework == "webarena"
- WebArena-specific attributes (task_id, action_type, etc.)
- Error recording
- Parent-child relationships
- Content capture gating (``capture_message_content``)

Implementation note:  Because ``wrapt.wrap_function_wrapper`` patches functions
/methods *in-place* on the module or class, any reference captured at import
time (``from agent.agent import construct_agent``) still points to the
*unwrapped* original.  We therefore access the wrapped versions at call-time
via ``sys.modules``.  For instance-method error paths, we temporarily swap the
``__wrapped__`` attribute on the class method descriptor so that the wrapper
sees the failure without subclass method overrides bypassing the descriptor.
"""

from __future__ import annotations

import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from opentelemetry.trace import StatusCode

# Import conftest helpers via the tests package
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
from conftest import (
    ActionTypes,
    make_action,
    make_ns_args,
    span_attr,
)

# State module for resetting between tests
from opentelemetry.instrumentation.webarena.internal import _state as state

# Attribute constants used by the production code
from opentelemetry.instrumentation.webarena.internal._attrs import (
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GEN_AI_SPAN_KIND,
    WEBARENA_ACTION_SET_TAG,
    WEBARENA_ACTION_TYPE,
    WEBARENA_BROWSER_ELEMENT_ID,
    WEBARENA_FAIL_ERROR,
    WEBARENA_MEMORY_TRAJECTORY_LENGTH,
    WEBARENA_OBSERVATION_MAIN_TYPE,
    WEBARENA_OBSERVATION_TYPE,
    WEBARENA_PAGE_URL_AFTER,
    WEBARENA_PAGE_URL_BEFORE,
    WEBARENA_PARSING_FAILURE_COUNT,
    WEBARENA_PREVIOUS_ACTION,
    WEBARENA_REQUIRE_LOGIN,
    WEBARENA_STEP_COUNT,
    WEBARENA_TASK_ID,
    WEBARENA_TOOL_COUNT,
)

# ---------------------------------------------------------------------------
# Accessor helpers: always fetch classes/functions from sys.modules so we
# use the *wrapped* versions after instrumentation patches them.
# ---------------------------------------------------------------------------


def _new_env():
    """Return a new ScriptBrowserEnv instance (wrapped class)."""
    return sys.modules["browser_env.envs"].ScriptBrowserEnv()


def _new_agent():
    """Return a new PromptAgent instance (wrapped class)."""
    return sys.modules["agent.agent"].PromptAgent()


def _construct_agent(ns_args):
    """Call construct_agent via sys.modules (wrapped version)."""
    return sys.modules["agent.agent"].construct_agent(ns_args)


def _generate_hf(*args, **kwargs):
    """Call generate_from_huggingface_completion via sys.modules (wrapped)."""
    fn = sys.modules[
        "llms.providers.hf_utils"
    ].generate_from_huggingface_completion
    return fn(*args, **kwargs)


def _new_direct_pc():
    """Return a new DirectPromptConstructor instance (wrapped class)."""
    return sys.modules[
        "agent.prompts.prompt_constructor"
    ].DirectPromptConstructor()


def _new_cot_pc():
    """Return a new CoTPromptConstructor instance (wrapped class)."""
    return sys.modules[
        "agent.prompts.prompt_constructor"
    ].CoTPromptConstructor()


def _make_tracer(span_exporter):
    """Create a tracer backed by the test exporter (for direct wrapper tests)."""
    from opentelemetry.sdk.trace import TracerProvider as _TP
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = _TP()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider.get_tracer("test-tracer")


@contextmanager
def _swap_wrapped(cls, method_name: str, replacement):
    """Temporarily replace the ``__wrapped__`` attribute of a wrapt-patched method.

    This lets us inject a failing / custom function into the wrapper's call
    path without subclassing (which would bypass the ``wrapt`` descriptor).
    """
    descriptor = getattr(cls, method_name)
    original = descriptor.__wrapped__
    descriptor.__wrapped__ = replacement
    try:
        yield
    finally:
        descriptor.__wrapped__ = original


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Ensure each test starts with a clean state machine."""
    state.end_task_spans()
    yield
    state.end_task_spans()


# ===================================================================
# Config helpers
# ===================================================================


class TestConfig:
    """Tests for config.py helper functions."""

    def test_int_env_default(self):
        from opentelemetry.instrumentation.webarena.config import _int_env

        val = _int_env("WEBARENA_TEST_NONEXISTENT_12345", "42")
        assert val == 42

    def test_int_env_override(self):
        from opentelemetry.instrumentation.webarena.config import _int_env

        os.environ["WEBARENA_TEST_INT"] = "99"
        try:
            assert _int_env("WEBARENA_TEST_INT", "42") == 99
        finally:
            os.environ.pop("WEBARENA_TEST_INT", None)

    def test_int_env_invalid_falls_back_to_default(self):
        from opentelemetry.instrumentation.webarena.config import _int_env

        os.environ["WEBARENA_TEST_INT_BAD"] = "not_a_number"
        try:
            assert _int_env("WEBARENA_TEST_INT_BAD", "10") == 10
        finally:
            os.environ.pop("WEBARENA_TEST_INT_BAD", None)

    def test_bool_env_default_false(self):
        from opentelemetry.instrumentation.webarena.config import _bool_env

        assert _bool_env("WEBARENA_BOOL_NONEXISTENT") is False

    def test_bool_env_truthy_values(self):
        from opentelemetry.instrumentation.webarena.config import _bool_env

        for val in ("1", "true", "yes", "on", "True", "YES", "ON"):
            os.environ["WEBARENA_BOOL_TEST"] = val
            assert _bool_env("WEBARENA_BOOL_TEST") is True, (
                f"Expected True for {val!r}"
            )
        os.environ.pop("WEBARENA_BOOL_TEST", None)

    def test_bool_env_falsy_values(self):
        from opentelemetry.instrumentation.webarena.config import _bool_env

        for val in ("0", "false", "no", "off", "random"):
            os.environ["WEBARENA_BOOL_TEST"] = val
            assert _bool_env("WEBARENA_BOOL_TEST") is False, (
                f"Expected False for {val!r}"
            )
        os.environ.pop("WEBARENA_BOOL_TEST", None)

    def test_capture_message_content_truthy(self):
        from opentelemetry.instrumentation.webarena.config import (
            capture_message_content,
        )

        for val in (
            "TRUE",
            "1",
            "YES",
            "ON",
            "SPAN_ONLY",
            "SPAN_AND_EVENT",
            "EVENT_ONLY",
        ):
            os.environ[
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
            ] = val
            assert capture_message_content() is True, (
                f"Expected True for {val!r}"
            )
        os.environ.pop(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
        )

    def test_capture_message_content_default_false(self):
        from opentelemetry.instrumentation.webarena.config import (
            capture_message_content,
        )

        os.environ.pop(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
        )
        assert capture_message_content() is False


# ===================================================================
# Attribute helpers (_attrs.py)
# ===================================================================


class TestAttrHelpers:
    """Tests for internal._attrs helper functions."""

    def test_truncate_short_string(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            truncate,
        )

        assert truncate("hello", 1024) == "hello"

    def test_truncate_long_string(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            truncate,
        )

        long = "x" * 2000
        result = truncate(long, 100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_truncate_none_returns_empty(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            truncate,
        )

        assert truncate(None, 100) == ""

    def test_truncate_non_string(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            truncate,
        )

        assert truncate(42, 100) == "42"

    def test_truncate_very_small_max(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            truncate,
        )

        assert truncate("abcdef", 3) == "abc"

    def test_safe_json_dumps_dict(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            safe_json_dumps,
        )

        result = safe_json_dumps({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_safe_json_dumps_with_max_len(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            safe_json_dumps,
        )

        result = safe_json_dumps({"key": "x" * 5000}, max_len=50)
        assert len(result) <= 50

    def test_safe_json_dumps_unencodable(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            safe_json_dumps,
        )

        class Weird:
            pass

        result = safe_json_dumps(Weird())
        assert isinstance(result, str)

    def test_action_type_name_dict_with_enum(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_type_name,
        )

        action = {"action_type": ActionTypes.CLICK}
        assert action_type_name(action) == "CLICK"

    def test_action_type_name_not_dict(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_type_name,
        )

        assert action_type_name("not a dict") == "UNKNOWN"

    def test_action_type_name_missing_key(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_type_name,
        )

        assert action_type_name({}) == "UNKNOWN"

    def test_action_arguments_filters_keys(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_arguments,
        )

        action = {
            "action_type": ActionTypes.CLICK,
            "element_id": "10",
            "coords": [100, 200],
            "raw_prediction": "click [10]",
            "page_screenshot": b"binary_data",
            "url": "http://example.com",
        }
        result = action_arguments(action)
        assert result["action_type"] == "CLICK"
        assert result["element_id"] == "10"
        assert result["url"] == "http://example.com"
        assert "coords" not in result
        assert "raw_prediction" not in result
        assert "page_screenshot" not in result

    def test_action_arguments_not_dict(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_arguments,
        )

        assert action_arguments("nope") == {}

    def test_messages_to_input_value_string(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            messages_to_input_value,
        )

        assert messages_to_input_value("hello") == "hello"

    def test_messages_to_input_value_list(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            messages_to_input_value,
        )

        msgs = [{"role": "user", "content": "hi"}]
        result = messages_to_input_value(msgs)
        assert "user" in result

    def test_messages_to_input_value_other(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            messages_to_input_value,
        )

        result = messages_to_input_value(42)
        assert result == "42"


# ===================================================================
# State machine (_state.py)
# ===================================================================


class TestState:
    """Tests for the _state module context-var helpers."""

    def test_initial_state(self):
        state.end_task_spans()
        assert state.in_task() is False
        assert state.step_count() == 0
        assert state.tool_count() == 0
        assert state.parsing_failure_count() == 0

    def test_mark_in_task(self):
        state.mark_in_task(True)
        assert state.in_task() is True
        state.mark_in_task(False)
        assert state.in_task() is False

    def test_increment_counters(self):
        state.end_task_spans()
        assert state.increment_step() == 1
        assert state.increment_step() == 2
        assert state.step_count() == 2

        assert state.increment_tool() == 1
        assert state.increment_tool() == 2
        assert state.tool_count() == 2

        assert state.increment_parsing_failure() == 1
        assert state.parsing_failure_count() == 1

    def test_end_task_spans_resets_counters(self):
        state.mark_in_task(True)
        state.increment_step()
        state.increment_tool()
        state.increment_parsing_failure()
        state.end_task_spans()
        assert state.in_task() is False
        assert state.step_count() == 0
        assert state.tool_count() == 0
        assert state.parsing_failure_count() == 0

    def test_end_step_returns_round_number(self):
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("test step")
        span.set_attribute(GEN_AI_REACT_ROUND, 5)

        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        token = otel_context.attach(set_span_in_context(span))
        state.set_step(span, token)

        round_no = state.end_step()
        assert round_no == 5

    def test_end_step_no_active_returns_zero(self):
        state.end_task_spans()
        assert state.end_step() == 0

    def test_get_span_accessors_none_by_default(self):
        state.end_task_spans()
        assert state.get_entry_span() is None
        assert state.get_chain_span() is None
        assert state.get_step_span() is None

    def test_detach_token_none_is_safe(self):
        state._detach_token(None)

    def test_detach_token_already_detached_is_safe(self):
        from opentelemetry import context as otel_context

        token = otel_context.attach(otel_context.get_current())
        otel_context.detach(token)
        state._detach_token(token)


# ===================================================================
# EnvResetWrapper  ->  ENTRY + CHAIN spans
# ===================================================================


class TestEnvResetWrapper:
    """Tests for ScriptBrowserEnv.reset -> ENTRY + CHAIN spans."""

    def test_reset_creates_entry_and_chain(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ]
        chain_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ]
        assert len(entry_spans) >= 1
        assert len(chain_spans) >= 1

    def test_entry_span_attributes(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        assert span_attr(entry, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(entry, "gen_ai.operation.name") == "enter"

    def test_chain_span_attributes(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        chain = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ][0]
        assert span_attr(chain, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(chain, "gen_ai.operation.name") == "workflow"
        assert "webarena_task" in chain.name

    def test_chain_is_child_of_entry(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        chain = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ][0]
        assert chain.parent is not None
        assert chain.parent.span_id == entry.context.span_id

    def test_reset_with_config_file(self, instrument, span_exporter, tmp_path):
        config = {
            "task_id": 42,
            "intent": "Find the cheapest flight",
            "sites": ["shopping", "reddit"],
            "storage_state": "/some/cookie.json",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config))

        env = _new_env()
        env.reset(options={"config_file": str(cfg_path)})
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        assert span_attr(entry, WEBARENA_TASK_ID) == "42"
        assert span_attr(entry, WEBARENA_REQUIRE_LOGIN) is True
        assert "42" in entry.name

    def test_reset_with_config_list(self, instrument, span_exporter, tmp_path):
        config = [{"task_id": 7, "intent": "do something", "sites": []}]
        cfg_path = tmp_path / "config_list.json"
        cfg_path.write_text(json.dumps(config))

        env = _new_env()
        env.reset(options={"config_file": str(cfg_path)})
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        assert span_attr(entry, WEBARENA_TASK_ID) == "7"

    def test_reset_error_records_exception(self, span_exporter):
        """If the wrapped reset() raises, the ENTRY span should record the exception.

        Tested by invoking the wrapper class directly with a failing ``wrapped``.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvResetWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvResetWrapper(tracer)

        def _failing_reset(*args, **kwargs):
            raise RuntimeError("browser init failed")

        env = _new_env()
        with pytest.raises(RuntimeError, match="browser init failed"):
            wrapper(_failing_reset, env, (), {"options": None})

        spans = local_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        assert entry.status.status_code == StatusCode.ERROR
        events = entry.events
        assert any("browser init failed" in str(e.attributes) for e in events)

    def test_consecutive_resets_close_previous_task(
        self, instrument, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)
        env.reset(options=None)
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ]
        assert len(entry_spans) == 2

    def test_reset_with_content_capture(
        self, instrument_with_content, span_exporter, tmp_path
    ):
        config = {"task_id": 1, "intent": "Buy the cheapest item", "sites": []}
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))

        env = _new_env()
        env.reset(options={"config_file": str(cfg_path)})
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        input_msgs = span_attr(entry, "gen_ai.input.messages")
        assert input_msgs is not None
        assert "Buy the cheapest item" in input_msgs

    def test_reset_output_messages_with_content_capture(
        self, instrument_with_content, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        entry = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ][0]
        output_msgs = span_attr(entry, "gen_ai.output.messages")
        assert output_msgs is not None
        assert "Initial observation" in output_msgs


# ===================================================================
# EnvCloseWrapper
# ===================================================================


class TestEnvCloseWrapper:
    """Tests for ScriptBrowserEnv.close -> finalizing open spans."""

    def test_close_finalizes_spans(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)
        env.close()

        spans = span_exporter.get_finished_spans()
        entry_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ]
        chain_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ]
        assert len(entry_spans) >= 1
        assert len(chain_spans) >= 1

    def test_close_without_reset_is_safe(self, instrument, span_exporter):
        env = _new_env()
        env.close()

    def test_close_sets_step_and_tool_counts(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "do something", {"action_history": []})
        env.step(make_action())

        env.close()

        spans = span_exporter.get_finished_spans()
        chain = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ][0]
        assert span_attr(chain, WEBARENA_STEP_COUNT) == 1
        assert span_attr(chain, WEBARENA_TOOL_COUNT) == 1


# ===================================================================
# NextActionWrapper  ->  STEP + AGENT spans
# ===================================================================


class TestNextActionWrapper:
    """Tests for PromptAgent.next_action -> STEP + AGENT spans."""

    def test_creates_step_and_agent_spans(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "click the button", {"action_history": []})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        step_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"
        ]
        agent_spans = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ]
        assert len(step_spans) >= 1
        assert len(agent_spans) >= 1

    def test_step_span_has_round_number(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        assert span_attr(step, GEN_AI_REACT_ROUND) == 1

    def test_multiple_rounds_increment_step_number(
        self, instrument, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs1", "intent", {"action_history": []})
        agent.next_action("obs2", "intent", {"action_history": ["click [1]"]})
        agent.next_action(
            "obs3", "intent", {"action_history": ["click [1]", "type [2]"]}
        )

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        step_spans = sorted(
            [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"],
            key=lambda s: span_attr(s, GEN_AI_REACT_ROUND),
        )
        assert len(step_spans) == 3
        assert span_attr(step_spans[0], GEN_AI_REACT_ROUND) == 1
        assert span_attr(step_spans[1], GEN_AI_REACT_ROUND) == 2
        assert span_attr(step_spans[2], GEN_AI_REACT_ROUND) == 3

    def test_agent_span_attributes(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        assert span_attr(agent_span, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(agent_span, "gen_ai.operation.name") == "invoke_agent"
        assert "PromptAgent" in span_attr(agent_span, "gen_ai.agent.name")
        assert span_attr(agent_span, "gen_ai.request.model") == "gpt-4"
        assert span_attr(agent_span, "gen_ai.provider.name") == "openai"

    def test_agent_span_records_action_type(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        assert span_attr(agent_span, WEBARENA_ACTION_TYPE) == "CLICK"

    def test_agent_span_previous_action(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": ["click [5]"]})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        assert "click [5]" in span_attr(agent_span, WEBARENA_PREVIOUS_ACTION)

    def test_agent_span_is_child_of_step(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        assert agent_span.parent is not None
        assert agent_span.parent.span_id == step.context.span_id

    def test_step_is_child_of_chain(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        chain = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ][0]
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        assert step.parent is not None
        assert step.parent.span_id == chain.context.span_id

    def test_agent_error_is_recorded(self, span_exporter):
        """When next_action raises, AGENT span should record the error.

        Tested by invoking the wrapper directly with a failing wrapped function.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            NextActionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)

        # Manually set up task state so the wrapper creates a STEP
        state.mark_in_task(True)

        wrapper = NextActionWrapper(tracer)
        agent = _new_agent()

        def _failing(*args, **kwargs):
            raise ValueError("LLM call failed")

        with pytest.raises(ValueError, match="LLM call failed"):
            wrapper(
                _failing, agent, ("obs", "intent", {"action_history": []}), {}
            )

        state.end_task_spans()

        spans = local_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        assert agent_span.status.status_code == StatusCode.ERROR
        assert (
            span_attr(agent_span, GEN_AI_REACT_FINISH_REASON) == "ValueError"
        )

    def test_stop_action_sets_finish_reason(self, span_exporter):
        """When next_action returns STOP, the STEP should record finish_reason=stop.

        Tested by invoking the wrapper directly.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            NextActionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)

        state.mark_in_task(True)

        wrapper = NextActionWrapper(tracer)
        agent = _new_agent()

        def _stop_action(*args, **kwargs):
            return {
                "action_type": ActionTypes.STOP,
                "answer": "The price is $42",
                "raw_prediction": "stop [The price is $42]",
            }

        wrapper(
            _stop_action, agent, ("obs", "intent", {"action_history": []}), {}
        )
        state.end_task_spans()

        spans = local_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        assert span_attr(step, GEN_AI_REACT_FINISH_REASON) == "stop"

    def test_none_action_increments_parsing_failure(self, span_exporter):
        """When next_action returns NONE action, parsing failure count should increment.

        Tested by invoking the wrapper directly.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            NextActionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)

        state.mark_in_task(True)

        wrapper = NextActionWrapper(tracer)
        agent = _new_agent()

        def _none_action(*args, **kwargs):
            return {
                "action_type": ActionTypes.NONE,
                "raw_prediction": "gibberish",
            }

        wrapper(
            _none_action, agent, ("obs", "intent", {"action_history": []}), {}
        )

        assert state.parsing_failure_count() == 1

        state.end_task_spans()

        spans = local_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        assert span_attr(step, GEN_AI_REACT_FINISH_REASON) == "parse_failure"

    def test_next_action_outside_task_no_step(self, instrument, span_exporter):
        state.end_task_spans()
        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ]
        step_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"
        ]
        assert len(agent_spans) >= 1
        assert len(step_spans) == 0

    def test_agent_content_capture(
        self, instrument_with_content, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action(
            "obs",
            "click the login button",
            {"action_history": ["type [1] hello"]},
        )

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        input_msgs = span_attr(agent_span, "gen_ai.input.messages")
        assert input_msgs is not None
        assert "click the login button" in input_msgs
        output_msgs = span_attr(agent_span, "gen_ai.output.messages")
        assert output_msgs is not None
        assert "click [42]" in output_msgs
        tool_defs = span_attr(agent_span, "gen_ai.tool.definitions")
        assert tool_defs is not None
        assert "click" in tool_defs
        sys_instr = span_attr(agent_span, "gen_ai.system_instructions")
        assert sys_instr is not None
        assert "web browsing agent" in sys_instr

    def test_agent_kwargs_intent_and_meta(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action(
            "obs",
            intent="find the product",
            meta_data={"action_history": ["goto [url]"]},
        )

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ][0]
        assert "goto [url]" in span_attr(agent_span, WEBARENA_PREVIOUS_ACTION)


# ===================================================================
# EnvStepWrapper  ->  TOOL spans
# ===================================================================


class TestEnvStepWrapper:
    """Tests for ScriptBrowserEnv.step -> TOOL(execute_tool) spans."""

    def test_step_creates_tool_span(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        env.step(make_action(ActionTypes.CLICK, element_id="10"))
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"
        ]
        assert len(tool_spans) == 1

    def test_tool_span_attributes(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        env.step(
            make_action(ActionTypes.GOTO, element_id="", url="http://shop.com")
        )
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(tool, "gen_ai.operation.name") == "execute_tool"
        assert span_attr(tool, "gen_ai.tool.name") == "GOTO"
        assert span_attr(tool, "gen_ai.tool.type") == "browser_action"
        assert "execute_tool GOTO" in tool.name

    def test_tool_span_records_element_id(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        env.step(make_action(ActionTypes.CLICK, element_id="77"))
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, WEBARENA_BROWSER_ELEMENT_ID) == "77"

    def test_tool_span_records_page_url_before(
        self, instrument, span_exporter
    ):
        env = _new_env()
        env.page = types.SimpleNamespace(url="http://before.example.com")
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        env.step(make_action())
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert "before.example.com" in span_attr(
            tool, WEBARENA_PAGE_URL_BEFORE
        )

    def test_tool_span_records_observation_main_type(
        self, instrument, span_exporter
    ):
        env = _new_env()
        env.main_observation_type = "image"
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        env.step(make_action())
        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, WEBARENA_OBSERVATION_MAIN_TYPE) == "image"

    def test_tool_span_is_child_of_step(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})
        env.step(make_action())

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert tool.parent is not None
        assert tool.parent.span_id == step.context.span_id

    def test_tool_span_error_path(self, span_exporter):
        """If the underlying step() raises, the TOOL span should record the error.

        Tested by invoking the wrapper directly.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvStepWrapper(tracer)
        env = _new_env()

        def _failing(*args, **kwargs):
            raise RuntimeError("Playwright timeout")

        action = make_action()
        with pytest.raises(RuntimeError, match="Playwright timeout"):
            wrapper(_failing, env, (action,), {})

        spans = local_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert tool.status.status_code == StatusCode.ERROR

    def test_tool_span_fail_error_attribute(self, span_exporter):
        """When step result contains fail_error, it should be recorded.

        Tested by invoking the wrapper directly.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvStepWrapper(tracer)
        env = _new_env()

        def _fail_result(*args, **kwargs):
            return (
                "obs",
                False,
                False,
                False,
                {"fail_error": "element not found"},
            )

        action = make_action()
        wrapper(_fail_result, env, (action,), {})

        spans = local_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, WEBARENA_FAIL_ERROR) == "element not found"
        assert tool.status.status_code == StatusCode.ERROR

    def test_tool_span_terminated_sets_step_finish_reason(
        self, instrument, span_exporter
    ):
        """When step returns terminated=True, the parent STEP should get finish_reason=terminated.

        Tested by invoking the wrapper directly within a STEP context.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
            _rotate_step,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)

        # Set up a STEP span so the wrapper can propagate to it
        state.mark_in_task(True)
        _rotate_step(tracer)

        wrapper = EnvStepWrapper(tracer)
        env = _new_env()

        def _terminated(*args, **kwargs):
            return ("obs", True, True, False, {"fail_error": ""})

        action = make_action()
        wrapper(_terminated, env, (action,), {})

        state.end_task_spans()

        spans = local_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        assert span_attr(step, GEN_AI_REACT_FINISH_REASON) == "terminated"

    def test_tool_span_with_content_capture(
        self, instrument_with_content, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        action = make_action(
            ActionTypes.TYPE, element_id="5", text="hello world"
        )
        env.step(action)

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        args_str = span_attr(tool, "gen_ai.tool.call.arguments")
        assert args_str is not None
        assert "TYPE" in args_str
        result_str = span_attr(tool, "gen_ai.tool.call.result")
        assert result_str is not None

    def test_multiple_tool_calls_counted(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})

        env.step(make_action(ActionTypes.CLICK))
        env.step(make_action(ActionTypes.TYPE))
        env.step(make_action(ActionTypes.SCROLL))

        env.close()

        spans = span_exporter.get_finished_spans()
        chain = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ][0]
        assert span_attr(chain, WEBARENA_TOOL_COUNT) == 3


# ===================================================================
# PromptConstructWrapper  ->  TASK spans
# ===================================================================


class TestPromptConstructWrapper:
    """Tests for PromptConstructor.construct -> TASK(build_prompt_context) spans."""

    def test_construct_creates_task_span(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        pc = _new_direct_pc()
        pc.construct(trajectory=[], intent="find the link", meta_data={})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        task_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"
        ]
        assert len(task_spans) >= 1

    def test_task_span_attributes(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        pc = _new_cot_pc()
        pc.construct(trajectory=[], intent="do the thing", meta_data={})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert span_attr(task, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(task, "gen_ai.operation.name") == "run_task"
        assert span_attr(task, "webarena.task.name") == "build_prompt_context"
        assert "run_task build_prompt_context" in task.name

    def test_task_span_trajectory_length(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        pc = _new_direct_pc()
        trajectory = [
            {"observation": {"text": "hello"}, "info": {}},
            {"observation": {}, "info": {}},
        ]
        pc.construct(trajectory=trajectory, intent="intent", meta_data={})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert span_attr(task, WEBARENA_MEMORY_TRAJECTORY_LENGTH) == 2

    def test_task_span_prompt_messages_count(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        pc = _new_direct_pc()
        result = pc.construct(trajectory=[], intent="intent", meta_data={})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert span_attr(task, "webarena.prompt.messages_count") == len(result)

    def test_task_span_error_path(self, span_exporter):
        """When construct() raises, the TASK span should record the error.

        Tested by invoking the wrapper directly.
        """
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            PromptConstructWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = PromptConstructWrapper(tracer)
        pc = _new_direct_pc()

        def _failing(*args, **kwargs):
            raise RuntimeError("prompt template error")

        with pytest.raises(RuntimeError, match="prompt template error"):
            wrapper(_failing, pc, ([], "intent", {}), {})

        spans = local_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert task.status.status_code == StatusCode.ERROR

    def test_task_span_with_content_capture(
        self, instrument_with_content, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)

        pc = _new_direct_pc()
        pc.construct(
            trajectory=[],
            intent="buy the item",
            meta_data={"action_history": ["goto [url]"]},
        )

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        input_val = span_attr(task, "input.value")
        assert input_val is not None
        assert "buy the item" in input_val
        output_val = span_attr(task, "output.value")
        assert output_val is not None
        assert span_attr(task, "input.mime_type") == "application/json"
        assert span_attr(task, "output.mime_type") == "application/json"

    def test_task_span_obs_text_length(self, instrument, span_exporter):
        env = _new_env()
        env.reset(options=None)

        pc = _new_direct_pc()
        pc.obs_modality = "text"
        trajectory = [
            {"observation": {"text": "A" * 500}, "info": {}},
        ]
        pc.construct(trajectory=trajectory, intent="intent", meta_data={})

        state.end_task_spans()

        spans = span_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert span_attr(task, "webarena.memory.obs_text_length") == 500


# ===================================================================
# ConstructAgentWrapper  ->  AGENT(create_agent) spans
# ===================================================================


class TestConstructAgentWrapper:
    """Tests for construct_agent -> AGENT(create_agent) spans."""

    def test_creates_agent_span(self, instrument, span_exporter):
        ns = make_ns_args()
        _construct_agent(ns)

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "create_agent" in s.name
        ]
        assert len(agent_spans) == 1

    def test_create_agent_span_attributes(self, instrument, span_exporter):
        ns = make_ns_args(
            agent_type="prompt",
            provider="openai",
            model="gpt-4-turbo",
            action_set_tag="id_accessibility_tree",
            observation_type="accessibility_tree",
        )
        _construct_agent(ns)

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "create_agent" in s.name
        ][0]

        assert span_attr(agent_span, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(agent_span, "gen_ai.operation.name") == "create_agent"
        assert "prompt" in span_attr(agent_span, "gen_ai.agent.name")
        assert span_attr(agent_span, "gen_ai.provider.name") == "openai"
        assert span_attr(agent_span, "gen_ai.request.model") == "gpt-4-turbo"
        assert (
            span_attr(agent_span, WEBARENA_ACTION_SET_TAG)
            == "id_accessibility_tree"
        )
        assert (
            span_attr(agent_span, WEBARENA_OBSERVATION_TYPE)
            == "accessibility_tree"
        )
        assert "create_agent webarena" in agent_span.name

    def test_create_agent_has_agent_id(self, instrument, span_exporter):
        ns = make_ns_args()
        _construct_agent(ns)

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "create_agent" in s.name
        ][0]
        agent_id = span_attr(agent_span, "gen_ai.agent.id")
        assert agent_id is not None
        assert len(agent_id) == 16

    def test_create_agent_has_description(self, instrument, span_exporter):
        ns = make_ns_args(provider="huggingface", model="llama-2")
        _construct_agent(ns)

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "create_agent" in s.name
        ][0]
        desc = span_attr(agent_span, "gen_ai.agent.description")
        assert "huggingface" in desc
        assert "llama-2" in desc

    def test_create_agent_error_path(self, instrument, span_exporter):
        agent_mod = sys.modules["agent.agent"]
        original = agent_mod.construct_agent.__wrapped__

        def fail_construct(args):
            raise TypeError("bad config")

        agent_mod.construct_agent.__wrapped__ = fail_construct
        try:
            with pytest.raises(TypeError, match="bad config"):
                _construct_agent(make_ns_args())
        finally:
            agent_mod.construct_agent.__wrapped__ = original

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "create_agent" in s.name
        ][0]
        assert agent_span.status.status_code == StatusCode.ERROR

    def test_create_agent_with_content_capture(
        self, instrument_with_content, span_exporter
    ):
        ns = make_ns_args()
        _construct_agent(ns)

        spans = span_exporter.get_finished_spans()
        agent_span = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "create_agent" in s.name
        ][0]
        tool_defs = span_attr(agent_span, "gen_ai.tool.definitions")
        assert tool_defs is not None
        assert "click" in tool_defs


# ===================================================================
# HuggingFaceCompletionWrapper  ->  LLM spans
# ===================================================================


class TestHuggingFaceCompletionWrapper:
    """Tests for generate_from_huggingface_completion -> LLM spans."""

    def test_creates_llm_span(self, instrument, span_exporter):
        result = _generate_hf(
            "What is the price?",
            "http://hf-endpoint:8080",
            0.5,
            0.9,
            128,
        )
        assert result == "Generated text response"

        spans = span_exporter.get_finished_spans()
        llm_spans = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"
        ]
        assert len(llm_spans) == 1

    def test_llm_span_attributes(self, instrument, span_exporter):
        _generate_hf(
            "prompt",
            "http://my-model:8080",
            0.7,
            0.95,
            256,
            ["\n", "```"],
        )

        spans = span_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert span_attr(llm, GEN_AI_FRAMEWORK) == FRAMEWORK_NAME
        assert span_attr(llm, "gen_ai.operation.name") == "text_completion"
        assert span_attr(llm, "gen_ai.provider.name") == "huggingface"
        assert span_attr(llm, "gen_ai.request.model") == "http://my-model:8080"
        assert (
            span_attr(llm, "gen_ai.response.model") == "http://my-model:8080"
        )
        assert span_attr(llm, "gen_ai.request.temperature") == 0.7
        assert span_attr(llm, "gen_ai.request.top_p") == 0.95
        assert span_attr(llm, "gen_ai.request.max_tokens") == 256
        stop_seqs = span_attr(llm, "gen_ai.request.stop_sequences")
        assert stop_seqs is not None
        assert "\n" in stop_seqs
        assert "text_completion" in llm.name
        assert span_attr(llm, "gen_ai.output.type") == "text"

    def test_llm_span_with_kwargs(self, instrument, span_exporter):
        _generate_hf("hello", "http://ep", 0.0, 1.0, 50)

        spans = span_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert span_attr(llm, "gen_ai.request.temperature") == 0.0

    def test_llm_span_error_path(self, instrument, span_exporter):
        hf_mod = sys.modules["llms.providers.hf_utils"]
        original = hf_mod.generate_from_huggingface_completion.__wrapped__

        def fail_hf(*args, **kwargs):
            raise ConnectionError("HF endpoint unreachable")

        hf_mod.generate_from_huggingface_completion.__wrapped__ = fail_hf
        try:
            with pytest.raises(
                ConnectionError, match="HF endpoint unreachable"
            ):
                _generate_hf("test")
        finally:
            hf_mod.generate_from_huggingface_completion.__wrapped__ = original

        spans = span_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert llm.status.status_code == StatusCode.ERROR

    def test_llm_span_with_content_capture(
        self, instrument_with_content, span_exporter
    ):
        _generate_hf(
            "What is the capital of France?",
            "http://hf:8080",
            0.0,
            1.0,
            64,
        )

        spans = span_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        input_val = span_attr(llm, "input.value")
        assert input_val is not None
        assert "capital of France" in input_val
        output_val = span_attr(llm, "output.value")
        assert output_val is not None
        assert "Generated text response" in output_val

    def test_llm_span_no_content_without_capture(
        self, instrument, span_exporter
    ):
        os.environ.pop(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
        )
        _generate_hf("secret prompt", "http://hf:8080", 0.0, 1.0, 64)

        spans = span_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert span_attr(llm, "input.value") is None
        assert span_attr(llm, "output.value") is None


# ===================================================================
# Full workflow (end-to-end parent-child hierarchy)
# ===================================================================


class TestFullWorkflow:
    """End-to-end test verifying the complete span hierarchy for a single task."""

    def test_complete_hierarchy(self, instrument, span_exporter, tmp_path):
        config = {
            "task_id": 100,
            "intent": "Buy the item",
            "sites": ["shopping"],
        }
        cfg_path = tmp_path / "task.json"
        cfg_path.write_text(json.dumps(config))

        env = _new_env()
        env.reset(options={"config_file": str(cfg_path)})

        # Round 1
        agent = _new_agent()
        pc = _new_direct_pc()
        pc.construct(trajectory=[], intent="Buy the item", meta_data={})
        agent.next_action("obs1", "Buy the item", {"action_history": []})
        env.step(make_action(ActionTypes.CLICK, element_id="3"))

        # Round 2
        pc.construct(
            trajectory=[{"observation": {"text": "page"}, "info": {}}],
            intent="Buy the item",
            meta_data={},
        )
        agent.next_action(
            "obs2", "Buy the item", {"action_history": ["click [3]"]}
        )
        env.step(make_action(ActionTypes.TYPE, element_id="7", text="42"))

        env.close()

        spans = span_exporter.get_finished_spans()

        entries = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ]
        chains = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ]
        steps = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"]
        agents = [
            s
            for s in spans
            if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
            and "invoke_agent" in s.name
        ]
        tools = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"]
        tasks = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"]

        assert len(entries) == 1
        assert len(chains) == 1
        assert len(steps) == 2
        assert len(agents) == 2
        assert len(tools) == 2
        assert len(tasks) >= 2

        # Hierarchy: CHAIN -> ENTRY
        assert chains[0].parent.span_id == entries[0].context.span_id

        # Hierarchy: STEP -> CHAIN
        for step in steps:
            assert step.parent.span_id == chains[0].context.span_id

        # Hierarchy: AGENT -> STEP
        for ag in agents:
            assert ag.parent is not None
            parent_step = [
                s for s in steps if s.context.span_id == ag.parent.span_id
            ]
            assert len(parent_step) == 1

        # Hierarchy: TOOL -> STEP
        for tool in tools:
            assert tool.parent is not None
            parent_step = [
                s for s in steps if s.context.span_id == tool.parent.span_id
            ]
            assert len(parent_step) == 1

        # Summary attributes on CHAIN
        chain = chains[0]
        assert span_attr(chain, WEBARENA_STEP_COUNT) == 2
        assert span_attr(chain, WEBARENA_TOOL_COUNT) == 2
        assert span_attr(chain, WEBARENA_PARSING_FAILURE_COUNT) == 0

    def test_multi_task_workflow(self, instrument, span_exporter):
        env = _new_env()

        # Task 1
        env.reset(options=None)
        agent = _new_agent()
        agent.next_action("obs", "task 1", {"action_history": []})
        env.step(make_action())

        # Task 2 (reset implicitly closes task 1)
        env.reset(options=None)
        agent.next_action("obs", "task 2", {"action_history": []})
        env.step(make_action())

        env.close()

        spans = span_exporter.get_finished_spans()
        entries = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
        ]
        chains = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ]
        assert len(entries) == 2
        assert len(chains) == 2

    def test_chain_output_value_with_content_capture(
        self, instrument_with_content, span_exporter
    ):
        env = _new_env()
        env.reset(options=None)

        agent = _new_agent()
        agent.next_action("obs", "intent", {"action_history": []})
        env.step(make_action())

        env.close()

        spans = span_exporter.get_finished_spans()
        chain = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
        ][0]
        output_val = span_attr(chain, "output.value")
        assert output_val is not None
        assert "1 steps" in output_val
        assert "1 tool calls" in output_val


# ===================================================================
# Additional coverage: edge cases, exception handlers, boundary paths
# ===================================================================


class TestReadConfigFile:
    """Tests for _read_config_file edge cases in _wrappers.py."""

    def test_config_file_invalid_path(self, span_exporter):
        """Non-existent config file should not crash; returns None."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _read_config_file,
        )

        result = _read_config_file({"config_file": "/nonexistent/path.json"})
        assert result is None

    def test_config_file_non_dict_non_list_data(self, span_exporter, tmp_path):
        """Config file with string data should return None."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _read_config_file,
        )

        cfg = tmp_path / "string.json"
        cfg.write_text('"just a string"')
        result = _read_config_file({"config_file": str(cfg)})
        assert result is None

    def test_config_file_empty_list(self, span_exporter, tmp_path):
        """Config file with empty list should return None."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _read_config_file,
        )

        cfg = tmp_path / "empty_list.json"
        cfg.write_text("[]")
        result = _read_config_file({"config_file": str(cfg)})
        assert result is None

    def test_config_file_no_config_file_key(self, span_exporter):
        """Options dict without config_file key."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _read_config_file,
        )

        result = _read_config_file({"other_key": "value"})
        assert result is None

    def test_config_file_none_options(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _read_config_file,
        )

        assert _read_config_file(None) is None
        assert _read_config_file({}) is None


class TestJsonDumps:
    """Tests for _json_dumps in _wrappers.py."""

    def test_json_dumps_normal(self):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _json_dumps,
        )

        result = _json_dumps({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_json_dumps_with_non_serializable_uses_default_str(self):
        import datetime

        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _json_dumps,
        )

        result = _json_dumps({"ts": datetime.datetime(2024, 1, 1)})
        assert "2024" in result


class TestSetCommonAttrs:
    """Tests for _set_common_attrs."""

    def test_sets_span_kind_and_framework(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _set_common_attrs,
        )
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("test")
        _set_common_attrs(span, "TOOL")
        span.end()
        assert span.attributes.get(GEN_AI_SPAN_KIND) == "TOOL"
        assert span.attributes.get(GEN_AI_FRAMEWORK) == FRAMEWORK_NAME


class TestSetAgentContentAttrs:
    """Tests for _set_agent_content_attrs edge cases."""

    def test_agent_content_attrs_no_prompt_constructor(self, span_exporter):
        """When instance has no prompt_constructor, should not crash."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _set_agent_content_attrs,
        )
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("test")
        instance = types.SimpleNamespace()  # no prompt_constructor
        _set_agent_content_attrs(
            span, instance, "intent text", {"action_history": ["click [1]"]}
        )
        span.end()
        # Should still set input messages and tool definitions
        assert "intent text" in span.attributes.get(
            "gen_ai.input.messages", ""
        )
        assert "click" in span.attributes.get("gen_ai.tool.definitions", "")

    def test_agent_content_attrs_with_intent_and_history(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _set_agent_content_attrs,
        )
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("test")
        instance = types.SimpleNamespace(
            prompt_constructor=types.SimpleNamespace(
                instruction={"intro": "You help with browsing."}
            )
        )
        _set_agent_content_attrs(
            span, instance, "find product", {"action_history": ["goto [url]"]}
        )
        span.end()
        sys_instr = span.attributes.get("gen_ai.system_instructions", "")
        assert "browsing" in sys_instr
        input_msgs = span.attributes.get("gen_ai.input.messages", "")
        assert "find product" in input_msgs
        assert "goto [url]" in input_msgs

    def test_agent_content_attrs_no_intent(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _set_agent_content_attrs,
        )
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("test")
        instance = types.SimpleNamespace()
        _set_agent_content_attrs(span, instance, None, {})
        span.end()
        # No intent -> no gen_ai.input.messages
        assert span.attributes.get("gen_ai.input.messages") is None


class TestEnvResetWrapperEdgeCases:
    """Edge case tests for EnvResetWrapper."""

    def test_reset_result_string_obs(self, span_exporter):
        """When reset returns a string observation instead of dict."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvResetWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        try:
            local_exporter = InMemorySpanExporter()
            tracer = _make_tracer(local_exporter)
            wrapper = EnvResetWrapper(tracer)

            def _string_reset(*args, **kwargs):
                return ("String observation text", {})

            env = _new_env()
            wrapper(_string_reset, env, (), {"options": None})

            state.end_task_spans()

            spans = local_exporter.get_finished_spans()
            entry = [
                s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
            ][0]
            output_msgs = span_attr(entry, "gen_ai.output.messages")
            assert output_msgs is not None
            assert "String observation text" in output_msgs
        finally:
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
            )

    def test_reset_result_non_tuple(self, span_exporter):
        """When reset returns a non-tuple, output messages should be skipped."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvResetWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        try:
            local_exporter = InMemorySpanExporter()
            tracer = _make_tracer(local_exporter)
            wrapper = EnvResetWrapper(tracer)

            def _non_tuple_reset(*args, **kwargs):
                return "just a string"

            env = _new_env()
            wrapper(_non_tuple_reset, env, (), {"options": None})

            state.end_task_spans()

            spans = local_exporter.get_finished_spans()
            entry = [
                s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "ENTRY"
            ][0]
            # No output messages since result is not a tuple
            assert span_attr(entry, "gen_ai.output.messages") is None
        finally:
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
            )


class TestEnvStepWrapperEdgeCases:
    """Edge case tests for EnvStepWrapper."""

    def test_tool_span_url_after_recorded(self, span_exporter):
        """After step, if page.url changed, url_after should be recorded."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvStepWrapper(tracer)

        env = _new_env()
        env.page = types.SimpleNamespace(url="http://before.com")

        def _step(*args, **kwargs):
            env.page.url = "http://after.com"
            return ("obs", True, False, False, {"fail_error": ""})

        wrapper(_step, env, (make_action(),), {})

        spans = local_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, WEBARENA_PAGE_URL_AFTER) == "http://after.com"
        assert span_attr(tool, WEBARENA_PAGE_URL_BEFORE) == "http://before.com"
        assert span_attr(tool, "webarena.tool.success") is True

    def test_tool_span_short_result_tuple(self, span_exporter):
        """When step returns tuple with fewer than 5 elements."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvStepWrapper(tracer)

        env = _new_env()

        def _short_step(*args, **kwargs):
            return ("obs", True)  # only 2 elements

        wrapper(_short_step, env, (make_action(),), {})

        spans = local_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        # success defaults to False since len(result) < 5
        assert span_attr(tool, "webarena.tool.success") is False

    def test_tool_span_no_page(self, span_exporter):
        """When env has no page attribute."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvStepWrapper(tracer)

        env = types.SimpleNamespace()  # no page attribute

        def _step(*args, **kwargs):
            return ("obs", False, False, False, {"fail_error": ""})

        wrapper(_step, env, (make_action(),), {})

        spans = local_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, WEBARENA_PAGE_URL_BEFORE) is None

    def test_tool_span_action_no_element_id(self, span_exporter):
        """When action dict has no element_id or empty element_id."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            EnvStepWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = EnvStepWrapper(tracer)

        env = _new_env()
        action = {"action_type": ActionTypes.SCROLL}  # no element_id

        def _step(*args, **kwargs):
            return ("obs", False, False, False, {"fail_error": ""})

        wrapper(_step, env, (action,), {})

        spans = local_exporter.get_finished_spans()
        tool = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TOOL"][
            0
        ]
        assert span_attr(tool, WEBARENA_BROWSER_ELEMENT_ID) is None


class TestNextActionWrapperEdgeCases:
    """Edge case tests for NextActionWrapper."""

    def test_agent_no_lm_config(self, span_exporter):
        """When agent has no lm_config, should still emit span."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            NextActionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = NextActionWrapper(tracer)

        class CustomAgent:
            def __init__(self):
                self.prompt_constructor = types.SimpleNamespace(
                    instruction_path=None
                )
                self.lm_config = None

        agent = CustomAgent()

        def _action(*args, **kwargs):
            return {"action_type": ActionTypes.CLICK, "element_id": "1"}

        wrapper(_action, agent, ("obs",), {})

        spans = local_exporter.get_finished_spans()
        agent_span = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
        ][0]
        assert span_attr(agent_span, "gen_ai.operation.name") == "invoke_agent"

    def test_agent_no_prompt_constructor(self, span_exporter):
        """When agent has no prompt_constructor, should handle gracefully."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            NextActionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = NextActionWrapper(tracer)

        class BareAgent:
            def __init__(self):
                self.lm_config = None

        agent = BareAgent()

        def _action(*args, **kwargs):
            return {"action_type": ActionTypes.HOVER}

        wrapper(
            _action, agent, ("obs",), {"intent": "do thing", "meta_data": {}}
        )

        spans = local_exporter.get_finished_spans()
        agent_span = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
        ][0]
        assert "BareAgent" in agent_span.name

    def test_agent_error_propagated_to_step(self, span_exporter):
        """When next_action raises, the STEP span should also record error status."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            NextActionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)

        # Set up STEP context
        state.mark_in_task(True)

        wrapper = NextActionWrapper(tracer)
        agent = _new_agent()

        def _failing(*args, **kwargs):
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError, match="crash"):
            wrapper(_failing, agent, ("obs", "intent", {}), {})

        state.end_task_spans()

        spans = local_exporter.get_finished_spans()
        step = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "STEP"][
            0
        ]
        assert step.status.status_code == StatusCode.ERROR
        assert span_attr(step, GEN_AI_REACT_FINISH_REASON) == "RuntimeError"


class TestConstructAgentWrapperEdgeCases:
    """Edge case tests for ConstructAgentWrapper."""

    def test_construct_agent_minimal_ns_args(self, span_exporter):
        """When ns_args has minimal attributes."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            ConstructAgentWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = ConstructAgentWrapper(tracer)

        ns = types.SimpleNamespace()  # no attributes at all

        def _construct(args):
            return "agent"

        wrapper(_construct, None, (ns,), {})

        spans = local_exporter.get_finished_spans()
        agent_span = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
        ][0]
        assert span_attr(agent_span, "gen_ai.operation.name") == "create_agent"
        assert "unknown" in span_attr(agent_span, "gen_ai.agent.name")

    def test_construct_agent_via_kwargs(self, span_exporter):
        """When args is passed via kwargs instead of positional."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            ConstructAgentWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = ConstructAgentWrapper(tracer)

        ns = make_ns_args(
            agent_type="custom", provider="aws", model="claude-3"
        )

        def _construct(args):
            return "agent"

        wrapper(_construct, None, (), {"args": ns})

        spans = local_exporter.get_finished_spans()
        agent_span = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
        ][0]
        assert "custom" in span_attr(agent_span, "gen_ai.agent.name")
        assert span_attr(agent_span, "gen_ai.provider.name") == "aws"
        assert span_attr(agent_span, "gen_ai.request.model") == "claude-3"

    def test_construct_agent_error(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            ConstructAgentWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = ConstructAgentWrapper(tracer)

        ns = make_ns_args()

        def _fail(args):
            raise TypeError("bad")

        with pytest.raises(TypeError, match="bad"):
            wrapper(_fail, None, (ns,), {})

        spans = local_exporter.get_finished_spans()
        agent_span = [
            s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "AGENT"
        ][0]
        assert agent_span.status.status_code == StatusCode.ERROR


class TestHuggingFaceCompletionWrapperEdgeCases:
    """Edge case tests for HuggingFaceCompletionWrapper."""

    def test_hf_none_temperature(self, span_exporter):
        """When temperature is None, attribute should not be set."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            HuggingFaceCompletionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = HuggingFaceCompletionWrapper(tracer)

        def _gen(*args, **kwargs):
            return "output"

        wrapper(_gen, None, ("prompt", "http://endpoint"), {})

        spans = local_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert span_attr(llm, "gen_ai.request.temperature") is None
        assert span_attr(llm, "gen_ai.request.top_p") is None
        assert span_attr(llm, "gen_ai.request.max_tokens") is None

    def test_hf_empty_model_endpoint(self, span_exporter):
        """When model_endpoint is empty."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            HuggingFaceCompletionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = HuggingFaceCompletionWrapper(tracer)

        def _gen(*args, **kwargs):
            return "output"

        wrapper(_gen, None, ("prompt", ""), {})

        spans = local_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert "huggingface" in llm.name
        assert span_attr(llm, "gen_ai.request.model") is None

    def test_hf_non_string_result(self, span_exporter):
        """When generation returns non-string, output.value should not be set."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            HuggingFaceCompletionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        try:
            local_exporter = InMemorySpanExporter()
            tracer = _make_tracer(local_exporter)
            wrapper = HuggingFaceCompletionWrapper(tracer)

            def _gen(*args, **kwargs):
                return 42  # non-string

            wrapper(_gen, None, ("prompt", "http://ep", 0.5, 0.9, 128), {})

            spans = local_exporter.get_finished_spans()
            llm = [
                s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"
            ][0]
            assert span_attr(llm, "output.value") is None
        finally:
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
            )

    def test_hf_via_kwargs(self, span_exporter):
        """When called with kwargs instead of positional args."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            HuggingFaceCompletionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = HuggingFaceCompletionWrapper(tracer)

        def _gen(*args, **kwargs):
            return "generated"

        wrapper(
            _gen,
            None,
            (),
            {
                "prompt": "hello",
                "model_endpoint": "http://model",
                "temperature": 0.3,
                "top_p": 0.8,
                "max_new_tokens": 100,
                "stop_sequences": ["END"],
            },
        )

        spans = local_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert span_attr(llm, "gen_ai.request.model") == "http://model"
        assert span_attr(llm, "gen_ai.request.temperature") == 0.3
        assert span_attr(llm, "gen_ai.request.max_tokens") == 100

    def test_hf_error_path(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            HuggingFaceCompletionWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = HuggingFaceCompletionWrapper(tracer)

        def _fail(*args, **kwargs):
            raise ConnectionError("unreachable")

        with pytest.raises(ConnectionError, match="unreachable"):
            wrapper(_fail, None, ("prompt", "http://ep", 0.0, 1.0, 64), {})

        spans = local_exporter.get_finished_spans()
        llm = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "LLM"][0]
        assert llm.status.status_code == StatusCode.ERROR


class TestPromptConstructWrapperEdgeCases:
    """Edge case tests for PromptConstructWrapper."""

    def test_construct_string_result(self, span_exporter):
        """When construct returns a string instead of a list."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            PromptConstructWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = PromptConstructWrapper(tracer)

        instance = types.SimpleNamespace(obs_modality=None)

        def _construct(*args, **kwargs):
            return "raw prompt text"

        wrapper(_construct, instance, ([], "intent", {}), {})

        spans = local_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert span_attr(task, "webarena.prompt.length") == len(
            "raw prompt text"
        )

    def test_construct_with_trajectory_url(self, span_exporter):
        """When trajectory has page URL info."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            PromptConstructWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = PromptConstructWrapper(tracer)

        instance = types.SimpleNamespace(obs_modality=None)
        page = types.SimpleNamespace(url="http://example.com/page")
        trajectory = [{"observation": {}, "info": {"page": page}}]

        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        try:

            def _construct(*args, **kwargs):
                return [{"role": "user", "content": "do it"}]

            wrapper(_construct, instance, (trajectory, "intent", {}), {})

            spans = local_exporter.get_finished_spans()
            task = [
                s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"
            ][0]
            input_val = span_attr(task, "input.value")
            assert "example.com" in input_val
        finally:
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
            )

    def test_construct_error_path(self, span_exporter):
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            PromptConstructWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = PromptConstructWrapper(tracer)

        instance = types.SimpleNamespace(obs_modality=None)

        def _fail(*args, **kwargs):
            raise RuntimeError("template error")

        with pytest.raises(RuntimeError, match="template error"):
            wrapper(_fail, instance, ([], "intent", {}), {})

        spans = local_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert task.status.status_code == StatusCode.ERROR

    def test_construct_via_kwargs(self, span_exporter):
        """When construct is called with keyword arguments."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            PromptConstructWrapper,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        local_exporter = InMemorySpanExporter()
        tracer = _make_tracer(local_exporter)
        wrapper = PromptConstructWrapper(tracer)

        instance = types.SimpleNamespace(obs_modality="text")

        def _construct(*args, **kwargs):
            return [{"role": "user", "content": "hi"}]

        wrapper(
            _construct,
            instance,
            (),
            {
                "trajectory": [
                    {"observation": {"text": "A" * 100}, "info": {}}
                ],
                "intent": "test",
                "meta_data": {"action_history": ["click [1]"]},
            },
        )

        spans = local_exporter.get_finished_spans()
        task = [s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "TASK"][
            0
        ]
        assert span_attr(task, WEBARENA_MEMORY_TRAJECTORY_LENGTH) == 1
        assert span_attr(task, "webarena.memory.obs_text_length") == 100


class TestCloseTaskSpans:
    """Tests for _close_task_spans edge cases."""

    def test_close_with_content_capture(self, span_exporter):
        """When content capture is on, CHAIN should get output.value."""
        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            _close_task_spans,
            _open_task_spans,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        try:
            local_exporter = InMemorySpanExporter()
            tracer = _make_tracer(local_exporter)

            _open_task_spans(tracer, None)
            state.increment_step()
            state.increment_tool()
            state.increment_parsing_failure()
            _close_task_spans()

            spans = local_exporter.get_finished_spans()
            chain = [
                s for s in spans if span_attr(s, GEN_AI_SPAN_KIND) == "CHAIN"
            ][0]
            output = span_attr(chain, "output.value")
            assert "1 steps" in output
            assert "1 tool calls" in output
            assert "1 parsing failures" in output
        finally:
            os.environ.pop(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
            )


class TestAttrHelpersAdditional:
    """Additional tests for _attrs.py."""

    def test_truncate_content(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            truncate_content,
        )

        short = "hello"
        assert truncate_content(short) == short

        long = "x" * 10000
        result = truncate_content(long)
        assert len(result) <= 4096

    def test_action_type_name_with_raw_int(self):
        """When action_type is a raw int matching an enum value."""
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_type_name,
        )

        action = {"action_type": 0}  # ActionTypes.CLICK.value
        result = action_type_name(action)
        assert result == "CLICK"

    def test_action_arguments_empty_values_excluded(self):
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_arguments,
        )

        action = {
            "action_type": ActionTypes.CLICK,
            "element_id": "",  # empty string
            "text": None,  # None
            "url": [],  # empty list
            "direction": {},  # empty dict
            "amount": "5",  # non-empty
        }
        result = action_arguments(action)
        assert "element_id" not in result  # empty string excluded
        assert "text" not in result
        assert "url" not in result
        assert "direction" not in result
        assert result["amount"] == "5"

    def test_safe_json_dumps_fallback_on_unjsonifiable(self):
        """When json.dumps raises, safe_json_dumps falls back to str()."""
        # Create an object that cannot be JSON-serialized even with default=str
        # by monkeypatching json.dumps to raise
        import json as _json

        from opentelemetry.instrumentation.webarena.internal._attrs import (
            safe_json_dumps,
        )

        with patch.object(_json, "dumps", side_effect=OverflowError("boom")):
            result = safe_json_dumps({"key": "value"})
        # Falls back to str(value)
        assert "key" in result
        assert "value" in result

    def test_action_type_name_invalid_int_fallback(self):
        """When action_type is an int that doesn't match any enum value,
        ActionTypes(raw) raises and we fall back to str(raw)."""
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            action_type_name,
        )

        action = {"action_type": 99999}  # Not a valid ActionTypes value
        result = action_type_name(action)
        assert result == "99999"

    def test_messages_to_input_value_list_exception_fallback(self):
        """When safe_json_dumps raises on a list, fall back to str()."""
        from opentelemetry.instrumentation.webarena.internal._attrs import (
            messages_to_input_value,
        )

        msgs = [{"role": "user", "content": "hello"}]
        with patch(
            "opentelemetry.instrumentation.webarena.internal._attrs.safe_json_dumps",
            side_effect=TypeError("boom"),
        ):
            result = messages_to_input_value(msgs)
        # Falls back to truncate_content(str(messages))
        assert "hello" in result


class TestStateAdditional:
    """Additional tests for _state.py."""

    def test_set_and_get_entry(self):
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("entry")

        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        token = otel_context.attach(set_span_in_context(span))
        state.set_entry(span, token)
        assert state.get_entry_span() is span
        state.end_task_spans()
        assert state.get_entry_span() is None

    def test_set_and_get_chain(self):
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")
        span = tracer.start_span("chain")

        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        token = otel_context.attach(set_span_in_context(span))
        state.set_chain(span, token)
        assert state.get_chain_span() is span
        state.end_task_spans()
        assert state.get_chain_span() is None

    def test_end_chain_and_end_entry(self):
        from opentelemetry.sdk.trace import TracerProvider as _TP

        provider = _TP()
        tracer = provider.get_tracer("test")

        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        entry = tracer.start_span("entry")
        entry_token = otel_context.attach(set_span_in_context(entry))
        state.set_entry(entry, entry_token)

        chain = tracer.start_span("chain")
        chain_token = otel_context.attach(set_span_in_context(chain))
        state.set_chain(chain, chain_token)

        state.end_chain()
        assert state.get_chain_span() is None
        assert state.get_entry_span() is entry  # entry still alive

        state.end_entry()
        assert state.get_entry_span() is None
