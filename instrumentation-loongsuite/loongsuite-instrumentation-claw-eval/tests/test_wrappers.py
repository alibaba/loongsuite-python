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

"""Unit tests for ``opentelemetry.instrumentation.claw_eval.internal.wrappers``.

Tests cover:
- Content helper functions (_safe_json, _extract_tool_result_text, _extract_system_prompt)
- Message serialization (_block_to_part, _message_to_chat_message, etc.)
- STEP lifecycle (_end_current_step, _rotate_step)
- All wrapper classes (Entry, RunSingleTask, RunTask, ProviderChat, etc.)
- Error paths for each wrapper
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest
from conftest import (
    ContentBlock,
    DispatchEvent,
    Message,
    Prompt,
    TaskDefinition,
    ToolResultBlock,
    ToolSpec,
    ToolUse,
    Usage,
)

from opentelemetry import context as otel_context

# Import wrappers module
from opentelemetry.instrumentation.claw_eval.internal.wrappers import (
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DEFINITIONS,
    DoAutoCompactWrapper,
    EntryWrapper,
    GetGraderWrapper,
    JudgeWrapper,
    LoadPeerGraderWrapper,
    ProviderChatWrapper,
    RunSingleTaskWrapper,
    RunTaskWrapper,
    ToolDispatchWrapper,
    _agent_capture,
    _agent_tool_definitions,
    _block_to_part,
    _build_user_text_messages,
    _compact_depth,
    _current_step_span,
    _current_step_token,
    _end_current_step,
    _entry_capture,
    _extract_dispatch_attrs,
    _extract_system_prompt,
    _extract_tool_result_text,
    _get_task_prompt,
    _in_agent_run,
    _in_tool_dispatch,
    _infer_finish_reason,
    _install_provider_chat_capture_shim,
    _maybe_suppress_llm_sdk,
    _message_to_chat_message,
    _populate_agent_span,
    _populate_entry_span,
    _rotate_step,
    _safe_json,
    _serialize_input_messages,
    _serialize_output_message,
    _serialize_system_instructions,
    _serialize_tool_definitions,
    _step_counter,
    _wrap_grader_eval_methods,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import StatusCode

# ---------------------------------------------------------------------------
# Helper to get a test tracer with in-memory exporter
# ---------------------------------------------------------------------------


def _make_tracer():
    """Return (tracer, exporter) for use in wrapper tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-claw-eval")
    return tracer, exporter, provider


# ===================================================================
# Content helpers
# ===================================================================


class TestSafeJson:
    """Tests for _safe_json."""

    def test_simple_dict(self):
        result = _safe_json({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_list(self):
        result = _safe_json([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_string(self):
        result = _safe_json("hello")
        assert json.loads(result) == "hello"

    def test_none(self):
        result = _safe_json(None)
        assert json.loads(result) is None

    def test_nested(self):
        obj = {"a": {"b": [1, 2]}, "c": True}
        result = _safe_json(obj)
        assert json.loads(result) == obj

    def test_non_serializable_uses_default_str(self):
        """Non-JSON-serializable objects fall back to str()."""
        obj = {"key": object()}
        result = _safe_json(obj)
        # Should not raise, result should be a string
        assert isinstance(result, str)

    def test_unicode(self):
        result = _safe_json({"text": "hello world"})
        assert "hello world" in result

    def test_ensure_ascii_false(self):
        result = _safe_json({"text": "hello world"})
        # With ensure_ascii=False, non-ASCII characters appear as-is
        assert isinstance(result, str)

    def test_empty_dict(self):
        result = _safe_json({})
        assert json.loads(result) == {}


class TestExtractToolResultText:
    """Tests for _extract_tool_result_text."""

    def test_with_text_blocks(self):
        result = ToolResultBlock(
            content=[
                ContentBlock(type="text", text="line 1"),
                ContentBlock(type="text", text="line 2"),
            ]
        )
        assert _extract_tool_result_text(result) == "line 1\nline 2"

    def test_empty_content(self):
        result = ToolResultBlock(content=[])
        assert _extract_tool_result_text(result) == ""

    def test_none_content(self):
        result = ToolResultBlock(content=None)
        assert _extract_tool_result_text(result) == ""

    def test_no_content_attr(self):
        result = MagicMock(spec=[])
        assert _extract_tool_result_text(result) == ""

    def test_mixed_blocks_only_text(self):
        result = ToolResultBlock(
            content=[
                ContentBlock(type="text", text="only text"),
                ContentBlock(type="image"),  # no text attr
            ]
        )
        assert _extract_tool_result_text(result) == "only text"

    def test_block_with_none_text(self):
        result = ToolResultBlock(
            content=[ContentBlock(type="text", text=None)]
        )
        assert _extract_tool_result_text(result) == ""


class TestExtractSystemPrompt:
    """Tests for _extract_system_prompt."""

    def test_system_message_present(self):
        messages = [
            Message(
                role="system",
                content=[ContentBlock(type="text", text="Be helpful.")],
            ),
            Message(
                role="user",
                content=[ContentBlock(type="text", text="Hi")],
            ),
        ]
        assert _extract_system_prompt(messages) == "Be helpful."

    def test_no_system_message(self):
        messages = [
            Message(
                role="user",
                content=[ContentBlock(type="text", text="Hi")],
            ),
        ]
        assert _extract_system_prompt(messages) == ""

    def test_empty_messages(self):
        assert _extract_system_prompt([]) == ""

    def test_none_messages(self):
        assert _extract_system_prompt(None) == ""

    def test_system_with_no_text_block(self):
        messages = [
            Message(
                role="system",
                content=[ContentBlock(type="image")],
            ),
        ]
        assert _extract_system_prompt(messages) == ""

    def test_system_with_none_content(self):
        messages = [Message(role="system", content=None)]
        assert _extract_system_prompt(messages) == ""

    def test_system_with_empty_text(self):
        messages = [
            Message(
                role="system",
                content=[ContentBlock(type="text", text="")],
            ),
        ]
        assert _extract_system_prompt(messages) == ""


# ===================================================================
# Message serialization
# ===================================================================


class TestBlockToPart:
    """Tests for _block_to_part."""

    def test_text_block(self):
        block = ContentBlock(type="text", text="hello")
        part = _block_to_part(block)
        assert part == {"type": "text", "content": "hello"}

    def test_text_block_none_text(self):
        block = ContentBlock(type="text", text=None)
        part = _block_to_part(block)
        assert part == {"type": "text", "content": ""}

    def test_tool_use_block(self):
        block = ContentBlock(
            type="tool_use",
            id="tu_001",
            name="bash",
            input={"cmd": "ls"},
        )
        part = _block_to_part(block)
        assert part == {
            "type": "tool_call",
            "id": "tu_001",
            "name": "bash",
            "arguments": {"cmd": "ls"},
        }

    def test_tool_use_block_none_fields(self):
        block = ContentBlock(type="tool_use")
        part = _block_to_part(block)
        assert part["type"] == "tool_call"
        assert part["id"] == ""
        assert part["name"] == ""
        assert part["arguments"] is None

    def test_tool_result_block(self):
        inner = ContentBlock(type="text", text="output line")
        block = ContentBlock(
            type="tool_result",
            tool_use_id="tu_001",
            content=[inner],
        )
        part = _block_to_part(block)
        assert part == {
            "type": "tool_call_response",
            "id": "tu_001",
            "response": "output line",
        }

    def test_tool_result_block_multiple_inner(self):
        block = ContentBlock(
            type="tool_result",
            tool_use_id="tu_002",
            content=[
                ContentBlock(type="text", text="line1"),
                ContentBlock(type="text", text="line2"),
            ],
        )
        part = _block_to_part(block)
        assert part["response"] == "line1\nline2"

    def test_tool_result_block_none_content(self):
        block = ContentBlock(
            type="tool_result", tool_use_id="tu_003", content=None
        )
        part = _block_to_part(block)
        assert part["response"] == ""

    def test_image_block(self):
        block = ContentBlock(type="image")
        part = _block_to_part(block)
        assert part == {"type": "image"}

    def test_audio_block(self):
        block = ContentBlock(type="audio")
        part = _block_to_part(block)
        assert part == {"type": "audio"}

    def test_video_block(self):
        block = ContentBlock(type="video")
        part = _block_to_part(block)
        assert part == {"type": "video"}

    def test_unknown_block_type(self):
        block = ContentBlock(type="custom_type")
        part = _block_to_part(block)
        assert part == {"type": "custom_type"}

    def test_empty_type(self):
        block = ContentBlock(type="")
        part = _block_to_part(block)
        assert part == {"type": "unknown"}

    def test_no_type_attribute(self):
        block = MagicMock(spec=[])
        part = _block_to_part(block)
        assert part == {"type": "unknown"}


class TestMessageToChatMessage:
    """Tests for _message_to_chat_message."""

    def test_user_message(self):
        msg = Message(
            role="user",
            content=[ContentBlock(type="text", text="Hello")],
        )
        result = _message_to_chat_message(msg)
        assert result == {
            "role": "user",
            "parts": [{"type": "text", "content": "Hello"}],
        }

    def test_assistant_message_with_tool_use(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type="text", text="Let me run a command."),
                ContentBlock(
                    type="tool_use",
                    id="tu_1",
                    name="bash",
                    input={"cmd": "ls"},
                ),
            ],
        )
        result = _message_to_chat_message(msg)
        assert result["role"] == "assistant"
        assert len(result["parts"]) == 2
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][1]["type"] == "tool_call"

    def test_empty_content(self):
        msg = Message(role="user", content=[])
        result = _message_to_chat_message(msg)
        assert result == {"role": "user", "parts": []}

    def test_none_content(self):
        msg = Message(role="user", content=None)
        result = _message_to_chat_message(msg)
        assert result == {"role": "user", "parts": []}

    def test_no_role_attribute(self):
        msg = MagicMock(spec=[])
        result = _message_to_chat_message(msg)
        assert result["role"] == "unknown"


class TestInferFinishReason:
    """Tests for _infer_finish_reason."""

    def test_with_tool_use(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock(type="tool_use", name="bash")],
        )
        assert _infer_finish_reason(msg) == "tool_call"

    def test_text_only_stop(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock(type="text", text="Done")],
        )
        assert _infer_finish_reason(msg) == "stop"

    def test_empty_content(self):
        msg = Message(role="assistant", content=[])
        assert _infer_finish_reason(msg) == "stop"

    def test_none_content(self):
        msg = Message(role="assistant", content=None)
        assert _infer_finish_reason(msg) == "stop"

    def test_mixed_text_and_tool_use(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type="text", text="I will use a tool"),
                ContentBlock(type="tool_use", name="bash"),
            ],
        )
        assert _infer_finish_reason(msg) == "tool_call"


class TestSerializeInputMessages:
    """Tests for _serialize_input_messages."""

    def test_basic(self):
        messages = [
            Message(
                role="user", content=[ContentBlock(type="text", text="Hi")]
            ),
        ]
        result = _serialize_input_messages(messages)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"
        assert parsed[0]["parts"][0]["content"] == "Hi"

    def test_multiple_messages(self):
        messages = [
            Message(
                role="user", content=[ContentBlock(type="text", text="q1")]
            ),
            Message(
                role="assistant",
                content=[ContentBlock(type="text", text="a1")],
            ),
        ]
        result = _serialize_input_messages(messages)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_empty_messages(self):
        result = _serialize_input_messages([])
        assert json.loads(result) == []

    def test_none_messages(self):
        result = _serialize_input_messages(None)
        assert json.loads(result) == []


class TestSerializeOutputMessage:
    """Tests for _serialize_output_message."""

    def test_basic_text_response(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock(type="text", text="Hello!")],
        )
        result = _serialize_output_message(msg)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "assistant"
        assert parsed[0]["finish_reason"] == "stop"
        assert parsed[0]["parts"][0]["content"] == "Hello!"

    def test_tool_call_response(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock(type="tool_use", name="bash", id="tu_1")],
        )
        result = _serialize_output_message(msg)
        parsed = json.loads(result)
        assert parsed[0]["finish_reason"] == "tool_call"

    def test_none_message(self):
        result = _serialize_output_message(None)
        assert result == ""

    def test_none_role_defaults(self):
        msg = MagicMock()
        msg.role = None
        msg.content = [ContentBlock(type="text", text="hi")]
        result = _serialize_output_message(msg)
        parsed = json.loads(result)
        assert parsed[0]["role"] == "assistant"


class TestSerializeSystemInstructions:
    """Tests for _serialize_system_instructions."""

    def test_basic(self):
        result = _serialize_system_instructions("Be helpful.")
        parsed = json.loads(result)
        assert parsed == [{"type": "text", "content": "Be helpful."}]

    def test_empty_string(self):
        result = _serialize_system_instructions("")
        assert result == ""

    def test_none_like(self):
        # Empty string is falsy
        result = _serialize_system_instructions("")
        assert result == ""


class TestBuildUserTextMessages:
    """Tests for _build_user_text_messages."""

    def test_basic(self):
        result = _build_user_text_messages("What is 2+2?")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"
        assert parsed[0]["parts"][0]["content"] == "What is 2+2?"

    def test_empty_string(self):
        result = _build_user_text_messages("")
        assert result == ""


class TestSerializeToolDefinitions:
    """Tests for _serialize_tool_definitions."""

    def test_basic_tools(self):
        tools = [
            ToolSpec(name="bash", description="Run bash commands"),
            ToolSpec(name="python", description="Run Python code"),
        ]
        result = _serialize_tool_definitions(tools)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["type"] == "function"
        assert parsed[0]["name"] == "bash"
        assert parsed[0]["description"] == "Run bash commands"

    def test_tool_with_input_schema(self):
        tools = [
            ToolSpec(
                name="bash",
                description="Run bash",
                input_schema={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                },
            ),
        ]
        result = _serialize_tool_definitions(tools)
        parsed = json.loads(result)
        assert "parameters" in parsed[0]
        assert parsed[0]["parameters"]["type"] == "object"

    def test_tool_with_parameters_fallback(self):
        """If input_schema is None, falls back to parameters."""
        tools = [
            ToolSpec(
                name="bash",
                description="Run bash",
                input_schema=None,
                parameters={"type": "object"},
            ),
        ]
        result = _serialize_tool_definitions(tools)
        parsed = json.loads(result)
        assert parsed[0]["parameters"] == {"type": "object"}

    def test_tool_without_name_skipped(self):
        tools = [ToolSpec(name="", description="no name")]
        result = _serialize_tool_definitions(tools)
        assert result == ""

    def test_tool_without_description(self):
        tools = [ToolSpec(name="bash", description=None)]
        result = _serialize_tool_definitions(tools)
        parsed = json.loads(result)
        assert "description" not in parsed[0]

    def test_empty_tools(self):
        assert _serialize_tool_definitions([]) == ""

    def test_none_tools(self):
        assert _serialize_tool_definitions(None) == ""


# ===================================================================
# STEP lifecycle
# ===================================================================


class TestEndCurrentStep:
    """Tests for _end_current_step."""

    def test_no_active_step(self):
        """Should be a safe no-op when no step span exists."""
        _end_current_step()

    def test_ends_step_and_detaches(self):
        tracer, exporter, provider = _make_tracer()
        span = tracer.start_span("test step")
        ctx = otel_context.set_value("test", True)
        token = otel_context.attach(ctx)

        _current_step_span.set(span)
        _current_step_token.set(token)

        try:
            _end_current_step()
            assert _current_step_span.get(None) is None
            assert _current_step_token.get(None) is None
        finally:
            _current_step_span.set(None)
            _current_step_token.set(None)
            provider.shutdown()


class TestRotateStep:
    """Tests for _rotate_step."""

    def test_rotate_increments_counter(self):
        tracer, exporter, provider = _make_tracer()
        tok_cnt = _step_counter.set(0)
        _current_step_span.set(None)
        _current_step_token.set(None)

        try:
            _rotate_step(tracer)
            assert _step_counter.get(0) == 1

            _rotate_step(tracer)
            assert _step_counter.get(0) == 2

            # Clean up the active step
            _end_current_step()
        finally:
            _step_counter.reset(tok_cnt)
            _current_step_span.set(None)
            _current_step_token.set(None)
            provider.shutdown()

    def test_rotate_creates_step_span(self):
        tracer, exporter, provider = _make_tracer()
        tok_cnt = _step_counter.set(0)
        _current_step_span.set(None)
        _current_step_token.set(None)

        try:
            _rotate_step(tracer)
            step_span = _current_step_span.get(None)
            assert step_span is not None
            _end_current_step()

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].name == "react step"
        finally:
            _step_counter.reset(tok_cnt)
            _current_step_span.set(None)
            _current_step_token.set(None)
            provider.shutdown()


# ===================================================================
# EntryWrapper
# ===================================================================


class TestEntryWrapper:
    """Tests for EntryWrapper."""

    def test_creates_entry_span(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = EntryWrapper(tracer, "run")

        def fake_wrapped(*args, **kwargs):
            return "result"

        result = wrapper(fake_wrapped, None, (), {})
        assert result == "result"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "claw-eval run"
        assert span.attributes[GEN_AI_SPAN_KIND] == "ENTRY"
        assert span.attributes[GEN_AI_FRAMEWORK] == "claw-eval"
        assert span.attributes["claw_eval.command"] == "run"
        provider.shutdown()

    def test_batch_command(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = EntryWrapper(tracer, "batch")

        def fake_wrapped(*args, **kwargs):
            return "batch_result"

        wrapper(fake_wrapped, None, (), {})
        spans = exporter.get_finished_spans()
        assert spans[0].name == "claw-eval batch"
        assert spans[0].attributes["claw_eval.command"] == "batch"
        provider.shutdown()

    def test_error_records_exception(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = EntryWrapper(tracer, "run")

        def fake_wrapped(*args, **kwargs):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            wrapper(fake_wrapped, None, (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        events = spans[0].events
        assert any(e.name == "exception" for e in events)
        provider.shutdown()

    def test_entry_capture_propagated(self):
        """EntryWrapper should set _entry_capture so nested AGENT spans can push to it."""
        tracer, exporter, provider = _make_tracer()
        wrapper = EntryWrapper(tracer, "run")
        captured_list = []

        def fake_wrapped(*args, **kwargs):
            caps = _entry_capture.get()
            captured_list.append(caps)
            return "ok"

        wrapper(fake_wrapped, None, (), {})
        # The list should have been set inside the wrapped call
        assert len(captured_list) == 1
        assert isinstance(captured_list[0], list)
        provider.shutdown()


# ===================================================================
# RunSingleTaskWrapper
# ===================================================================


class TestRunSingleTaskWrapper:
    """Tests for RunSingleTaskWrapper."""

    def test_creates_entry_span_with_task_dir(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunSingleTaskWrapper(tracer)

        def fake_wrapped(task_dir, *args, **kwargs):
            return {"task_id": "T001", "score": 1.0}

        result = wrapper(fake_wrapped, None, ("/path/to/task",), {})
        assert result["task_id"] == "T001"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "claw-eval batch_worker"
        assert span.attributes[GEN_AI_SPAN_KIND] == "ENTRY"
        assert span.attributes["claw_eval.command"] == "batch_worker"
        assert span.attributes["claw_eval.task_dir"] == "/path/to/task"
        assert span.attributes["claw_eval.task_id"] == "T001"
        provider.shutdown()

    def test_task_dir_from_kwargs(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunSingleTaskWrapper(tracer)

        def fake_wrapped(task_dir="", **kwargs):
            return {"task_id": "T002"}

        wrapper(fake_wrapped, None, (), {"task_dir": "/my/task"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["claw_eval.task_dir"] == "/my/task"
        provider.shutdown()

    def test_non_dict_result_no_task_id(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunSingleTaskWrapper(tracer)

        def fake_wrapped(*args, **kwargs):
            return "not a dict"

        result = wrapper(fake_wrapped, None, ("dir",), {})
        assert result == "not a dict"
        spans = exporter.get_finished_spans()
        assert "claw_eval.task_id" not in spans[0].attributes
        provider.shutdown()

    def test_error_records_exception(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunSingleTaskWrapper(tracer)

        def fake_wrapped(*args, **kwargs):
            raise ValueError("task failed")

        with pytest.raises(ValueError, match="task failed"):
            wrapper(fake_wrapped, None, ("dir",), {})

        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
        provider.shutdown()

    def test_no_task_dir_arg(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunSingleTaskWrapper(tracer)

        def fake_wrapped(*args, **kwargs):
            return {}

        wrapper(fake_wrapped, None, (), {})
        spans = exporter.get_finished_spans()
        # Should not have task_dir attribute when empty string
        assert "claw_eval.task_dir" not in spans[0].attributes
        provider.shutdown()


# ===================================================================
# RunTaskWrapper
# ===================================================================


class TestRunTaskWrapper:
    """Tests for RunTaskWrapper."""

    def test_creates_agent_span(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)
        task = TaskDefinition(
            task_id="T001", prompt=Prompt(text="Do something")
        )

        class FakeProvider:
            model_id = "gpt-4o"

            def chat(self, messages, *args, **kwargs):
                return (
                    Message(
                        role="assistant",
                        content=[ContentBlock(type="text", text="Done")],
                    ),
                    Usage(input_tokens=100, output_tokens=50),
                )

        prov = FakeProvider()

        def fake_run_task(t, p, *args, **kwargs):
            # Simulate a chat call
            p.chat(
                [
                    Message(
                        role="user",
                        content=[ContentBlock(type="text", text="Hi")],
                    )
                ]
            )
            return {"task_id": "T001"}

        result = wrapper(fake_run_task, None, (task, prov), {})
        assert result["task_id"] == "T001"

        spans = exporter.get_finished_spans()
        # Should have at least the AGENT span (step spans may also exist)
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert len(agent_spans) == 1
        agent = agent_spans[0]
        assert agent.name == "invoke_agent claw-eval"
        assert agent.attributes[GEN_AI_FRAMEWORK] == "claw-eval"
        assert agent.attributes["claw_eval.task_id"] == "T001"
        assert agent.attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4o"
        provider.shutdown()

    def test_agent_description_set_from_task_prompt(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)
        task = TaskDefinition(
            task_id="T002", prompt=Prompt(text="Evaluate this code")
        )

        class FakeProvider:
            model_id = "gpt-4"

            def chat(self, messages, *args, **kwargs):
                return (
                    Message(role="assistant", content=[]),
                    Usage(),
                )

        prov = FakeProvider()

        def fake_run_task(t, p, *a, **kw):
            return {}

        wrapper(fake_run_task, None, (task, prov), {})
        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert (
            agent_spans[0].attributes[GenAI.GEN_AI_AGENT_DESCRIPTION]
            == "Evaluate this code"
        )
        provider.shutdown()

    def test_error_records_exception(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)
        task = TaskDefinition(task_id="T003")

        class FakeProvider:
            model_id = "gpt-4o"

            def chat(self, messages, *args, **kwargs):
                return Message(), Usage()

        prov = FakeProvider()

        def fake_run_task(t, p, *a, **kw):
            raise RuntimeError("agent crashed")

        with pytest.raises(RuntimeError, match="agent crashed"):
            wrapper(fake_run_task, None, (task, prov), {})

        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert agent_spans[0].status.status_code == StatusCode.ERROR
        provider.shutdown()

    def test_none_provider(self):
        """Should not raise when provider is None."""
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)
        task = TaskDefinition(task_id="T004")

        def fake_run_task(t, p, *a, **kw):
            return {}

        wrapper(fake_run_task, None, (task, None), {})
        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert len(agent_spans) == 1
        # No model attribute should be set
        assert GenAI.GEN_AI_REQUEST_MODEL not in agent_spans[0].attributes
        provider.shutdown()

    def test_none_task(self):
        """Should not raise when task is None."""
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)

        class FakeProvider:
            model_id = "gpt-4o"

            def chat(self, messages, *args, **kwargs):
                return Message(), Usage()

        prov = FakeProvider()

        def fake_run_task(t, p, *a, **kw):
            return {}

        wrapper(fake_run_task, None, (None, prov), {})
        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert agent_spans[0].attributes["claw_eval.task_id"] == "unknown"
        provider.shutdown()

    def test_total_turns_attribute(self):
        """When steps are rotated inside the agent, total_turns should be set."""
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)
        task = TaskDefinition(task_id="T005", prompt=Prompt(text="multi step"))

        class FakeProvider:
            model_id = "gpt-4o"

            def chat(self, messages, *args, **kwargs):
                return (
                    Message(
                        role="assistant",
                        content=[ContentBlock(type="text", text="ok")],
                    ),
                    Usage(input_tokens=50, output_tokens=25),
                )

        prov = FakeProvider()

        def fake_run_task(t, p, *a, **kw):
            # Simulate multiple step rotations via _step_counter
            _step_counter.set(3)
            return {}

        wrapper(fake_run_task, None, (task, prov), {})
        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert agent_spans[0].attributes["claw_eval.total_turns"] == 3
        provider.shutdown()

    def test_kwargs_task_and_provider(self):
        """RunTaskWrapper should extract task/provider from kwargs."""
        tracer, exporter, provider = _make_tracer()
        wrapper = RunTaskWrapper(tracer)
        task = TaskDefinition(task_id="T_kw")

        class FakeProvider:
            model_id = "model-kw"

            def chat(self, messages, *args, **kwargs):
                return Message(), Usage()

        prov = FakeProvider()

        def fake_run_task(task=None, provider=None, **kw):
            return {}

        wrapper(fake_run_task, None, (), {"task": task, "provider": prov})
        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "AGENT"
        ]
        assert agent_spans[0].attributes["claw_eval.task_id"] == "T_kw"
        assert (
            agent_spans[0].attributes[GenAI.GEN_AI_REQUEST_MODEL] == "model-kw"
        )
        provider.shutdown()


# ===================================================================
# _install_provider_chat_capture_shim
# ===================================================================


class TestInstallProviderChatCaptureShim:
    """Tests for _install_provider_chat_capture_shim."""

    def test_installs_shim_on_provider(self):
        class FakeProvider:
            def chat(self, messages, *args, **kwargs):
                return Message(), Usage()

        prov = FakeProvider()
        _install_provider_chat_capture_shim(prov)
        assert hasattr(prov.chat, "_claw_eval_capture_shim")
        assert prov.chat._claw_eval_capture_shim is True

    def test_idempotent(self):
        """Second install should not double-wrap."""

        class FakeProvider:
            def chat(self, messages, *args, **kwargs):
                return Message(), Usage()

        prov = FakeProvider()
        _install_provider_chat_capture_shim(prov)
        first_chat = prov.chat
        _install_provider_chat_capture_shim(prov)
        assert prov.chat is first_chat

    def test_none_provider(self):
        """Should be a safe no-op."""
        _install_provider_chat_capture_shim(None)

    def test_shim_captures_tokens(self):
        class FakeProvider:
            def chat(self, messages, *args, **kwargs):
                return (
                    Message(
                        role="assistant",
                        content=[ContentBlock(type="text", text="hi")],
                    ),
                    Usage(input_tokens=100, output_tokens=50),
                )

        prov = FakeProvider()
        _install_provider_chat_capture_shim(prov)

        capture = {
            "input_tokens": 0,
            "output_tokens": 0,
            "system_instructions": "",
            "input_messages_str": "",
            "last_response_str": "",
            "task_prompt": "",
            "first_call_done": False,
        }
        tok = _agent_capture.set(capture)
        try:
            prov.chat([])
            assert capture["input_tokens"] == 100
            assert capture["output_tokens"] == 50
        finally:
            _agent_capture.reset(tok)

    def test_shim_skips_during_compact(self):
        """Token accumulation should be skipped when compact_depth > 0."""

        class FakeProvider:
            def chat(self, messages, *args, **kwargs):
                return (
                    Message(role="assistant", content=[]),
                    Usage(input_tokens=200, output_tokens=100),
                )

        prov = FakeProvider()
        _install_provider_chat_capture_shim(prov)

        capture = {
            "input_tokens": 0,
            "output_tokens": 0,
            "system_instructions": "",
            "input_messages_str": "",
            "last_response_str": "",
            "task_prompt": "",
            "first_call_done": False,
        }
        tok_cap = _agent_capture.set(capture)
        tok_depth = _compact_depth.set(1)
        try:
            prov.chat([])
            assert capture["input_tokens"] == 0
            assert capture["output_tokens"] == 0
        finally:
            _agent_capture.reset(tok_cap)
            _compact_depth.reset(tok_depth)

    def test_shim_captures_system_prompt_on_first_call(self):
        class FakeProvider:
            def chat(self, messages, *args, **kwargs):
                return (
                    Message(role="assistant", content=[]),
                    Usage(input_tokens=10, output_tokens=5),
                )

        prov = FakeProvider()
        _install_provider_chat_capture_shim(prov)

        capture = {
            "input_tokens": 0,
            "output_tokens": 0,
            "system_instructions": "",
            "input_messages_str": "",
            "last_response_str": "",
            "task_prompt": "",
            "first_call_done": False,
        }
        tok = _agent_capture.set(capture)
        try:
            messages = [
                Message(
                    role="system",
                    content=[ContentBlock(type="text", text="Be helpful")],
                ),
                Message(
                    role="user",
                    content=[ContentBlock(type="text", text="Hello")],
                ),
            ]
            prov.chat(messages)
            assert capture["system_instructions"] == "Be helpful"
            assert capture["first_call_done"] is True
            # input_messages_str should exclude the system message
            parsed = json.loads(capture["input_messages_str"])
            assert len(parsed) == 1
            assert parsed[0]["role"] == "user"
        finally:
            _agent_capture.reset(tok)

    def test_shim_captures_tool_definitions(self):
        class FakeProvider:
            def chat(self, messages, *args, **kwargs):
                return (
                    Message(role="assistant", content=[]),
                    Usage(input_tokens=10, output_tokens=5),
                )

        prov = FakeProvider()
        _install_provider_chat_capture_shim(prov)

        capture = {
            "input_tokens": 0,
            "output_tokens": 0,
            "system_instructions": "",
            "input_messages_str": "",
            "last_response_str": "",
            "task_prompt": "",
            "first_call_done": False,
        }
        tools = [ToolSpec(name="bash", description="Run bash")]
        tok_cap = _agent_capture.set(capture)
        tok_tools = _agent_tool_definitions.set("")
        try:
            prov.chat([], tools)
            tool_defs = _agent_tool_definitions.get("")
            assert tool_defs != ""
            parsed = json.loads(tool_defs)
            assert parsed[0]["name"] == "bash"
        finally:
            _agent_capture.reset(tok_cap)
            _agent_tool_definitions.reset(tok_tools)


# ===================================================================
# ProviderChatWrapper
# ===================================================================


class TestProviderChatWrapper:
    """Tests for ProviderChatWrapper."""

    def test_rotates_step_inside_agent_run(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ProviderChatWrapper(tracer)

        tok_agent = _in_agent_run.set(True)
        tok_cnt = _step_counter.set(0)
        _current_step_span.set(None)
        _current_step_token.set(None)

        def fake_chat(*args, **kwargs):
            return Message(), Usage()

        try:
            wrapper(fake_chat, None, ([],), {})
            assert _step_counter.get(0) == 1

            wrapper(fake_chat, None, ([],), {})
            assert _step_counter.get(0) == 2

            _end_current_step()
        finally:
            _in_agent_run.reset(tok_agent)
            _step_counter.reset(tok_cnt)
            _current_step_span.set(None)
            _current_step_token.set(None)
            provider.shutdown()

    def test_no_rotation_outside_agent_run(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ProviderChatWrapper(tracer)

        tok_agent = _in_agent_run.set(False)
        tok_cnt = _step_counter.set(0)

        def fake_chat(*args, **kwargs):
            return Message(), Usage()

        try:
            wrapper(fake_chat, None, ([],), {})
            assert _step_counter.get(0) == 0
        finally:
            _in_agent_run.reset(tok_agent)
            _step_counter.reset(tok_cnt)
            provider.shutdown()

    def test_no_rotation_during_compact(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ProviderChatWrapper(tracer)

        tok_agent = _in_agent_run.set(True)
        tok_cnt = _step_counter.set(0)
        tok_depth = _compact_depth.set(1)
        _current_step_span.set(None)
        _current_step_token.set(None)

        def fake_chat(*args, **kwargs):
            return Message(), Usage()

        try:
            wrapper(fake_chat, None, ([],), {})
            assert _step_counter.get(0) == 0
        finally:
            _in_agent_run.reset(tok_agent)
            _step_counter.reset(tok_cnt)
            _compact_depth.reset(tok_depth)
            _current_step_span.set(None)
            _current_step_token.set(None)
            provider.shutdown()


# ===================================================================
# DoAutoCompactWrapper
# ===================================================================


class TestDoAutoCompactWrapper:
    """Tests for DoAutoCompactWrapper."""

    def test_creates_chain_span(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = DoAutoCompactWrapper(tracer)

        def fake_compact(*args, **kwargs):
            return "compacted"

        result = wrapper(fake_compact, None, (), {})
        assert result == "compacted"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "compact"
        assert span.attributes[GEN_AI_SPAN_KIND] == "CHAIN"
        assert span.attributes[GEN_AI_FRAMEWORK] == "claw-eval"
        assert span.attributes["claw_eval.compact.layer"] == "auto"
        provider.shutdown()

    def test_manual_layer_when_focus_provided(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = DoAutoCompactWrapper(tracer)

        def fake_compact(*args, **kwargs):
            return "compacted"

        wrapper(fake_compact, None, (), {"focus": "some_focus"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["claw_eval.compact.layer"] == "manual"
        provider.shutdown()

    def test_increments_compact_depth(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = DoAutoCompactWrapper(tracer)
        observed_depth = []

        def fake_compact(*args, **kwargs):
            observed_depth.append(_compact_depth.get(0))
            return "ok"

        wrapper(fake_compact, None, (), {})
        assert observed_depth == [1]
        # After the wrapper returns, depth should be restored
        assert _compact_depth.get(0) == 0
        provider.shutdown()

    def test_nested_compact_depth(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = DoAutoCompactWrapper(tracer)
        observed_depths = []

        def fake_compact_outer(*args, **kwargs):
            observed_depths.append(("outer", _compact_depth.get(0)))

            # Simulate a nested compact call
            def fake_compact_inner(*a, **kw):
                observed_depths.append(("inner", _compact_depth.get(0)))
                return "inner_result"

            wrapper(fake_compact_inner, None, (), {})
            return "outer_result"

        wrapper(fake_compact_outer, None, (), {})
        assert observed_depths == [("outer", 1), ("inner", 2)]
        assert _compact_depth.get(0) == 0
        provider.shutdown()

    def test_error_records_exception(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = DoAutoCompactWrapper(tracer)

        def fake_compact(*args, **kwargs):
            raise RuntimeError("compact failed")

        with pytest.raises(RuntimeError, match="compact failed"):
            wrapper(fake_compact, None, (), {})

        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
        # Depth should still be restored
        assert _compact_depth.get(0) == 0
        provider.shutdown()


# ===================================================================
# ToolDispatchWrapper
# ===================================================================


class TestToolDispatchWrapper:
    """Tests for ToolDispatchWrapper."""

    def test_creates_tool_span(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_001", input={"cmd": "ls"})

        result_block = ToolResultBlock(
            content=[ContentBlock(type="text", text="file1.txt")]
        )
        event = DispatchEvent(latency_ms=42.0, response_status=200)

        def fake_dispatch(tu, *args, **kwargs):
            return result_block, event

        result = wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})
        assert result == (result_block, event)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "execute_tool bash"
        assert span.attributes[GEN_AI_SPAN_KIND] == "TOOL"
        assert span.attributes[GEN_AI_FRAMEWORK] == "claw-eval"
        assert span.attributes[GenAI.GEN_AI_TOOL_NAME] == "bash"
        assert span.attributes[GenAI.GEN_AI_TOOL_TYPE] == "function"
        assert span.attributes[GenAI.GEN_AI_TOOL_CALL_ID] == "tu_001"
        provider.shutdown()

    def test_tool_input_serialized(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_002", input={"cmd": "echo hi"})

        def fake_dispatch(tu, *args, **kwargs):
            return ToolResultBlock(), DispatchEvent()

        wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})
        spans = exporter.get_finished_spans()
        args_attr = spans[0].attributes.get(GEN_AI_TOOL_CALL_ARGUMENTS)
        assert args_attr is not None
        parsed = json.loads(args_attr)
        assert parsed["cmd"] == "echo hi"
        provider.shutdown()

    def test_tool_result_text_captured(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_003")

        result_block = ToolResultBlock(
            content=[ContentBlock(type="text", text="output data")]
        )
        event = DispatchEvent(latency_ms=10.0, response_status=200)

        def fake_dispatch(tu, *args, **kwargs):
            return result_block, event

        wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes[GEN_AI_TOOL_CALL_RESULT] == "output data"
        provider.shutdown()

    def test_sandbox_dispatcher_attributes(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_004")

        # Instance with _http attr signals sandbox dispatcher
        instance = MagicMock()
        instance._http = True
        instance._sandbox_url = "http://sandbox:8080"

        def fake_dispatch(tu, *args, **kwargs):
            return ToolResultBlock(), DispatchEvent()

        wrapper(fake_dispatch, instance, (tool_use,), {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["claw_eval.sandbox.remote"] is True
        provider.shutdown()

    def test_sandbox_dispatcher_no_url(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_005")

        instance = MagicMock()
        instance._http = True
        instance._sandbox_url = None

        def fake_dispatch(tu, *args, **kwargs):
            return ToolResultBlock(), DispatchEvent()

        wrapper(fake_dispatch, instance, (tool_use,), {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["claw_eval.sandbox.remote"] is False
        provider.shutdown()

    def test_dispatch_guard_prevents_double_span(self):
        """When _in_tool_dispatch is True, wrapper should pass through."""
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_006")
        guard_tok = _in_tool_dispatch.set(True)

        call_count = 0

        def fake_dispatch(tu, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return ToolResultBlock(), DispatchEvent()

        try:
            wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})
            # Should have called the wrapped function
            assert call_count == 1
            # But no span should be created
            spans = exporter.get_finished_spans()
            assert len(spans) == 0
        finally:
            _in_tool_dispatch.reset(guard_tok)
            provider.shutdown()

    def test_error_records_exception(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_007")

        def fake_dispatch(tu, *args, **kwargs):
            raise RuntimeError("dispatch failed")

        with pytest.raises(RuntimeError, match="dispatch failed"):
            wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})

        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
        # Guard should be reset even on error
        assert _in_tool_dispatch.get(False) is False
        provider.shutdown()

    def test_tool_use_from_kwargs(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="python", id="tu_008")

        def fake_dispatch(tool_use=None, *args, **kwargs):
            return ToolResultBlock(), DispatchEvent()

        wrapper(fake_dispatch, MagicMock(spec=[]), (), {"tool_use": tool_use})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes[GenAI.GEN_AI_TOOL_NAME] == "python"
        provider.shutdown()

    def test_no_tool_use_arg(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)

        def fake_dispatch(*args, **kwargs):
            return ToolResultBlock(), DispatchEvent()

        wrapper(fake_dispatch, MagicMock(spec=[]), (), {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes[GenAI.GEN_AI_TOOL_NAME] == "unknown"
        provider.shutdown()

    def test_tool_definitions_propagated(self):
        """When _agent_tool_definitions is set, TOOL span should carry it."""
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_009")
        defs_json = json.dumps([{"type": "function", "name": "bash"}])
        tok_defs = _agent_tool_definitions.set(defs_json)

        def fake_dispatch(tu, *args, **kwargs):
            return ToolResultBlock(), DispatchEvent()

        try:
            wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})
            spans = exporter.get_finished_spans()
            assert spans[0].attributes[GEN_AI_TOOL_DEFINITIONS] == defs_json
        finally:
            _agent_tool_definitions.reset(tok_defs)
            provider.shutdown()

    def test_error_in_dispatch_result(self):
        """When tool_result.is_error is True, span status should be ERROR."""
        tracer, exporter, provider = _make_tracer()
        wrapper = ToolDispatchWrapper(tracer)
        tool_use = ToolUse(name="bash", id="tu_010")

        error_result = ToolResultBlock(
            content=[ContentBlock(type="text", text="error message")],
            is_error=True,
        )
        event = DispatchEvent(latency_ms=5.0, response_status=500)

        def fake_dispatch(tu, *args, **kwargs):
            return error_result, event

        wrapper(fake_dispatch, MagicMock(spec=[]), (tool_use,), {})
        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
        assert spans[0].attributes["http.response.status_code"] == 500
        provider.shutdown()


# ===================================================================
# _extract_dispatch_attrs
# ===================================================================


class TestExtractDispatchAttrs:
    """Tests for _extract_dispatch_attrs."""

    def test_with_valid_tuple(self):
        span = MagicMock()
        tool_result = ToolResultBlock(
            content=[ContentBlock(type="text", text="output")],
            is_error=False,
        )
        event = DispatchEvent(latency_ms=42.0, response_status=200)

        _extract_dispatch_attrs(span, (tool_result, event))
        span.set_attribute.assert_any_call(
            "claw_eval.dispatch.latency_ms", 42.0
        )
        span.set_attribute.assert_any_call("http.response.status_code", 200)
        span.set_attribute.assert_any_call(GEN_AI_TOOL_CALL_RESULT, "output")

    def test_with_error_result(self):
        span = MagicMock()
        tool_result = ToolResultBlock(is_error=True)
        event = DispatchEvent()

        _extract_dispatch_attrs(span, (tool_result, event))
        span.set_status.assert_called_once()

    def test_not_tuple(self):
        span = MagicMock()
        _extract_dispatch_attrs(span, "not a tuple")
        span.set_attribute.assert_not_called()

    def test_short_tuple(self):
        span = MagicMock()
        _extract_dispatch_attrs(span, ("only one",))
        span.set_attribute.assert_not_called()

    def test_no_latency(self):
        span = MagicMock()
        tool_result = ToolResultBlock()
        event = MagicMock(spec=[])  # no latency_ms attr

        _extract_dispatch_attrs(span, (tool_result, event))
        # Should not set latency attribute
        for call in span.set_attribute.call_args_list:
            assert call[0][0] != "claw_eval.dispatch.latency_ms"


# ===================================================================
# JudgeWrapper
# ===================================================================


class TestJudgeWrapper:
    """Tests for JudgeWrapper."""

    def test_suppresses_and_returns_result(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = JudgeWrapper(tracer, "evaluate")

        def fake_evaluate(*args, **kwargs):
            return {"score": 0.9}

        result = wrapper(fake_evaluate, None, (), {})
        assert result == {"score": 0.9}

        # JudgeWrapper does NOT create a span
        spans = exporter.get_finished_spans()
        assert len(spans) == 0
        provider.shutdown()

    def test_error_still_detaches(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = JudgeWrapper(tracer, "evaluate_actions")

        def fake_evaluate(*args, **kwargs):
            raise RuntimeError("judge error")

        with pytest.raises(RuntimeError, match="judge error"):
            wrapper(fake_evaluate, None, (), {})

        # No span created even on error
        spans = exporter.get_finished_spans()
        assert len(spans) == 0
        provider.shutdown()

    def test_different_method_names(self):
        tracer, exporter, provider = _make_tracer()

        for method in ("evaluate", "evaluate_actions", "evaluate_visual"):
            wrapper = JudgeWrapper(tracer, method)

            def fake(*a, **kw):
                return {"score": 0.5}

            result = wrapper(fake, None, (), {})
            assert result == {"score": 0.5}

        provider.shutdown()


# ===================================================================
# GetGraderWrapper
# ===================================================================


class TestGetGraderWrapper:
    """Tests for GetGraderWrapper."""

    def test_wraps_grader_eval_methods(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = GetGraderWrapper(tracer)

        class MyGrader:
            def _llm_score_classifications(self, *a, **kw):
                return {"classifications": []}

        def fake_get_grader(*args, **kwargs):
            return MyGrader()

        grader = wrapper(fake_get_grader, None, (), {})
        # The method should have been wrapped
        assert hasattr(
            MyGrader._llm_score_classifications, "_claw_eval_judge_wrapped"
        )
        # The grader should still work
        result = grader._llm_score_classifications()
        assert result == {"classifications": []}
        provider.shutdown()

    def test_returns_grader_on_wrap_failure(self):
        """Even if wrapping fails, the grader should still be returned."""
        tracer, exporter, provider = _make_tracer()
        wrapper = GetGraderWrapper(tracer)

        class WeirdGrader:
            pass

        def fake_get_grader(*args, **kwargs):
            return WeirdGrader()

        grader = wrapper(fake_get_grader, None, (), {})
        assert isinstance(grader, WeirdGrader)
        provider.shutdown()


# ===================================================================
# LoadPeerGraderWrapper
# ===================================================================


class TestLoadPeerGraderWrapper:
    """Tests for LoadPeerGraderWrapper."""

    def test_wraps_peer_grader_class(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = LoadPeerGraderWrapper(tracer)

        class PeerGrader:
            def _llm_score_classifications(self, *a, **kw):
                return {"peer": True}

        def fake_load_peer(*args, **kwargs):
            return PeerGrader

        cls = wrapper(fake_load_peer, None, (), {})
        assert cls is PeerGrader
        assert hasattr(
            PeerGrader._llm_score_classifications, "_claw_eval_judge_wrapped"
        )
        provider.shutdown()

    def test_returns_class_on_wrap_failure(self):
        tracer, exporter, provider = _make_tracer()
        wrapper = LoadPeerGraderWrapper(tracer)

        class SimplePeerGrader:
            pass

        def fake_load_peer(*args, **kwargs):
            return SimplePeerGrader

        cls = wrapper(fake_load_peer, None, (), {})
        assert cls is SimplePeerGrader
        provider.shutdown()


# ===================================================================
# _wrap_grader_eval_methods
# ===================================================================


class TestWrapGraderEvalMethods:
    """Tests for _wrap_grader_eval_methods."""

    def test_wraps_matching_method(self):
        tracer, _, provider = _make_tracer()

        class Grader:
            def _llm_score_classifications(self):
                return "result"

        _wrap_grader_eval_methods(Grader, tracer)
        assert hasattr(
            Grader._llm_score_classifications, "_claw_eval_judge_wrapped"
        )
        provider.shutdown()

    def test_idempotent(self):
        tracer, _, provider = _make_tracer()

        class Grader:
            def _llm_score_classifications(self):
                return "result"

        _wrap_grader_eval_methods(Grader, tracer)
        # Capture the raw dict entry (the FunctionWrapper itself, not the
        # BoundFunctionWrapper descriptor returned by attribute access).
        first_raw = Grader.__dict__["_llm_score_classifications"]
        _wrap_grader_eval_methods(Grader, tracer)
        second_raw = Grader.__dict__["_llm_score_classifications"]
        # The marker prevents double-wrapping, so the underlying
        # FunctionWrapper in __dict__ must be the same object.
        assert second_raw is first_raw
        # And calling it should still work.
        g = Grader()
        assert g._llm_score_classifications() == "result"
        provider.shutdown()

    def test_none_class(self):
        tracer, _, provider = _make_tracer()
        _wrap_grader_eval_methods(None, tracer)
        provider.shutdown()

    def test_object_class(self):
        tracer, _, provider = _make_tracer()
        _wrap_grader_eval_methods(object, tracer)
        provider.shutdown()

    def test_walks_mro(self):
        tracer, _, provider = _make_tracer()

        class BaseGrader:
            def _llm_score_classifications(self):
                return "base"

        class DerivedGrader(BaseGrader):
            pass

        _wrap_grader_eval_methods(DerivedGrader, tracer)
        # Base class method should be wrapped via MRO walk
        assert hasattr(
            BaseGrader._llm_score_classifications, "_claw_eval_judge_wrapped"
        )
        provider.shutdown()

    def test_class_without_eval_method(self):
        tracer, _, provider = _make_tracer()

        class NoEvalGrader:
            def grade(self):
                return "score"

        _wrap_grader_eval_methods(NoEvalGrader, tracer)
        # Should not raise, no methods to wrap
        assert not hasattr(NoEvalGrader.grade, "_claw_eval_judge_wrapped")
        provider.shutdown()


# ===================================================================
# _maybe_suppress_llm_sdk
# ===================================================================


class TestMaybeSuppressLlmSdk:
    """Tests for _maybe_suppress_llm_sdk."""

    def test_returns_token(self):
        token = _maybe_suppress_llm_sdk()
        assert token is not None
        otel_context.detach(token)

    def test_suppression_key_set(self):
        from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

        token = _maybe_suppress_llm_sdk()
        try:
            val = otel_context.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
            assert val is True
        finally:
            otel_context.detach(token)


# ===================================================================
# _get_task_prompt
# ===================================================================


class TestGetTaskPrompt:
    """Tests for _get_task_prompt."""

    def test_with_prompt(self):
        task = TaskDefinition(
            task_id="T001", prompt=Prompt(text="Do the thing")
        )
        assert _get_task_prompt(task) == "Do the thing"

    def test_none_task(self):
        assert _get_task_prompt(None) == ""

    def test_none_prompt(self):
        task = MagicMock()
        task.prompt = None
        assert _get_task_prompt(task) == ""

    def test_empty_text(self):
        task = TaskDefinition(task_id="T002", prompt=Prompt(text=""))
        assert _get_task_prompt(task) == ""


# ===================================================================
# _populate_agent_span
# ===================================================================


class TestPopulateAgentSpan:
    """Tests for _populate_agent_span."""

    def test_with_tokens(self):
        span = MagicMock()
        capture = {
            "input_tokens": 500,
            "output_tokens": 200,
            "system_instructions": "Be helpful",
            "input_messages_str": '[{"role":"user","parts":[]}]',
            "last_response_str": '[{"role":"assistant","parts":[],"finish_reason":"stop"}]',
        }
        _populate_agent_span(span, capture, "prompt text")
        span.set_attribute.assert_any_call(
            GenAI.GEN_AI_USAGE_INPUT_TOKENS, 500
        )
        span.set_attribute.assert_any_call(
            GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, 200
        )

    def test_fallback_to_task_prompt(self):
        span = MagicMock()
        capture = {
            "input_tokens": 0,
            "output_tokens": 0,
            "system_instructions": "",
            "input_messages_str": "",
            "last_response_str": "",
        }
        _populate_agent_span(span, capture, "fallback prompt")
        # Should set input messages from task prompt
        found = False
        for call in span.set_attribute.call_args_list:
            if call[0][0] == GenAI.GEN_AI_INPUT_MESSAGES:
                found = True
                parsed = json.loads(call[0][1])
                assert parsed[0]["role"] == "user"
                assert parsed[0]["parts"][0]["content"] == "fallback prompt"
        assert found

    def test_no_tokens_no_attrs(self):
        span = MagicMock()
        capture = {
            "input_tokens": 0,
            "output_tokens": 0,
            "system_instructions": "",
            "input_messages_str": "",
            "last_response_str": "",
        }
        _populate_agent_span(span, capture, "")
        # No token or message attributes should be set
        for call in span.set_attribute.call_args_list:
            attr_name = call[0][0]
            assert attr_name not in (
                GenAI.GEN_AI_USAGE_INPUT_TOKENS,
                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS,
                GenAI.GEN_AI_INPUT_MESSAGES,
                GenAI.GEN_AI_OUTPUT_MESSAGES,
            )


# ===================================================================
# _populate_entry_span
# ===================================================================


class TestPopulateEntrySpan:
    """Tests for _populate_entry_span."""

    def test_with_captures(self):
        span = MagicMock()
        captures = [
            {
                "input_messages_str": '[{"role":"user","parts":[{"type":"text","content":"Hi"}]}]',
                "last_response_str": '[{"role":"assistant","parts":[],"finish_reason":"stop"}]',
                "task_prompt": "Hi",
            },
        ]
        _populate_entry_span(span, captures)
        input_set = False
        output_set = False
        for call in span.set_attribute.call_args_list:
            if call[0][0] == GenAI.GEN_AI_INPUT_MESSAGES:
                input_set = True
            if call[0][0] == GenAI.GEN_AI_OUTPUT_MESSAGES:
                output_set = True
        assert input_set
        assert output_set

    def test_empty_captures(self):
        span = MagicMock()
        _populate_entry_span(span, [])
        span.set_attribute.assert_not_called()

    def test_none_captures(self):
        span = MagicMock()
        _populate_entry_span(span, None)
        span.set_attribute.assert_not_called()

    def test_fallback_to_task_prompt(self):
        span = MagicMock()
        captures = [
            {
                "input_messages_str": "",
                "last_response_str": "",
                "task_prompt": "Fallback prompt",
            },
        ]
        _populate_entry_span(span, captures)
        found = False
        for call in span.set_attribute.call_args_list:
            if call[0][0] == GenAI.GEN_AI_INPUT_MESSAGES:
                found = True
                parsed = json.loads(call[0][1])
                assert parsed[0]["parts"][0]["content"] == "Fallback prompt"
        assert found

    def test_last_response_from_last_capture(self):
        span = MagicMock()
        captures = [
            {
                "input_messages_str": '[{"role":"user","parts":[]}]',
                "last_response_str": "first",
                "task_prompt": "",
            },
            {
                "input_messages_str": "",
                "last_response_str": "last",
                "task_prompt": "",
            },
        ]
        _populate_entry_span(span, captures)
        for call in span.set_attribute.call_args_list:
            if call[0][0] == GenAI.GEN_AI_OUTPUT_MESSAGES:
                assert call[0][1] == "last"


# ===================================================================
# Integration: full trace hierarchy via the instrument fixture
# ===================================================================


class TestFullTraceHierarchy:
    """Integration tests verifying the complete span hierarchy via instrument fixture."""

    def test_entry_agent_step_trace(self, instrument, span_exporter):
        """Full lifecycle: ENTRY -> AGENT -> STEP."""
        import claw_eval.cli as cli
        import claw_eval.runner.providers.openai_compat as oc

        provider = oc.OpenAICompatProvider()
        task = TaskDefinition(
            task_id="T_full", prompt=Prompt(text="Full test")
        )

        # Call cmd_run which internally calls run_task via the mock
        # The mock run_task calls provider.chat which triggers step rotation
        import claw_eval.runner.loop as loop

        # Simulate: cmd_run calls run_task
        def patched_cmd_run(*args, **kwargs):
            return loop.run_task(task, provider)

        # Temporarily replace the mock with our version
        (
            cli.cmd_run.__wrapped__
            if hasattr(cli.cmd_run, "__wrapped__")
            else None
        )

        # We can just call the instrumented cmd_run directly
        # because the mock claw_eval.cli.cmd_run is now wrapped
        # But we need it to invoke run_task. Let's just call run_task directly.
        loop.run_task(task, provider)

        spans = span_exporter.get_finished_spans()
        span_kinds = {s.attributes.get(GEN_AI_SPAN_KIND) for s in spans}
        # Should have at least AGENT and STEP spans
        assert "AGENT" in span_kinds
        assert "STEP" in span_kinds

    def test_tool_dispatch_produces_tool_span(self, instrument, span_exporter):
        """ToolDispatcher.dispatch should produce a TOOL span."""
        import claw_eval.runner.dispatcher as disp

        dispatcher = disp.ToolDispatcher()
        tool_use = ToolUse(name="bash", id="tu_int", input={"cmd": "echo hi"})
        dispatcher.dispatch(tool_use)

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "TOOL"
        ]
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes[GenAI.GEN_AI_TOOL_NAME] == "bash"

    def test_compact_produces_chain_span(self, instrument, span_exporter):
        """do_auto_compact should produce a CHAIN span."""
        import claw_eval.runner.compact as compact

        compact.do_auto_compact()

        spans = span_exporter.get_finished_spans()
        chain_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "CHAIN"
        ]
        assert len(chain_spans) == 1

    def test_judge_suppresses_spans(self, instrument, span_exporter):
        """LLMJudge methods should NOT produce spans."""
        import claw_eval.graders.llm_judge as lj

        judge = lj.LLMJudge()
        result = judge.evaluate()
        assert result == {"score": 0.9}

        result2 = judge.evaluate_actions()
        assert result2 == {"score": 0.8}

        result3 = judge.evaluate_visual()
        assert result3 == {"score": 0.7}

        spans = span_exporter.get_finished_spans()
        # No judge spans should be created
        assert len(spans) == 0

    def test_get_grader_wraps_eval_methods(self, instrument, span_exporter):
        """get_grader should wrap the returned grader's eval methods."""
        import claw_eval.graders.registry as reg

        grader = reg.get_grader()
        # The method should be wrapped with judge suppression
        result = grader._llm_score_classifications()
        assert result == {"classifications": []}
        # No spans should be created (suppressed)
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_load_peer_grader_wraps_class(self, instrument, span_exporter):
        """load_peer_grader should wrap the returned class's eval methods."""
        import claw_eval.graders.base as base

        cls = base.load_peer_grader()
        instance = cls()
        result = instance._llm_score_classifications()
        assert result == {"peer_classifications": []}

    def test_sandbox_dispatch_produces_tool_span(
        self, instrument, span_exporter
    ):
        """SandboxToolDispatcher.dispatch should produce a TOOL span."""
        import claw_eval.runner.sandbox_dispatcher as sdisp

        dispatcher = sdisp.SandboxToolDispatcher()
        tool_use = ToolUse(
            name="sandbox_bash", id="tu_sand", input={"cmd": "ls"}
        )
        dispatcher.dispatch(tool_use)

        spans = span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans if s.attributes.get(GEN_AI_SPAN_KIND) == "TOOL"
        ]
        assert len(tool_spans) == 1
        assert (
            tool_spans[0].attributes[GenAI.GEN_AI_TOOL_NAME] == "sandbox_bash"
        )
        assert tool_spans[0].attributes["claw_eval.sandbox.remote"] is True
