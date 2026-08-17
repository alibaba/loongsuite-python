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

"""Unit tests for the text_completion / responses request shapes.

The helpers landed with the DSPy-driven §6.1 decision-A closure: LiteLLM's
``text_completion`` / ``responses`` entry points share the completion
lifecycle but not the payload shape, so they need their own normalize /
create-invocation / apply-response helpers and the wrapper must dispatch by
``kind``. These tests exercise each helper directly and then round-trip a
fake ``litellm.text_completion`` / ``litellm.responses`` call through the
parameterized wrapper to verify the dispatch is wired end to end.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from opentelemetry.instrumentation.litellm._utils import (
    apply_litellm_responses_response_to_invocation,
    apply_litellm_text_completion_response_to_invocation,
    create_llm_invocation_from_litellm_responses,
    create_llm_invocation_from_litellm_text_completion,
    normalize_litellm_responses_kwargs,
    normalize_litellm_text_completion_kwargs,
)
from opentelemetry.instrumentation.litellm._wrapper import (
    CHAT_COMPLETION,
    RESPONSES,
    TEXT_COMPLETION,
    AsyncCompletionWrapper,
    CompletionWrapper,
    _CompletionAdviceState,
)


def _text_completion_signature(model, prompt, **kwargs):
    """Stand-in for ``litellm.text_completion``'s signature."""
    return model, prompt, kwargs


def _responses_signature(model, input, **kwargs):  # noqa: A002 (mirrors litellm)
    """Stand-in for ``litellm.responses``'s signature."""
    return model, input, kwargs


class TestNormalizeTextCompletionKwargs(unittest.TestCase):
    def test_positional_model_and_prompt_promoted_to_kwargs(self):
        result = normalize_litellm_text_completion_kwargs(
            _text_completion_signature,
            ("openai/gpt-3.5-turbo-instruct", "Say hi"),
            {"temperature": 0.2},
        )
        self.assertEqual(result["model"], "openai/gpt-3.5-turbo-instruct")
        self.assertEqual(result["prompt"], "Say hi")
        self.assertEqual(result["temperature"], 0.2)

    def test_kwargs_win_over_positional_defaults(self):
        result = normalize_litellm_text_completion_kwargs(
            _text_completion_signature,
            ("openai/gpt-3.5-turbo-instruct",),
            {"prompt": "kwargs prompt"},
        )
        self.assertEqual(result["prompt"], "kwargs prompt")


class TestNormalizeResponsesKwargs(unittest.TestCase):
    def test_input_positional_is_promoted(self):
        result = normalize_litellm_responses_kwargs(
            _responses_signature,
            ("openai/gpt-4o-mini", "Hi"),
            {"max_output_tokens": 32},
        )
        self.assertEqual(result["model"], "openai/gpt-4o-mini")
        self.assertEqual(result["input"], "Hi")
        self.assertEqual(result["max_output_tokens"], 32)


class TestCreateTextCompletionInvocation(unittest.TestCase):
    def test_string_prompt_becomes_single_user_message(self):
        invocation = create_llm_invocation_from_litellm_text_completion(
            model="openai/gpt-3.5-turbo-instruct",
            prompt="Explain retrieval",
            temperature=0.1,
        )

        self.assertEqual(invocation.request_model, "gpt-3.5-turbo-instruct")
        self.assertEqual(invocation.provider, "openai")
        self.assertEqual(invocation.operation_name, "text_completion")
        self.assertEqual(len(invocation.input_messages), 1)
        message = invocation.input_messages[0]
        self.assertEqual(message.role, "user")
        self.assertEqual(len(message.parts), 1)
        self.assertEqual(message.parts[0].content, "Explain retrieval")
        self.assertEqual(invocation.temperature, 0.1)

    def test_list_prompt_produces_one_message_with_many_parts(self):
        invocation = create_llm_invocation_from_litellm_text_completion(
            model="openai/gpt-3.5-turbo-instruct",
            prompt=["First", "", "Second"],
        )

        self.assertEqual(len(invocation.input_messages), 1)
        parts = invocation.input_messages[0].parts
        self.assertEqual([p.content for p in parts], ["First", "Second"])

    def test_non_string_prompt_yields_no_messages(self):
        invocation = create_llm_invocation_from_litellm_text_completion(
            model="openai/gpt-3.5-turbo-instruct",
            prompt=42,
        )
        self.assertEqual(invocation.input_messages, [])


class TestApplyTextCompletionResponse(unittest.TestCase):
    def _make_invocation(self):
        return create_llm_invocation_from_litellm_text_completion(
            model="openai/gpt-3.5-turbo-instruct",
            prompt="hi",
        )

    def test_choices_text_becomes_assistant_output(self):
        invocation = self._make_invocation()
        response = SimpleNamespace(
            id="resp-1",
            model="gpt-3.5-turbo-instruct",
            usage=SimpleNamespace(
                prompt_tokens=3, completion_tokens=5, total_tokens=8
            ),
            choices=[
                SimpleNamespace(text="Hello", finish_reason="stop"),
            ],
        )

        apply_litellm_text_completion_response_to_invocation(
            invocation, response
        )

        self.assertEqual(invocation.response_id, "resp-1")
        self.assertEqual(invocation.response_model_name, "gpt-3.5-turbo-instruct")
        self.assertEqual(invocation.input_tokens, 3)
        self.assertEqual(invocation.output_tokens, 5)
        self.assertEqual(invocation.finish_reasons, ["stop"])
        self.assertEqual(len(invocation.output_messages), 1)
        out = invocation.output_messages[0]
        self.assertEqual(out.role, "assistant")
        self.assertEqual(out.finish_reason, "stop")
        self.assertEqual(out.parts[0].content, "Hello")

    def test_missing_finish_reason_falls_back_to_stop(self):
        invocation = self._make_invocation()
        response = SimpleNamespace(
            choices=[SimpleNamespace(text="hi", finish_reason=None)]
        )
        apply_litellm_text_completion_response_to_invocation(
            invocation, response
        )
        self.assertEqual(
            invocation.output_messages[0].finish_reason, "stop"
        )


class TestCreateResponsesInvocation(unittest.TestCase):
    def test_string_input_becomes_user_message(self):
        invocation = create_llm_invocation_from_litellm_responses(
            model="openai/gpt-4o-mini",
            input="What is retrieval augmented generation?",
        )
        self.assertEqual(invocation.operation_name, "chat")
        self.assertEqual(len(invocation.input_messages), 1)
        self.assertEqual(invocation.input_messages[0].role, "user")

    def test_list_input_with_nested_content(self):
        invocation = create_llm_invocation_from_litellm_responses(
            model="openai/gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {"type": "input_text", "text": "world"},
                    ],
                },
                {"role": "assistant", "content": "prior turn"},
            ],
            instructions="Be terse.",
            max_output_tokens=64,
        )

        self.assertEqual(len(invocation.input_messages), 2)
        parts = invocation.input_messages[0].parts
        self.assertEqual([p.content for p in parts], ["hello", "world"])
        self.assertEqual(invocation.input_messages[1].role, "assistant")
        self.assertEqual(invocation.max_tokens, 64)
        self.assertEqual(
            [t.content for t in invocation.system_instruction], ["Be terse."]
        )


class TestApplyResponsesResponse(unittest.TestCase):
    def _make_invocation(self):
        return create_llm_invocation_from_litellm_responses(
            model="openai/gpt-4o-mini",
            input="hi",
        )

    def test_text_and_tool_call_output(self):
        invocation = self._make_invocation()
        response = SimpleNamespace(
            id="rsp-9",
            model="gpt-4o-mini",
            status="completed",
            usage=SimpleNamespace(
                input_tokens=4,
                output_tokens=7,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
            ),
            output=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "text", "text": "thinking"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "tool-1",
                    "name": "lookup",
                    "arguments": '{"q": "otel"}',
                },
            ],
        )

        apply_litellm_responses_response_to_invocation(invocation, response)

        self.assertEqual(invocation.response_id, "rsp-9")
        self.assertEqual(invocation.input_tokens, 4)
        self.assertEqual(invocation.output_tokens, 7)
        self.assertEqual(invocation.usage_cache_read_input_tokens, 2)
        self.assertEqual(invocation.finish_reasons, ["stop"])

        self.assertEqual(len(invocation.output_messages), 1)
        parts = invocation.output_messages[0].parts
        # Order preserved: reasoning, text, tool call.
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0].content, "thinking")
        self.assertEqual(parts[1].content, "hello")
        self.assertEqual(parts[2].name, "lookup")
        self.assertEqual(parts[2].arguments, {"q": "otel"})

    def test_status_incomplete_maps_to_length(self):
        invocation = self._make_invocation()
        response = SimpleNamespace(status="incomplete", output=[])
        apply_litellm_responses_response_to_invocation(invocation, response)
        self.assertEqual(invocation.finish_reasons, ["length"])


class _FakeHandler:
    """Minimal handler capturing lifecycle events without a real OTel provider."""

    def __init__(self):
        self.events: list[tuple[str, Any]] = []

    def start_llm(self, invocation):
        self.events.append(("start", invocation))

    def stop_llm(self, invocation):
        self.events.append(("stop", invocation))

    def fail_llm(self, invocation, error):
        self.events.append(("fail", invocation, error))

    def abandon_llm(self, invocation):
        self.events.append(("abandon", invocation))

    def detach_llm_context(self, invocation):
        self.events.append(("detach", invocation))


class TestCompletionWrapperDispatch(unittest.TestCase):
    """The parameterized wrappers must route by ``kind`` end to end."""

    def test_text_completion_kind_uses_text_completion_helpers(self):
        handler = _FakeHandler()

        def fake_text_completion(model, prompt, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(text="pong", finish_reason="stop")],
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2
                ),
            )

        wrapper = CompletionWrapper(
            handler, fake_text_completion, kind=TEXT_COMPLETION
        )
        response = wrapper("openai/gpt-3.5-turbo-instruct", "ping")

        self.assertEqual(response.choices[0].text, "pong")
        kinds = [event[0] for event in handler.events]
        self.assertEqual(kinds, ["start", "stop"])

        invocation = handler.events[0][1]
        # Only the text_completion helper produces this operation name.
        self.assertEqual(invocation.operation_name, "text_completion")
        self.assertEqual(invocation.input_messages[0].parts[0].content, "ping")
        self.assertEqual(invocation.output_messages[0].parts[0].content, "pong")

    def test_responses_kind_uses_responses_helpers(self):
        handler = _FakeHandler()

        def fake_responses(model, input, **kwargs):  # noqa: A002
            return SimpleNamespace(
                status="completed",
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "answer"}],
                    }
                ],
                usage=SimpleNamespace(input_tokens=2, output_tokens=3),
            )

        wrapper = CompletionWrapper(
            handler, fake_responses, kind=RESPONSES
        )
        wrapper("openai/gpt-4o-mini", "question")

        invocation = handler.events[0][1]
        # Responses maps onto the chat operation but its unique usage field
        # names only apply here.
        self.assertEqual(invocation.operation_name, "chat")
        self.assertEqual(invocation.input_tokens, 2)
        self.assertEqual(invocation.output_tokens, 3)
        self.assertEqual(
            invocation.output_messages[0].parts[0].content, "answer"
        )

    def test_stream_kwarg_ignored_for_non_stream_kinds(self):
        """``supports_stream=False`` kinds must never enter the stream path."""
        handler = _FakeHandler()

        def fake_text_completion(model, prompt, **kwargs):
            # Even if the caller asked for stream=True, the wrapper must
            # short-circuit to the non-stream success path.
            return SimpleNamespace(
                choices=[SimpleNamespace(text="x", finish_reason="stop")]
            )

        wrapper = CompletionWrapper(
            handler, fake_text_completion, kind=TEXT_COMPLETION
        )
        wrapper("openai/gpt-3.5-turbo-instruct", "hi", stream=True)

        kinds = [event[0] for event in handler.events]
        self.assertEqual(kinds, ["start", "stop"])
        self.assertNotIn("detach", kinds)

    def test_error_path_records_failure(self):
        handler = _FakeHandler()

        def fake_text_completion(model, prompt, **kwargs):
            raise RuntimeError("boom")

        wrapper = CompletionWrapper(
            handler, fake_text_completion, kind=TEXT_COMPLETION
        )
        with self.assertRaises(RuntimeError):
            wrapper("openai/gpt-3.5-turbo-instruct", "hi")

        kinds = [event[0] for event in handler.events]
        self.assertEqual(kinds, ["start", "fail"])

    def test_default_kind_is_chat_completion(self):
        state = _CompletionAdviceState(invocation=None, is_stream=False)
        self.assertIs(state.kind, CHAT_COMPLETION)


class TestAsyncCompletionWrapperDispatch(unittest.TestCase):
    def test_async_responses_kind(self):
        handler = _FakeHandler()

        async def fake_aresponses(model, input, **kwargs):  # noqa: A002
            return SimpleNamespace(
                status="completed",
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

        wrapper = AsyncCompletionWrapper(
            handler, fake_aresponses, kind=RESPONSES
        )
        asyncio.run(wrapper("openai/gpt-4o-mini", "q"))

        kinds = [event[0] for event in handler.events]
        self.assertEqual(kinds, ["start", "stop"])
        invocation = handler.events[0][1]
        self.assertEqual(invocation.operation_name, "chat")
        self.assertEqual(
            invocation.output_messages[0].parts[0].content, "ok"
        )


if __name__ == "__main__":
    unittest.main()
