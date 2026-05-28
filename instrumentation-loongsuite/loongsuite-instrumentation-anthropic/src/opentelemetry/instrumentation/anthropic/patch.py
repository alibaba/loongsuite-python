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

"""Patching functions for Anthropic instrumentation."""

from __future__ import annotations

import json
import timeit
from typing import TYPE_CHECKING, Any, Callable, Optional

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    LLMInvocation,
    MessagePart,
    OutputMessage,
    Text,
    ToolCall,
)

from .utils import (
    create_anthropic_invocation,
    is_streaming,
    populate_response,
)

if TYPE_CHECKING:
    from anthropic.resources.messages import AsyncMessages, Messages
    from anthropic.types import Message


def messages_create(
    handler: TelemetryHandler,
    content_capturing_mode: ContentCapturingMode,
) -> Callable[..., "Message"]:
    """Wrap the `create` method of the `Messages` class to trace it."""
    capture_content = content_capturing_mode != ContentCapturingMode.NO_CONTENT

    def traced_method(
        wrapped: Callable[..., "Message"],
        instance: "Messages",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> "Message":
        invocation = handler.start_llm(
            create_anthropic_invocation(kwargs, instance, capture_content)
        )

        try:
            result = wrapped(*args, **kwargs)

            if is_streaming(kwargs):
                return AnthropicStreamWrapper(
                    result, handler, invocation, capture_content
                )

            populate_response(invocation, result, capture_content)
            handler.stop_llm(invocation)
            return result

        except Exception as error:
            handler.fail_llm(
                invocation, Error(type=type(error), message=str(error))
            )
            raise

    return traced_method


def async_messages_create(
    handler: TelemetryHandler,
    content_capturing_mode: ContentCapturingMode,
) -> Callable[..., "Message"]:
    """Wrap the `create` method of the `AsyncMessages` class to trace it."""
    capture_content = content_capturing_mode != ContentCapturingMode.NO_CONTENT

    async def traced_method(
        wrapped: Callable[..., "Message"],
        instance: "AsyncMessages",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> "Message":
        invocation = handler.start_llm(
            create_anthropic_invocation(kwargs, instance, capture_content)
        )

        try:
            result = await wrapped(*args, **kwargs)

            if is_streaming(kwargs):
                return AsyncAnthropicStreamWrapper(
                    result, handler, invocation, capture_content
                )

            populate_response(invocation, result, capture_content)
            handler.stop_llm(invocation)
            return result

        except Exception as error:
            handler.fail_llm(
                invocation, Error(type=type(error), message=str(error))
            )
            raise

    return traced_method


class _ContentBlockAccumulator:
    """Accumulates content blocks from streaming events."""

    def __init__(self):
        self.text_parts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._current_block_type: str | None = None
        self._current_block_data: dict[str, Any] = {}

    def on_content_block_start(self, content_block: Any) -> None:
        """Handle a content_block_start event."""
        block_type = getattr(content_block, "type", None)
        self._current_block_type = block_type
        self._current_block_data = {}

        if block_type == "tool_use":
            self._current_block_data = {
                "name": getattr(content_block, "name", ""),
                "id": getattr(content_block, "id", None),
                "input_json": "",
            }

    def on_content_block_delta(self, delta: Any) -> None:
        """Handle a content_block_delta event."""
        delta_type = getattr(delta, "type", None)

        if delta_type == "text_delta":
            text = getattr(delta, "text", "")
            self.text_parts.append(text)
        elif delta_type == "input_json_delta":
            partial_json = getattr(delta, "partial_json", "")
            if self._current_block_data:
                self._current_block_data["input_json"] += partial_json

    def on_content_block_stop(self) -> None:
        """Handle a content_block_stop event."""
        if self._current_block_type == "tool_use" and self._current_block_data:
            input_json_str = self._current_block_data.get("input_json", "")
            try:
                arguments = json.loads(input_json_str) if input_json_str else None
            except (json.JSONDecodeError, ValueError):
                arguments = input_json_str

            self.tool_calls.append({
                "name": self._current_block_data.get("name", ""),
                "id": self._current_block_data.get("id"),
                "arguments": arguments,
            })

        self._current_block_type = None
        self._current_block_data = {}

    def build_output_messages(
        self, stop_reason: str | None
    ) -> list[OutputMessage]:
        """Build OutputMessage list from accumulated data."""
        parts: list[MessagePart] = []

        text = "".join(self.text_parts)
        if text:
            parts.append(Text(content=text))

        for tc in self.tool_calls:
            parts.append(
                ToolCall(
                    name=tc["name"],
                    id=tc.get("id"),
                    arguments=tc.get("arguments"),
                )
            )

        if not parts:
            return []

        return [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason=stop_reason or "error",
            )
        ]


class AnthropicStreamWrapper:
    """Wraps an Anthropic streaming response to capture telemetry."""

    def __init__(
        self,
        stream: Any,
        handler: TelemetryHandler,
        invocation: LLMInvocation,
        capture_content: bool,
    ):
        self.stream = stream
        self.handler = handler
        self.invocation = invocation
        self.capture_content = capture_content
        self._accumulator = _ContentBlockAccumulator()
        self._started = True
        self._response_model: str | None = None
        self._response_id: str | None = None
        self._stop_reason: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cache_creation_input_tokens: int | None = None
        self._cache_read_input_tokens: int | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        error = exc_val if exc_type else None
        self._cleanup(error)
        return False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self.stream)
            self._process_event(event)
            return event
        except StopIteration:
            self._cleanup()
            raise
        except Exception as error:
            self._cleanup(error)
            raise

    def close(self):
        if hasattr(self.stream, "close"):
            self.stream.close()
        self._cleanup()

    def _process_event(self, event: Any) -> None:
        """Process a single streaming event."""
        event_type = getattr(event, "type", None)

        # Record time to first token
        if (
            self.invocation.monotonic_first_token_s is None
            and event_type in ("content_block_delta",)
        ):
            self.invocation.monotonic_first_token_s = timeit.default_timer()

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message:
                self._response_model = getattr(message, "model", None)
                self._response_id = getattr(message, "id", None)
                usage = getattr(message, "usage", None)
                if usage:
                    self._input_tokens = getattr(usage, "input_tokens", None)
                    if hasattr(usage, "cache_creation_input_tokens"):
                        self._cache_creation_input_tokens = getattr(
                            usage, "cache_creation_input_tokens", None
                        )
                    if hasattr(usage, "cache_read_input_tokens"):
                        self._cache_read_input_tokens = getattr(
                            usage, "cache_read_input_tokens", None
                        )

        elif event_type == "content_block_start":
            content_block = getattr(event, "content_block", None)
            if content_block and self.capture_content:
                self._accumulator.on_content_block_start(content_block)

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and self.capture_content:
                self._accumulator.on_content_block_delta(delta)

        elif event_type == "content_block_stop":
            if self.capture_content:
                self._accumulator.on_content_block_stop()

        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                self._stop_reason = getattr(delta, "stop_reason", None)
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = getattr(usage, "output_tokens", None)

        elif event_type == "message_stop":
            pass  # Cleanup happens in __next__ StopIteration or __exit__

    def _cleanup(self, error: Optional[BaseException] = None) -> None:
        """Finalize the invocation with accumulated data."""
        if not self._started:
            return
        self._started = False

        self.invocation.response_model_name = self._response_model
        self.invocation.response_id = self._response_id
        self.invocation.input_tokens = self._input_tokens
        self.invocation.output_tokens = self._output_tokens

        if self._cache_creation_input_tokens:
            self.invocation.usage_cache_creation_input_tokens = (
                self._cache_creation_input_tokens
            )
        if self._cache_read_input_tokens:
            self.invocation.usage_cache_read_input_tokens = (
                self._cache_read_input_tokens
            )

        if self._stop_reason:
            self.invocation.finish_reasons = [self._stop_reason]

        if self.capture_content:
            self.invocation.output_messages = (
                self._accumulator.build_output_messages(self._stop_reason)
            )

        if error:
            self.handler.fail_llm(
                self.invocation,
                Error(type=type(error), message=str(error)),
            )
        else:
            self.handler.stop_llm(self.invocation)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying stream."""
        return getattr(self.stream, name)


class AsyncAnthropicStreamWrapper:
    """Wraps an Anthropic async streaming response to capture telemetry."""

    def __init__(
        self,
        stream: Any,
        handler: TelemetryHandler,
        invocation: LLMInvocation,
        capture_content: bool,
    ):
        self.stream = stream
        self.handler = handler
        self.invocation = invocation
        self.capture_content = capture_content
        self._accumulator = _ContentBlockAccumulator()
        self._started = True
        self._response_model: str | None = None
        self._response_id: str | None = None
        self._stop_reason: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cache_creation_input_tokens: int | None = None
        self._cache_read_input_tokens: int | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        error = exc_val if exc_type else None
        self._cleanup(error)
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self.stream.__anext__()
            self._process_event(event)
            return event
        except StopAsyncIteration:
            self._cleanup()
            raise
        except Exception as error:
            self._cleanup(error)
            raise

    async def close(self):
        if hasattr(self.stream, "close"):
            await self.stream.close()
        self._cleanup()

    def _process_event(self, event: Any) -> None:
        """Process a single streaming event - same logic as sync."""
        event_type = getattr(event, "type", None)

        if (
            self.invocation.monotonic_first_token_s is None
            and event_type in ("content_block_delta",)
        ):
            self.invocation.monotonic_first_token_s = timeit.default_timer()

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message:
                self._response_model = getattr(message, "model", None)
                self._response_id = getattr(message, "id", None)
                usage = getattr(message, "usage", None)
                if usage:
                    self._input_tokens = getattr(usage, "input_tokens", None)
                    if hasattr(usage, "cache_creation_input_tokens"):
                        self._cache_creation_input_tokens = getattr(
                            usage, "cache_creation_input_tokens", None
                        )
                    if hasattr(usage, "cache_read_input_tokens"):
                        self._cache_read_input_tokens = getattr(
                            usage, "cache_read_input_tokens", None
                        )

        elif event_type == "content_block_start":
            content_block = getattr(event, "content_block", None)
            if content_block and self.capture_content:
                self._accumulator.on_content_block_start(content_block)

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and self.capture_content:
                self._accumulator.on_content_block_delta(delta)

        elif event_type == "content_block_stop":
            if self.capture_content:
                self._accumulator.on_content_block_stop()

        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                self._stop_reason = getattr(delta, "stop_reason", None)
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = getattr(usage, "output_tokens", None)

    def _cleanup(self, error: Optional[BaseException] = None) -> None:
        """Finalize the invocation with accumulated data."""
        if not self._started:
            return
        self._started = False

        self.invocation.response_model_name = self._response_model
        self.invocation.response_id = self._response_id
        self.invocation.input_tokens = self._input_tokens
        self.invocation.output_tokens = self._output_tokens

        if self._cache_creation_input_tokens:
            self.invocation.usage_cache_creation_input_tokens = (
                self._cache_creation_input_tokens
            )
        if self._cache_read_input_tokens:
            self.invocation.usage_cache_read_input_tokens = (
                self._cache_read_input_tokens
            )

        if self._stop_reason:
            self.invocation.finish_reasons = [self._stop_reason]

        if self.capture_content:
            self.invocation.output_messages = (
                self._accumulator.build_output_messages(self._stop_reason)
            )

        if error:
            self.handler.fail_llm(
                self.invocation,
                Error(type=type(error), message=str(error)),
            )
        else:
            self.handler.stop_llm(self.invocation)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying stream."""
        return getattr(self.stream, name)
