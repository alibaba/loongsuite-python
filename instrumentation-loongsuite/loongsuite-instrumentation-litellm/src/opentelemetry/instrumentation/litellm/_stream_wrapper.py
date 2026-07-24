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

"""
Stream wrapper for LiteLLM streaming responses.
"""

import logging
import timeit
from threading import Lock
from typing import Any, Iterator, Optional

from opentelemetry.instrumentation.litellm._utils import (
    extract_litellm_text_parts,
    get_litellm_value,
    parse_tool_call_arguments,
)
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.types import (
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
)

logger = logging.getLogger(__name__)


@hook_advice("litellm", "stream_chunk")
def _record_stream_chunk(accumulator: Any, chunk: Any) -> None:
    """Record probe-owned chunk state without affecting stream delivery."""
    accumulator.record_chunk(chunk)


@hook_advice("litellm", "stream_finalize")
def _invoke_stream_callback(
    callback: Any,
    span: Any,
    last_chunk: Any,
    error: Optional[BaseException],
) -> None:
    """Run the external finalization callback as fail-open probe advice."""
    callback(span, last_chunk, error)


class _StreamAccumulator:
    """Accumulate LiteLLM streaming deltas into GenAI output messages."""

    def __init__(self, invocation: Any = None):
        self.invocation = invocation
        self._choice_states: dict[int, dict[str, Any]] = {}

    def record_chunk(self, chunk: Any) -> None:
        choices = get_litellm_value(chunk, "choices") or []
        if not choices:
            return

        saw_token = False
        for default_index, choice in enumerate(choices):
            index = get_litellm_value(choice, "index", default_index)
            if not isinstance(index, int):
                index = default_index

            state = self._choice_states.setdefault(
                index,
                {
                    "role": "assistant",
                    "reasoning": [],
                    "content": [],
                    "finish_reason": None,
                    "tool_calls": {},
                },
            )

            finish_reason = get_litellm_value(choice, "finish_reason")
            if finish_reason:
                state["finish_reason"] = finish_reason

            delta = get_litellm_value(choice, "delta")
            if delta is None:
                continue

            role = get_litellm_value(delta, "role")
            if role:
                state["role"] = role

            content = get_litellm_value(delta, "content")
            content_parts = extract_litellm_text_parts(content)
            if content_parts:
                state["content"].extend(content_parts)
                saw_token = True

            reasoning_content = get_litellm_value(delta, "reasoning_content")
            if reasoning_content is None:
                reasoning_content = get_litellm_value(delta, "reasoning")
            reasoning_parts = extract_litellm_text_parts(reasoning_content)
            if reasoning_parts:
                state["reasoning"].extend(reasoning_parts)
                saw_token = True

            tool_calls = get_litellm_value(delta, "tool_calls")
            if tool_calls:
                saw_token = True
                self._record_tool_calls(state, tool_calls)

        if saw_token and self.invocation is not None:
            first_token_time = getattr(
                self.invocation, "monotonic_first_token_s", None
            )
            if first_token_time is None:
                self.invocation.monotonic_first_token_s = (
                    timeit.default_timer()
                )

    def get_output_messages(self) -> list[OutputMessage]:
        output_messages = []
        for index in sorted(self._choice_states):
            state = self._choice_states[index]
            parts = []
            reasoning = "".join(state["reasoning"])
            if reasoning:
                parts.append(Reasoning(content=reasoning))

            content = "".join(state["content"])
            if content:
                parts.append(Text(content=content))

            for tool_index in sorted(state["tool_calls"]):
                tool_call = state["tool_calls"][tool_index]
                arguments = parse_tool_call_arguments(
                    tool_call.get("arguments", "")
                )
                if (
                    tool_call.get("id")
                    or tool_call.get("name")
                    or arguments not in (None, "")
                ):
                    parts.append(
                        ToolCall(
                            id=tool_call.get("id"),
                            name=tool_call.get("name", ""),
                            arguments=arguments,
                        )
                    )

            if not parts:
                parts.append(Text(content=""))

            output_messages.append(
                OutputMessage(
                    role=state["role"] or "assistant",
                    parts=parts,
                    finish_reason=state["finish_reason"] or "stop",
                )
            )
        return output_messages

    def finish_reasons(self) -> list[str]:
        finish_reasons = []
        for index in sorted(self._choice_states):
            state = self._choice_states[index]
            if state["finish_reason"]:
                finish_reasons.append(state["finish_reason"])
        return finish_reasons

    @staticmethod
    def _record_tool_calls(
        state: dict[str, Any], tool_calls: list[Any]
    ) -> None:
        for fallback_index, tool_call in enumerate(tool_calls):
            tool_index = get_litellm_value(tool_call, "index", fallback_index)
            if not isinstance(tool_index, int):
                tool_index = fallback_index

            stored = state["tool_calls"].setdefault(
                tool_index,
                {"id": None, "name": "", "arguments": ""},
            )

            tool_id = get_litellm_value(tool_call, "id")
            if tool_id:
                stored["id"] = tool_id

            function = get_litellm_value(tool_call, "function")
            function_name = get_litellm_value(function, "name")
            if function_name:
                stored["name"] = function_name

            arguments = get_litellm_value(function, "arguments")
            if isinstance(arguments, str):
                stored["arguments"] += arguments
            elif arguments:
                logger.debug(
                    "Skipping non-string LiteLLM streamed tool-call arguments"
                )


class StreamWrapper:
    """
    Wrapper for synchronous streaming responses.
    Note: To avoid memory leaks, we only keep the last chunk instead of all chunks.
    This is sufficient for extracting usage information which is typically in the last chunk.

    Supports context manager protocol for reliable cleanup.
    """

    _warned_unclosed_stream = False

    def __init__(
        self,
        stream: Iterator[Any],
        span: Any,
        callback: callable,
        invocation: Any = None,
    ):
        self.stream = stream
        self.span = span
        self.callback = callback
        self._accumulator = _StreamAccumulator(invocation)
        self.last_chunk = None  # Only keep last chunk to avoid memory leak
        self.chunk_count = 0
        self._finalized = False
        self._finalize_lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)

            _record_stream_chunk(self._accumulator, chunk)

            # Only keep the last chunk (contains usage info)
            self.last_chunk = chunk
            self.chunk_count += 1

            return chunk
        except StopIteration:
            # Stream ended normally, finalize span
            self._finalize()
            raise
        except GeneratorExit:
            # Generator close is an early, successful termination signal.
            self._finalize()
            raise
        except BaseException as e:
            # Error during streaming
            logger.debug("Error during streaming: %s", e, exc_info=True)
            self._finalize(error=e)
            raise

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure finalization on context exit."""
        if exc_type is not None:
            # Exception occurred during iteration
            self._finalize(error=exc_val)
        else:
            # Normal exit (may have completed or early terminated)
            self._finalize()
        return False

    def close(self):
        """Explicitly close and finalize the stream."""
        self._finalize()

    def __del__(self):
        if getattr(self, "_finalized", True):
            return

        if not StreamWrapper._warned_unclosed_stream:
            StreamWrapper._warned_unclosed_stream = True
            logger.warning(
                "LiteLLM stream wrapper was garbage-collected before close; "
                "finalizing the span. Use a context manager or call close() "
                "when terminating streams early."
            )

        try:
            self._finalize()
        except Exception:
            pass

    def _close_stream(self) -> None:
        close = getattr(self.stream, "close", None)
        if not callable(close):
            return

        try:
            close()
        except Exception as exc:
            logger.debug(
                "Error closing LiteLLM stream: %s", exc, exc_info=True
            )

    def _finalize(self, error: Optional[BaseException] = None):
        """Finalize the span with data from last chunk."""
        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True
        try:
            self._close_stream()
        finally:
            try:
                # The callback is probe advice and must not affect stream cleanup.
                if self.callback:
                    _invoke_stream_callback(
                        self.callback,
                        self.span,
                        self.last_chunk,
                        error,
                    )
            finally:
                self.last_chunk = None

    def get_output_messages(self) -> list[OutputMessage]:
        return self._accumulator.get_output_messages()

    def finish_reasons(self) -> list[str]:
        return self._accumulator.finish_reasons()


class AsyncStreamWrapper:
    """
    Wrapper for asynchronous streaming responses.
    Note: To avoid memory leaks, we only keep the last chunk instead of all chunks.
    This is sufficient for extracting usage information which is typically in the last chunk.

    Important: AsyncStreamWrapper must be consumed within an async context that ensures
    finalization, either by:
    1. Using as an async context manager: async with response: ...
    2. Explicitly calling close() after iteration
    3. Letting the wrapper detect stream exhaustion
    """

    def __init__(
        self,
        stream,
        span: Any,
        callback: callable,
        invocation: Any = None,
    ):
        self.stream = stream
        self.span = span
        self.callback = callback
        self._accumulator = _StreamAccumulator(invocation)
        self.last_chunk = None  # Only keep last chunk to avoid memory leak
        self.chunk_count = 0
        self._finalized = False
        self._finalize_lock = Lock()
        self._stream_exhausted = False
        self._stream_closed = False

    def __aiter__(self):
        # Return an async generator that wraps the stream and ensures finalization
        return self._wrapped_iteration()

    async def _wrapped_iteration(self):
        """
        Async generator that wraps the underlying stream and ensures finalization.
        This approach guarantees that _finalize() is called when:
        1. The stream is exhausted normally
        2. An exception occurs
        3. The generator is closed early (via aclose())
        """
        error = None
        try:
            async for chunk in self.stream:
                _record_stream_chunk(self._accumulator, chunk)

                # Only keep the last chunk (contains usage info)
                self.last_chunk = chunk
                self.chunk_count += 1

                yield chunk

            # Stream exhausted normally
            logger.debug(
                "AsyncStreamWrapper: Stream completed (chunks: %s)",
                self.chunk_count,
            )
        except GeneratorExit:
            # ``aclose()`` injects GeneratorExit into this wrapper generator.
            raise
        except BaseException as e:
            # Error during streaming
            logger.debug(
                "AsyncStreamWrapper: Error during streaming: %s",
                e,
                exc_info=True,
            )
            error = e
            raise
        finally:
            # Always finalize, whether completed normally, with error, or closed early
            try:
                await self._aclose_stream()
            finally:
                self._finalize(error=error)

    async def __aenter__(self):
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure finalization on async context exit."""
        try:
            await self._aclose_stream()
        finally:
            if exc_type is not None:
                # Exception occurred during iteration
                self._finalize(error=exc_val)
            else:
                # Normal exit (may have completed or early terminated)
                self._finalize()
        return False

    async def aclose(self):
        """Explicitly close and finalize the async stream."""
        try:
            await self._aclose_stream()
        finally:
            self._finalize()

    def close(self):
        """Synchronous close method for compatibility."""
        try:
            self._close_stream()
        finally:
            self._finalize()

    def _close_stream(self) -> None:
        if self._stream_closed:
            return

        self._stream_closed = self._close_sync_stream()

    def _close_sync_stream(self) -> bool:
        close = getattr(self.stream, "close", None)
        if not callable(close):
            return False

        try:
            close()
        except Exception as exc:
            logger.debug(
                "Error closing LiteLLM async stream: %s",
                exc,
                exc_info=True,
            )
            return False
        return True

    async def _aclose_stream(self) -> None:
        if self._stream_closed:
            return

        aclose = getattr(self.stream, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as exc:
                logger.debug(
                    "Error closing LiteLLM async stream: %s",
                    exc,
                    exc_info=True,
                )
            else:
                self._stream_closed = True
                return

        self._stream_closed = self._close_sync_stream()

    def _finalize(self, error: Optional[BaseException] = None):
        """Finalize the span with data from last chunk."""
        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True
        try:
            if self.callback:
                _invoke_stream_callback(
                    self.callback,
                    self.span,
                    self.last_chunk,
                    error,
                )
        finally:
            self.last_chunk = None

    def get_output_messages(self) -> list[OutputMessage]:
        return self._accumulator.get_output_messages()

    def finish_reasons(self) -> list[str]:
        return self._accumulator.finish_reasons()
